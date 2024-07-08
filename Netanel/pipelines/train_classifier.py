import argparse
import json
import os
from collections import defaultdict
from sklearn.metrics import log_loss
from torch import topk
from training import losses
from training.datasets.classifier_dataset import DeepFakeClassifierDataset
from training.losses import WeightedLosses
from training.tools.config import load_config
from training.tools.utils import create_optimizer, AverageMeter
from training.transforms.albu import IsotropicResize
from training.zoo import classifiers

# Set environment variables to limit the number of threads
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import numpy as np
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter
from apex import amp
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

# Enable cuDNN benchmark mode for optimal performance
torch.backends.cudnn.benchmark = True

# Create training data transformations
def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ])

# Create validation data transformations
def create_val_transforms(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def main():
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument

    # Define command-line arguments
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--workers', type=int, default=6, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='classifier_')
    arg('--data-dir', type=str, default="/mnt/sota/datasets/deepfake")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--crops-dir', type=str, default='crops')
    arg('--label-smoothing', type=float, default=0.01)
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg('--freeze-epochs', type=int, default=0)
    arg("--local_rank", default=0, type=int)
    arg("--seed", default=777, type=int)
    arg("--padding-part", default=3, type=int)
    arg("--opt-level", default='O1', type=str)
    arg("--test_every", type=int, default=1)
    arg("--no-oversample", action="store_true")
    arg("--no-hardcore", action="store_true")
    arg("--only-changed-frames", action="store_true")

    # Parse arguments
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed training setup
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    # Load configuration file
    conf = load_config(args.config)

    # Initialize model
    model = classifiers.__dict__[conf['network']](encoder=conf['encoder'])
    model = model.cuda()

    if args.distributed:
        model = convert_syncbn_model(model)

    # Setup loss functions
    ohem = conf.get("ohem_samples", None)
    reduction = "mean" if not ohem else "none"
    loss_fn = []
    weights = []

    for loss_name, weight in conf["losses"].items():
        loss_fn.append(losses.__dict__[loss_name](reduction=reduction).cuda())
        weights.append(weight)

    loss = WeightedLosses(loss_fn, weights)
    loss_functions = {"classifier_loss": loss}

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)

    bce_best = 100
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    # Initialize datasets
    data_train = DeepFakeClassifierDataset(
        mode="train", oversample_real=not args.no_oversample, fold=args.fold,
        padding_part=args.padding_part, hardcore=not args.no_hardcore,
        crops_dir=args.crops_dir, data_path=args.data_dir,
        label_smoothing=args.label_smoothing, folds_csv=args.folds_csv,
        transforms=create_train_transforms(conf["size"]),
        normalize=conf.get("normalize", None)
    )

    data_val = DeepFakeClassifierDataset(
        mode="val", fold=args.fold, padding_part=args.padding_part,
        crops_dir=args.crops_dir, data_path=args.data_dir,
        folds_csv=args.folds_csv, transforms=create_val_transforms(conf["size"]),
        normalize=conf.get("normalize", None)
    )

    # Create data loaders
    val_data_loader = DataLoader(
        data_val, batch_size=batch_size * 2, num_workers=args.workers, shuffle=False, pin_memory=False
    )

    # Initialize SummaryWriter for TensorBoard logging
    os.makedirs(args.logdir, exist_ok=True)
    summary_writer = SummaryWriter(args.logdir + '/' + conf.get("prefix", args.prefix) + conf['encoder'] + "_" + str(args.fold))

    # Load checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                if not args.zero_score:
                    bce_best = checkpoint.get('bce_best', 0)
            print(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, bce_best {checkpoint['bce_best']})"
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.from_zero:
        start_epoch = 0

    current_epoch = start_epoch

    # Initialize mixed precision training if enabled
    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale='dynamic')

    snapshot_name = "{}{}_{}_{}".format(conf.get("prefix", args.prefix), conf['network'], conf['encoder'], args.fold)

    # Distributed or DataParallel model
    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = DataParallel(model).cuda()

    # Reset validation dataset
    data_val.reset(1, args.seed)
    max_epochs = conf['optimizer']['schedule']['epochs']

    for epoch in range(current_epoch, max_epochs):
        data_train.reset(epoch, args.seed)
        train_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
            train_sampler.set_epoch(epoch)

        # Freeze encoder for specified epochs
        if epoch < args.freeze_epochs:
            print("Freezing encoder!!!")
            model.module.encoder.eval()
            for p in model.module.encoder.parameters():
                p.requires_grad = False
        else:
            model.module.encoder.train()
            for p in model.module.encoder.parameters():
                p.requires_grad = True

        # Create training data loader
        train_data_loader = DataLoader(
            data_train, batch_size=batch_size, num_workers=args.workers,
            shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False, drop_last=True
        )

        # Train for one epoch
        train_loss = train_epoch(train_data_loader, model, loss_functions, optimizer, epoch, scheduler, args.distributed)

        # Evaluate the model
        if dist.is_primary():
            if epoch % args.test_every == 0:
                bce_best = evaluate(val_data_loader, model, loss_functions, current_epoch, bce_best, snapshot_name)
            summary_writer.add_scalar('bce_best', bce_best, epoch)
            summary_writer.add_scalar('loss', train_loss, epoch)
            current_epoch += 1

def train_epoch(train_data_loader, model, loss_functions, optimizer, epoch, scheduler, distributed):
    losses = AverageMeter()
    model.train()

    tbar = tqdm(train_data_loader)
    for i, (inputs, targets) in enumerate(tbar):
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)

        loss = 0
        for name, criterion in loss_functions.items():
            l = criterion(outputs, targets)
            loss += l

        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if not distributed:
            scheduler.step(epoch + i / len(tbar))

        losses.update(loss.item(), inputs.size(0))
        tbar.set_description('Epoch: {}; Loss: {:.4f}'.format(epoch, losses.avg))

    if distributed:
        scheduler.step()

    return losses.avg

def evaluate(val_data_loader, model, loss_functions, epoch, bce_best, snapshot_name):
    losses = AverageMeter()
    model.eval()
    targets_all = []
    outputs_all = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(val_data_loader)):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)

            loss = 0
            for name, criterion in loss_functions.items():
                l = criterion(outputs, targets)
                loss += l

            losses.update(loss.item(), inputs.size(0))
            targets_all.append(targets)
            outputs_all.append(outputs)

        targets_all = torch.cat(targets_all)
        outputs_all = torch.cat(outputs_all)

        bce = log_loss(targets_all.cpu().numpy(), torch.sigmoid(outputs_all).cpu().numpy())
        if bce < bce_best:
            print("Epoch {}: Improved log_loss from {:.4f} to {:.4f}. Saving checkpoint.".format(epoch, bce_best, bce))
            bce_best = bce
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'bce_best': bce_best}, snapshot_name + "_best.pth")
        else:
            print("Epoch {}: log_loss did not improve from {:.4f}".format(epoch, bce_best))

        return bce_best

if __name__ == "__main__":
    main()
