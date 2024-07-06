# Resources
- [Presentation](https://technionmail-my.sharepoint.com/:p:/g/personal/victormel_campus_technion_ac_il/EcEUP5kI1DNHmBq4n9l2GhUBRhw4wOpnqjLmhpYRtvUpAw?e=nfDrg2)
- [Monday board](https://ai-course.monday.com/boards/1524530148)

## Useful links
- Deepfake detection challenge [overview](https://www.kaggle.com/competitions/deepfake-detection-challenge/overview) & [starter kit](https://www.kaggle.com/code/gpreda/deepfake-starter-kit/notebook).

- [Deepfake detection is super hard!!!](https://towardsdatascience.com/deepfake-detection-is-super-hard-38f98241ee49) - article describing some statistics about the best models from the comptetition.
- [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) - according to the article above, the best models for deep fake detection used this algorithm. [HuggingFace link](https://huggingface.co/docs/transformers/en/model_doc/efficientnet).

## Dealing with the data's scale

The go-to library for dealing with big data sets with PyTorch is webdataset ([docs](https://webdataset.github.io/webdataset/), [repo](https://webdataset.github.io/webdataset/)).
It seems that its primary use case is for datasets that are too big to store/process on a single machine - in that sense, our dataset seems "small enough" to train locally.
It also seems to speed IO up when working in a local machine, but this should be compared to alternatives; also, it requires a specific structure for the data, different from what we have.

We should also look into PyTorch Lightning and AIStore, in this context and in general.

Other options are `torchvision`'s `io` package (https://pytorch.org/vision/0.18/io.html) which we would use in combination with PyTorch's DataLoader/DataSet, or the [PyTorchVideo](https://pytorchvideo.org/) library, which seems powerful but less accessible.