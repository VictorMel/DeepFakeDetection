from torch.utils.data import Dataset
import json
from os import path
from typing import Self, Union, Optional, Literal, Callable, List
import glob
import torch

Label = Union[Literal[0], Literal[1]]

class VideoTensorDataset(Dataset):
  """
  Handles reading downsized video tensor files from the given directories,
  as well as their corresponding labels from the original data.
  """

  # path to the root directory of the original data 
  original_data_path: str
  # path to the root directories of the tensor data
  tensor_data_paths: List[str]

  # dict from video file id (e.g. 'zbpwazdhtz') to tensor file path
  tensor_paths: dict[str, str]
  
  # dict from video file id (e.g. 'zbpwazdhtz') to label
  labels: dict[str, Label]

  # list of video file ids
  tensor_ids: list[str]

  transform: Optional[Callable]
  target_transform: Optional[Callable]

  def __init__(self: Self, original_data_path: str, tensor_data_paths: List[str], transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    self.original_data_path = original_data_path
    self.tensor_data_paths = tensor_data_paths
    self.transform = transform
    self.target_transform = target_transform

    # Init labels
    self.labels = {}
    metadata_glob = glob.iglob('**/metadata.json', recursive=True, root_dir=original_data_path)
    metadata_paths = (path.join(original_data_path, p) for p in metadata_glob)

    for metadata_path in metadata_paths:
      metadata = json.load(open(metadata_path))
      for filename, data in metadata.items():
        key, _ = path.splitext(filename)
        self.labels[key] = 1 if data['label'] == 'FAKE' else 0
    
    # Init tensor paths & ids
    self.tensor_paths = {}
    self.tensor_ids = []

    for tensor_root_path in tensor_data_paths:
      tensor_glob = glob.iglob('**/*.tns', recursive=True, root_dir=tensor_root_path)
      tensor_paths = [path.join(tensor_root_path, p) for p in tensor_glob]
      for tensor_path in tensor_paths:
        key, _ = path.splitext(path.basename(tensor_path))
        self.tensor_paths[key] = tensor_path
        self.tensor_ids.append(key)

  def __getitem__(self: Self, index: int):
    tensor_id = self.tensor_ids[index]
    tensor_path = self.tensor_paths[tensor_id]
    tensor = torch.load(tensor_path)

    if self.transform:
      tensor = self.transform(tensor)

    label = self.labels[tensor_id]
    if self.target_transform:
      label = self.target_transform(label)

    return tensor, label

  def __len__(self: Self):
    return len(self.tensor_ids)