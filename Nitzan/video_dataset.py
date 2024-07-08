import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torch import nn
import json
from os import path
from typing import Self, TypedDict, Union, Optional, Literal, List, Tuple, Callable
import glob

# Split = Union[Literal['train'], Literal['validation']]
Label = Union[Literal[0], Literal[1]]

class FileMetadata(TypedDict):
  path: str
  label: Label
  original: Optional[str]

class VideoDataset(Dataset):
  def __init__(self: Self, root_path: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    self.root_path = root_path
    self.transform = transform
    self.target_transform = target_transform

    # Init metadata
    self.metadata: List[FileMetadata] = []
    metadata_glob = glob.iglob('**/metadata.json', recursive=True, root_dir=root_path)
    metadata_paths = [path.join(root_path, p) for p in metadata_glob]
    for metadata_path in metadata_paths:
      # print(f'reading metadata from {metadata_path}...')
      # start = time.time()
      metadata = json.load(open(metadata_path))
      # post_load = time.time()
      # print(f'loading json took {1_000*(post_load - start):.2f}ms')
      for k, data in metadata.items():
        video_path = path.join(path.dirname(metadata_path), k)
        data['path'] = video_path
        data['label'] = 1 if data['label'] == 'FAKE' else 0
        del data['split']

        self.metadata.append(data)
      
      # post_parse = time.time()
      # print(f'parsing json took {1_000*(post_parse - post_load):.2f}ms')


  def __getitem__(self: Self, index: int) -> Tuple[torch.Tensor, Label]:
    metadata = self.metadata[index]
    # How do we handle the audio as well?
    video, _, _ = read_video(metadata['path'], pts_unit='sec', end_pts=9)
    if self.transform:
      video = self.transform(video)
    label = metadata['label']
    if self.target_transform:
      label = self.target_transform(label)

    return video, label
  
  def __len__(self: Self):
    return len(self.metadata)