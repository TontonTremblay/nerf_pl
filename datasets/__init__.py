from .blender import BlenderDataset
from .llff import LLFFDataset
from .nvisii import NvisiiDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'nvisii': NvisiiDataset}