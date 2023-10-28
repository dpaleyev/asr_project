from torch import Tensor
import torch_audiomentations as T

from hw_asr.augmentations.base import AugmentationBase

class Noise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = T.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self.aug(data.unsqueeze(1)).squeeze(1)