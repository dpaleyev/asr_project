from torch import Tensor
import torchaudio.transforms as T
import random

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs) -> None:
        self.aug = T.TimeMasking(*args, **kwargs)
    
    def __call__(self, data: Tensor) -> Tensor:
        return self.aug(data.unsqueeze(1)).squeeze(1)