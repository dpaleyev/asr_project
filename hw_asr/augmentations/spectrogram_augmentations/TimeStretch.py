from torch import Tensor
import torchaudio.transforms as T
import random

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, p, min_alpha, max_alpha, *args, **kwargs) -> None:
        self.aug = T.TimeStretch(*args, **kwargs)
        self.p = p
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
    
    def __call__(self, data: Tensor) -> Tensor:
        alpha = random.uniform(self.min_alpha, self.max_alpha)
        if random.random() < self.p:
            return self.aug(data.unsqueeze(1), overriding_rate=alpha).squeeze(1) 
        else:
            return data