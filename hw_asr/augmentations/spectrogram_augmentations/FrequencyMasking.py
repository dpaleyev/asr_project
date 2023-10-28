from torch import Tensor
import torchaudio.transforms as T
import random


from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, p, *args, **kwargs) -> None:
        self.aug = T.FrequencyMasking(*args, **kwargs)
        self.p = p
    
    def __call__(self, data: Tensor) -> Tensor:
        if random.random() < self.p:
            return self.aug(data.unsqueeze(1)).squeeze(1) 
        else:
            return data