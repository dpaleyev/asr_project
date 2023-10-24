import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    #print(dataset_items[0].keys())

    result_batch = {}
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["text_encoded"] = pad_sequence([item["text_encoded"][0] for item in dataset_items], batch_first=True)
    result_batch["text_encoded_length"] = Tensor([item["text_encoded"][0].shape[0] for item in dataset_items])
    result_batch["spectrogram"] = pad_sequence([item["spectrogram"][0].T for item in dataset_items], batch_first=True).transpose(1, 2)
    result_batch["spectrogram_length"] = Tensor([item["spectrogram"].shape[2] for item in dataset_items])
    result_batch["audio"] = pad_sequence([item["audio"][0] for item in dataset_items], batch_first=True)
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]
    
    return result_batch