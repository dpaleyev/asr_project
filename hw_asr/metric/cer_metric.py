from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
    
    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        lengths = log_probs_length.detach().cpu()
        for log_prob_vec, length, target_text in zip(log_probs.detach().cpu(), lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            hyps = self.text_encoder.ctc_beam_search(log_prob_vec, length, beam_size=self.beam_size)
            pred_text = hyps[0].text
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetricWithLM(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
    
    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        lm_texts = self.text_encoder.lm_decode(log_probs, log_probs_length, beam_size=self.beam_size)

        for lm_text, target_text in zip(lm_texts, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, lm_text))
        return sum(cers) / len(cers)
