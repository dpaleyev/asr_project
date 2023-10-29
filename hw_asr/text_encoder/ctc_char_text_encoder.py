from typing import List, NamedTuple

import torch
from collections import defaultdict

from .char_text_encoder import CharTextEncoder
from pyctcdecode.decoder import build_ctcdecoder
import multiprocessing
import numpy as np

class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, use_lm: bool = False, lm_path: str = None, grams_path: str = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.use_lm = use_lm

        if use_lm:
            vocab_copy = [""] + [elem.upper() for elem in vocab[1:]]
            with open(grams_path) as f:
                unigrams = [line.strip() for line in f.readlines()]
            self.decoder = build_ctcdecoder(vocab_copy, unigrams=unigrams, lm_path=lm_path)

    def ctc_decode(self, inds: List[int]) -> str:
        """
        Decodes CTC output.
        """
        last_char = self.EMPTY_TOK
        res = []
        for ind in inds:
            char = self.ind2char[ind]
            if char == self.EMPTY_TOK:
                last_char = self.EMPTY_TOK
            elif char != last_char:
                res.append(char)
                last_char = char
        return "".join(res)
    
    def lm_decode(self, probs: torch.Tensor, probs_length: torch.Tensor, beam_size: int = 100) -> List[str]:
        logits_list = np.array([probs[i][:probs_length[i]].detach().cpu().numpy() for i in range(probs_length.shape[0])])

        with multiprocessing.get_context("fork").Pool() as p:
            lm_texts = self.model.decode_batch(p, logits_list, beam_width=beam_size)

        lm_texts = [elem.replace("|", "").replace("??", "").replace("'", "").lower().strip() for elem in lm_texts]

        return lm_texts

    def _extend_and_merge(self, frame, state) -> dict:
        new_state = defaultdict(float)
        for next_char_idx, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = self.ind2char[next_char_idx]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba * next_char_proba
        return new_state
    
    def _truncate_state(self, state: dict, beam_size: int):
        state_trunc = sorted(list(state.items()), key=lambda x: x[1], reverse=True)[:beam_size]
        return dict(state_trunc)

    def ctc_beam_search(self, probs: torch.Tensor, probs_length: int, beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        probs = probs[:probs_length]
        
        state = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = self._truncate_state(self._extend_and_merge(frame, state), beam_size)

        hypos = [Hypothesis(text, prob) for (text, _), prob in state.items()]

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
        

