from typing import List, NamedTuple

import torch
from collections import defaultdict

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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

    def ctc_beam_search(self, probs: torch.tensor, probs_length: int, beam_size: int = 100) -> List[Hypothesis]:
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
        

