# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
# Multilingual version of AlignmentStreamAnalyzer
import logging
import torch
from dataclasses import dataclass


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    false_start: bool
    long_tail: bool
    repetition: bool
    discontinuity: bool
    complete: bool
    position: int


class AlignmentStreamAnalyzerMTL:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j - i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        # Track generated tokens for repetition detection
        self.generated_tokens = []

        # Multi-head attention spies
        self.last_aligned_attns = []
        for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns.append(None)
            self._add_attention_spy(tfmr, i, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        def attention_forward_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1].cpu()
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]

        target_layer = tfmr.layers[layer_idx].self_attn
        target_layer.register_forward_hook(attention_forward_hook)
        if hasattr(tfmr, 'config') and hasattr(tfmr.config, 'output_attentions'):
            tfmr.config.output_attentions = True

    def reset(self, text_tokens_slice):
        """Reset state for a new request."""
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.alignment = torch.zeros(0, j - i)
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        self.generated_tokens = []
        for k in range(len(self.last_aligned_attns)):
            self.last_aligned_attns[k] = None

    def step(self, logits, next_token=None):
        # Average attention across tracked heads
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            A_chunk = aligned_attn[j:, i:j].clone().cpu()
        else:
            A_chunk = aligned_attn[:, i:j].clone().cpu()

        A_chunk[:, self.curr_frame_pos + 1:] = 0

        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # Update position
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        # False start detection
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Completion detection
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        last_text_token_duration = A[15:, -3:].sum()

        # Long tail detection (more aggressive threshold than English: 5 vs 10)
        long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5)

        # Alignment-based repetition
        alignment_repetition = self.complete and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        # Token-level repetition detection
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        token_repetition = (
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-2:])) == 1
        )

        # Suppress EOS first (only for longer texts)
        if cur_text_posn < S - 3 and S > 5:
            logits[..., self.eos_idx] = -2**15

        # Force EOS overrides suppress when hallucination detected
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits
