# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
# Multilingual version of AlignmentStreamAnalyzer
import logging
import torch
from dataclasses import dataclass
from types import MethodType


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
        self._hook_handles = []
        for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns.append(None)
            self._add_attention_spy(tfmr, i, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        def attention_forward_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1].cpu()
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]

        target_layer = tfmr.layers[layer_idx].self_attn

        # Patch forward to ensure output_attentions=True for this layer
        if not hasattr(target_layer, '_mtl_patched'):
            original_forward = target_layer.forward
            def patched_forward(*args, **kwargs):
                kwargs['output_attentions'] = True
                return original_forward(*args, **kwargs)
            target_layer.forward = patched_forward
            target_layer._mtl_patched = True

        handle = target_layer.register_forward_hook(attention_forward_hook)
        self._hook_handles.append(handle)

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
        # Check hooks fired
        none_count = sum(1 for a in self.last_aligned_attns if a is None)
        if none_count > 0:
            logger.warning(f"[ASA] frame={self.curr_frame_pos}: {none_count}/{len(self.last_aligned_attns)} attention heads are None (hooks not firing!)")
            self.curr_frame_pos += 1
            return logits

        # Average attention across tracked heads
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            A_chunk = aligned_attn[j:, i:j].clone().cpu()
        else:
            A_chunk = aligned_attn[:, i:j].clone().cpu()

        A_chunk[:, min(self.curr_frame_pos + 1, A_chunk.size(-1)):] = 0

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

        # Diagnostic logging: first 5 frames detailed, then every 10 frames
        if self.curr_frame_pos < 5 or self.curr_frame_pos % 10 == 0:
            attn_max = A_chunk[-1].max().item()
            logger.warning(
                f"[ASA] frame={self.curr_frame_pos} text_pos={self.text_position}/{S} "
                f"cur_argmax={cur_text_posn.item()} attn_max={attn_max:.4f} "
                f"complete={self.complete} started={self.started} "
                f"chunk_shape={list(A_chunk.shape)}"
            )

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

        # Safety limit: max frames before we stop suppressing EOS
        # Typical speech is ~5-10 audio frames per text token
        max_suppress_frames = max(12 * S, 100)
        # Hard limit: force EOS after this many frames
        max_generation_frames = max(15 * S, 150)

        past_suppress_limit = self.curr_frame_pos >= max_suppress_frames
        past_hard_limit = self.curr_frame_pos >= max_generation_frames

        # Suppress EOS first (only for longer texts, and only within safety limit)
        if cur_text_posn < S - 3 and S > 5 and not past_suppress_limit:
            logits[..., self.eos_idx] = -2**15

        if past_suppress_limit and not self.complete:
            logger.warning(
                f"[ASA] frame={self.curr_frame_pos}: past suppress limit ({max_suppress_frames}), "
                f"text_pos={self.text_position}/{S} - no longer suppressing EOS"
            )

        # Force EOS overrides suppress when hallucination detected OR hard limit hit
        if long_tail or alignment_repetition or token_repetition or past_hard_limit:
            if past_hard_limit:
                logger.warning(
                    f"[ASA] forcing EOS: hard frame limit ({max_generation_frames}) reached. "
                    f"text_pos={self.text_position}/{S}, complete={self.complete}"
                )
            else:
                logger.warning(f"[ASA] forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits
