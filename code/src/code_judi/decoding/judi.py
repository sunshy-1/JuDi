"""JuDi KL-based speculative decoding."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .common import (
    accept_and_update,
    crop_cache,
    draft_window_cached,
    finish,
    initialize_cache,
    result,
    target_window_cached,
)
from code_judi.models.loader import model_device


@torch.no_grad()
def run_judi(
    input_ids: torch.Tensor,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    *,
    draft_length: int = 40,
    max_new_tokens: int = 2048,
    judi_threshold: float = 0.0,
    confidence_threshold: float = 0.9,
) -> dict:
    """Run JuDi with strict matching above the confidence threshold."""
    target_device = model_device(target_model, str(input_ids.device))
    sequence = input_ids.to(target_device)
    input_length = sequence.shape[1]
    draft_lengths, accepted_lengths = [], []

    # Keep both caches at the accepted prefix. The target's first greedy token
    # becomes the pending last token, matching the cache flow used by AJ-V2.
    target_prompt = initialize_cache(sequence, target_model)
    draft_prompt = initialize_cache(sequence, draft_model)
    target_cache = target_prompt.past_key_values
    draft_cache = draft_prompt.past_key_values
    first_token = target_prompt.logits[:, -1:].argmax(dim=-1)
    sequence = torch.cat((sequence, first_token.to(sequence.device)), dim=-1)

    while True:
        checkpoint = sequence.clone()
        sequence, n_draft, _, draft_logits_list, draft_cache = draft_window_cached(
            sequence,
            draft_model,
            tokenizer,
            draft_length,
            draft_cache,
            collect_logits=True,
        )
        if n_draft == 0:
            break
        target_outputs, target_tokens, target_last, _, target_cache = target_window_cached(
            sequence,
            target_model,
            n_draft,
            target_cache,
        )
        draft_tokens = sequence[:, -n_draft:]
        draft_logits = torch.cat(draft_logits_list[:n_draft], dim=0).unsqueeze(0).to(target_device)
        target_logits = target_outputs.logits[:, -(n_draft + 1) : -1, :]
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        draft_log_probs = F.log_softmax(draft_logits, dim=-1)
        judi_divergence = F.kl_div(draft_log_probs, target_log_probs, reduction="none", log_target=True).sum(dim=-1).squeeze(0)
        confidence = target_log_probs.exp().amax(dim=-1).squeeze(0)
        exact = (target_tokens == draft_tokens).flatten()
        semantic = judi_divergence < judi_threshold
        accept_mask = exact | torch.where(confidence > confidence_threshold, exact, semantic)
        sequence, accepted, _ = accept_and_update(
            sequence, checkpoint, draft_tokens, target_tokens, exact | accept_mask, target_last
        )
        if accepted <= n_draft:
            cache_length = sequence.shape[1] - 1
            draft_cache = crop_cache(draft_cache, cache_length)
            target_cache = crop_cache(target_cache, cache_length)
        draft_lengths.append(n_draft)
        accepted_lengths.append(accepted)
        sequence, done = finish(sequence, input_length, tokenizer, max_new_tokens)
        if done:
            break
    return result(sequence, draft_lengths, accepted_lengths)
