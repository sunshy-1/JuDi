"""Last-layer AutoJudge speculative decoding."""

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
def run_autojudge(
    input_ids: torch.Tensor,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    scaler: Any,
    head: Any,
    *,
    draft_length: int = 40,
    max_new_tokens: int = 2048,
    threshold: float = 0.0,
    confidence_threshold: float = 0.9,
) -> dict:
    """Run last-layer AutoJudge with strict matching above target confidence."""
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
        sequence, n_draft, draft_hidden_list, _, draft_cache = draft_window_cached(
            sequence,
            draft_model,
            tokenizer,
            draft_length,
            draft_cache,
            collect_hidden=True,
        )
        if n_draft == 0:
            break
        target_outputs, target_tokens, target_last, target_hidden, target_cache = target_window_cached(
            sequence,
            target_model,
            n_draft,
            target_cache,
            collect_hidden=True,
        )
        draft_tokens = sequence[:, -n_draft:]
        draft_hidden = torch.cat(draft_hidden_list, dim=0)
        features = torch.cat((draft_hidden.cpu(), target_hidden.cpu()), dim=-1).float().numpy()
        probabilities = torch.as_tensor(head.predict_proba(scaler.transform(features))[:, 1], device=target_device)
        exact = (target_tokens == draft_tokens).flatten()
        target_logits = target_outputs.logits[:, -(n_draft + 1) : -1, :]
        confidence = F.softmax(target_logits, dim=-1).amax(dim=-1).flatten()
        classifier_accept = probabilities < threshold
        accept_mask = exact | torch.where(confidence > confidence_threshold, exact, classifier_accept)
        sequence, accepted, _ = accept_and_update(
            sequence, checkpoint, draft_tokens, target_tokens, accept_mask, target_last
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
