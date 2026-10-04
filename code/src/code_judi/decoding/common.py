"""Shared speculative-decoding operations."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from code_judi.models.loader import model_device


def _append(sequence: torch.Tensor, token: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.cat((sequence, token.to(device)), dim=-1)


@torch.no_grad()
def initialize_cache(sequence: torch.Tensor, model: Any, *, output_hidden_states: bool = False) -> Any:
    """Run the prompt once and return the model outputs with a populated KV cache."""
    device = model_device(model, str(sequence.device))
    return model(
        input_ids=sequence.to(device),
        output_hidden_states=output_hidden_states,
        use_cache=True,
        return_dict=True,
    )


@torch.no_grad()
def draft_window_cached(
    sequence: torch.Tensor,
    draft_model: Any,
    tokenizer: Any,
    draft_length: int,
    past_key_values: Any,
    *,
    collect_hidden: bool = False,
    collect_logits: bool = False,
) -> Tuple[torch.Tensor, int, List[torch.Tensor], List[torch.Tensor], Any]:
    """Generate one draft window using a cache that excludes ``sequence[:, -1]``."""
    draft_device = model_device(draft_model, str(sequence.device))
    checkpoint = sequence.clone()
    hidden_states: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    reached_eos = False

    # The cache contains the prefix before the current last token. Feeding one
    # token at a time keeps it aligned with the sequence after every append.
    for position in range(draft_length + 1):
        outputs = draft_model(
            input_ids=sequence[:, -1:].to(draft_device),
            past_key_values=past_key_values,
            output_hidden_states=collect_hidden,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        if collect_hidden:
            hidden_states.append(outputs.hidden_states[-1][0, -1].unsqueeze(0).detach())
        if collect_logits:
            logits.append(outputs.logits[:, -1, :].detach().cpu())
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if position < draft_length:
            if reached_eos:
                break
            sequence = _append(sequence, next_token, sequence.device)
            if int(next_token.item()) == tokenizer.eos_token_id:
                reached_eos = True

    n_draft_tokens = sequence.shape[1] - checkpoint.shape[1]
    if collect_hidden:
        hidden_states = hidden_states[1 : n_draft_tokens + 1]
    if collect_logits:
        logits = logits[:n_draft_tokens]
    return sequence, n_draft_tokens, hidden_states, logits, past_key_values


@torch.no_grad()
def target_window_cached(
    sequence: torch.Tensor,
    target_model: Any,
    n_draft_tokens: int,
    past_key_values: Any,
    *,
    collect_hidden: bool = False,
) -> Tuple[Any, torch.Tensor, torch.Tensor, Any, Any]:
    """Verify a draft window using a cache populated through the prompt.

    ``sequence`` includes the target-generated first token followed by the
    draft window. The returned cache contains all of those tokens.
    """
    target_device = model_device(target_model, str(sequence.device))
    new_tokens = sequence[:, -(n_draft_tokens + 1) :].to(target_device)
    outputs = target_model(
        input_ids=new_tokens,
        past_key_values=past_key_values,
        output_hidden_states=collect_hidden,
        use_cache=True,
        return_dict=True,
    )
    target_tokens = outputs.logits.argmax(dim=-1)
    window_tokens = target_tokens[:, :-1]
    last_token = target_tokens[:, -1:]
    hidden = outputs.hidden_states[-1][0, -n_draft_tokens:].detach() if collect_hidden else None
    return outputs, window_tokens, last_token, hidden, outputs.past_key_values


def crop_cache(past_key_values: Any, max_length: int) -> Any:
    """Crop a Transformers cache to the accepted prefix length."""
    if past_key_values is None:
        return None
    crop = getattr(past_key_values, "crop", None)
    if callable(crop):
        crop(max_length)
        return past_key_values

    # Transformers versions that return legacy tuples need a tensor-level
    # fallback. Each layer stores key/value tensors with sequence at -2.
    if isinstance(past_key_values, (tuple, list)):
        cropped_layers = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("Unsupported legacy past_key_values layer format")
            values = list(layer)
            values[0] = values[0][..., :max_length, :]
            values[1] = values[1][..., :max_length, :]
            cropped_layers.append(tuple(values))
        return type(past_key_values)(cropped_layers)
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def accept_and_update(
    sequence: torch.Tensor,
    checkpoint: torch.Tensor,
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    accept_mask: torch.Tensor,
    target_last_token: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    exact_mask = (target_tokens == draft_tokens).flatten().to(torch.int32)
    accepted_without_correction = int(exact_mask.cumprod(dim=-1).sum().item())
    accepted = int(accept_mask.to(torch.int32).cumprod(dim=-1).sum().item())

    if accepted != draft_tokens.shape[1]:
        sequence = sequence.clone()
        replacement_index = -draft_tokens.shape[1] + accepted
        sequence[:, replacement_index] = target_tokens[:, accepted]
        accepted += 1
        sequence = sequence[:, : checkpoint.shape[1] + accepted]
    else:
        sequence = torch.cat((sequence, target_last_token.to(sequence.device)), dim=-1)
        accepted += 1
    return sequence, accepted, accepted_without_correction


def finish(sequence: torch.Tensor, input_length: int, tokenizer: Any, max_new_tokens: int) -> Tuple[torch.Tensor, bool]:
    generated = sequence[:, input_length:]
    eos = tokenizer.eos_token_id
    if eos is not None:
        positions = torch.nonzero(generated == eos, as_tuple=False)
        if len(positions):
            sequence = sequence[:, : input_length + int(positions[0, 1]) + 1]
            return sequence, True
    if sequence.shape[1] - input_length >= max_new_tokens:
        sequence = sequence[:, : input_length + max_new_tokens]
        return sequence, True
    return sequence, False


def result(sequence: torch.Tensor, draft_lengths: List[int], accepted_lengths: List[int]) -> Dict[str, Any]:
    return {
        "generated_ids": sequence,
        "draft_lengths": draft_lengths,
        "accepted_lengths": accepted_lengths,
        "steps": len(accepted_lengths),
    }
