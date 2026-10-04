"""Engine-local JuDi runtime patch for CUDA / vLLM 0.8.3, V0.

Run with /data0/sunshengyin/envs/vllm083/bin/python, from this directory:

    from judi_vllm import create_judi_llm, SamplingParams

    llm = create_judi_llm(
        model="models/target/Llama-3.1-8B-Instruct",
        draft_model="models/draft/Llama-3.2-1B-Instruct",
        judi_threshold=0.3, draft_length=40, max_model_len=4096,
    )
    outputs = llm.generate(["Question: What is 12 + 7?\\nAnswer:"],
                           SamplingParams(temperature=0, max_tokens=128))

Set CUDA_VISIBLE_DEVICES before launching Python. For a sharded target pass
tensor_parallel_size explicitly; draft_tensor_parallel_size defaults to 1.
Keep this module importable (cwd or PYTHONPATH) in all worker processes.
Use an ``if __name__ == "__main__":`` guard in scripts that spawn TP workers.
On this machine, TP=2 was validated with NCCL_P2P_DISABLE=1 and
disable_custom_all_reduce=True; default NCCL initialization stalled. These
machine-specific communication settings are deliberately not forced here.

Algorithm: greedy draft; full-vocabulary KL(target || draft); accept iff exact
OR (target confidence <= confidence_threshold AND KL < judi_threshold). Emit
only the accepted prefix, then target argmax at the first rejection, or the
target bonus token if all drafts pass. vLLM handles prefill, KV rollback,
bonus-token catch-up, batching, EOS and the output limit. It may compute past
EOS within a draft window; those tokens are discarded by its output processor.
Its speculative metrics count verification output before EOS/length clipping,
not the reference runner's MAT/DT on shortened EOS windows.

The algorithm matches src/code_judi/decoding/judi.py. Reductions use FP32, as
does vLLM's sampler; Transformers BF16 logits and different model kernels can
produce different rounding, particularly near the KL/confidence boundaries.
This is not a promise of bitwise identical Transformers generation.

No installed package or other engine is patched. The worker factory applies
instance patches after worker construction, including in spawned TP workers.
The private vLLM "probs" transport carries log probabilities within this engine
only. That avoids log(softmax) underflow and a second full-vocabulary buffer.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from importlib.metadata import version
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _kl_tiles(TARGET, DRAFT, KL, MAXIMUM, ARGMAX,
              TS0: tl.constexpr, TS1: tl.constexpr,
              DS0: tl.constexpr, DS1: tl.constexpr,
              K: tl.constexpr, V: tl.constexpr, TILES: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    b, k = row // K, row % K
    v = tile * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(TARGET + b * TS0 + k * TS1 + v, v < V, other=-float("inf"))
    q = tl.load(DRAFT + b * DS0 + k * DS1 + v, v < V, other=0.0)
    # The contribution of zero target mass is zero, including 0 * (inf-inf).
    terms = tl.where(p == -float("inf"), 0.0, tl.exp(p) * (p - q))
    maximum = tl.max(p, 0)
    argmax = tl.min(tl.where((p == maximum) & (v < V), v, 2147483647), 0)
    offset = row * TILES + tile
    tl.store(KL + offset, tl.sum(terms, 0))
    tl.store(MAXIMUM + offset, maximum)
    tl.store(ARGMAX + offset, argmax)


@triton.jit
def _accept_prefix(KL, MAXIMUM, ARGMAX, DRAFT_IDS, BONUS_IDS, OUTPUT,
                   ACCEPTED_COUNT, EMITTED_COUNT,
                   threshold, confidence_threshold,
                   IDS0: tl.constexpr, IDS1: tl.constexpr,
                   BONUS0: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                   KP: tl.constexpr, TP: tl.constexpr):
    b = tl.program_id(0)
    k = tl.arange(0, KP)
    t = tl.arange(0, TP)
    offset = (b * K + k[:, None]) * TILES + t[None, :]
    valid = (k[:, None] < K) & (t[None, :] < TILES)
    divergence = tl.sum(tl.load(KL + offset, valid, other=0.0), 1)
    maxima = tl.load(MAXIMUM + offset, valid, other=-float("inf"))
    maximum = tl.max(maxima, 1)
    indices = tl.load(ARGMAX + offset, valid, other=2147483647)
    target_id = tl.min(tl.where(maxima == maximum[:, None], indices, 2147483647), 1)
    draft_id = tl.load(DRAFT_IDS + b * IDS0 + k * IDS1, k < K, other=-1)
    accept = ((target_id == draft_id) |
              ((tl.exp(maximum) <= confidence_threshold) & (divergence < threshold)))
    first_reject = tl.min(tl.where((k < K) & ~accept, k, K), 0)
    out = tl.where(k < first_reject, draft_id,
                   tl.where(k == first_reject, target_id, -1))
    tl.store(OUTPUT + b * (K + 1) + k, out, k < K)
    bonus = tl.load(BONUS_IDS + b * BONUS0)
    tl.store(OUTPUT + b * (K + 1) + K, tl.where(first_reject == K, bonus, -1))
    # Count the causal prefix, not acceptable positions after the first failure.
    tl.atomic_add(ACCEPTED_COUNT, first_reject.to(tl.int64), sem="relaxed")
    tl.atomic_add(EMITTED_COUNT, (first_reject + 1).to(tl.int64), sem="relaxed")


class JudiSampler(torch.nn.Module):
    """vLLM deterministic sampler interface; both *_probs inputs are logprobs.

    Two launches, O(batch * K * ceil(vocab / 4096)) scratch storage, no device
    to host transfer. No [batch, K, vocab] KL/exp/difference intermediates.
    """

    probs_dtype = torch.float32
    token_id_dtype = torch.int64

    def __init__(self, judi_threshold: float, confidence_threshold: float = 0.9):
        super().__init__()
        if not math.isfinite(judi_threshold):
            raise ValueError("judi_threshold must be finite")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.judi_threshold = float(judi_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.num_accepted_tokens = None
        self.num_emitted_tokens = None
        self.num_draft_tokens = 0

    def init_tensors(self, device, device_type="cuda"):
        if isinstance(device, int):
            device = f"{torch.device(device_type).type}:{device}"
        self.num_accepted_tokens = torch.zeros((), dtype=torch.long, device=device)
        self.num_emitted_tokens = torch.zeros((), dtype=torch.long, device=device)

    def forward(self, target_with_bonus_probs, bonus_token_ids, draft_probs,
                draft_token_ids):
        batch, k, vocab = draft_probs.shape
        output = torch.empty((batch, k + 1), dtype=torch.long, device=draft_probs.device)
        if batch == 0:
            return output
        if k == 0:
            output.copy_(bonus_token_ids)
            self.num_emitted_tokens.add_(batch)
            return output
        tiles = triton.cdiv(vocab, 4096)
        kl = torch.empty((batch, k, tiles), device=draft_probs.device, dtype=torch.float32)
        maximum = torch.empty_like(kl)
        argmax = torch.empty((batch, k, tiles), device=draft_probs.device, dtype=torch.int32)
        _kl_tiles[(batch * k, tiles)](
            target_with_bonus_probs, draft_probs, kl, maximum, argmax,
            *target_with_bonus_probs.stride()[:2], *draft_probs.stride()[:2],
            k, vocab, tiles, 4096, enable_fp_fusion=False)
        _accept_prefix[(batch,)](
            kl, maximum, argmax, draft_token_ids, bonus_token_ids, output,
            self.num_accepted_tokens, self.num_emitted_tokens,
            self.judi_threshold, self.confidence_threshold,
            *draft_token_ids.stride(), bonus_token_ids.stride(0), k, tiles,
            triton.next_power_of_2(k), triton.next_power_of_2(tiles),
            enable_fp_fusion=False)
        self.num_draft_tokens += batch * k
        return output


def _validate_sampling(params):
    # Called during prefill, before logits processors or draft sampling run.
    expected = dict(temperature=0, n=1, top_p=1, top_k=-1, min_p=0,
                    presence_penalty=0, frequency_penalty=0,
                    repetition_penalty=1, min_tokens=0, ignore_eos=False)
    for key, value in expected.items():
        if getattr(params, key) != value:
            raise ValueError(f"JuDi requires {key}={value!r}")
    for key in ("logits_processors", "guided_decoding", "logit_bias",
                "allowed_token_ids", "bad_words", "stop", "stop_token_ids"):
        if getattr(params, key, None):
            raise ValueError(f"JuDi requires {key} to be unset")
    for key in ("logprobs", "prompt_logprobs"):
        if getattr(params, key) is not None:
            raise ValueError(f"This JuDi integration does not export {key}")


def _add_request(self, *args, **kwargs):
    # Reject unsupported requests before scheduling, including with TP workers.
    params = kwargs.get("params", args[2] if len(args) > 2 else None)
    if params is None:
        raise ValueError("Provide SamplingParams(temperature=0, max_tokens=...)")
    _validate_sampling(params)
    return self._judi_original_add_request(*args, **kwargs)


def _greedy_logprob_forward(self, logits, sampling_metadata):
    """Keep vLLM result bookkeeping, omitting unused softmax and one-hot work."""
    from vllm.model_executor.layers.sampler import (
        _build_sampler_output, _sample_with_torch, get_logprobs)

    # vLLM's synthetic memory-profile request uses random sampling. Real
    # requests are validated before either model executes.
    if any(g.sampling_params.temperature != 0 for g in sampling_metadata.seq_groups):
        return self._judi_original_forward(logits, sampling_metadata)
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    # All requests were validated as greedy. The greedy branch does not read
    # probs or SamplingTensors. Passing logprobs here never samples from logs.
    results, token_ids = _sample_with_torch(
        logprobs, logprobs, sampling_metadata, None,
        include_gpu_probs_tensor=True, modify_greedy_probs=False)
    prompt_logprobs = sample_logprobs = None
    if not sampling_metadata.skip_sampler_cpu_output:
        prompt_logprobs, sample_logprobs = get_logprobs(logprobs, sampling_metadata, results)
    return _build_sampler_output(
        results, sampling_metadata, prompt_logprobs, sample_logprobs,
        on_device_tensors=(logprobs, logprobs, token_ids),
        skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)


def _merge_draft_outputs(self, batch_size, proposal_len, maybe_sampler_output,
                         proposal_lens, nonzero_proposal_len_indices, sampler_transposed):
    """Stack logprobs once; avoid the native duplicate logprob/probability copy."""
    if maybe_sampler_output is None:
        return self._judi_original_merge(batch_size, proposal_len, None, proposal_lens,
                                         nonzero_proposal_len_indices, sampler_transposed)
    tokens = torch.stack([x.sampled_token_ids.flatten() for x in maybe_sampler_output])
    logprobs = torch.stack([x.logprobs for x in maybe_sampler_output])
    if sampler_transposed:
        tokens, logprobs = tokens.transpose(0, 1), logprobs.transpose(0, 1)
    if len(nonzero_proposal_len_indices) != batch_size:
        all_tokens = tokens.new_full((batch_size, proposal_len), -1)
        all_logs = logprobs.new_zeros((batch_size, proposal_len, self._vocab_size))
        all_tokens[nonzero_proposal_len_indices] = tokens
        all_logs[nonzero_proposal_len_indices] = logprobs
        tokens, logprobs = all_tokens, all_logs
    lengths = torch.tensor(proposal_lens, dtype=torch.long, device=tokens.device)
    return tokens, logprobs, lengths


def _filter_draft_output(expanded_batch_outputs, output_indices_to_retain):
    from vllm.model_executor.layers.sampler import SamplerOutput

    filtered = []
    for output in expanded_batch_outputs:
        logs = output.logprobs[output_indices_to_retain]
        filtered.append(SamplerOutput(
            outputs=[output.outputs[i] for i in output_indices_to_retain]
            if output.outputs else [],
            sampled_token_probs=logs, logprobs=logs,
            sampled_token_ids=output.sampled_token_ids[output_indices_to_retain]))
    return filtered


def _verify_tokens(self, seq_group_metadata_list, proposal_scores, proposals,
                   max_proposal_len):
    # The common all-decode batch needs no full-vocabulary gather or output
    # reordering. Keep vLLM's general path for mixed prefill/decode batches.
    lengths = proposals.proposal_lens.tolist()
    if (lengths and all(n == max_proposal_len for n in lengths)
            and proposal_scores.hidden_states is None):
        tokens = self.spec_decode_sampler(
            proposal_scores.probs, proposal_scores.token_ids[:, -1:],
            proposals.proposal_probs, proposals.proposal_token_ids)
        return tokens, proposal_scores.logprobs
    return self._judi_original_verify(seq_group_metadata_list, proposal_scores,
                                       proposals, max_proposal_len)


def _configure_samplers(self):
    from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker

    samplers = [self.scorer_worker.model_runner.model.sampler]
    proposer = self.proposer_worker
    if isinstance(proposer, SmallerTpProposerWorker):
        proposer = None if proposer._is_dummy else proposer._worker
    if proposer is not None:
        samplers.append(proposer.model_runner.model.sampler)
        proposer._filter_model_output = _filter_draft_output
        merger = proposer._proposer
        merger._judi_original_merge = merger._merge_outputs
        merger._merge_outputs = MethodType(_merge_draft_outputs, merger)
    for sampler in samplers:
        sampler.include_gpu_probs_tensor = True
        sampler.should_modify_greedy_probs_inplace = False
        sampler._judi_original_forward = sampler.forward
        sampler.forward = MethodType(_greedy_logprob_forward, sampler)


def _execute_model(self, execute_model_req=None):
    if execute_model_req is not None:
        for group in execute_model_req.seq_group_metadata_list:
            if group.is_prompt:
                _validate_sampling(group.sampling_params)
    return self._judi_original_execute(execute_model_req)


@dataclass(frozen=True)
class _JudiConfig:
    judi_threshold: float
    confidence_threshold: float

    def compute_hash(self):
        return hashlib.sha256(repr(self).encode()).hexdigest()


def create_judi_worker(*args, **kwargs):
    """Importable worker factory: applied in every vLLM worker process."""
    from .runtime import activate
    activate()
    from vllm import envs
    from vllm.spec_decode.spec_decode_worker import create_spec_worker

    if version("vllm") != "0.8.3" or envs.VLLM_USE_V1:
        raise RuntimeError("JuDi patch requires vllm==0.8.3 and VLLM_USE_V1=0")
    config = kwargs["vllm_config"]
    options = config.additional_config
    if not isinstance(options, _JudiConfig):
        raise ValueError("Construct this worker using create_judi_llm")
    spec = config.speculative_config
    if spec is None or spec.draft_model_config.hf_config.model_type in (
            "eagle", "medusa", "mlp_speculator", "deepseek_mtp"):
        raise ValueError("JuDi requires an ordinary autoregressive draft model")
    config.parallel_config.sd_worker_cls = "vllm.worker.worker.Worker"
    worker = create_spec_worker(*args, **kwargs)
    sampler = JudiSampler(options.judi_threshold, options.confidence_threshold)
    worker.spec_decode_sampler = sampler
    worker._metrics.spec_decode_sampler = sampler
    worker._configure_model_sampler_for_spec_decode = MethodType(_configure_samplers, worker)
    worker._judi_original_execute = worker.execute_model
    worker.execute_model = MethodType(_execute_model, worker)
    worker._judi_original_verify = worker._verify_tokens
    worker._verify_tokens = MethodType(_verify_tokens, worker)
    from vllm.logger import init_logger
    init_logger("vllm.judi_patch").info(
        "JuDi runtime patch active: KL(target || draft) < %g; confidence <= %g",
        options.judi_threshold, options.confidence_threshold)
    return worker


def create_judi_llm(*, model: str, draft_model: str, judi_threshold: float = 0.0,
                    confidence_threshold: float = 0.9, draft_length: int = 40,
                    draft_tensor_parallel_size: int = 1, **llm_kwargs):
    """Build a patched LLM; use SamplingParams(temperature=0, max_tokens=...).

    All remaining arguments are vLLM LLM options. Defaults enable the MQA scorer
    (eager target, FlashAttention) to verify a whole draft window in one pass.
    Setting enforce_eager=False instead uses vLLM's batch-expansion scorer.
    """
    from .runtime import activate
    activate()
    if version("vllm") != "0.8.3":
        raise RuntimeError("Use the vllm083 environment (vllm==0.8.3)")
    os.environ.setdefault("VLLM_USE_V1", "0")
    from vllm import LLM, envs
    from transformers import AutoTokenizer

    if envs.VLLM_USE_V1:
        raise RuntimeError("Set VLLM_USE_V1=0 before importing/initializing vLLM")
    JudiSampler(judi_threshold, confidence_threshold)  # Validate before loading weights.
    if not isinstance(draft_length, int) or draft_length <= 0:
        raise ValueError("draft_length must be a positive integer")
    for key in ("worker_cls", "additional_config", "speculative_config"):
        if key in llm_kwargs:
            raise ValueError(f"{key} is managed by the JuDi patch")
    if llm_kwargs.get("skip_tokenizer_init", False):
        raise ValueError("JuDi requires tokenizer initialization to preserve EOS semantics")
    llm_kwargs.setdefault("enforce_eager", True)
    llm_kwargs.setdefault("generation_config", "vllm")
    llm = LLM(
        model=model, worker_cls="judi_vllm.patch.create_judi_worker",
        additional_config=_JudiConfig(float(judi_threshold), float(confidence_threshold)),
        speculative_config={"model": draft_model,
                            "num_speculative_tokens": draft_length,
                            "draft_tensor_parallel_size": draft_tensor_parallel_size},
        **llm_kwargs)
    tokenizer = llm.get_tokenizer()
    draft_tokenizer = AutoTokenizer.from_pretrained(
        draft_model, trust_remote_code=llm_kwargs.get("trust_remote_code", False))
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("JuDi requires identical target/draft token-to-ID mappings")
    # vLLM 0.8.3 reads extra EOS IDs from generation_config even when its other
    # defaults are disabled. The reference stops only on tokenizer.eos_token_id.
    llm.llm_engine.generation_config_fields = {"eos_token_id": tokenizer.eos_token_id}
    engine = llm.llm_engine
    engine._judi_original_add_request = engine.add_request
    engine.add_request = MethodType(_add_request, engine)
    return llm
