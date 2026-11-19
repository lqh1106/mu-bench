# AUTO-GENERATED — do not edit manually. Run build_hub_files.py to regenerate.
from __future__ import annotations

import argparse
import enum
import json
import logging
import math
import os
import queue
import random
import re
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, partial
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, cast, overload
from typing_extensions import NotRequired, Self

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
from tqdm.auto import trange

from transformers import (
    Cache,
    DynamicCache,
    GenerationMixin,
    LogitsProcessorList,
    MimiConfig,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2TokenizerFast,
    Qwen3Config,
    Qwen3Model,
    Qwen3OmniMoePreTrainedModel,
    Qwen3OmniMoeTalkerCodePredictorModel,
    StaticCache,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    WhisperFeatureExtractor,
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache as _Cache, DynamicCache as _DynCache
from transformers.configuration_utils import PretrainedConfig as _PConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel as _PTModel
from transformers.models.mimi import MimiConfig as _MimiCfg, MimiModel
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiConv1dPaddingCache,
    MimiConvTranspose1d,
    MimiEncoder,
    MimiEncoderOutput,
    MimiResnetBlock,
    MimiTransformerModel,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoePreTrainedModel as _QOMMPreTrained,
    Qwen3OmniMoeTalkerCodePredictorOutputWithPast,
    Qwen3OmniMoeThinkerTextModel,
    SinusoidsPositionEmbedding,
)
from transformers.utils.generic import ModelOutput
from transformers.utils.import_utils import is_torchdynamo_compiling

from .configuration_raon import (
    EmbeddingAdaptorConfig,
    RaonConfig,
    RaonDuplexConfig,
    SpeakerEncoderConfig,
    VoxtralRealtimeEncoderConfig,
)

# ── from utils/special_tokens.py ──

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecialToken:
    """Frozen container for a special token's id and surface text."""

    id: int
    text: str

    def __int__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return self.text


PAD = SpecialToken(id=151679, text="<|endoftext|>")


IM_START = SpecialToken(id=151644, text="<|im_start|>")


IM_END = SpecialToken(id=151645, text="<|im_end|>")


AUDIO_START = SpecialToken(id=151669, text="<|audio_start|>")


AUDIO_END = SpecialToken(id=151670, text="<|audio_end|>")


SPEAKER_EMBEDDING_PLACEHOLDER = SpecialToken(id=151671, text="<|speaker_embedding_placeholder|>")


AUDIO_OUTPUT_PLACEHOLDER = SpecialToken(id=151675, text="<|audio_output_placeholder|>")


AUDIO_INPUT_PLACEHOLDER = SpecialToken(id=151676, text="<|audio_input_placeholder|>")


AUDIO_OUTPUT_PAD = SpecialToken(id=151677, text="<|audio_output_pad|>")


AUDIO_OUTPUT_END_PAD = SpecialToken(id=151678, text="<|audio_output_end_pad|>")

# Duplex SIL token (dedicated token, not repurposed FIM)


DUPLEX_SIL = SpecialToken(id=151672, text="<|audio_output_sil|>")

# Backchannel onset token (marks "uh-huh", "mm-hmm" turns instead of EPAD)


AUDIO_OUTPUT_BC = SpecialToken(id=151673, text="<|audio_output_backchannel|>")


PRETRAINING_AUDIO_TAG = "<audio>"


AUDIO_PLACEHOLDER = "<|audio_placeholder|>"


LOSS_IGNORE_INDEX = -100


ALL_SPECIAL_TOKENS: list[SpecialToken] = [
    PAD,
    IM_START,
    IM_END,
    AUDIO_START,
    AUDIO_END,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_END_PAD,
    DUPLEX_SIL,
    AUDIO_OUTPUT_BC,
]


def _mk_added_token_payload(token_id: int, content: str) -> dict[str, Any]:
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def _tokenizer_is_aligned(tokenizer: Any) -> bool:
    for token in ALL_SPECIAL_TOKENS:
        encoded = tokenizer.encode(token.text, add_special_tokens=False)
        if encoded != [token.id]:
            return False
    return True


def patch_tokenizer_files(tokenizer_dir: Path) -> None:
    """Patch tokenizer files on disk to align special token ids and surface text.

    Modifies vocab.json, tokenizer.json, added_tokens.json, tokenizer_config.json,
    and special_tokens_map.json in place, and ensures ALL_SPECIAL_TOKENS are
    correctly registered.

    Args:
        tokenizer_dir: Directory containing the tokenizer files to patch.
    """
    expected_by_id = {token.id: token.text for token in ALL_SPECIAL_TOKENS}
    vocab_path = tokenizer_dir / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        for token_id, token_text in expected_by_id.items():
            vocab[token_text] = token_id
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    tokenizer_json = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))

    model_vocab = tokenizer_json.get("model", {}).get("vocab")
    if isinstance(model_vocab, dict):
        for token_id, token_text in expected_by_id.items():
            model_vocab[token_text] = token_id

    added_tokens: list[dict[str, Any]] = tokenizer_json.get("added_tokens", [])
    by_id: dict[int, dict[str, Any]] = {}
    for entry in added_tokens:
        token_id = int(entry["id"])
        by_id[token_id] = entry

    for token_id, token_text in expected_by_id.items():
        entry = by_id.get(token_id)
        if entry is None:
            by_id[token_id] = _mk_added_token_payload(token_id, token_text)
            continue
        entry["content"] = token_text
        entry["single_word"] = False
        entry["lstrip"] = False
        entry["rstrip"] = False
        entry["normalized"] = False
        entry["special"] = True

    tokenizer_json["added_tokens"] = [by_id[token_id] for token_id in sorted(by_id.keys())]
    tokenizer_json_path.write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=2), encoding="utf-8")

    added_tokens_path = tokenizer_dir / "added_tokens.json"
    if added_tokens_path.exists():
        added_tokens_map = json.loads(added_tokens_path.read_text(encoding="utf-8"))
        for token in ALL_SPECIAL_TOKENS:
            added_tokens_map[token.text] = token.id
        added_tokens_path.write_text(json.dumps(added_tokens_map, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["additional_special_tokens"] = [AUDIO_INPUT_PLACEHOLDER.text]
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2), encoding="utf-8")

    special_tokens_map_path = tokenizer_dir / "special_tokens_map.json"
    if special_tokens_map_path.exists():
        special_tokens_map = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))
        special_tokens_map["audio_bos_token"] = AUDIO_START.text
        special_tokens_map["audio_eos_token"] = AUDIO_END.text
        special_tokens_map["audio_token"] = AUDIO_OUTPUT_PLACEHOLDER.text
        special_tokens_map["additional_special_tokens"] = [AUDIO_INPUT_PLACEHOLDER.text]
        special_tokens_map_path.write_text(json.dumps(special_tokens_map, ensure_ascii=False, indent=2), encoding="utf-8")


def update_tokenizer(tokenizer: Any) -> Any:
    """Ensure tokenizer special tokens match expected mapping; patch in-place if needed.

    Saves tokenizer to a temp directory, patches files via patch_tokenizer_files,
    loads the patched tokenizer, and updates the original tokenizer's attributes.
    Raises RuntimeError if alignment cannot be achieved.

    Args:
        tokenizer: HuggingFace tokenizer instance to update.

    Returns:
        The same tokenizer instance with updated special token mappings.
    """
    if _tokenizer_is_aligned(tokenizer):
        return tokenizer

    logger.warning("Tokenizer special token mapping is outdated. Applying overrides.")
    for token in ALL_SPECIAL_TOKENS:
        current_text = tokenizer.convert_ids_to_tokens(token.id)
        if current_text != token.text:
            logger.warning(f"id {token.id} token mismatch: current={current_text!r}, expected={token.text!r}. Overriding.")

    tokenizer_cls = tokenizer.__class__
    with tempfile.TemporaryDirectory(prefix="raon_tokenizer_patch_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        tokenizer.save_pretrained(tmp_path)
        patch_tokenizer_files(tmp_path)
        patched = tokenizer_cls.from_pretrained(tmp_path)

    tokenizer.__dict__.update(patched.__dict__)

    if not _tokenizer_is_aligned(tokenizer):
        raise RuntimeError("Failed to align tokenizer special tokens with required mapping.")

    return tokenizer


# ── from utils/misc.py ──


import json
import os
from pathlib import Path
from typing import Any, overload

import torch
from torch import nn


def load_safetensors_by_prefix(
    model_path: str | Path,
    prefixes: dict[str, str],
    dtype: torch.dtype | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load safetensors shards and split state_dict by key prefixes.

    Handles both sharded (``model.safetensors.index.json``) and single-file
    checkpoints, and works with both local directories and HuggingFace Hub IDs.
    Only shards containing at least one key matching a requested prefix are
    opened.

    Args:
        model_path: Path to directory containing safetensors files, or a
            HuggingFace Hub model ID.
        prefixes: Mapping of ``{name: prefix_string}`` to extract.  Each
            prefix should include a trailing ``"."`` if applicable.
        dtype: Optional dtype to cast tensors to.  When ``None`` tensors are
            returned in their stored dtype.

    Returns:
        Dict mapping each name from ``prefixes`` to a prefix-stripped
        state_dict containing the matching tensors.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    model_path_str = str(model_path)
    is_local = os.path.isdir(model_path_str)

    def _resolve(filename: str) -> str:
        if is_local:
            return os.path.join(model_path_str, filename)
        return hf_hub_download(repo_id=model_path_str, filename=filename)

    # Build shard -> list-of-keys mapping from the index, or fall back to a
    # single safetensors file.
    weight_map: dict[str, str] | None = None
    try:
        index_path = _resolve("model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_files: list[str] = sorted(set(weight_map.values()))
    except Exception:
        shard_files = ["model.safetensors"]

    # Initialise result containers.
    result: dict[str, dict[str, torch.Tensor]] = {name: {} for name in prefixes}

    for shard_name in shard_files:
        # When we have a weight map, skip shards that contain no relevant keys.
        if weight_map is not None:
            shard_keys = [k for k, v in weight_map.items() if v == shard_name]
            if not any(
                k.startswith(prefix)
                for k in shard_keys
                for prefix in prefixes.values()
            ):
                continue

        shard_path = _resolve(shard_name)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                for name, prefix in prefixes.items():
                    if key.startswith(prefix):
                        stripped_key = key.removeprefix(prefix)
                        tensor = f.get_tensor(key)
                        if dtype is not None:
                            tensor = tensor.to(dtype)
                        result[name][stripped_key] = tensor
                        break  # a key can only match one prefix

    return result


def _read_loss_param(env_key: str, config: Any, attr_name: str, default: float) -> float:
    """Read a loss parameter with precedence: environment variable > config attribute > default.

    Args:
        env_key: Environment variable name (e.g. ``RAON_TEXT_LOSS_WEIGHT``).
        config: Model config object to fall back to.
        attr_name: Attribute name on the config (e.g. ``text_loss_weight``).
        default: Default value if neither env var nor config attribute is set.

    Returns:
        The resolved parameter value as a float.
    """
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return float(env_val)
    return float(getattr(config, attr_name, default))


def _read_acoustic_loss_weights(config: Any, num_code_groups: int) -> list[float]:
    """Read acoustic loss weights with precedence: env var > config > default.

    The environment variable ``RAON_ACOUSTIC_LOSS_WEIGHTS`` is a comma-separated
    string of floats (one per acoustic codebook, i.e. ``num_code_groups - 1`` values).

    Args:
        config: Model config object to fall back to.
        num_code_groups: Total number of code groups (semantic + acoustic).

    Returns:
        List of acoustic loss weights with length ``num_code_groups - 1``.
    """
    env_val = os.environ.get("RAON_ACOUSTIC_LOSS_WEIGHTS")
    if env_val is not None:
        weights = [float(w.strip()) for w in env_val.split(",")]
        assert len(weights) == num_code_groups - 1, (
            f"RAON_ACOUSTIC_LOSS_WEIGHTS has {len(weights)} values, expected {num_code_groups - 1}"
        )
        return weights
    config_weights = getattr(config, "acoustic_loss_weights", None)
    if config_weights is not None:
        return [float(w) for w in config_weights]
    return [0.1] * (num_code_groups - 1)


def _get_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


@overload
def cast_float_inputs(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor: ...
@overload
def cast_float_inputs(tensor: None, target_dtype: torch.dtype) -> None: ...


def cast_float_inputs(tensor: torch.Tensor | None, target_dtype: torch.dtype) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.is_floating_point() and tensor.dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


@overload
def cast_to_module_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor: ...
@overload
def cast_to_module_dtype(tensor: None, module: nn.Module) -> None: ...


def cast_to_module_dtype(tensor: torch.Tensor | None, module: nn.Module) -> torch.Tensor | None:
    return cast_float_inputs(tensor, _get_module_dtype(module))


DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert a dtype string to a torch.dtype.

    Args:
        dtype_str: One of ``"bfloat16"``, ``"float16"``, ``"float32"``.

    Returns:
        Corresponding ``torch.dtype``.
    """
    return DTYPE_MAP[dtype_str]


# ── from utils/delay.py ──

import torch


def delay_audio_codes(
    delays: list[int],
    audio_codes: torch.Tensor,
    padding_value: int = 0,
) -> torch.Tensor:
    """Apply per-codebook delays to audio codes for training.

    Args:
        delays: List of delay values for each codebook
        audio_codes: Audio codes tensor, either (B, T, K) or (T, K)
        padding_value: Value to use for padding delayed positions

    Returns:
        Delayed audio codes with same shape as input
    """
    # Handle both 2D (T, K) and 3D (B, T, K) inputs
    squeeze_batch = False
    if audio_codes.dim() == 2:
        audio_codes = audio_codes.unsqueeze(0)  # (T, K) -> (1, T, K)
        squeeze_batch = True

    B, T, K = audio_codes.shape
    audio_codes_t = audio_codes.transpose(1, 2)  # (B, K, T)
    delayed = []
    for k, delay in enumerate(delays):
        if delay == 0:
            delayed.append(audio_codes_t[:, k])
        else:
            line = audio_codes_t[:, k].roll(delay, dims=1)
            line[:, :delay] = padding_value
            delayed.append(line)
    result = torch.stack(delayed, dim=1).transpose(1, 2)  # (B, T, K)

    if squeeze_batch:
        result = result.squeeze(0)  # (1, T, K) -> (T, K)

    return result


def undelay_audio_codes(
    delays: list[int],
    audio_codes: torch.Tensor,
    padding_value: int = 0,
) -> torch.Tensor:
    """Inverse of delay_audio_codes: shift codes back to original alignment."""
    if all(d == 0 for d in delays):
        return audio_codes
    squeeze_batch = False
    if audio_codes.dim() == 2:
        audio_codes = audio_codes.unsqueeze(0)
        squeeze_batch = True
    B, T, K = audio_codes.shape
    audio_codes_t = audio_codes.transpose(1, 2)
    undelayed = []
    for k, delay in enumerate(delays):
        if delay == 0:
            undelayed.append(audio_codes_t[:, k])
        else:
            line = audio_codes_t[:, k].roll(-delay, dims=1)
            line[:, -delay:] = padding_value
            undelayed.append(line)
    result = torch.stack(undelayed, dim=1).transpose(1, 2)
    if squeeze_batch:
        result = result.squeeze(0)
    return result




# ── from types.py ──


from dataclasses import dataclass
from typing import TypedDict
from typing_extensions import NotRequired

import torch
from transformers.utils.generic import ModelOutput


class RaonInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    audio_input: torch.Tensor | None
    audio_output: torch.Tensor | None
    speaker_encoder_audio: NotRequired[torch.Tensor | None]
    audio_input_lengths: torch.Tensor | None
    audio_output_lengths: torch.Tensor | None
    speaker_encoder_audio_lengths: NotRequired[torch.Tensor | None]
    labels: torch.Tensor
    sample_slot: NotRequired[int]


class DuplexInputs(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    audio_input: torch.Tensor
    audio_output: torch.Tensor
    audio_input_lengths: torch.Tensor
    audio_output_lengths: torch.Tensor
    speaker_encoder_audio: torch.Tensor | None
    speaker_encoder_audio_lengths: torch.Tensor | None
    sample_slot: int


@dataclass
class AudioEncoderOutput(ModelOutput):
    """Output of the audio encoder forward pass.

    Attributes:
        audio_embeds: Encoded audio representations. Shape: [batch, frames, hidden_size].
        audio_embeds_mask: Boolean mask indicating valid frames. Shape: [batch, frames].
    """

    audio_embeds: torch.Tensor | None = None
    audio_embeds_mask: torch.Tensor | None = None
    encoder_cache: tuple | None = None  # (encoder_past_key_values, conv_padding_cache) for streaming


@dataclass
class AudioTokenizerOutput(ModelOutput):
    """Output of the audio tokenizer (encoder + quantizer) forward pass.

    Attributes:
        audio_codes: Discrete codec codes per frame and codebook group.
            Shape: [batch, num_code_groups, num_frames].
        audio_codes_mask: Boolean mask indicating valid code frames. Shape: [batch, num_frames].
        mimi_features: Pre-quantization encoder features.
            Shape: [batch_size, num_frames, 512].
        encoder_cache: Streaming encoder cache tuple, if available.
    """

    audio_codes: torch.Tensor | None = None
    audio_codes_mask: torch.Tensor | None = None
    mimi_features: torch.Tensor | None = None  # [batch_size, num_frames, 512] pre-quantization features
    encoder_cache: tuple | None = None  # (encoder_past_key_values, conv_padding_cache) for streaming


@dataclass
class AudioDecoderOutput(ModelOutput):
    """Output of the audio decoder (codec decoder) forward pass.

    Attributes:
        audio: Reconstructed waveform. Shape: [batch, num_samples].
        decoder_cache: Streaming decoder cache tuple, if available.
    """

    audio: torch.Tensor
    decoder_cache: tuple | None = None  # (decoder_past_key_values, conv1d_padding_cache, convtranspose1d_padding_cache)


# ── from utils/state_machine.py ──


import enum
from dataclasses import dataclass
from typing import Literal

import torch



class DuplexPhase(enum.Enum):
    """Two-phase duplex state: silence or active speech."""

    SIL = "SIL"
    SPEECH = "SPEECH"


@dataclass
class DuplexMachineState:
    """Current Mealy machine state with last emitted frame tokens."""

    phase: DuplexPhase
    last_frame_tokens: list[int]

    @property
    def num_input_tokens(self) -> int:
        return len(self.last_frame_tokens)

    @property
    def emitted_audio(self) -> bool:
        return AUDIO_OUTPUT_PLACEHOLDER.id in self.last_frame_tokens or AUDIO_START.id in self.last_frame_tokens


@dataclass(frozen=True)
class DuplexStateConfig:
    """Immutable configuration for DuplexStateManager."""

    use_duplex_end_pad: bool = False
    use_sil_token: bool = False
    no_audio_in_sil: bool = False
    sequence_mode: Literal["tua", "uta"] | None = None
    duplex_pad_token_id: int = AUDIO_OUTPUT_PAD.id
    duplex_end_pad_token_id: int = AUDIO_OUTPUT_END_PAD.id
    duplex_sil_token_id: int = DUPLEX_SIL.id
    use_backchannel_token: bool = False
    duplex_bc_token_id: int = AUDIO_OUTPUT_BC.id

    @property
    def effective_sequence_mode(self) -> Literal["tua", "uta"]:
        return self.sequence_mode or "tua"


# Structural tokens that must never be sampled as text predictions.
_BLOCKED_STRUCTURAL: frozenset[int] = frozenset(
    {
        AUDIO_INPUT_PLACEHOLDER.id,
        AUDIO_OUTPUT_PLACEHOLDER.id,
        AUDIO_START.id,
        AUDIO_END.id,
        IM_START.id,
        IM_END.id,
        SPEAKER_EMBEDDING_PLACEHOLDER.id,
    }
)


class DuplexStateManager:
    """Mealy state machine for duplex inference: transitions + logit masking."""

    def __init__(self, config: DuplexStateConfig) -> None:
        self._config = config

    def initial_state(self, speak_first: bool = False) -> DuplexMachineState:
        """Return the initial machine state (SIL phase)."""
        return DuplexMachineState(
            phase=DuplexPhase.SIL,
            last_frame_tokens=[AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id],
        )

    def initial_forced_prediction_id(self, speak_first: bool) -> int | None:
        """Return the token ID to force for the first [U] prediction.

        Speak/listen mode is controlled via runtime token forcing.
        The first text-side prediction must be forced:
        - speak-first: force EPAD to enter onset/speech mode
        - listen-first: force SIL to remain in silence mode

        Returns:
            Token ID to force, or None when no explicit override is configured.
        """
        cfg = self._config
        if speak_first:
            if cfg.use_duplex_end_pad:
                return cfg.duplex_end_pad_token_id
            return None
        if cfg.use_sil_token:
            return cfg.duplex_sil_token_id
        return None

    def transition(
        self,
        state: DuplexMachineState,
        predicted_id: int,
        device: torch.device,
    ) -> tuple[DuplexMachineState, list[int], bool]:
        """Compute the next state and emitted frame tokens from a text prediction.

        Returns:
            Tuple of (new_state, frame_tokens, emitted_audio).
        """
        cfg = self._config
        aip = AUDIO_INPUT_PLACEHOLDER.id
        aop = AUDIO_OUTPUT_PLACEHOLDER.id
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id
        is_uta = cfg.effective_sequence_mode == "uta"

        is_sil_prediction = predicted_id == sil_id

        if state.phase == DuplexPhase.SIL:
            if is_sil_prediction:
                tokens = [aip, aop]
                return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

            if cfg.use_duplex_end_pad and predicted_id == epad_id:
                tokens = [aip, epad_id, aop] if is_uta else [epad_id, aip, aop]
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            # SIL -> SPEECH via backchannel onset.
            if cfg.use_backchannel_token and predicted_id == bc_id:
                tokens = [aip, bc_id, aop] if is_uta else [bc_id, aip, aop]
                return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

            # SIL -> SPEECH via direct text.
            if is_uta:
                tokens = [aip, predicted_id, aop]
            else:
                tokens = [predicted_id, aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        # SPEECH phase
        if is_sil_prediction:
            tokens = [aip, aop]
            return DuplexMachineState(DuplexPhase.SIL, tokens), tokens, True

        if predicted_id == pad_id:
            tokens = [aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        if predicted_id == epad_id:
            tokens = [aip, epad_id, aop] if is_uta else [epad_id, aip, aop]
            return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

        # SPEECH -> SPEECH (text token).
        if is_uta:
            tokens = [aip, predicted_id, aop]
        else:
            tokens = [predicted_id, aip, aop]
        return DuplexMachineState(DuplexPhase.SPEECH, tokens), tokens, True

    def apply_logit_mask(
        self,
        user_logits: torch.Tensor,
        state: DuplexMachineState,
        vocab_size: int,
    ) -> torch.Tensor:
        """Mask logits to enforce valid state-machine transitions.

        Returns:
            Masked logits with invalid tokens set to -inf.
        """
        cfg = self._config
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id

        # Build onset token set (EPAD and optionally BC).
        onset_ids = {epad_id}
        if cfg.use_backchannel_token:
            onset_ids.add(bc_id)

        user_logits = user_logits.clone()
        mask = torch.full_like(user_logits, float("-inf"))
        max_token_id = mask.shape[-1]

        def _allow_token(token_id: int) -> None:
            if token_id < max_token_id:
                mask[:, :, token_id] = 0.0

        if state.phase == DuplexPhase.SIL:
            # SILENCE: only SIL, EPAD, or BC allowed.
            _allow_token(sil_id)
            if cfg.use_duplex_end_pad:
                _allow_token(epad_id)
            if cfg.use_backchannel_token:
                _allow_token(bc_id)
        elif state.phase == DuplexPhase.SPEECH:
            context_token = self._extract_context_token(state)

            if context_token is not None and context_token in onset_ids:
                # After EPAD/BC onset: only text tokens allowed.
                mask[:, :, :vocab_size] = 0.0
                for block_id in _BLOCKED_STRUCTURAL | {sil_id, pad_id} | onset_ids:
                    if block_id < mask.shape[-1]:
                        mask[:, :, block_id] = float("-inf")
            elif context_token is not None and context_token not in (_BLOCKED_STRUCTURAL | onset_ids | {pad_id, sil_id}):
                # After text: text + PAD + EPAD + SIL allowed (BC only from SIL phase).
                mask[:, :, :vocab_size] = 0.0
                _allow_token(pad_id)
                _allow_token(epad_id)
                _allow_token(sil_id)
                for block_id in _BLOCKED_STRUCTURAL:
                    if block_id < mask.shape[-1]:
                        mask[:, :, block_id] = float("-inf")
            else:
                # PAD frame (no context token): PAD + EPAD + SIL allowed.
                _allow_token(pad_id)
                _allow_token(epad_id)
                _allow_token(sil_id)

        return user_logits + mask

    def _extract_context_token(self, state: DuplexMachineState) -> int | None:
        """Extract the text or EPAD context token from the last frame."""
        tokens = state.last_frame_tokens
        if len(tokens) != 3:
            return None

        is_uta = self._config.effective_sequence_mode == "uta"
        if is_uta:
            return tokens[1]
        else:
            return tokens[0]


# ── from modules/embedding.py ──

@dataclass
class EmbeddingAdaptorOutput(ModelOutput):
    """Output of EmbeddingAdaptor.forward().

    Attributes:
        outputs_embeds: Projected embeddings in the LM space.
            Shape: [batch, out_seq_len, output_size].
        mask: Boolean validity mask for the output sequence, or None if no mask
            was provided. Shape: [batch, out_seq_len].
    """

    outputs_embeds: torch.Tensor
    mask: torch.Tensor | None = None


class EmbeddingAdaptor(nn.Module):
    """Projects audio encoder embeddings into LM embedding space with optional time rescaling.

    Supports three backends controlled by constructor arguments:
    - 1-layer linear MLP (default)
    - 2-layer MLP with GELU activation
    - Lightweight Qwen3 transformer decoder (when ``decoder_config`` is provided)

    Time rescaling via ``output_time_scale`` allows matching different encoder/LM
    frame rates: values >= 1 upsample (reshape + linear), values < 1 downsample
    (stack adjacent frames then project).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: Qwen3Config | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        post_norm_init_scale: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.decoder_config = decoder_config

        if output_time_scale >= 1:
            scale = int(output_time_scale)
            assert scale == output_time_scale, (
                f"`output_time_scale` must be an integer when >= 1, got `{output_time_scale}`."
            )
            proj_input_size = input_size
            final_output_size = output_size * scale
        else:
            scale = int(1 / output_time_scale)
            assert scale == 1 / output_time_scale, (
                f"`1/output_time_scale` must be an integer when < 1, got `{output_time_scale}`."
            )
            proj_input_size = input_size * scale
            final_output_size = output_size

        # Check if we should use transformer mode
        if decoder_config is not None:
            # Transformer adaptor mode
            self.is_linear = False
            decoder_hidden_size = decoder_config.hidden_size
            self.input_proj = nn.Linear(
                proj_input_size,
                int(decoder_hidden_size * output_time_scale),
                bias=False,
                dtype=dtype,
            )
            self.decoder = Qwen3Model._from_config(decoder_config, dtype=dtype)
            # Remove unused embedding layer to save memory
            del self.decoder.embed_tokens
            self.decoder.embed_tokens = None  # type: ignore
            self.output_proj = nn.Linear(decoder_hidden_size, output_size, bias=False, dtype=dtype)
        elif num_layers == 1:
            # MLP mode (1 layer)
            self.is_linear = True
            self.proj = nn.Linear(proj_input_size, final_output_size, bias=False, dtype=dtype)
        elif num_layers == 2:
            # MLP mode (2 layers)
            self.is_linear = True
            hidden = hidden_size or final_output_size
            self.proj = nn.Sequential(
                nn.Linear(proj_input_size, hidden, bias=False, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden, final_output_size, bias=False, dtype=dtype),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

        self.post_norm = nn.RMSNorm(output_size, eps=norm_eps, dtype=dtype) if use_post_norm else None
        if self.post_norm is not None and post_norm_init_scale is not None:
            self.post_norm.weight.data.fill_(post_norm_init_scale)

    @classmethod
    def from_config(cls, config: Any, *, dtype: torch.dtype | None = None) -> EmbeddingAdaptor:
        """Create an EmbeddingAdaptor from a config object."""
        return cls(
            input_size=config.input_size,
            output_size=config.output_size,
            output_time_scale=config.output_time_scale,
            num_layers=getattr(config, "num_layers", 1),
            hidden_size=getattr(config, "hidden_size", None),
            decoder_config=getattr(config, "decoder_config", None),
            use_post_norm=getattr(config, "use_post_norm", False),
            norm_eps=getattr(config, "norm_eps", 1e-6),
            post_norm_init_scale=getattr(config, "post_norm_init_scale", None),
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EmbeddingAdaptorOutput:
        """Project encoder embeddings to LM space, applying time rescaling.

        Args:
            inputs: Encoder output embeddings. Shape: [batch, seq_len, input_size].
            mask: Optional boolean validity mask. Shape: [batch, seq_len].
                True indicates a valid (non-padded) frame.

        Returns:
            EmbeddingAdaptorOutput with projected embeddings of shape
            [batch, out_seq_len, output_size] and a corresponding mask.
            When output_time_scale >= 1, out_seq_len = seq_len * scale (upsample).
            When output_time_scale < 1, out_seq_len = ceil(seq_len / scale) (downsample).
        """
        batch_size, seq_length, _ = inputs.shape

        # output_time_scale >= 1: upsample -- each input frame expands to `scale` output frames.
        #   The projection output dimension is output_size * scale; a view() then splits
        #   it into (seq_len * scale, output_size).
        # output_time_scale < 1: downsample -- every `scale` consecutive input frames are
        #   concatenated along the feature axis before projection, reducing sequence length
        #   by a factor of `scale`. The sequence is right-padded with the last frame if
        #   its length is not divisible by `scale`.
        if self.output_time_scale >= 1:
            scale = int(self.output_time_scale)

            if self.is_linear:
                # MLP mode
                outputs_embeds = self.proj(inputs)
            else:
                # Transformer mode
                inputs_embeds = self.input_proj(inputs)
                # Convert mask to attention mask format if provided
                attention_mask = mask.to(inputs_embeds.dtype) if mask is not None else None
                decoder_outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                outputs_embeds = self.output_proj(decoder_outputs.last_hidden_state)

            outputs_embeds = outputs_embeds.view(batch_size, seq_length * scale, self.output_size)

            if mask is not None:
                output_mask = mask.repeat_interleave(scale, dim=1)
            else:
                output_mask = None
        else:
            scale = int(1 / self.output_time_scale)
            remainder = seq_length % scale
            if remainder != 0:
                padding_length = scale - remainder
                last_embed = inputs[:, -1:].expand(-1, padding_length, -1)
                inputs = torch.cat([inputs, last_embed], dim=1)
                if mask is not None:
                    mask = F.pad(mask, (0, padding_length), value=False)

            new_seq_length = inputs.shape[1] // scale
            inputs = inputs.view(batch_size, new_seq_length, scale * self.input_size)

            if self.is_linear:
                # MLP mode
                outputs_embeds = self.proj(inputs)
            else:
                # Transformer mode
                inputs_embeds = self.input_proj(inputs)
                attention_mask = mask.to(inputs_embeds.dtype) if mask is not None else None
                decoder_outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                outputs_embeds = self.output_proj(decoder_outputs.last_hidden_state)

            if mask is not None:
                output_mask = mask.view(batch_size, new_seq_length, scale).any(dim=-1)
            else:
                output_mask = None

        if self.post_norm is not None:
            outputs_embeds = self.post_norm(outputs_embeds)

        return EmbeddingAdaptorOutput(outputs_embeds=outputs_embeds, mask=output_mask)


# ── from modules/adaptor.py ──


import torch
from torch import nn


class ThinkerToTalkerProjection(nn.Module):
    """Projection from thinker hidden states to talker input space.

    Supports two modes:
    - ``"linear"``: RMSNorm (optional) followed by a single linear layer (no bias).
    - ``"mlp"``: Optional RMSNorm followed by a two-layer MLP with SiLU activation
      and bias, matching the original TalkerResizeMLP design.

    Args:
        thinker_hidden_size: Dimension of thinker hidden states.
        talker_hidden_size: Dimension of talker input.
        intermediate_size: Hidden dimension for the MLP mode. Required when
            ``mode="mlp"``, ignored for ``"linear"``.
        mode: Projection type — ``"linear"`` or ``"mlp"``.
        use_norm: If True, apply RMSNorm before projection (both modes).
        rms_norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        thinker_hidden_size: int,
        talker_hidden_size: int,
        intermediate_size: int | None = None,
        mode: str = "linear",
        use_norm: bool = True,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.norm: nn.RMSNorm | None = nn.RMSNorm(thinker_hidden_size, eps=rms_norm_eps) if use_norm else None
        if mode == "mlp":
            assert intermediate_size is not None, "intermediate_size is required for mlp mode."
            self.linear_fc1 = nn.Linear(thinker_hidden_size, intermediate_size, bias=True)
            self.linear_fc2 = nn.Linear(intermediate_size, talker_hidden_size, bias=True)
            self.act_fn = nn.SiLU()
            self.linear = None
        else:
            self.linear = nn.Linear(thinker_hidden_size, talker_hidden_size, bias=False)
            self.linear_fc1 = None
            self.linear_fc2 = None
            self.act_fn = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project thinker hidden states to talker input space.

        Args:
            hidden_states: Thinker hidden states.
                Shape: [batch_size, seq_len, thinker_hidden_size]. Dtype: float.

        Returns:
            Projected hidden states. Shape: [batch_size, seq_len, talker_hidden_size]. Dtype: float.
        """
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.mode == "mlp":
            return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))
        return self.linear(hidden_states)


# ── from modules/speaker_encoder.py ──

class PretrainedSpeakerEncoder(nn.Module):
    """Frozen pretrained speaker encoder with a trainable projection.

    This encoder consumes raw 24kHz audio, resamples internally to 16kHz, runs
    a frozen SpeechBrain ECAPA model, and projects the pretrained embedding to
    the raon hidden size.
    """

    def __init__(self, config: SpeakerEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Initialize ECAPA speaker encoder wrapper.

        Args:
            config: Speaker encoder configuration.
            dtype: Dtype for the trainable projection layer.
        """
        super().__init__()
        assert config.encoder_type == "ecapa_tdnn", f"Only ecapa_tdnn is supported, got: {config.encoder_type}"
        assert config.pretrained_dim is not None, (
            f"The `pretrained_dim` attribute must be set for encoder_type={config.encoder_type}."
        )

        self.encoder_type = config.encoder_type
        self.pretrained_model_id = config.pretrained_model_id
        self.pretrained_dim = config.pretrained_dim
        self.output_size = config.output_size
        self.source_sample_rate = 24000
        self.target_sample_rate = 16000
        self.min_seconds = config.min_seconds
        self.max_seconds = config.max_seconds

        self.projection = nn.Linear(config.pretrained_dim, config.output_size, bias=False, dtype=dtype)

        # Backend is loaded lazily on first forward to avoid redundant loads.
        self._backend: Any = None
        self._backend_device = torch.device("cpu")

    def _load_backend(self) -> None:
        """Load the frozen ECAPA backend for a single local process."""
        import torchaudio

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []

        # Patch huggingface_hub for speechbrain compat: speechbrain 1.0.x
        # passes the removed `use_auth_token` kwarg to hf_hub_download.
        import huggingface_hub as _hfhub

        _orig_hf_download = _hfhub.hf_hub_download

        def _patched_hf_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            return _orig_hf_download(*args, **kwargs)

        _hfhub.hf_hub_download = _patched_hf_download

        EncoderClassifier = __import__('importlib').import_module('speechbrain.inference.speaker').EncoderClassifier

        model_id = self.pretrained_model_id or "speechbrain/spkrec-ecapa-voxceleb"

        if os.path.isdir(model_id):
            # Local path: use directly without downloading.
            local_dir = model_id
        else:
            # HuggingFace repo ID: download to cache.
            cache_root = os.environ.get(
                "SPEECHBRAIN_ECAPA_SAVEDIR",
                os.path.expanduser("~/.cache/raon/speechbrain"),
            )
            os.makedirs(cache_root, exist_ok=True)
            local_dir = os.path.join(cache_root, model_id.replace("/", "_"))
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(model_id, local_dir=local_dir)

        backend = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            run_opts={"device": "cpu"},
        )

        object.__setattr__(self, "_backend", backend)
        if hasattr(self._backend, "parameters"):
            for param in self._backend.parameters():
                param.requires_grad = False

        self._backend_device = torch.device("cpu")

    def _ensure_backend_device(self) -> None:
        """Move backend to the projection device if needed."""
        target_device = self.projection.weight.device
        if self._backend is None:
            self._load_backend()
        if self._backend_device == target_device:
            return

        # SpeechBrain Pretrained keeps modules in backend.mods.
        self._backend.device = str(target_device)
        for mod in self._backend.mods.values():
            mod.to(target_device)
        self._backend_device = target_device

    def _extract_embedding(self, audio_16k: torch.Tensor, lengths_16k: torch.Tensor) -> torch.Tensor:
        """Extract frozen ECAPA embeddings from 16kHz audio.

        Args:
            audio_16k: Resampled mono audio. Shape: [batch_size, num_samples_16k]. Dtype: float.
            lengths_16k: Valid sample lengths. Shape: [batch_size]. Dtype: long.

        Returns:
            Pretrained embedding. Shape: [batch_size, pretrained_dim]. Dtype: float.
        """
        min_samples_16k = 1600
        batch_size = audio_16k.shape[0]
        if audio_16k.shape[1] < min_samples_16k:
            return torch.zeros(
                batch_size,
                self.pretrained_dim,
                device=audio_16k.device,
                dtype=audio_16k.dtype,
            )

        lengths_16k = lengths_16k.clamp(min=min_samples_16k)
        reference_length = max(1, int(audio_16k.shape[1]))
        wav_lens = lengths_16k.float() / float(reference_length)
        wav_lens = wav_lens.clamp(max=1.0)
        autocast_dtype: torch.dtype | None = None
        if audio_16k.device.type == "cuda" and self.projection.weight.dtype in {torch.float16, torch.bfloat16}:
            autocast_dtype = self.projection.weight.dtype

        with torch.no_grad():
            with torch.autocast(
                device_type=audio_16k.device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                embeddings = self._backend.encode_batch(audio_16k, wav_lens)

        return embeddings.squeeze(1)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute speaker embedding from raw 24kHz audio.

        Args:
            audio: Raw mono waveform at 24kHz. Shape: [batch_size, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.

        Returns:
            Speaker embedding. Shape: [batch_size, 1, output_size]. Dtype: float.
        """
        self._ensure_backend_device()
        import torchaudio.functional

        audio_input = audio
        if (
            audio_input.device.type == "cuda"
            and self.projection.weight.dtype in {torch.float16, torch.bfloat16}
            and audio_input.dtype != self.projection.weight.dtype
        ):
            audio_input = audio_input.to(dtype=self.projection.weight.dtype)
        elif audio_input.dtype not in {torch.float32, torch.float64, torch.float16, torch.bfloat16}:
            audio_input = audio_input.float()

        audio_16k = torchaudio.functional.resample(
            audio_input,
            orig_freq=self.source_sample_rate,
            new_freq=self.target_sample_rate,
        )
        lengths_16k = (audio_lengths.float() * self.target_sample_rate / self.source_sample_rate).long()

        # Random crop (training) or front-truncate (inference).
        min_samples = int(self.min_seconds * self.target_sample_rate)
        max_samples = int(self.max_seconds * self.target_sample_rate)
        min_valid = int(lengths_16k.min().item())

        if self.training and min_valid > min_samples:
            target_len = int(torch.randint(min_samples, min(max_samples, min_valid) + 1, (1,)).item())
            if min_valid > target_len:
                start = int(torch.randint(0, min_valid - target_len + 1, (1,)).item())
                audio_16k = audio_16k[:, start : start + target_len]
                lengths_16k = (lengths_16k - start).clamp(min=0, max=target_len)
        else:
            if audio_16k.shape[1] > max_samples:
                audio_16k = audio_16k[:, :max_samples]
                lengths_16k = lengths_16k.clamp(max=max_samples)

        raw_embedding = self._extract_embedding(audio_16k, lengths_16k)
        raw_embedding = raw_embedding.to(dtype=self.projection.weight.dtype)
        projected = self.projection(raw_embedding)
        return projected.unsqueeze(1)


# ── from modules/audio_tokenizer.py ──


from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mimi import MimiConfig, MimiModel
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiConv1dPaddingCache,
    MimiConvTranspose1d,
    MimiEncoder,
    MimiResnetBlock,
    MimiTransformerModel,
)
from transformers.utils.generic import ModelOutput
from transformers.utils.import_utils import is_torchdynamo_compiling


class StaticMimiConv1dPaddingCache(MimiConv1dPaddingCache):
    def __init__(self, per_layer_padding: list[int], padding_cache: list[torch.Tensor]) -> None:
        self.per_layer_padding = per_layer_padding
        self.padding_cache: list[torch.Tensor | None] = padding_cache  # type: ignore[assignment]
        self.is_initialized = True
        if not is_torchdynamo_compiling():
            for i in range(len(padding_cache)):
                torch._dynamo.mark_static_address(self.padding_cache[i])

    def update(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        assert self.is_initialized
        padding = self.per_layer_padding[layer_idx]
        cache_item = self.padding_cache[layer_idx]
        assert cache_item is not None
        current_cache = cache_item.clone()
        cache_item.copy_(hidden_states[:, :, hidden_states.shape[2] - padding :])
        return current_cache

    def reset(self) -> None:
        self.is_initialized = False

    def initialize(self, padding_cache: list[torch.Tensor]) -> None:
        for i in range(len(self.padding_cache)):
            cache_item = self.padding_cache[i]
            assert cache_item is not None
            cache_item.copy_(padding_cache[i])

        self.is_initialized = True


@dataclass
class CausalAudioEncoderOutput(ModelOutput):
    embeds: torch.Tensor | None = None
    encoder_past_key_values: Cache | None = None
    padding_cache: MimiConv1dPaddingCache | None = None
    streaming_state: object | None = None


class MimiConvTranspose1dPaddingCache:
    """
    Padding cache for MimiConvTranspose1d causal convolutions in order to support streaming via cache padding.

    A padding cache is a list of cached partial hidden states for each convolution layer.
    Hidden states are cached from the previous call to the MimiConvTranspose1d forward pass, given the padding size.
    """

    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[torch.Tensor],
        per_layer_in_channels: list[int],
    ):
        # ensure correct number of layers for each arg
        from_args_num_layers = {len(per_layer_padding), len(per_layer_in_channels)}

        if len(from_args_num_layers) != 1 or from_args_num_layers.pop() != num_layers:
            raise ValueError(
                f"Expected `num_layers` ({num_layers}) values in `per_layer_padding`, "
                "`per_layer_padding_mode` and `per_layer_in_channels`"
            )
        self.per_layer_padding = per_layer_padding
        self.per_layer_in_channels = per_layer_in_channels
        self.per_layer_is_init = [True] * num_layers
        self.padding_cache: list[torch.Tensor | None] = [None] * num_layers

    def update(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch_size, dtype, device = (
            hidden_states.shape[0],
            hidden_states.dtype,
            hidden_states.device,
        )
        padding = int(self.per_layer_padding[layer_idx].long().item())
        in_channels = self.per_layer_in_channels[layer_idx]

        cached = self.padding_cache[layer_idx]
        if cached is None:
            current_cache = torch.zeros(
                batch_size,
                in_channels,
                padding,
                device=device,
                dtype=dtype,
            )
        else:
            current_cache = cached

        if padding > 0:
            padding_states = hidden_states[:, :, -padding:]
        else:
            padding_states = torch.empty(batch_size, in_channels, padding, dtype=dtype, device=device)
        self.padding_cache[layer_idx] = padding_states

        return current_cache


@dataclass
class StreamingMimiOutput(ModelOutput):
    audio_codes: torch.LongTensor | None = None
    audio_values: torch.FloatTensor | None = None
    encoder_past_key_values: Cache | list[torch.FloatTensor] | None = None
    decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None
    conv1d_padding_cache: MimiConv1dPaddingCache | None = None
    convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None


@dataclass
class StreamingMimiDecoderOutput(ModelOutput):
    audio_values: torch.FloatTensor | None = None
    decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None
    conv1d_padding_cache: MimiConv1dPaddingCache | None = None
    convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None


class StreamingMimiConvTranspose1d(MimiConvTranspose1d):
    def __init__(
        self,
        config: MimiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(config, in_channels, out_channels, kernel_size, stride, groups, bias)

        self.in_channels = in_channels
        self.layer_idx = layer_idx

    @property
    def kernel_size(self) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor(self.conv.kernel_size[0], dtype=torch.int64)

    @property
    def stride(self) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor(self.conv.stride[0], dtype=torch.int64)

    @property
    def padding_total(self) -> torch.Tensor:  # type: ignore[override]
        return self.kernel_size - self.stride

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: MimiConvTranspose1dPaddingCache | None = None,
    ) -> torch.Tensor:
        if not self.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is only defined for causal convolutions.")
        if self.causal and padding_cache is not None:
            assert self.layer_idx is not None
            layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
            padding_len = padding_cache.per_layer_padding[self.layer_idx]
            extra_padding = padding_len - layer_padding_cache.shape[-1]
            if extra_padding > 0:
                layer_padding_cache = nn.functional.pad(
                    layer_padding_cache,
                    (int(extra_padding), 0),
                    mode="constant",
                    value=0,
                )
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=-1)
            padding_left = layer_padding_cache.shape[-1] * self.stride + self.padding_left  # type: ignore
        else:
            padding_left = self.padding_left
        hidden_states = self.conv(hidden_states)

        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., padding_left:end]

        return hidden_states


class StreamingMimiDecoder(nn.Module):
    """SEANet decoder as used by Mimi."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model: list[nn.Module] = [
            MimiConv1d(
                config,
                config.hidden_size,
                scaling * config.num_filters,
                config.kernel_size,
            )
        ]
        mimiconv1d_layer_names = ["layers.0"]
        mimiconvtranspose1d_layer_names: list[str] = []

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            # Add  upsampling layers
            model += [nn.ELU()]
            mimiconvtranspose1d_layer_names.append(f"layers.{len(model)}")
            model += [
                StreamingMimiConvTranspose1d(
                    config,
                    current_scale,
                    current_scale // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                mimiconv1d_layer_names.extend([f"layers.{len(model)}.block.{1}", f"layers.{len(model)}.block.{3}"])
                model += [MimiResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]  # type: ignore
            scaling //= 2

        # Add final layers
        model += [nn.ELU()]
        mimiconv1d_layer_names.append(f"layers.{len(model)}")
        model += [
            MimiConv1d(
                config,
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
            )
        ]
        self.layers = nn.ModuleList(model)

        self._mimiconv1d_layer_names = mimiconv1d_layer_names
        self._mimiconvtranspose1d_layer_names = mimiconvtranspose1d_layer_names

        # initialize layer_idx for MimiConv1d submodules, necessary for padding_cache
        for layer_idx, layername in enumerate(self._mimiconv1d_layer_names):
            conv_layer = self.get_submodule(layername)
            conv_layer.layer_idx = layer_idx  # type: ignore
        # initialize layer_idx for MimiConvTranspose1d submodules, necessary for padding_cache
        for layer_idx, layername in enumerate(self._mimiconvtranspose1d_layer_names):
            convtranspose_layer = self.get_submodule(layername)
            convtranspose_layer.layer_idx = layer_idx  # type: ignore

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden_states = layer(hidden_states, padding_cache=conv1d_padding_cache)
            elif isinstance(layer, MimiConvTranspose1d):
                hidden_states = layer(hidden_states, padding_cache=convtranspose1d_padding_cache)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class StreamingMimiModel(MimiModel):
    def __init__(self, config: MimiConfig):
        # Use Flash Attention 2 when available so that Mimi's sliding_window is applied.
        # SDPA and eager attention ignore sliding_window, which can break long-audio
        # (>20s) encoding/decoding where SWA matters.
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401

                config._attn_implementation = "flash_attention_2"
            except ImportError:
                import warnings

                warnings.warn(
                    "flash_attn is not installed. Mimi will use SDPA attention, which ignores "
                    "sliding_window. Audio longer than ~20s may produce artifacts. "
                    "Install flash-attn for full correctness: pip install flash-attn",
                    stacklevel=2,
                )
        super().__init__(config)
        self.decoder = StreamingMimiDecoder(config)
        self.upsample = StreamingMimiConvTranspose1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
            stride=2,
            bias=False,
            groups=config.upsample_groups,
            layer_idx=len(self.decoder._mimiconvtranspose1d_layer_names),
        )
        # targets = [self.encoder, self.downsample, self.decoder]
        targets = [self.decoder]
        for target in targets:
            for module in target.modules():
                if isinstance(module, MimiConv1d):
                    module.forward = partial(self.mimi_conv1d_forward, module)  # type: ignore[method-assign]

    def mimi_conv1d_forward(
        self,
        module: MimiConv1d,
        hidden_states: torch.Tensor,
        padding_cache: MimiConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        extra_padding = module._get_extra_padding_for_conv1d(hidden_states)

        if not module.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is not supported for non-causal convolutions.")

        if module.causal and padding_cache is not None:
            assert module.layer_idx is not None
            layer_padding_cache = padding_cache.update(hidden_states, module.layer_idx)
            assert layer_padding_cache is not None
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
            assert not isinstance(module.padding_total, nn.Module)
            hidden_states = module._pad1d(
                hidden_states,
                (
                    max(0, module.padding_total - layer_padding_cache.shape[2]),  # type: ignore
                    extra_padding,  # type: ignore
                ),
                mode=module.pad_mode,
            )

        elif module.causal and padding_cache is None:
            hidden_states = module._pad1d(
                hidden_states,
                (module.padding_total, extra_padding),  # type: ignore
                mode=module.pad_mode,
            )

        else:
            hidden_states = module._pad1d(
                hidden_states,
                (module.padding_left, module.padding_right + extra_padding),  # type: ignore
                mode=module.pad_mode,
            )

        hidden_states = module.conv(hidden_states)
        return hidden_states

    def _decode_frame(  # type: ignore[override]
        self,
        codes: torch.Tensor,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        return_dict: bool | None = None,
    ) -> tuple[
        torch.Tensor,
        Cache | list[torch.FloatTensor] | None,
        MimiConv1dPaddingCache | None,
        MimiConvTranspose1dPaddingCache | None,
    ]:
        embeddings = self.quantizer.decode(codes)

        assert self.upsample is not None, "_decode_frame: `self.upsample` is None."
        embeddings = self.upsample(embeddings, padding_cache=convtranspose1d_padding_cache)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=return_dict,
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            past_key_values = decoder_outputs[1]
        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(
            embeddings,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
        )
        return outputs, past_key_values, conv1d_padding_cache, convtranspose1d_padding_cache

    def decode(  # type: ignore
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        use_streaming: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | StreamingMimiDecoderOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_streaming = use_streaming if use_streaming is not None else self.config.use_streaming

        if use_streaming and conv1d_padding_cache is None:
            per_layer_padding, per_layer_padding_mode, per_layer_in_channels = (
                [],
                [],
                [],
            )
            for layer_name in self.decoder._mimiconv1d_layer_names:
                per_layer_padding.append(self.decoder.get_submodule(layer_name).padding_total)  # type: ignore
                per_layer_padding_mode.append(self.decoder.get_submodule(layer_name).pad_mode)
                per_layer_in_channels.append(self.decoder.get_submodule(layer_name).in_channels)  # type: ignore

            conv1d_padding_cache = MimiConv1dPaddingCache(
                num_layers=len(self.decoder._mimiconv1d_layer_names),
                per_layer_padding=per_layer_padding,  # type: ignore
                per_layer_padding_mode=per_layer_padding_mode,
                per_layer_in_channels=per_layer_in_channels,
            )

        if use_streaming and convtranspose1d_padding_cache is None:
            convtranspose_per_layer_padding: list[torch.Tensor] = []
            convtranspose_per_layer_in_channels: list[int] = []
            for layer_name in self.decoder._mimiconvtranspose1d_layer_names:
                k = self.decoder.get_submodule(layer_name).kernel_size
                s = self.decoder.get_submodule(layer_name).stride
                if k % s == 0:  # type: ignore
                    padding_tmp = (k / s - 1) * s  # type: ignore
                else:
                    padding_tmp = torch.floor(k / s) * s  # type: ignore
                convtranspose_per_layer_padding.append(padding_tmp)
                convtranspose_per_layer_in_channels.append(self.decoder.get_submodule(layer_name).in_channels)  # type: ignore

            assert self.upsample is not None
            k = self.upsample.kernel_size
            s = self.upsample.stride
            if k % s == 0:  # type: ignore
                padding_tmp = (k / s - 1) * s  # type: ignore
            else:
                padding_tmp = torch.floor(k / s) * s  # type: ignore

            convtranspose_per_layer_padding.append(padding_tmp)
            convtranspose_per_layer_in_channels.append(self.upsample.in_channels)  # type: ignore

            convtranspose1d_padding_cache = MimiConvTranspose1dPaddingCache(
                num_layers=len(self.decoder._mimiconvtranspose1d_layer_names) + 1,
                per_layer_padding=convtranspose_per_layer_padding,
                per_layer_in_channels=convtranspose_per_layer_in_channels,
            )

        (
            audio_values,
            decoder_past_key_values,
            conv1d_padding_cache,
            convtranspose1d_padding_cache,
        ) = self._decode_frame(
            audio_codes,
            past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
            return_dict=return_dict,
        )

        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (  # type: ignore
                audio_values,
                decoder_past_key_values,
                conv1d_padding_cache,
                convtranspose1d_padding_cache,
            )
        return StreamingMimiDecoderOutput(
            audio_values=audio_values,  # type: ignore
            decoder_past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        audio_codes: torch.Tensor | None = None,
        encoder_past_key_values: Cache | list[torch.FloatTensor] | None = None,
        decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | StreamingMimiOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if audio_codes is None:
            encoder_outputs = self.encode(
                input_values,
                padding_mask,
                num_quantizers,
                encoder_past_key_values,
                return_dict=return_dict,
            )
            audio_codes = encoder_outputs[0]
            if return_dict:
                encoder_past_key_values = encoder_outputs.get("past_key_values")  # type: ignore[union-attr]
            elif len(encoder_outputs) > 1:
                encoder_past_key_values = encoder_outputs[1]  # type: ignore[assignment]

        decoder_outputs = self.decode(audio_codes, padding_mask, decoder_past_key_values, return_dict=return_dict)
        audio_values = decoder_outputs[0]
        if return_dict:
            decoder_past_key_values = decoder_outputs.get("past_key_values")  # type: ignore[union-attr]
            conv1d_padding_cache = decoder_outputs.get("conv1d_padding_cache")  # type: ignore[union-attr]
            convtranspose1d_padding_cache = decoder_outputs.get("convtranspose1d_padding_cache")  # type: ignore[union-attr]
        elif len(decoder_outputs) > 1:
            decoder_past_key_values = decoder_outputs[1]  # type: ignore[assignment]
            conv1d_padding_cache = decoder_outputs[2]  # type: ignore[misc]
            convtranspose1d_padding_cache = decoder_outputs[3]  # type: ignore[misc]

        if not return_dict:
            return (  # type: ignore
                audio_codes,
                audio_values,
                encoder_past_key_values,
                decoder_past_key_values,
                conv1d_padding_cache,
                convtranspose1d_padding_cache,
            )

        return StreamingMimiOutput(
            audio_codes=audio_codes,  # type: ignore
            audio_values=audio_values,  # type: ignore
            encoder_past_key_values=encoder_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,  # type: ignore
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,  # type: ignore
        )

    def get_input_embeddings(self):
        """Return None as audio models don't have traditional input embeddings."""
        return None

    def set_input_embeddings(self, value):
        """No-op as audio models don't have traditional input embeddings."""
        pass


class CausalAudioEncoder(PreTrainedModel):
    config_class: type[MimiConfig] = MimiConfig  # type: ignore

    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.config = config

        self.encoder = MimiEncoder(config)
        self.encoder_transformer = MimiTransformerModel(config)

        self.downsample = None
        assert config.hidden_size is not None
        if config.frame_rate != config.encodec_frame_rate:
            self.downsample = MimiConv1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                pad_mode="replicate",
                layer_idx=len(self.encoder._mimiconv1d_layer_names),  # type: ignore
            )

        self.post_init()

    def encode_frame(
        self,
        audio: torch.Tensor,
        past_key_values: Cache | None = None,
        padding_cache: MimiConv1dPaddingCache | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, Cache | None, MimiConv1dPaddingCache | None]:
        embeds = self.encoder(audio, padding_cache=padding_cache)
        encoder_outputs = self.encoder_transformer(
            embeds.transpose(1, 2),
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        if return_dict:
            past_key_values = encoder_outputs.get("past_key_values")
        elif len(encoder_outputs) > 1:
            past_key_values = encoder_outputs[1]

        embeds = encoder_outputs[0].transpose(1, 2)
        if self.downsample is not None:
            embeds = self.downsample(embeds, padding_cache=padding_cache)

        return embeds, past_key_values, padding_cache

    def forward(
        self,
        audio: torch.Tensor,
        encoder_past_key_values: Cache | None = None,
        padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> tuple[torch.Tensor, ...] | CausalAudioEncoderOutput:
        if use_streaming is None:
            use_streaming = self.config.use_streaming

        assert 1 <= audio.shape[1] <= 2, f"Number of audio channels must be 1 or 2, but got {audio.shape[1]}."

        if use_streaming and padding_cache is None:
            per_layer_padding: list[int] = []
            per_layer_padding_mode: list[str] = []
            per_layer_in_channels: list[int] = []
            for layer_name in self.encoder._mimiconv1d_layer_names:  # type: ignore
                layer = self.encoder.get_submodule(layer_name)
                per_layer_padding.append(int(layer.padding_total))  # type: ignore
                per_layer_padding_mode.append(str(layer.pad_mode))
                per_layer_in_channels.append(int(layer.in_channels))  # type: ignore

            if self.downsample is not None:
                per_layer_padding.append(int(self.downsample.padding_total))  # type: ignore
                per_layer_padding_mode.append(str(self.downsample.pad_mode))
                per_layer_in_channels.append(int(self.downsample.in_channels))

            padding_cache = MimiConv1dPaddingCache(
                num_layers=len(self.encoder._mimiconv1d_layer_names) + (1 if self.downsample is not None else 0),  # type: ignore
                per_layer_padding=per_layer_padding,
                per_layer_padding_mode=per_layer_padding_mode,
                per_layer_in_channels=per_layer_in_channels,
            )

        embeds, encoder_past_key_values, padding_cache = self.encode_frame(
            audio,
            past_key_values=encoder_past_key_values,
            padding_cache=padding_cache,
            return_dict=True,
        )

        embeds = embeds.transpose(1, 2)

        return CausalAudioEncoderOutput(
            embeds=embeds,
            encoder_past_key_values=encoder_past_key_values,
            padding_cache=padding_cache,
        )


# ── from modules/audio_encoder.py ──


import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import WhisperFeatureExtractor
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoePreTrainedModel,
    SinusoidsPositionEmbedding,
)


_AUDIO_TOWER_PREFIX = "thinker.audio_tower."


@dataclass
class AuTStreamingState:
    """All mutable state needed for frame-by-frame streaming of the causal AuT encoder.

    A single instance is created via ``AuTEncoder.init_streaming_state`` (or
    ``AuTWrapper.init_streaming_state``) and passed into each successive
    ``forward`` call.  The caller must **not** modify the tensors in-place;
    updated state is returned from the forward methods.
    """

    stft_cache: torch.Tensor | None
    """Cached tail waveform samples for STFT overlap between chunks.
    Shape: [num_leftover_samples]. Dtype: float.  ``None`` before the first call."""

    running_max: float
    """Running maximum of per-frame log-mel maxima across all chunks so far.
    Used for causal running-max normalization in feature extraction."""

    conv_caches: list[torch.Tensor]
    """Per-Conv2d-layer border cache on the time axis (3 entries).
    Each tensor has shape [1, channels, freq_dim, 0..2]. Dtype: float."""

    kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
    """Per-transformer-layer past key and value projections.
    Each tuple is (past_keys, past_values) with shape
    [1, num_heads, past_seq_len, head_dim]. Dtype: float.
    Empty (past_seq_len=0) before the first call."""

    num_frames_produced: int
    """Total number of output frames produced so far (positional embedding offset)."""


@dataclass
class AuTEncoderOutput:
    """Output of ``AuTEncoder.forward``.

    Contains the projected encoder hidden states and per-sample output frame
    counts.
    """

    last_hidden_state: torch.Tensor
    """Padded batched encoder output.
    Shape: [batch_size, max_output_frames, output_dim]. Dtype: float."""

    output_lens: list[int]
    """Per-sample output frame counts."""

    streaming_state: AuTStreamingState | None = None
    """Updated streaming state, present only when streaming mode is active."""


class AudioAttention(nn.Module):
    """Multi-headed attention for the AuT audio encoder.

    This is a standalone duplicate of the HuggingFace audio attention module that
    can be modified independently. The forward signature and weight layout are
    identical so that pretrained weights can be loaded directly.
    """

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim: int = config.d_model
        self.num_heads: int = config.encoder_attention_heads
        self.dropout: float = config.attention_dropout
        self.head_dim: int = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.scaling: float = self.head_dim**-0.5
        self.attention_dropout: float = 0.0
        self.is_decoder: bool = False
        self.is_causal: bool = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute multi-headed self-attention.

        Args:
            hidden_states: Input tensor.
                Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
            attention_mask: Per-token padding mask (True = valid).
                Shape: [batch_size, seq_len]. Dtype: bool.
            causal: If True, use causal (autoregressive) attention where each
                position can only attend to itself and earlier positions.

        Returns:
            Attention output. Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
        """
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        additive_mask = None
        if not causal and attention_mask is not None:
            key_padding = ~attention_mask.to(torch.bool)
            additive_mask = torch.zeros((batch_size, 1, 1, seq_len), device=hidden_states.device, dtype=hidden_states.dtype)
            additive_mask = additive_mask.masked_fill(
                key_padding.unsqueeze(1).unsqueeze(1), torch.finfo(hidden_states.dtype).min
            )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=additive_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=causal,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute causal self-attention with a KV cache for streaming.

        Projects the new hidden states into Q/K/V, concatenates K/V with the
        cached past, runs scaled dot-product attention (new queries attend to
        all past + new keys), and returns the updated cache.

        Args:
            hidden_states: New input tokens for this streaming step.
                Shape: [new_seq_len, embed_dim]. Dtype: float.
            past_key_value: Tuple of (past_keys, past_values) from previous
                streaming steps.
                Each shape: [1, num_heads, past_seq_len, head_dim]. Dtype: float.

        Returns:
            A tuple of:
                - Attention output for the new tokens only.
                    Shape: [new_seq_len, embed_dim]. Dtype: float.
                - Updated (past_keys, past_values) including the new tokens.
                    Each shape: [1, num_heads, past_seq_len + new_seq_len, head_dim].
                    Dtype: float.
        """
        new_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(new_len, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(new_len, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(new_len, self.num_heads, -1)

        # [new_len, num_heads, head_dim] -> [1, num_heads, new_len, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Concatenate with past KV cache.
        past_keys, past_values = past_key_value
        key_states = torch.cat([past_keys, key_states], dim=2)
        value_states = torch.cat([past_values, value_states], dim=2)

        # Causal mask: new queries can attend to all past + current keys,
        # but not to future keys within the new chunk.
        total_len = key_states.shape[2]
        causal_mask = torch.full(
            (1, 1, new_len, total_len),
            torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        past_len = total_len - new_len
        # All new queries can attend to all past positions.
        causal_mask[:, :, :, :past_len] = 0.0
        # Within the new chunk, apply lower-triangular mask.
        new_block = (
            torch.triu(
                torch.ones(new_len, new_len, device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=1,
            )
            * torch.finfo(hidden_states.dtype).min
        )
        causal_mask[:, :, :, past_len:] = new_block

        # Use explicit matmul instead of scaled_dot_product_attention so that
        # the streaming (1-token-at-a-time) and non-streaming (full-sequence)
        # paths produce identical results on CPU.  SDPA dispatches to different
        # fused kernels depending on input shape, causing ~1e-3 diffs.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(new_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, (key_states, value_states)


class AuTEncoderLayer(GradientCheckpointingLayer):
    """Single transformer encoder layer for the AuT audio encoder.

    Uses the local ``AudioAttention`` duplicate for self-attention while keeping
    the FFN, layer norms, and activation function from HuggingFace.
    """

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim: int = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout: float = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout: float = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        """Run one encoder layer (pre-norm self-attention + FFN).

        Args:
            hidden_states: Batched input hidden states.
                Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
            attention_mask: Per-token padding mask (True = valid).
                Shape: [batch_size, seq_len]. Dtype: bool.
            causal: If True, use causal attention.
            padding_mask: Per-token padding mask used to zero padded positions.
                Shape: [batch_size, seq_len]. Dtype: bool.

        Returns:
            Single-element tuple containing the output hidden states.
            Shape: [batch_size, seq_len, embed_dim]. Dtype: float.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal=causal,
        )
        hidden_states = residual + hidden_states
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask.unsqueeze(-1).to(hidden_states.dtype)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask.unsqueeze(-1).to(hidden_states.dtype)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs

    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one encoder layer in streaming mode with a KV cache.

        Args:
            hidden_states: New hidden states for this streaming step.
                Shape: [new_seq_len, embed_dim]. Dtype: float.
            past_key_value: Tuple of (past_keys, past_values) for this layer.
                Each shape: [1, num_heads, past_seq_len, head_dim]. Dtype: float.

        Returns:
            A tuple of:
                - Output hidden states for the new tokens.
                    Shape: [new_seq_len, embed_dim]. Dtype: float.
                - Updated (past_keys, past_values).
                    Each shape: [1, num_heads, past_seq_len + new_seq_len, head_dim].
                    Dtype: float.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, new_kv = self.self_attn.forward_streaming(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, new_kv


class AuTEncoder(Qwen3OmniMoePreTrainedModel):
    """Audio Transformer encoder with a duplicated attention submodule.

    Architecturally identical to the HuggingFace audio encoder but uses
    ``AuTEncoderLayer`` (which contains the local ``AudioAttention`` duplicate)
    instead of the original encoder layer. All other components (convolutions,
    positional embeddings, projection layers) are unchanged.

    Pretrained weights from the HuggingFace audio encoder can be loaded directly
    because the state dict keys are identical (the attention weight names inside
    each layer match one-to-one).
    """

    config: Qwen3OmniMoeAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["AuTEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__(config)
        self.dropout: float = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins: int = config.num_mel_bins
        self.max_source_positions: int = config.max_source_positions
        self.embed_scale: float = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window: int = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([AuTEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer: int = self.config.n_window_infer
        self.conv_chunksize: int = self.config.conv_chunksize
        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return the first convolutional layer as the input embedding."""
        return self.conv2d1

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the first convolutional layer."""
        self.conv2d1 = value

    def _prepare_batched_attention_mask(
        self,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a standard per-token padding mask for batched attention.

        Args:
            valid_mask: True at valid time positions and False at padded positions.
                Shape: [batch_size, seq_len]. Dtype: bool.

        Returns:
            Padding mask with True for valid tokens.
            Shape: [batch_size, seq_len]. Dtype: bool.
        """
        return valid_mask

    def _get_noncausal_chunk_length(self) -> int:
        """Return the expected non-causal attention chunk length after CNN.

        Returns:
            Chunk length in encoder frames. Dtype: int.
        """
        full_chunk_frames = self.n_window * 2
        frames_after_cnn = full_chunk_frames
        for _ in range(3):
            frames_after_cnn = (frames_after_cnn - 1) // 2 + 1
        scale = self.n_window_infer // (self.n_window * 2)
        return frames_after_cnn * scale

    def _apply_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three Conv2d downsampling layers with GELU activations.

        Args:
            x: Input feature map.
                Shape: [batch_size, 1, num_mel_bins, num_frames]. Dtype: float.

        Returns:
            Downsampled feature map.
            Shape: [batch_size, hidden_channels, mel_bins_downsampled, frames_downsampled].
            Dtype: float.
        """
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        return x

    def _apply_causal_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three Conv2d downsampling layers with causal padding on the time axis.

        Each layer uses left-only padding along the time dimension (no future
        leakage) and symmetric padding along the frequency dimension. This
        ensures that each output time frame depends only on current and past
        input frames.

        Args:
            x: Input feature map.
                Shape: [batch_size, channels, num_mel_bins, num_frames]. Dtype: float.

        Returns:
            Downsampled feature map with causal time alignment.
            Shape: [batch_size, hidden_channels, mel_bins_downsampled, frames_downsampled].
            Dtype: float.
        """
        for conv in [self.conv2d1, self.conv2d2, self.conv2d3]:
            # Pad: (time_left, time_right, freq_left, freq_right)
            # Causal: left-pad time by kernel_size-1=2, no right-pad.
            # Symmetric: pad freq by 1 on each side (same as padding=1).
            x = F.pad(x, (2, 0, 1, 1))
            x = F.gelu(F.conv2d(x, conv.weight, conv.bias, stride=2))
        return x

    def _apply_causal_conv_stack_streaming(
        self,
        x: torch.Tensor,
        conv_caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the causal Conv2d stack incrementally using cached border pixels.

        On each call the unconsumed tail frames (stored in ``conv_caches``) are
        prepended to the new input before padding and convolution, so that
        boundary frames are computed identically to the non-streaming path.

        The cache size varies depending on stride alignment: with kernel=3 and
        stride=2, the conv consumes frames in pairs.  If the combined input
        length is even, 1 frame is left over; if odd, 2 frames are left over.

        Args:
            x: New input feature map chunk.
                Shape: [1, channels, freq_dim, new_time_frames]. Dtype: float.
            conv_caches: List of 3 tensors, one per Conv2d layer.  Each holds
                the unconsumed tail frames from the previous call.
                Shape: [1, channels, freq_dim, 0..2]. Dtype: float.

        Returns:
            A tuple of:
                - Output feature map containing only the **new** output frames.
                    Shape: [1, hidden_channels, freq_downsampled, new_output_frames].
                    Dtype: float.
                - Updated ``conv_caches`` list (same structure, new tensors).
        """
        new_caches: list[torch.Tensor] = []
        for conv, cache in zip([self.conv2d1, self.conv2d2, self.conv2d3], conv_caches, strict=True):
            # Prepend unconsumed tail from the previous chunk.
            x_with_cache = torch.cat([cache, x], dim=3)
            combined_len = x_with_cache.shape[3]

            if combined_len < 3:
                # Not enough for even one conv output -- carry everything.
                new_caches.append(x_with_cache.clone())
                x = x_with_cache[:, :, :, :0]  # Empty time dim
                continue

            # Compute how many frames the conv will produce.
            num_outputs = (combined_len - 3) // 2 + 1
            # The next output would start at position 2 * num_outputs.
            # Everything from that position onward is unconsumed.
            unconsumed_start = 2 * num_outputs
            new_caches.append(x_with_cache[:, :, :, unconsumed_start:].clone())

            # Causal pad: freq symmetric (1,1), time already handled by cache.
            x_padded = F.pad(x_with_cache, (0, 0, 1, 1))
            x = F.gelu(F.conv2d(x_padded, conv.weight, conv.bias, stride=2))  # type: ignore
        return x, new_caches

    def _flatten_conv_output(self, conv_out_4d: torch.Tensor) -> torch.Tensor:
        """Flatten and project a 4D conv output to embedding space.

        Args:
            conv_out_4d: Output of the conv stack.
                Shape: [batch_size, hidden_channels, mel_bins_downsampled, frames_downsampled].
                Dtype: float.

        Returns:
            Projected embeddings.
            Shape: [batch_size, frames_downsampled, embed_dim]. Dtype: float.
        """
        batch_size, channels, freq, time = conv_out_4d.size()
        # [batch_size, channels, freq, time] -> [batch_size, time, channels * freq]
        return self.conv_out(conv_out_4d.permute(0, 3, 1, 2).contiguous().view(batch_size, time, channels * freq))

    def _downsample_chunked(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Downsample using the original chunked windowing strategy.

        Splits each sample's mel features into fixed-size ``n_window * 2``
        chunks, pads chunk tensors for Conv2d batching, and merges chunk
        outputs back per sample.

        This reproduces the exact behavior of the original HuggingFace
        audio encoder forward pass.

        Args:
            input_features: Padded batched mel spectrogram.
                Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
            feature_lens: Per-sample mel frame counts.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - hidden_states: Padded batched embeddings.
                    Shape: [batch_size, max_output_frames, embed_dim]. Dtype: float.
                - valid_mask: True at valid output positions.
                    Shape: [batch_size, max_output_frames]. Dtype: bool.
                - output_lens: Per-sample output frame counts.
        """
        output_lens: list[int] = []
        merged_sample_embeds: list[torch.Tensor] = []
        for sample_idx, sample_len_tensor in enumerate(feature_lens):
            sample_len = int(sample_len_tensor.item())
            chunk_count = math.ceil(sample_len / (self.n_window * 2))
            chunk_lengths = torch.tensor(
                [self.n_window * 2] * chunk_count,
                dtype=torch.long,
                device=input_features.device,
            )
            remainder = sample_len % (self.n_window * 2)
            if remainder != 0:
                chunk_lengths[-1] = remainder

            sample_feature = input_features[sample_idx, :, :sample_len]
            chunk_list = list(sample_feature.T.split(chunk_lengths.tolist(), dim=0))
            padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
            padded_feature = padded_feature.unsqueeze(1)

            feature_lens_after_cnn = chunk_lengths
            for _ in range(3):
                feature_lens_after_cnn = (feature_lens_after_cnn - 1) // 2 + 1

            padded_embeds: list[torch.Tensor] = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                padded_embeds.append(self._apply_conv_stack(chunk))
            padded_embed = torch.cat(padded_embeds, dim=0)
            padded_embed = self._flatten_conv_output(padded_embed)

            pos_embed_buffer: torch.Tensor = self.positional_embedding(padded_embed.shape[1])
            positional_embedding = pos_embed_buffer.unsqueeze(0).to(padded_embed.dtype)
            padded_embed = padded_embed + positional_embedding

            sample_chunk_embeds: list[torch.Tensor] = []
            for i in range(chunk_count):
                chunk_len = int(feature_lens_after_cnn[i].item())
                sample_chunk_embeds.append(padded_embed[i, :chunk_len])
            merged_embed = torch.cat(sample_chunk_embeds, dim=0)
            sample_len = int(merged_embed.shape[0])
            merged_sample_embeds.append(merged_embed)
            output_lens.append(sample_len)

        hidden_states = nn.utils.rnn.pad_sequence(merged_sample_embeds, batch_first=True)
        valid_mask = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=hidden_states.device) for length in output_lens],
            batch_first=True,
        )

        return hidden_states, valid_mask, output_lens

    def _downsample_causal(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Downsample without chunking using causal (left-only) time padding.

        Processes each sample's mel features independently through the causal
        conv stack (no windowed chunking), then pads outputs back to a batched
        tensor.

        Each Conv2d layer uses left-only padding along the time dimension,
        ensuring each output frame depends only on current and past input frames.

        Args:
            input_features: Padded batched mel spectrogram.
                Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
            feature_lens: Per-sample mel frame counts.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - hidden_states: Padded batched embeddings.
                    Shape: [batch_size, max_output_frames, embed_dim]. Dtype: float.
                - valid_mask: True at valid output positions.
                    Shape: [batch_size, max_output_frames]. Dtype: bool.
                - output_lens: Per-sample output frame counts.
        """
        all_embeds: list[torch.Tensor] = []
        output_lens: list[int] = []

        for sample_idx, sample_len_tensor in enumerate(feature_lens):
            sample_len = int(sample_len_tensor.item())
            # [num_frames, num_mel_bins] -> [1, 1, num_mel_bins, num_frames]
            x = input_features[sample_idx, :, :sample_len].unsqueeze(0).unsqueeze(0)
            conv_out = self._apply_causal_conv_stack(x)
            embed = self._flatten_conv_output(conv_out)  # [1, output_frames, embed_dim]

            pos_embed_buffer: torch.Tensor = self.positional_embedding(embed.shape[1])
            positional_embedding = pos_embed_buffer.unsqueeze(0).to(embed.dtype)
            embed = embed + positional_embedding

            output_frames = embed.shape[1]
            all_embeds.append(embed.squeeze(0))  # [output_frames, embed_dim]
            output_lens.append(output_frames)

        hidden_states = nn.utils.rnn.pad_sequence(all_embeds, batch_first=True)
        valid_mask = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=hidden_states.device) for length in output_lens],
            batch_first=True,
        )
        return hidden_states, valid_mask, output_lens

    def _downsample_causal_streaming(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        streaming_state: AuTStreamingState,
    ) -> tuple[torch.Tensor, list[int], AuTStreamingState]:
        """Downsample a single streaming chunk using cached conv border pixels.

        Processes the new mel frames through the causal conv stack (with cached
        border pixels from the previous call), flattens, and adds positional
        embeddings at the correct offset.

        Only supports ``batch_size=1`` (single-stream).

        Args:
            input_features: Packed mel spectrogram for this chunk (no batch dim).
                Shape: [num_mel_bins, new_frames]. Dtype: float.
            feature_lens: Frame count for this chunk.
                Shape: [1]. Dtype: long.
            streaming_state: Current streaming state with conv caches and
                positional offset.

        Returns:
            A tuple of:
                - hidden_states: Embeddings for the new output frames.
                    Shape: [new_output_frames, embed_dim]. Dtype: float.
                - output_lens: Single-element list with the number of new
                    output frames.
                - Updated ``AuTStreamingState``.
        """
        num_frames = int(feature_lens[0].item())
        # [num_mel_bins, num_frames] -> [1, 1, num_mel_bins, num_frames]
        x = input_features[:, :num_frames].unsqueeze(0).unsqueeze(0)

        conv_out, new_conv_caches = self._apply_causal_conv_stack_streaming(x, streaming_state.conv_caches)
        embed = self._flatten_conv_output(conv_out)  # [1, new_output_frames, embed_dim]

        new_output_frames = embed.shape[1]
        offset = streaming_state.num_frames_produced

        pos_embed_buffer: torch.Tensor = self.positional_embedding(offset + new_output_frames)
        positional_embedding = pos_embed_buffer[offset : offset + new_output_frames].unsqueeze(0).to(embed.dtype)
        embed = embed + positional_embedding

        hidden_states = embed.squeeze(0)  # [new_output_frames, embed_dim]

        new_state = AuTStreamingState(
            stft_cache=streaming_state.stft_cache,
            running_max=streaming_state.running_max,
            conv_caches=new_conv_caches,
            kv_caches=streaming_state.kv_caches,
            num_frames_produced=offset + new_output_frames,
        )

        return hidden_states, [new_output_frames], new_state

    def init_streaming_state(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> AuTStreamingState:
        """Create an initial (empty) streaming state for incremental encoding.

        The returned state should be passed to the first ``forward(...,
        streaming_state=...)`` call and then replaced with the state returned
        in the output on each subsequent call.

        Args:
            device: Device for the cache tensors.
            dtype: Dtype for the cache tensors.

        Returns:
            A fresh ``AuTStreamingState`` with zero-filled conv and KV caches.
        """
        num_mel_bins = self.num_mel_bins
        ds_hidden = self.conv2d1.out_channels  # downsample_hidden_size

        freq1 = (num_mel_bins + 1) // 2
        freq2 = (freq1 + 1) // 2
        freq3 = (freq2 + 1) // 2  # noqa: F841 — Kept for documentation

        conv_caches = [
            torch.zeros(1, 1, num_mel_bins, 2, device=device, dtype=dtype),
            torch.zeros(1, ds_hidden, freq1, 2, device=device, dtype=dtype),  # type: ignore
            torch.zeros(1, ds_hidden, freq2, 2, device=device, dtype=dtype),  # type: ignore
        ]

        num_heads = self.layers[0].self_attn.num_heads  # type: ignore
        head_dim = self.layers[0].self_attn.head_dim  # type: ignore
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in self.layers:
            kv_caches.append(
                (
                    torch.zeros(1, num_heads, 0, head_dim, device=device, dtype=dtype),  # type: ignore
                    torch.zeros(1, num_heads, 0, head_dim, device=device, dtype=dtype),  # type: ignore
                )
            )

        return AuTStreamingState(
            stft_cache=None,
            running_max=float("-inf"),
            conv_caches=conv_caches,
            kv_caches=kv_caches,
            num_frames_produced=0,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        causal: bool = False,
        streaming_state: AuTStreamingState | None = None,
    ) -> AuTEncoderOutput:
        """Run the full AuT encoder forward pass.

        Dispatches to the appropriate downsampling strategy, then runs the
        transformer encoder layers and output projection.

        When ``causal=False`` (default), uses the original chunked windowing
        downsampling with bidirectional attention (preserves HuggingFace
        equivalence).

        When ``causal=True``, uses non-chunked causal convolution downsampling
        and causal (autoregressive) attention.

        When ``streaming_state`` is provided, uses the streaming path (implies
        ``causal=True``, batch_size=1).

        Args:
            input_features: Mel spectrogram inputs.
                Non-streaming shape: [batch_size, num_mel_bins, max_frames].
                Streaming shape: [num_mel_bins, total_frames]. Dtype: float.
            feature_lens: Per-sample mel frame counts.
                Shape: [batch_size]. Dtype: long.
            causal: If True, use causal convolution and causal attention.
            streaming_state: If provided, run in streaming mode with cached
                state from the previous call.

        Returns:
            ``AuTEncoderOutput`` with ``last_hidden_state`` containing the
            projected encoder output, ``output_lens`` with per-sample output
            frame counts, and ``streaming_state`` if streaming mode is active.
        """
        if streaming_state is not None:
            # Streaming path: incremental causal encoding with KV cache.
            hidden_states, output_lens, streaming_state = self._downsample_causal_streaming(
                input_features, feature_lens, streaming_state
            )

            new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for encoder_layer, past_kv in zip(self.layers, streaming_state.kv_caches, strict=True):
                hidden_states, new_kv = encoder_layer.forward_streaming(hidden_states, past_kv)  # type: ignore
                new_kv_caches.append(new_kv)

            streaming_state = AuTStreamingState(
                stft_cache=streaming_state.stft_cache,
                running_max=streaming_state.running_max,
                conv_caches=streaming_state.conv_caches,
                kv_caches=new_kv_caches,
                num_frames_produced=streaming_state.num_frames_produced,
            )

            hidden_states = self.ln_post(hidden_states)
            hidden_states = self.proj1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.proj2(hidden_states)
            return AuTEncoderOutput(
                last_hidden_state=hidden_states,
                output_lens=output_lens,
                streaming_state=streaming_state,
            )

        # Non-streaming path.
        if causal:
            hidden_states, valid_mask, output_lens = self._downsample_causal(input_features, feature_lens)
        else:
            hidden_states, valid_mask, output_lens = self._downsample_chunked(input_features, feature_lens)
            max_output_len = max(output_lens)
            chunk_length = self._get_noncausal_chunk_length()
            assert max_output_len <= chunk_length, (
                f"Non-causal AuT requires one attention chunk per sample. "
                f"Got max output length {max_output_len}, chunk length {chunk_length}."
            )

        attention_mask = self._prepare_batched_attention_mask(valid_mask=valid_mask)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                causal=causal,
                padding_mask=valid_mask,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        hidden_states = hidden_states * valid_mask.unsqueeze(-1).to(hidden_states.dtype)
        return AuTEncoderOutput(last_hidden_state=hidden_states, output_lens=output_lens)


def _load_audio_tower_state_dict(
    model_path: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Load audio tower weights from an ASR checkpoint.

    Reads the safetensors index (or a single safetensors file) to find which
    shards contain audio tower weights, then loads only those tensors with
    the ``thinker.audio_tower.`` prefix stripped.

    Args:
        model_path: HuggingFace model ID or local directory path.
        dtype: Target dtype for the loaded tensors.

    Returns:
        State dict with keys matching ``AuTEncoder`` (prefix stripped).
    """

    result = load_safetensors_by_prefix(
        model_path,
        prefixes={"audio_tower": _AUDIO_TOWER_PREFIX},
        dtype=dtype,
    )
    return result["audio_tower"]


def _load_audio_encoder_config(model_path: str) -> Qwen3OmniMoeAudioEncoderConfig:
    """Build an ``AuTEncoder``-compatible config from an ASR model config.json.

    Reads the nested ``thinker_config.audio_config`` and maps its fields into
    a ``Qwen3OmniMoeAudioEncoderConfig`` (which has an identical field set).

    Args:
        model_path: HuggingFace model ID or local directory path.

    Returns:
        Audio encoder config suitable for constructing ``AuTEncoder``.
    """
    is_local = os.path.isdir(model_path)
    if is_local:
        config_path = os.path.join(model_path, "config.json")
    else:
        config_path = hf_hub_download(repo_id=model_path, filename="config.json")

    with open(config_path) as f:
        full_config = json.load(f)

    audio_cfg = full_config["thinker_config"]["audio_config"]
    return Qwen3OmniMoeAudioEncoderConfig(
        d_model=audio_cfg["d_model"],
        encoder_layers=audio_cfg["encoder_layers"],
        encoder_attention_heads=audio_cfg["encoder_attention_heads"],
        encoder_ffn_dim=audio_cfg["encoder_ffn_dim"],
        num_mel_bins=audio_cfg["num_mel_bins"],
        max_source_positions=audio_cfg["max_source_positions"],
        n_window=audio_cfg["n_window"],
        n_window_infer=audio_cfg["n_window_infer"],
        output_dim=audio_cfg["output_dim"],
        conv_chunksize=audio_cfg["conv_chunksize"],
        downsample_hidden_size=audio_cfg["downsample_hidden_size"],
        scale_embedding=audio_cfg["scale_embedding"],
        activation_function=audio_cfg["activation_function"],
        dropout=audio_cfg.get("dropout", 0),
        attention_dropout=audio_cfg.get("attention_dropout", 0),
        activation_dropout=audio_cfg.get("activation_dropout", 0),
    )


class AuTWrapper(nn.Module):
    """Wrapper that runs the AuT encoder on raw audio waveforms.

    Handles resampling, feature extraction, and output length alignment.
    """

    config: Qwen3OmniMoeAudioEncoderConfig

    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        feature_extractor: WhisperFeatureExtractor,
        encoder: AuTEncoder,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = feature_extractor
        self.encoder = encoder

        self.input_sample_rate = 24000
        self.encoder_sample_rate: int = feature_extractor.sampling_rate
        self.frame_rate = 12.5
        self.hidden_size = config.output_dim

        # Whisper STFT requires at least n_fft samples; pad shorter audio to avoid errors.
        self._min_encoder_samples: int = feature_extractor.n_fft

        self.config.sampling_rate = self.input_sample_rate

    @classmethod
    def from_config(
        cls,
        config: Qwen3OmniMoeAudioEncoderConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Create an AuTWrapper with randomly-initialized weights from a config.

        Useful for building test checkpoints or when pretrained weights are not
        needed (e.g. the weights will be loaded from a saved checkpoint later).

        Args:
            config: Audio encoder config specifying architecture dimensions.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with random weights.
        """
        feature_extractor = WhisperFeatureExtractor(
            feature_size=config.num_mel_bins,
            sampling_rate=16000,
        )
        aut_encoder = AuTEncoder(config)
        aut_encoder.to(dtype)  # type: ignore
        return cls(config=config, feature_extractor=feature_extractor, encoder=aut_encoder)

    @classmethod
    def from_asr_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        config: Qwen3OmniMoeAudioEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Load an AuTWrapper from an ASR checkpoint.

        Extracts the audio tower weights from the full ASR model
        (``thinker.audio_tower.*``) and loads them into the local ``AuTEncoder``.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local
                directory path.
            config: Optional override config. If None, derived from the
                checkpoint's ``config.json``.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with pretrained audio encoder weights.
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
        )

        if config is None:
            config = _load_audio_encoder_config(pretrained_model_name_or_path)

        state_dict = _load_audio_tower_state_dict(pretrained_model_name_or_path, dtype)

        aut_encoder = AuTEncoder(config)
        aut_encoder.load_state_dict(state_dict)
        aut_encoder.to(dtype)  # type: ignore

        return cls(config=config, feature_extractor=feature_extractor, encoder=aut_encoder)

    @classmethod
    def from_omni_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        config: Qwen3OmniMoeAudioEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Load an AuTWrapper from a standalone audio encoder checkpoint.

        Loads the HuggingFace audio encoder weights and transfers them into
        the local ``AuTEncoder`` (which has an identical state dict layout).

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
                pointing to a standalone audio encoder checkpoint.
            config: Optional override config. If None, uses the checkpoint config.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with pretrained weights.
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
        )

        hf_encoder: Qwen3OmniMoeAudioEncoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
        if config is None:
            config = hf_encoder.config

        aut_encoder = AuTEncoder(config)
        aut_encoder.load_state_dict(hf_encoder.state_dict())
        aut_encoder.to(dtype)  # type: ignore

        return cls(config=config, feature_extractor=feature_extractor, encoder=aut_encoder)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Qwen3OmniMoeAudioEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AuTWrapper:
        """Load an AuTWrapper from a pretrained checkpoint.

        Automatically detects whether the checkpoint is an ASR model (with
        nested ``thinker_config.audio_config``) or a standalone audio encoder,
        and dispatches to the appropriate loader.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            config: Optional override config. If None, derived from checkpoint.
            dtype: Parameter dtype for the encoder.

        Returns:
            Initialized ``AuTWrapper`` with pretrained weights.
        """
        is_local = os.path.isdir(pretrained_model_name_or_path)

        # Detect checkpoint type by checking for thinker_config in config.json.
        is_asr_checkpoint = False
        try:
            if is_local:
                config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            else:
                config_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="config.json",
                )
            with open(config_path) as f:
                raw_config = json.load(f)
            is_asr_checkpoint = "thinker_config" in raw_config
        except Exception:
            pass

        if is_asr_checkpoint:
            return cls.from_asr_checkpoint(pretrained_model_name_or_path, config=config, dtype=dtype)
        return cls.from_omni_checkpoint(pretrained_model_name_or_path, config=config, dtype=dtype)

    @property
    def device(self) -> torch.device:
        """Return the device of the encoder parameters."""
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder parameters."""
        return next(self.encoder.parameters()).dtype

    def compute_expected_output_length(self, num_samples: int) -> int:
        """Compute the expected number of output frames for a given number of audio samples.

        Args:
            num_samples: Number of input audio samples at ``input_sample_rate``.

        Returns:
            Expected number of output frames.
        """
        samples_per_frame = int(self.input_sample_rate / self.frame_rate)
        return math.ceil(num_samples / samples_per_frame)

    def _get_stft_params(self, device: torch.device) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        """Return STFT parameters for mel feature extraction.

        Returns:
            A tuple of (n_fft, hop_length, window, mel_filters).
        """
        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length
        window = torch.hann_window(n_fft, device=device)
        mel_filters = torch.from_numpy(np.array(self.feature_extractor.mel_filters)).to(device=device, dtype=torch.float32)
        return n_fft, hop_length, window, mel_filters

    def _preprocess_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Downmix stereo to mono, squeeze channel dim, and resample if needed.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long. May be ``None``.

        Returns:
            A tuple of (audio, audio_lengths) with the channel dim removed and
            sample rate converted to ``encoder_sample_rate``.
        """
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)

        audio = audio.squeeze(1)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
            audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        if self.input_sample_rate != self.encoder_sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            )
            if audio_lengths is not None:
                audio_lengths = torch.floor(audio_lengths.float() * self.encoder_sample_rate / self.input_sample_rate).to(
                    torch.long
                )
                audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        return audio, audio_lengths

    def _extract_features(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Whisper mel features from raw audio waveforms.

        Uses the standard centered STFT from the ``WhisperFeatureExtractor``.
        Downmixes stereo to mono, resamples to the encoder sample rate, runs
        the Whisper feature extractor, and returns padded batched mel features
        with per-sample frame lengths.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        audio, audio_lengths = self._preprocess_audio(audio, audio_lengths)

        n_fft, hop_length, window, mel_filters = self._get_stft_params(device=self.device)

        if audio_lengths is None:
            audio_lengths = torch.full(
                (audio.shape[0],),
                audio.shape[-1],
                dtype=torch.long,
                device=audio.device,
            )

        # Match WhisperFeatureExtractor behavior for short clips:
        # each sample is padded to at least n_fft before batch padding.
        effective_lengths = torch.maximum(audio_lengths, torch.full_like(audio_lengths, n_fft))
        target_len = int(effective_lengths.max().item())

        if audio.shape[-1] < target_len:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))
        elif audio.shape[-1] > target_len:
            audio = audio[:, :target_len]

        # Remove tail samples beyond each sequence's valid length so padded
        # regions do not affect STFT/mel features.
        sample_indices = torch.arange(target_len, device=audio.device).unsqueeze(0)
        sample_mask = sample_indices < effective_lengths.unsqueeze(1)
        waveform = audio.to(device=self.device, dtype=torch.float32) * sample_mask.to(dtype=torch.float32)

        # Mirror WhisperFeatureExtractor._torch_extract_fbank_features.
        log_spec = compute_log_mel_spectrogram(waveform, window, mel_filters, n_fft, hop_length)

        # Match WhisperFeatureExtractor.__call__ mask rescaling exactly.
        feature_attention_mask = sample_mask[:, ::hop_length]
        if target_len % hop_length != 0:
            feature_attention_mask = feature_attention_mask[:, :-1]

        audio_feature_lengths = feature_attention_mask.sum(dim=1).to(dtype=torch.long, device=self.device)
        batched_features = log_spec.to(device=self.device, dtype=self.dtype)
        return batched_features, audio_feature_lengths

    def _extract_features_causal(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mel features using a left-padded causal STFT.

        Prepends ``n_fft // 2`` zeros before running ``center=False`` STFT so
        that each frame is aligned to the same window center as the noncausal
        centered STFT.  The last STFT frame is dropped to match the noncausal
        frame count exactly.  Causal running-max normalization is applied: for
        each frame the floor is ``running_max - 8.0`` where ``running_max`` is
        the cumulative maximum of per-frame maxima up to that point.  This
        produces identical output whether the full utterance is processed at
        once (parallel) or chunk-by-chunk (streaming).

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        audio, audio_lengths = self._preprocess_audio(audio, audio_lengths)

        if audio_lengths is None:
            if audio.shape[-1] < self._min_encoder_samples:
                audio = F.pad(audio, (0, self._min_encoder_samples - audio.shape[-1]))
            audio_lengths = torch.full(
                (audio.shape[0],),
                audio.shape[-1],
                dtype=torch.long,
                device=audio.device,
            )

        n_fft, hop_length, window, mel_filters = self._get_stft_params(device=self.device)

        all_log_specs: list[torch.Tensor] = []
        frame_counts: list[int] = []

        for waveform, length in zip(audio, audio_lengths, strict=True):
            waveform_f32 = waveform[: int(length.item())].to(device=self.device, dtype=torch.float32)

            if waveform_f32.numel() < n_fft:
                waveform_f32 = F.pad(waveform_f32, (0, n_fft - waveform_f32.numel()))

            # Left-pad by n_fft // 2 to align frame centers with centered STFT.
            padded = F.pad(waveform_f32, (n_fft // 2, 0))
            stft = torch.stft(
                padded,
                n_fft,
                hop_length,
                window=window,
                center=False,
                return_complex=True,
            )
            # Drop the last frame to match centered STFT frame count.
            magnitudes = stft[..., :-1].abs() ** 2

            mel_spec = mel_filters.T @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()

            # Causal running-max normalization: the floor at each frame is
            # determined by the cumulative maximum of per-frame maxima seen so
            # far, matching what the streaming path computes incrementally.
            per_frame_max = log_spec.max(dim=0, keepdim=True)[0]  # [1, num_frames]
            running_max = torch.cummax(per_frame_max, dim=1)[0]  # [1, num_frames]
            log_spec = torch.maximum(log_spec, running_max.expand_as(log_spec) - 8.0)
            log_spec = (log_spec + 4.0) / 4.0

            num_frames = log_spec.shape[1]
            all_log_specs.append(log_spec)
            frame_counts.append(num_frames)

        batched_features = nn.utils.rnn.pad_sequence(
            [log_spec.T for log_spec in all_log_specs],
            batch_first=True,
        ).permute(0, 2, 1)
        batched_features = batched_features.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor(frame_counts, dtype=torch.long, device=self.device)

        return batched_features, audio_feature_lengths

    def _extract_features_causal_streaming(
        self,
        audio: torch.Tensor,
        stft_cache: torch.Tensor | None,
        running_max: float = float("-inf"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Extract causal mel features for a single streaming chunk.

        Uses left-padded causal STFT (``n_fft // 2`` zeros prepended on the
        first chunk) and causal running-max normalization to match the parallel
        ``_extract_features_causal`` output exactly.

        The parallel path drops the last STFT frame (``stft[..., :-1]``).  To
        match this in streaming, the waveform cache always retains enough
        samples so that the last STFT frame of the full sequence is never
        emitted until the next chunk proves it is not the final frame.  This is
        achieved by keeping ``hop_length`` extra tail samples in the cache
        beyond what is strictly unconsumed.

        Only supports ``batch_size=1``.

        Args:
            audio: Raw audio waveform chunk (mono, already at encoder sample rate).
                Shape: [1, num_samples]. Dtype: float.
            stft_cache: Unconsumed waveform samples carried over from the
                previous chunk.  Shape: [num_leftover_samples]. Dtype: float.
                ``None`` on the very first call.
            running_max: Running maximum of per-frame log-mel maxima from
                previous chunks.

        Returns:
            A tuple of:
                - packed_features: Mel spectrogram for the new frames only.
                    Shape: [num_mel_bins, new_frames]. Dtype: float.
                - audio_feature_lengths: Frame count for this chunk.
                    Shape: [1]. Dtype: long.
                - new_stft_cache: Unconsumed tail samples for the next call.
                    Shape: [num_leftover_samples]. Dtype: float.
                - new_running_max: Updated running maximum.
        """
        n_fft, hop_length, window, mel_filters = self._get_stft_params(device=self.device)

        waveform = audio[0].to(device=self.device, dtype=torch.float32)

        is_first_chunk = stft_cache is None
        if is_first_chunk:
            waveform = F.pad(waveform, (n_fft // 2, 0))
        else:
            waveform = torch.cat([stft_cache, waveform], dim=0)

        total_samples = waveform.shape[0]

        if total_samples < n_fft:
            packed_features = torch.zeros(
                mel_filters.shape[0],
                0,
                device=self.device,
                dtype=self.dtype,
            )
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        num_frames = (total_samples - n_fft) // hop_length + 1

        # Hold back the last frame to match the parallel path's [:-1] drop.
        # The held-back frame's samples stay in the cache and will be
        # re-computed (and emitted) when the next chunk arrives.
        if num_frames <= 1:
            packed_features = torch.zeros(
                mel_filters.shape[0],
                0,
                device=self.device,
                dtype=self.dtype,
            )
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        emit_frames = num_frames - 1
        consumed = (emit_frames - 1) * hop_length + n_fft

        stft = torch.stft(
            waveform[:consumed],
            n_fft,
            hop_length,
            window=window,
            center=False,
            return_complex=True,
        )
        magnitudes = stft.abs() ** 2
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        # Causal running-max normalization matching the parallel path.
        per_frame_max = log_spec.max(dim=0)[0]  # [emit_frames]
        new_running_max = running_max
        normalized_frames: list[torch.Tensor] = []
        for t in range(log_spec.shape[1]):
            frame_max = per_frame_max[t].item()
            new_running_max = max(new_running_max, frame_max)
            frame = torch.maximum(log_spec[:, t], log_spec.new_tensor(new_running_max - 8.0))
            normalized_frames.append(frame)
        if normalized_frames:
            log_spec = torch.stack(normalized_frames, dim=1)

        log_spec = (log_spec + 4.0) / 4.0

        # Cache starts at the beginning of the held-back frame.
        leftover_start = emit_frames * hop_length
        new_stft_cache = waveform[leftover_start:].clone()

        packed_features = log_spec.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor([emit_frames], dtype=torch.long, device=self.device)

        return packed_features, audio_feature_lengths, new_stft_cache, new_running_max

    def init_streaming_state(self) -> AuTStreamingState:
        """Create an initial streaming state for frame-by-frame encoding.

        Returns:
            A fresh ``AuTStreamingState`` ready for the first ``forward`` call
            with ``streaming_state=...``.
        """
        return self.encoder.init_streaming_state(device=self.device, dtype=self.dtype)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        causal: bool = False,
        streaming_state: AuTStreamingState | None = None,
    ) -> CausalAudioEncoderOutput:
        """Encode raw audio waveforms into hidden state embeddings.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths in ``audio``.
                Shape: [batch_size]. Dtype: long.
            causal: If True, use causal feature extraction, causal convolution,
                and causal attention throughout the pipeline.
            streaming_state: If provided, run in incremental streaming mode
                (implies ``causal=True``, ``batch_size=1``). Pass the returned
                ``streaming_state`` from the output into the next call.

        Returns:
            ``CausalAudioEncoderOutput`` with the encoder embeddings.
            When streaming, ``embeds`` contains only the **new** output frames
            and ``streaming_state`` holds the updated state.
        """
        assert 1 <= audio.shape[1] <= 2, f"Number of audio channels must be 1 or 2, but got {audio.shape[1]}."

        if streaming_state is not None:
            # Streaming path: single-sample, causal only.
            audio, _ = self._preprocess_audio(audio, None)

            packed_features, audio_feature_lengths, new_stft_cache, new_running_max = (
                self._extract_features_causal_streaming(audio, streaming_state.stft_cache, streaming_state.running_max)
            )

            streaming_state = AuTStreamingState(
                stft_cache=new_stft_cache,
                running_max=new_running_max,
                conv_caches=streaming_state.conv_caches,
                kv_caches=streaming_state.kv_caches,
                num_frames_produced=streaming_state.num_frames_produced,
            )

            if audio_feature_lengths[0].item() == 0:
                # Not enough audio to produce any mel frames yet.
                return CausalAudioEncoderOutput(
                    embeds=torch.zeros(1, 0, self.hidden_size, device=self.device, dtype=self.dtype),
                )

            encoder_output = self.encoder(
                input_features=packed_features,
                feature_lens=audio_feature_lengths,
                streaming_state=streaming_state,
            )

            embeds = encoder_output.last_hidden_state.unsqueeze(0)  # [1, new_frames, output_dim]
            return CausalAudioEncoderOutput(
                embeds=embeds,
            )

        # Non-streaming path.
        num_samples = audio.shape[2]
        expected_output_length = self.compute_expected_output_length(num_samples)

        if causal:
            batched_features, audio_feature_lengths = self._extract_features_causal(audio, audio_lengths=audio_lengths)
        else:
            batched_features, audio_feature_lengths = self._extract_features(audio, audio_lengths=audio_lengths)

        encoder_output = self.encoder(
            input_features=batched_features,
            feature_lens=audio_feature_lengths,
            causal=causal,
        )
        last_hidden_state = encoder_output.last_hidden_state
        embeds = last_hidden_state

        actual_output_length = embeds.shape[1]
        if actual_output_length > expected_output_length:
            embeds = embeds[:, :expected_output_length]
        elif actual_output_length < expected_output_length:
            # Pad with zeros to match the expected length. The extra frames will be
            # masked out by audio_embeds_mask in get_audio_input_embeds, so they do
            # not affect the model. This can happen in causal mode where cumulative
            # rounding in the stride-2 conv stack produces fewer frames than the
            # ceil(num_samples / samples_per_frame) formula predicts.
            embeds = F.pad(embeds, (0, 0, 0, expected_output_length - actual_output_length))

        return CausalAudioEncoderOutput(embeds=embeds)


# ── from modules/voxtral_encoder.py ──

class VoxtralRealtimeConv1dCacheLayer:
    """Cache for a single causal Conv1d layer's left-padding state."""

    def __init__(self) -> None:
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(
        self,
        hidden_states: torch.Tensor,
        conv_module: "VoxtralRealtimeCausalConv1d",
    ) -> None:
        """Initialize the cache on first use."""
        self.left_pad = conv_module.left_pad
        self.in_channels = conv_module.in_channels
        self.cache = torch.zeros(
            hidden_states.shape[0],
            self.in_channels,
            self.left_pad,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.is_initialized = True

    def update(
        self,
        hidden_states: torch.Tensor,
        conv_module: "VoxtralRealtimeCausalConv1d | None" = None,
    ) -> torch.Tensor:
        """Return the current padding and update the cache with new states."""
        if not self.is_initialized and conv_module is not None:
            self.lazy_initialization(hidden_states, conv_module)
        elif not self.is_initialized:
            raise ValueError("Cache not initialized. Provide conv_module on first call.")

        assert self.cache is not None
        if self.left_pad > 0:
            shortfall = max(0, self.left_pad - hidden_states.shape[-1])
            if shortfall > 0:
                padding_states = torch.cat([self.cache[:, :, -shortfall:], hidden_states], dim=-1)
            else:
                padding_states = hidden_states[:, :, -self.left_pad :]
        else:
            padding_states = torch.empty(
                hidden_states.shape[0],
                self.in_channels,
                0,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        current_cache = self.cache.clone()
        self.cache.copy_(padding_states)
        return current_cache


class VoxtralRealtimeConv1dPaddingCache:
    """Container for per-layer conv1d padding caches used during streaming."""

    def __init__(self) -> None:
        self.layers: dict[str, VoxtralRealtimeConv1dCacheLayer] = {}

    def update(
        self,
        hidden_states: torch.Tensor,
        cache_key: str,
        conv_module: "VoxtralRealtimeCausalConv1d",
    ) -> torch.Tensor:
        """Pad hidden_states using cached left-padding for the given layer."""
        if cache_key not in self.layers:
            self.layers[cache_key] = VoxtralRealtimeConv1dCacheLayer()
        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        return torch.cat([padding_states, hidden_states], dim=-1)


class SlidingWindowVoxtralKVCache(Cache):
    """Fixed-size sliding-window KV cache for compile-compatible streaming.

    Uses shift-left + append updates and always returns fixed-shape buffers.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        window_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._window_size = window_size
        self._num_layers = num_layers
        self._total_seen_tokens = 0
        self._valid_len = 0
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        for _ in range(num_layers):
            k = torch.zeros(batch_size, num_kv_heads, window_size, head_dim, dtype=dtype, device=device)
            v = torch.zeros(batch_size, num_kv_heads, window_size, head_dim, dtype=dtype, device=device)
            self.key_cache.append(k)
            self.value_cache.append(v)
            torch._dynamo.mark_static_address(k)
            torch._dynamo.mark_static_address(v)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_buf = self.key_cache[layer_idx]
        v_buf = self.value_cache[layer_idx]
        k_buf[:, :, :-1, :].copy_(k_buf[:, :, 1:, :].clone())
        k_buf[:, :, -1:, :].copy_(key_states)
        v_buf[:, :, :-1, :].copy_(v_buf[:, :, 1:, :].clone())
        v_buf[:, :, -1:, :].copy_(value_states)
        return k_buf, v_buf

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seen_tokens

    def get_kv_len(self) -> int:
        return self._valid_len

    def get_max_cache_shape(self) -> list[int]:
        return [self._window_size]

    def step(self) -> None:
        self._total_seen_tokens += 1
        self._valid_len = min(self._valid_len + 1, self._window_size)

    def reset(self) -> None:
        self._total_seen_tokens = 0
        self._valid_len = 0
        for k, v in zip(self.key_cache, self.value_cache):
            k.zero_()
            v.zero_()


class StaticVoxtralConv1dPaddingCache(VoxtralRealtimeConv1dPaddingCache):
    """Pre-allocated conv1d padding cache with static tensor addresses."""

    def __init__(
        self,
        layer_specs: list[tuple[str, int, int]],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        for cache_key, in_channels, left_pad in layer_specs:
            layer = VoxtralRealtimeConv1dCacheLayer()
            layer.left_pad = left_pad
            layer.in_channels = in_channels
            layer.cache = torch.zeros(batch_size, in_channels, left_pad, dtype=dtype, device=device)
            layer.is_initialized = True
            torch._dynamo.mark_static_address(layer.cache)
            self.layers[cache_key] = layer

    def reset(self) -> None:
        for layer in self.layers.values():
            if layer.cache is not None:
                layer.cache.zero_()


# ---------------------------------------------------------------------------
# Encoder output
# ---------------------------------------------------------------------------


@dataclass
class VoxtralRealtimeEncoderOutput(BaseModelOutputWithPast):
    """Output type for the Voxtral encoder, adding a padding cache field."""

    padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------


class VoxtralRealtimeRotaryEmbedding(nn.Module):
    """RoPE implementation for the Voxtral encoder."""

    inv_freq: torch.Tensor

    def __init__(self, config: VoxtralRealtimeEncoderConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        dim = config.head_dim
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary position embeddings."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Causal Conv1d
# ---------------------------------------------------------------------------


class VoxtralRealtimeCausalConv1d(nn.Conv1d):
    """Causal Conv1d that supports streaming via a padding cache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.cache_key = cache_key

    @cached_property
    def left_pad(self) -> int:
        """Number of left-padding samples needed for causal convolution."""
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        """Run causal conv1d, using padding_cache if in streaming mode."""
        if padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


# ---------------------------------------------------------------------------
# RMS Norm
# ---------------------------------------------------------------------------


class VoxtralRealtimeRMSNorm(nn.Module):
    """RMS normalization layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class VoxtralRealtimeAttention(nn.Module):
    """Multi-headed attention with RoPE and sliding-window causal masking."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run multi-head attention with RoPE."""
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        return self.o_proj(attn_output)


class VoxtralRealtimeSdpaAttention(VoxtralRealtimeAttention):
    """SDPA-backed attention using torch.nn.functional.scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run multi-head attention via SDPA with RoPE.

        Args:
            hidden_states: Input to the attention layer.
                Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
            position_embeddings: Tuple of (cos, sin) rotary embeddings.
                Each shape: [batch_size, seq_len, head_dim]. Dtype: float.
            attention_mask: Additive causal mask applied before softmax.
                Shape: [1, 1, seq_len, total_seq_len]. Dtype: float.
            past_key_values: KV cache for streaming inference.

        Returns:
            Attention output. Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
        """
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Convert additive float mask to bool mask for SDPA (True = attend, False = mask).
        # SDPA expects attn_mask as either a boolean mask or an additive float mask.
        # The existing mask is already additive (0.0 = attend, large negative = mask),
        # so we pass it directly.
        sdpa_mask = attention_mask

        # Handle GQA by expanding KV heads to match query heads.
        if self.num_key_value_groups > 1:
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        return self.o_proj(attn_output)


class VoxtralRealtimeFlashAttention2(VoxtralRealtimeAttention):
    """Flash Attention 2 backed attention using flash_attn.flash_attn_func."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run multi-head attention via Flash Attention 2 with RoPE.

        Args:
            hidden_states: Input to the attention layer.
                Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
            position_embeddings: Tuple of (cos, sin) rotary embeddings.
                Each shape: [batch_size, seq_len, head_dim]. Dtype: float.
            attention_mask: Unused; Flash Attention handles causality natively.
            past_key_values: KV cache for streaming inference.

        Returns:
            Attention output. Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
        """
        try:
            from flash_attn import flash_attn_func  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "flash_attn is required for flash_attention_2 backend. Install it with: pip install flash-attn"
            ) from exc

        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        # Projections: [batch_size, seq_len, num_heads, head_dim] (no transpose for flash_attn).
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE: temporarily transpose to [batch_size, num_heads, seq_len, head_dim].
        cos, sin = position_embeddings
        query_states_t = query_states.transpose(1, 2)
        key_states_t = key_states.transpose(1, 2)
        query_states_t, key_states_t = _apply_rotary_pos_emb(query_states_t, key_states_t, cos, sin)
        # Transpose back to [batch_size, seq_len, num_heads, head_dim].
        query_states = query_states_t.transpose(1, 2)
        key_states = key_states_t.transpose(1, 2)

        if past_key_values is not None:
            # Cache expects [batch_size, num_heads, seq_len, head_dim].
            key_states_c, value_states_c = past_key_values.update(
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                self.layer_idx,
            )
            # Transpose back to [batch_size, seq_len, num_heads, head_dim].
            key_states = key_states_c.transpose(1, 2)
            value_states = value_states_c.transpose(1, 2)

        dropout_p = self.attention_dropout if self.training else 0.0
        # flash_attn_func expects [batch_size, seq_len, num_heads, head_dim].
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_p,
            softmax_scale=self.scaling,
            causal=True,
            window_size=(self.config.sliding_window - 1, 0),
        )

        # Reshape: [batch_size, seq_len, num_heads * head_dim].
        attn_output = attn_output.reshape(bsz, attn_output.shape[1], -1).contiguous()
        return self.o_proj(attn_output)


VOXTRAL_ATTENTION_CLASSES: dict[str, type[VoxtralRealtimeAttention]] = {
    "eager": VoxtralRealtimeAttention,
    "sdpa": VoxtralRealtimeSdpaAttention,
    "flash_attention_2": VoxtralRealtimeFlashAttention2,
}


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class VoxtralRealtimeMLP(nn.Module):
    """Gated MLP (SwiGLU-style) used in encoder layers."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated MLP."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Embedder (mel -> hidden via two causal convs)
# ---------------------------------------------------------------------------


class VoxtralRealtimeEmbedder(nn.Module):
    """Front-end: two causal Conv1d layers mapping mel bins to hidden_size at 50Hz."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__()
        self.conv1 = VoxtralRealtimeCausalConv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, cache_key="conv1")
        self.conv2 = VoxtralRealtimeCausalConv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=2, cache_key="conv2"
        )

    def forward(
        self,
        input_features: torch.Tensor,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        """Convert mel spectrogram to encoder input embeddings.

        Args:
            input_features: Mel spectrogram.
                Shape: [batch_size, num_mel_bins, num_frames]. Dtype: float.
            padding_cache: Optional streaming conv padding cache.

        Returns:
            Embeddings. Shape: [batch_size, num_encoder_tokens, hidden_size]. Dtype: float.
        """
        x = F.gelu(self.conv1(input_features, padding_cache=padding_cache))
        x = F.gelu(self.conv2(x, padding_cache=padding_cache))
        return x.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class VoxtralRealtimeEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm attention and MLP."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        attn_cls = VOXTRAL_ATTENTION_CLASSES.get(config._attn_implementation, VoxtralRealtimeAttention)
        self.self_attn = attn_cls(config, layer_idx)
        self.self_attn_layer_norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layer_norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VoxtralRealtimeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run one encoder layer."""
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Sliding-window causal mask builder
# ---------------------------------------------------------------------------


def _make_sliding_window_causal_mask(
    seq_len: int,
    past_len: int,
    sliding_window: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a [1, 1, seq_len, past_len + seq_len] causal attention mask.

    Positions outside the sliding window are masked with a large negative value.
    """
    total_len = past_len + seq_len
    mask = torch.full((seq_len, total_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    for i in range(seq_len):
        abs_pos = past_len + i
        start = 0
        if sliding_window is not None:
            start = max(0, abs_pos - sliding_window + 1)
        mask[i, start : abs_pos + 1] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


def _make_fixed_sliding_window_mask(
    valid_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a fixed-shape ``[1, 1, 1, window_size]`` mask for sliding KV buffers."""
    mask = torch.zeros(1, 1, 1, window_size, dtype=dtype, device=device)
    if valid_len < window_size:
        mask[:, :, :, : window_size - valid_len] = torch.finfo(dtype).min
    return mask


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class VoxtralRealtimeEncoder(PreTrainedModel):
    """Voxtral Realtime audio encoder (causal transformer over mel spectrograms).

    Produces hidden states at 50Hz (one token per 2 mel frames).  Supports
    streaming via KV cache and conv1d padding cache.
    """

    config_class = VoxtralRealtimeEncoderConfig  # type: ignore[assignment]
    main_input_name = "input_features"
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embedder = VoxtralRealtimeEmbedder(config)
        self.layers = nn.ModuleList(
            [VoxtralRealtimeEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralRealtimeRotaryEmbedding(config)
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        use_padding_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> VoxtralRealtimeEncoderOutput:
        """Run the encoder on mel features or pre-computed embeddings.

        Args:
            input_features: Log-mel spectrogram.
                Shape: [batch_size, num_mel_bins, num_frames]. Dtype: float.
            past_key_values: KV cache for streaming.
            padding_cache: Conv1d padding cache for streaming.
            inputs_embeds: Pre-computed embeddings (alternative to input_features).
                Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
            use_cache: Whether to use/return KV cache.
            use_padding_cache: Whether to use/return padding cache.
            position_ids: Optional pre-computed position IDs for RoPE.
            attention_mask: Optional pre-computed attention mask.

        Returns:
            VoxtralRealtimeEncoderOutput with last_hidden_state, past_key_values,
            and padding_cache.
        """
        if (input_features is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_features or inputs_embeds.")

        if use_padding_cache and padding_cache is None:
            padding_cache = VoxtralRealtimeConv1dPaddingCache()

        if inputs_embeds is None:
            inputs_embeds = self.embedder(input_features, padding_cache if use_padding_cache else None)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens  # type: ignore
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is not None:
            causal_mask = attention_mask
        elif self.config._attn_implementation == "flash_attention_2":
            # Flash Attention handles causality and sliding window natively.
            causal_mask = None
        else:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            causal_mask = _make_sliding_window_causal_mask(
                seq_len=inputs_embeds.shape[1],  # type: ignore
                past_len=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                device=inputs_embeds.device,  # type: ignore
                dtype=inputs_embeds.dtype,  # type: ignore
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values if use_cache else None,
            )

        hidden_states = self.norm(hidden_states)
        return VoxtralRealtimeEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            padding_cache=padding_cache if use_padding_cache else None,
        )

    def transformer_forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache,
    ) -> torch.Tensor:
        """Run transformer-only path for compile-friendly streaming."""
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Multi-modal projector (frame-stack + MLP)
# ---------------------------------------------------------------------------


class VoxtralRealtimeMultiModalProjector(nn.Module):
    """Frame-stacks encoder tokens by downsample_factor, then projects via MLP.

    Reduces the 50Hz encoder output to 12.5Hz adapter embeddings matching the
    LLM hidden size.
    """

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__()
        output_size = config.projector_output_size or config.hidden_size
        self.linear_1 = nn.Linear(config.hidden_size * config.downsample_factor, output_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(output_size, output_size, bias=False)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Project frame-stacked encoder features.

        Args:
            audio_features: Frame-stacked encoder output.
                Shape: [batch_size, num_adapter_tokens, hidden_size * downsample_factor].
                Dtype: float.

        Returns:
            Projected embeddings.
            Shape: [batch_size, num_adapter_tokens, output_size]. Dtype: float.
        """
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ── from modules/voxtral_wrapper.py ──

import math
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from transformers import WhisperFeatureExtractor
from transformers.cache_utils import DynamicCache



DOWNSAMPLE_FACTOR = 4
ENCODER_STRIDE = 2
MEL_FRAMES_PER_ADAPTER_TOKEN = DOWNSAMPLE_FACTOR * ENCODER_STRIDE  # 8


@dataclass
class VoxtralStreamingState:
    """Streaming state for the Voxtral encoder.

    Holds the KV cache, conv1d padding cache, and mel feature buffer needed
    to run the encoder incrementally at 12.5Hz (80ms per adapter token).
    """

    kv_cache: DynamicCache | SlidingWindowVoxtralKVCache
    padding_cache: VoxtralRealtimeConv1dPaddingCache
    stft_cache: torch.Tensor | None = None
    running_max: float = float("-inf")
    mel_buffer: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset all caches for session reuse without reallocating tensors."""
        if hasattr(self.kv_cache, "reset"):
            self.kv_cache.reset()
        if hasattr(self.padding_cache, "reset"):
            self.padding_cache.reset()
        self.stft_cache = None
        self.running_max = float("-inf")
        self.mel_buffer = None
        pool_idx = getattr(self, "_pool_idx", None)
        pool_owner = getattr(self, "_pool_owner", None)
        if pool_idx is not None and pool_owner is not None:
            if pool_idx not in pool_owner._cache_available:
                pool_owner._cache_available.append(pool_idx)


def _load_encoder_and_projector_state_dicts(
    pretrained_model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load ``audio_tower`` and ``multi_modal_projector`` weights from safetensors.

    Handles both sharded (``model.safetensors.index.json``) and single-file
    checkpoints, and works with both local directories and HuggingFace Hub IDs.

    Args:
        pretrained_model_name_or_path: HuggingFace model ID or local path.
        dtype: Target dtype for the loaded tensors.

    Returns:
        Tuple of (encoder_state_dict, projector_state_dict) with prefixes
        stripped (e.g. ``audio_tower.layers.0.`` becomes ``layers.0.``).
    """

    result = load_safetensors_by_prefix(
        pretrained_model_name_or_path,
        prefixes={
            "encoder": "audio_tower.",
            "projector": "multi_modal_projector.",
        },
        dtype=dtype,
    )
    return result["encoder"], result["projector"]


def _collect_conv_layer_specs(
    encoder: VoxtralRealtimeEncoder,
) -> list[tuple[str, int, int]]:
    """Walk encoder causal conv1d layers and return cache specs."""
    specs: list[tuple[str, int, int]] = []
    for module in encoder.modules():
        if isinstance(module, VoxtralRealtimeCausalConv1d):
            specs.append((module.cache_key, module.in_channels, module.left_pad))
    return specs


class VoxtralWrapper(nn.Module):
    """Wrapper that runs the Voxtral Realtime encoder on raw audio waveforms.

    Handles resampling, mel feature extraction, encoder + projector forward,
    and streaming state management.
    """

    config: VoxtralRealtimeEncoderConfig

    def __init__(
        self,
        config: VoxtralRealtimeEncoderConfig,
        feature_extractor: WhisperFeatureExtractor,
        encoder: VoxtralRealtimeEncoder,
        projector: VoxtralRealtimeMultiModalProjector | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.projector = projector
        self.register_buffer(
            "_mel_filters",
            torch.as_tensor(np.array(feature_extractor.mel_filters), dtype=torch.float32).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_stft_window",
            torch.hann_window(feature_extractor.n_fft, periodic=True, dtype=torch.float32),
            persistent=False,
        )
        self._use_static_cache = False
        self._cache_pool: list[tuple[SlidingWindowVoxtralKVCache, StaticVoxtralConv1dPaddingCache]] | None = None
        self._cache_available: list[int] = []
        self._compiled_transformer: object | None = None

        self.input_sample_rate = 24000
        self.encoder_sample_rate: int = feature_extractor.sampling_rate
        self.frame_rate = 12.5

        if config.skip_projector:
            self.hidden_size = config.hidden_size * config.downsample_factor
        else:
            self.hidden_size = config.projector_output_size or config.hidden_size

        self._min_encoder_samples: int = feature_extractor.n_fft
        self.config.sampling_rate = self.input_sample_rate

    @classmethod
    def from_config(
        cls,
        config: VoxtralRealtimeEncoderConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "VoxtralWrapper":
        """Create a VoxtralWrapper with randomly-initialized weights.

        Args:
            config: Encoder config specifying architecture dimensions.
            dtype: Parameter dtype.

        Returns:
            Initialized ``VoxtralWrapper`` with random weights.
        """
        feature_extractor = WhisperFeatureExtractor(
            feature_size=config.num_mel_bins,
            sampling_rate=16000,
        )
        encoder = VoxtralRealtimeEncoder(config)
        encoder.to(dtype)  # type: ignore
        projector: VoxtralRealtimeMultiModalProjector | None = None
        if not config.skip_projector:
            projector = VoxtralRealtimeMultiModalProjector(config)
            projector.to(dtype)  # type: ignore
        return cls(config=config, feature_extractor=feature_extractor, encoder=encoder, projector=projector)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
        config: VoxtralRealtimeEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "VoxtralWrapper":
        """Load a VoxtralWrapper from a pretrained HuggingFace checkpoint.

        Reads the config directly from ``config.json`` (no ``AutoConfig``
        dependency on the ``voxtral_realtime`` model type) and loads only the
        ``audio_tower.*`` and ``multi_modal_projector.*`` weights from the
        safetensors shards.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            config: Optional override config. If None, derived from checkpoint.
            dtype: Parameter dtype.

        Returns:
            Initialized ``VoxtralWrapper`` with pretrained encoder and projector.
        """
        if config is None:
            config = VoxtralRealtimeEncoderConfig.from_pretrained(pretrained_model_name_or_path)

        wrapper = cls.from_config(config, dtype=dtype)

        encoder_state_dict, projector_state_dict = _load_encoder_and_projector_state_dicts(
            pretrained_model_name_or_path, dtype=dtype
        )

        wrapper.encoder.load_state_dict(encoder_state_dict, strict=False)
        if wrapper.projector is not None:
            wrapper.projector.load_state_dict(projector_state_dict)

        return wrapper

    @property
    def output_embedding_scale(self) -> float:
        """Return the scalar applied after the downsampling projector."""
        return self.config.output_embedding_scale

    @property
    def device(self) -> torch.device:
        """Return the device of the encoder parameters."""
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder parameters."""
        return next(self.encoder.parameters()).dtype

    def compute_expected_output_length(self, num_samples: int) -> int:
        """Compute the expected number of output frames for a given number of audio samples.

        Args:
            num_samples: Number of input audio samples at ``input_sample_rate``.

        Returns:
            Expected number of adapter output frames at 12.5Hz.
        """
        samples_per_frame = int(self.input_sample_rate / self.frame_rate)
        return math.ceil(num_samples / samples_per_frame)

    def _preprocess_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Downmix, squeeze, clamp lengths, and resample audio to encoder sample rate.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths, or None.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - audio: Mono waveform. Shape: [batch_size, num_samples].
                - audio_lengths: Updated lengths at encoder sample rate, or None.
        """
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)
        audio = audio.squeeze(1)

        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
            audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        if self.input_sample_rate != self.encoder_sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            )
            if audio_lengths is not None:
                audio_lengths = torch.floor(audio_lengths.float() * self.encoder_sample_rate / self.input_sample_rate).to(
                    torch.long
                )
                audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        return audio, audio_lengths

    def _extract_features(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Whisper mel features from raw audio waveforms.

        Uses the standard centered STFT from the ``WhisperFeatureExtractor``.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        audio, audio_lengths = self._preprocess_audio(audio, audio_lengths)

        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length

        if audio_lengths is None:
            audio_lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long, device=audio.device)

        effective_lengths = torch.maximum(audio_lengths, torch.full_like(audio_lengths, n_fft))
        target_len = int(effective_lengths.max().item())

        if audio.shape[-1] < target_len:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))
        elif audio.shape[-1] > target_len:
            audio = audio[:, :target_len]

        sample_indices = torch.arange(target_len, device=audio.device).unsqueeze(0)
        sample_mask = sample_indices < effective_lengths.unsqueeze(1)
        waveform = audio.to(device=self.device, dtype=torch.float32) * sample_mask.to(dtype=torch.float32)

        window = self._stft_window
        mel_filters = self._mel_filters
        log_spec = compute_log_mel_spectrogram(waveform, window, mel_filters, n_fft, hop_length)

        feature_attention_mask = sample_mask[:, ::hop_length]
        if target_len % hop_length != 0:
            feature_attention_mask = feature_attention_mask[:, :-1]

        audio_feature_lengths = feature_attention_mask.sum(dim=1).to(dtype=torch.long, device=self.device)
        batched_features = log_spec.to(device=self.device, dtype=self.dtype)
        return batched_features, audio_feature_lengths

    def _extract_features_causal(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mel features using a left-padded causal STFT.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        audio, audio_lengths = self._preprocess_audio(audio, audio_lengths)

        if audio_lengths is None:
            if audio.shape[-1] < self._min_encoder_samples:
                audio = F.pad(audio, (0, self._min_encoder_samples - audio.shape[-1]))
            audio_lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long, device=audio.device)

        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length
        mel_filters = self._mel_filters
        window = self._stft_window

        all_log_specs: list[torch.Tensor] = []
        frame_counts: list[int] = []

        for waveform, length in zip(audio, audio_lengths, strict=True):
            waveform_f32 = waveform[: int(length.item())].to(device=self.device, dtype=torch.float32)
            if waveform_f32.numel() < n_fft:
                waveform_f32 = F.pad(waveform_f32, (0, n_fft - waveform_f32.numel()))

            padded = F.pad(waveform_f32, (n_fft // 2, 0))
            stft = torch.stft(padded, n_fft, hop_length, window=window, center=False, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = mel_filters.T @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()

            per_frame_max = log_spec.max(dim=0, keepdim=True)[0]
            running_max = torch.cummax(per_frame_max, dim=1)[0]
            log_spec = torch.maximum(log_spec, running_max.expand_as(log_spec) - 8.0)
            log_spec = (log_spec + 4.0) / 4.0

            all_log_specs.append(log_spec)
            frame_counts.append(log_spec.shape[1])

        batched_features = nn.utils.rnn.pad_sequence([s.T for s in all_log_specs], batch_first=True).permute(0, 2, 1)
        batched_features = batched_features.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor(frame_counts, dtype=torch.long, device=self.device)
        return batched_features, audio_feature_lengths

    def _extract_features_causal_streaming(
        self,
        audio: torch.Tensor,
        stft_cache: torch.Tensor | None,
        running_max: float = float("-inf"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Extract causal mel features for a single streaming chunk.

        Args:
            audio: Raw audio chunk (mono, at encoder sample rate).
                Shape: [1, num_samples]. Dtype: float.
            stft_cache: Leftover waveform from previous chunk.
                Shape: [num_leftover_samples]. Dtype: float. None on first call.
            running_max: Running maximum of per-frame log-mel maxima.

        Returns:
            Tuple of (packed_features, audio_feature_lengths, new_stft_cache, new_running_max).
        """
        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length
        mel_filters = self._mel_filters
        window = self._stft_window

        waveform = audio[0].to(device=self.device, dtype=torch.float32)

        is_first_chunk = stft_cache is None
        if is_first_chunk:
            waveform = F.pad(waveform, (n_fft // 2, 0))
        else:
            waveform = torch.cat([stft_cache, waveform], dim=0)

        total_samples = waveform.shape[0]

        if total_samples < n_fft:
            packed_features = torch.zeros(mel_filters.shape[0], 0, device=self.device, dtype=self.dtype)
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        num_frames = (total_samples - n_fft) // hop_length + 1
        if num_frames <= 1:
            packed_features = torch.zeros(mel_filters.shape[0], 0, device=self.device, dtype=self.dtype)
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        emit_frames = num_frames - 1
        consumed = (emit_frames - 1) * hop_length + n_fft

        stft = torch.stft(waveform[:consumed], n_fft, hop_length, window=window, center=False, return_complex=True)
        magnitudes = stft.abs() ** 2
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        per_frame_max = log_spec.max(dim=0)[0]
        running_max_tensor = torch.full_like(per_frame_max, running_max)
        running_max_vals = torch.cummax(torch.maximum(per_frame_max, running_max_tensor), dim=0)[0]
        new_running_max = running_max_vals[-1].item()
        log_spec = torch.maximum(log_spec, (running_max_vals - 8.0).unsqueeze(0).expand_as(log_spec))
        log_spec = (log_spec + 4.0) / 4.0

        leftover_start = emit_frames * hop_length
        new_stft_cache = waveform[leftover_start:].clone()

        packed_features = log_spec.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor([emit_frames], dtype=torch.long, device=self.device)
        return packed_features, audio_feature_lengths, new_stft_cache, new_running_max

    def _frame_stack_and_project(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """Frame-stack encoder tokens by downsample_factor, then optionally project and scale.

        When ``config.skip_projector`` is True the frame-stacked tensor is
        returned directly without projection or scaling.

        Args:
            encoder_hidden: Encoder output.
                Shape: [batch_size, num_encoder_tokens, hidden_size]. Dtype: float.

        Returns:
            Adapter embeddings.
            Shape: [batch_size, num_adapter_tokens, hidden_size * downsample_factor]
            (skip_projector) or [batch_size, num_adapter_tokens, projector_output_size].
            Dtype: float.
        """
        stacked = encoder_hidden.reshape(
            encoder_hidden.shape[0], -1, self.config.hidden_size * self.config.downsample_factor
        )
        if self.config.skip_projector:
            return stacked
        assert self.projector is not None
        projected = self.projector(stacked)
        if self.output_embedding_scale != 1.0:
            projected = projected * self.output_embedding_scale
        return projected

    def init_streaming_state(self) -> VoxtralStreamingState:
        """Create an initial streaming state for frame-by-frame encoding.

        When the encoder has been compiled via ``compile_encoder()``, this
        acquires a pre-allocated cache set from the pool so the same Python
        objects are reused across sessions.

        Returns:
            A fresh ``VoxtralStreamingState`` ready for the first ``forward`` call.
        """
        if self._use_static_cache and self._cache_pool is not None and self._cache_available:
            idx = self._cache_available.pop()
            kv_cache, padding_cache = self._cache_pool[idx]
            kv_cache.reset()
            padding_cache.reset()
            state = VoxtralStreamingState(kv_cache=kv_cache, padding_cache=padding_cache)
            state._pool_idx = idx  # type: ignore[attr-defined]
            state._pool_owner = self  # type: ignore[attr-defined]
            return state
        if self._use_static_cache:
            return self.init_static_streaming_state()
        return VoxtralStreamingState(
            kv_cache=DynamicCache(),
            padding_cache=VoxtralRealtimeConv1dPaddingCache(),
        )

    def init_static_streaming_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> VoxtralStreamingState:
        """Create a streaming state with pre-allocated static caches."""
        device = device or self.device
        dtype = dtype or self.dtype
        config = self.config

        kv_cache = SlidingWindowVoxtralKVCache(
            num_layers=config.num_hidden_layers,
            batch_size=batch_size,
            num_kv_heads=config.num_key_value_heads,
            window_size=config.sliding_window,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )

        conv_layer_specs = _collect_conv_layer_specs(self.encoder)
        padding_cache = StaticVoxtralConv1dPaddingCache(
            layer_specs=conv_layer_specs,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

        return VoxtralStreamingState(
            kv_cache=kv_cache,
            padding_cache=padding_cache,
        )

    def compile_encoder(self, max_sessions: int = 2) -> None:
        """Compile encoder transformer core with static sliding-window caches."""
        self._use_static_cache = True
        device = self.device
        dtype = self.dtype
        config = self.config
        conv_specs = _collect_conv_layer_specs(self.encoder)

        self._cache_pool = []
        for _ in range(max_sessions):
            kv = SlidingWindowVoxtralKVCache(
                num_layers=config.num_hidden_layers,
                batch_size=1,
                num_kv_heads=config.num_key_value_heads,
                window_size=config.sliding_window,
                head_dim=config.head_dim,
                dtype=dtype,
                device=device,
            )
            pad = StaticVoxtralConv1dPaddingCache(
                layer_specs=conv_specs,
                batch_size=1,
                dtype=dtype,
                device=device,
            )
            self._cache_pool.append((kv, pad))
        self._cache_available = list(range(max_sessions))

        self._compiled_transformer = torch.compile(
            self.encoder.transformer_forward,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        encoder_past_key_values: object = None,
        padding_cache: object = None,
        use_streaming: bool | None = None,
        causal: bool = False,
        streaming_state: VoxtralStreamingState | None = None,
    ) -> CausalAudioEncoderOutput:
        """Encode raw audio waveforms into hidden state embeddings.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.
            encoder_past_key_values: Must be None (not used in this path).
            padding_cache: Must be None (not used in this path).
            use_streaming: Must be False (streaming_state controls streaming mode).
            causal: If True, use causal feature extraction.
            streaming_state: If provided, run in incremental streaming mode.

        Returns:
            ``CausalAudioEncoderOutput`` with encoder + projector embeddings.
        """
        assert encoder_past_key_values is None, "VoxtralWrapper: encoder_past_key_values must be None."
        assert padding_cache is None, "VoxtralWrapper: padding_cache must be None."
        assert not use_streaming, "VoxtralWrapper: use_streaming must be False."
        assert 1 <= audio.shape[1] <= 2, f"Number of audio channels must be 1 or 2, got {audio.shape[1]}."

        if streaming_state is not None:
            return self._forward_streaming(audio, streaming_state)

        return self._forward_batch(audio, audio_lengths, causal=causal)

    def _forward_streaming(
        self,
        audio: torch.Tensor,
        streaming_state: VoxtralStreamingState,
    ) -> CausalAudioEncoderOutput:
        """Streaming forward: process one chunk at 12.5Hz.

        Args:
            audio: Raw audio chunk.
                Shape: [1, num_channels, num_samples]. Dtype: float.
            streaming_state: Current streaming state.

        Returns:
            ``CausalAudioEncoderOutput`` with new adapter embeddings and updated state.
        """
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)
        audio = audio.squeeze(1)  # [1, num_samples]

        if self.input_sample_rate != self.encoder_sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            )

        packed_features, audio_feature_lengths, new_stft_cache, new_running_max = self._extract_features_causal_streaming(
            audio, streaming_state.stft_cache, streaming_state.running_max
        )

        new_state = VoxtralStreamingState(
            kv_cache=streaming_state.kv_cache,
            padding_cache=streaming_state.padding_cache,
            stft_cache=new_stft_cache,
            running_max=new_running_max,
            mel_buffer=streaming_state.mel_buffer,
        )
        if hasattr(streaming_state, "_pool_idx"):
            new_state._pool_idx = streaming_state._pool_idx  # type: ignore[attr-defined]
            new_state._pool_owner = streaming_state._pool_owner  # type: ignore[attr-defined]

        if audio_feature_lengths[0].item() == 0:
            return CausalAudioEncoderOutput(
                embeds=torch.zeros(1, 0, self.hidden_size, device=self.device, dtype=self.dtype),
                streaming_state=new_state,
            )

        # Append new mel frames to the buffer.
        # Shape: [num_mel_bins, new_frames] -> [1, num_mel_bins, new_frames]
        new_mel = packed_features.unsqueeze(0)
        if new_state.mel_buffer is not None:
            mel_all = torch.cat([new_state.mel_buffer, new_mel], dim=2)
        else:
            mel_all = new_mel

        # Process complete groups of MEL_FRAMES_PER_ADAPTER_TOKEN mel frames.
        total_mel_frames = mel_all.shape[2]
        num_complete_groups = total_mel_frames // MEL_FRAMES_PER_ADAPTER_TOKEN
        consumed_mel = num_complete_groups * MEL_FRAMES_PER_ADAPTER_TOKEN

        if num_complete_groups == 0:
            new_state.mel_buffer = mel_all
            return CausalAudioEncoderOutput(
                embeds=torch.zeros(1, 0, self.hidden_size, device=self.device, dtype=self.dtype),
                streaming_state=new_state,
            )

        # Feed mel frames through encoder in ENCODER_STRIDE chunks.
        mel_to_process = mel_all[:, :, :consumed_mel]
        all_adapter_tokens: list[torch.Tensor] = []
        use_sliding_window = isinstance(new_state.kv_cache, SlidingWindowVoxtralKVCache)

        for group_idx in range(num_complete_groups):
            step_enc_tokens: list[torch.Tensor] = []
            for sub in range(DOWNSAMPLE_FACTOR):
                mel_start = group_idx * MEL_FRAMES_PER_ADAPTER_TOKEN + sub * ENCODER_STRIDE
                chunk = mel_to_process[:, :, mel_start : mel_start + ENCODER_STRIDE]
                if use_sliding_window:
                    kv_cache = new_state.kv_cache
                    assert isinstance(kv_cache, SlidingWindowVoxtralKVCache)

                    inputs_embeds = self.encoder.embedder(chunk, new_state.padding_cache)
                    position_ids = torch.tensor([[kv_cache._total_seen_tokens]], device=self.device, dtype=torch.long)
                    mask = _make_fixed_sliding_window_mask(
                        valid_len=kv_cache.get_kv_len() + 1,
                        window_size=kv_cache._window_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    kv_cache.step()

                    transformer_fn = self._compiled_transformer or self.encoder.transformer_forward
                    hidden = transformer_fn(inputs_embeds, position_ids, mask, kv_cache)
                    if self._compiled_transformer is not None:
                        hidden = hidden.clone()
                    step_enc_tokens.append(hidden)
                else:
                    out = self.encoder(
                        input_features=chunk,
                        past_key_values=new_state.kv_cache,
                        padding_cache=new_state.padding_cache,
                        use_cache=True,
                        use_padding_cache=True,
                    )
                    new_state.kv_cache = out.past_key_values
                    new_state.padding_cache = out.padding_cache
                    step_enc_tokens.append(out.last_hidden_state)

            enc_group = torch.cat(step_enc_tokens, dim=1)
            adapter_token = self._frame_stack_and_project(enc_group)
            all_adapter_tokens.append(adapter_token)

        # Save leftover mel frames.
        if consumed_mel < total_mel_frames:
            new_state.mel_buffer = mel_all[:, :, consumed_mel:]
        else:
            new_state.mel_buffer = None

        embeds = torch.cat(all_adapter_tokens, dim=1)
        return CausalAudioEncoderOutput(
            embeds=embeds,
            streaming_state=new_state,
        )

    def _forward_batch(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        causal: bool = False,
    ) -> CausalAudioEncoderOutput:
        """Non-streaming batch forward.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.
            causal: Whether to use causal feature extraction.

        Returns:
            ``CausalAudioEncoderOutput`` with adapter embeddings.
        """
        num_samples = audio.shape[2]
        expected_output_length = self.compute_expected_output_length(num_samples)

        if causal:
            mel_features, mel_lengths = self._extract_features_causal(audio, audio_lengths)
        else:
            mel_features, mel_lengths = self._extract_features(audio, audio_lengths)

        # Ensure at least MEL_FRAMES_PER_ADAPTER_TOKEN mel frames so the encoder
        # produces at least one adapter token.  Very short audio (< ~125 ms in
        # causal mode) can yield fewer frames; right-pad with zeros to reach the
        # minimum.  The padding is silence and gets masked out downstream.
        max_mel = mel_features.shape[2]
        if max_mel < MEL_FRAMES_PER_ADAPTER_TOKEN:
            import warnings

            warnings.warn(
                f"[VoxtralWrapper] Short audio: {max_mel} mel frames < {MEL_FRAMES_PER_ADAPTER_TOKEN} required. "
                f"Right-padding with silence. audio_samples={num_samples}, causal={causal}",
                stacklevel=2,
            )
            mel_features = F.pad(mel_features, (0, MEL_FRAMES_PER_ADAPTER_TOKEN - max_mel))
            max_mel = MEL_FRAMES_PER_ADAPTER_TOKEN

        # Truncate mel to a multiple of MEL_FRAMES_PER_ADAPTER_TOKEN for clean frame-stacking.
        usable_mel = (max_mel // MEL_FRAMES_PER_ADAPTER_TOKEN) * MEL_FRAMES_PER_ADAPTER_TOKEN
        mel_features = mel_features[:, :, :usable_mel]

        enc_out = self.encoder(
            input_features=mel_features,
            use_cache=False,
            use_padding_cache=False,
        )
        enc_hidden = enc_out.last_hidden_state
        embeds = self._frame_stack_and_project(enc_hidden)

        actual_output_length = embeds.shape[1]
        if actual_output_length > expected_output_length:
            embeds = embeds[:, :expected_output_length]
        elif actual_output_length < expected_output_length:
            embeds = F.pad(embeds, (0, 0, 0, expected_output_length - actual_output_length))

        return CausalAudioEncoderOutput(embeds=embeds)


# ── from modules/code_predictor.py ──


from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    GenerationMixin,
    Qwen3OmniMoePreTrainedModel,
    Qwen3OmniMoeTalkerCodePredictorModel,
    StaticCache,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorOutputWithPast



class RaonCodePredictorModel(Qwen3OmniMoePreTrainedModel, GenerationMixin):  # type: ignore
    """Code predictor for autoregressive audio code generation with fused codec embedding."""

    config_class: type[Qwen3OmniMoeTalkerCodePredictorConfig] = Qwen3OmniMoeTalkerCodePredictorConfig  # type: ignore[assignment]

    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig):
        super().__init__(config)
        self.num_code_groups = config.num_code_groups
        _dtype = getattr(config, "torch_dtype", None) or torch.float32
        if isinstance(_dtype, str):
            _dtype = getattr(torch, _dtype, torch.float32)
        self.model = Qwen3OmniMoeTalkerCodePredictorModel._from_config(config, dtype=_dtype)
        input_embeddings = self.model.get_input_embeddings()
        assert isinstance(input_embeddings, nn.ModuleList), "Expected input embeddings to be a ModuleList."
        weights: list[torch.Tensor] = []
        for i in range(self.num_code_groups):
            embed = input_embeddings[i - 1]
            assert isinstance(embed, nn.Embedding)
            weights.append(embed.weight)

        fused_code_embed_weight = torch.cat(weights)
        self.codec_embedding = nn.Embedding(
            fused_code_embed_weight.shape[0],
            fused_code_embed_weight.shape[1],
            dtype=fused_code_embed_weight.dtype,
        )
        with torch.no_grad():
            self.codec_embedding.weight.copy_(fused_code_embed_weight)

        del self.model.codec_embedding
        self.vocab_size = config.vocab_size
        self.fused_lm_head = nn.Parameter(
            torch.randn(
                self.num_code_groups - 1,
                self.vocab_size,
                self.config.hidden_size,
                dtype=_dtype,
            )
            * (self.config.hidden_size**-0.5)
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        generation_steps: int | None = None,
        **kwargs: Any,
    ) -> Qwen3OmniMoeTalkerCodePredictorOutputWithPast:
        """Run one autoregressive step of code-group prediction.

        During prefill (inputs_embeds provided with seq_len > 1), generation_steps
        is derived from the sequence length. During decoding (input_ids provided),
        generation_steps must be supplied to select the correct per-step LM head
        and to offset the fused codec embedding lookup.

        Args:
            input_ids: Token ids for single-step decoding. Shape: [batch, 1]. Dtype: long.
            attention_mask: Attention mask for the full sequence. Shape: [batch, seq_len].
            position_ids: Position ids. Shape: [batch, seq_len].
            past_key_values: KV cache from previous steps.
            inputs_embeds: Precomputed embeddings (used during prefill).
                Shape: [batch, seq_len, hidden_size].
            use_cache: Whether to return updated KV cache.
            cache_position: Absolute positions of the current tokens in the cache.
            generation_steps: Which code group is being predicted (0-indexed).
                Inferred from inputs_embeds during prefill; required during decoding.
            **kwargs: Additional arguments forwarded to the inner model.

        Returns:
            Qwen3OmniMoeTalkerCodePredictorOutputWithPast with logits of shape
            [batch, seq_len, vocab_size], updated past_key_values, and
            generation_steps incremented by 1.
        """
        inputs_embeds = cast_to_module_dtype(inputs_embeds, self)
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2
        else:
            assert input_ids is not None and generation_steps is not None, f"{input_ids=}, {generation_steps=}"
            inputs_embeds = self.get_input_embeddings()(input_ids + generation_steps * self.vocab_size)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        logits = F.linear(outputs.last_hidden_state, self.fused_lm_head[generation_steps])
        return Qwen3OmniMoeTalkerCodePredictorOutputWithPast(
            logits=logits,  # type: ignore
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def parallel_forward(self, hidden_embeds: torch.Tensor, audio_codes: torch.Tensor) -> torch.Tensor:
        """Predict all code groups in parallel given hidden states and teacher-forced codes.

        Args:
            hidden_embeds: Hidden states from the LM. Shape: [batch_size, hidden_size].
                Dtype: float.
            audio_codes: Teacher-forced audio codes (all but last group).
                Shape: [batch_size, num_code_groups]. Dtype: long.

        Returns:
            Logits for the next code group at each position. Shape: [batch_size,
            num_code_groups - 1, vocab_size]. Dtype: float.
        """
        hidden_embeds = cast_to_module_dtype(hidden_embeds, self)
        generation_step = torch.arange(self.config.num_code_groups - 1, device=audio_codes.device)
        audio_code_embeds = self.codec_embedding(audio_codes[:, :-1] + generation_step * self.vocab_size)
        inputs_embeds = torch.cat((hidden_embeds[:, None], audio_code_embeds), dim=1).contiguous()
        last_hidden_state = self.model(inputs_embeds=inputs_embeds).last_hidden_state
        logits: torch.Tensor = torch.einsum("bsh,sch->bsc", last_hidden_state[:, 1:], self.fused_lm_head)
        return logits

    def generate_greedy(self, inputs_embeds: torch.Tensor, past_key_values: StaticCache) -> torch.Tensor:
        """Generate audio codes greedily given initial embeddings and KV cache.

        Args:
            inputs_embeds: Initial input embeddings. Shape: [batch_size, seq_length,
                hidden_size]. Dtype: float.
            past_key_values: StaticCache holding past KV for incremental decoding.

        Returns:
            Greedily sampled code sequence. Shape: [batch_size, num_code_groups - 1].
            Dtype: long.
        """
        cache_position = torch.arange(2, device=inputs_embeds.device)
        optional_input_ids: torch.Tensor | None = None
        optional_inputs_embeds: torch.Tensor | None = inputs_embeds
        sequences = torch.empty(
            (inputs_embeds.shape[0], self.num_code_groups - 1),
            dtype=torch.int64,
            device=inputs_embeds.device,
        )
        for i in range(self.num_code_groups - 1):
            logits: torch.Tensor = self(
                input_ids=optional_input_ids,
                inputs_embeds=optional_inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                generation_steps=i,
            ).logits
            optional_inputs_embeds = None
            optional_input_ids = logits[:, -1:].argmax(dim=-1)
            cache_position = cache_position[-1:] + 1
            sequences[:, i] = optional_input_ids[:, -1]

        return sequences

    def _update_model_kwargs_for_generation(  # type: ignore
        self,
        outputs: Qwen3OmniMoeTalkerCodePredictorOutputWithPast,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs

    def predict_codes(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Predict full audio code sequence from input embeddings via greedy generation.

        Args:
            inputs_embeds: Input embeddings. Shape: [batch_size, seq_length, hidden_size].
                Dtype: float.

        Returns:
            Predicted audio codes. Shape: [batch_size, num_code_groups - 1]. Dtype: long.
        """
        inputs_embeds = cast_to_module_dtype(inputs_embeds, self)
        past_key_values = StaticCache(self.config, max_cache_len=self.num_code_groups, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return self.generate_greedy(inputs_embeds=inputs_embeds, past_key_values=past_key_values)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the fused codec embedding layer."""
        return self.codec_embedding


# ── from modules/concurrent_audio_decoder.py ──

import queue
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.multiprocessing as mp
from transformers import Cache


StreamDecoderState: TypeAlias = tuple[Cache | None, MimiConv1dPaddingCache | None, MimiConvTranspose1dPaddingCache | None]

@torch.inference_mode()
def audio_decoder_worker(
    audio_tokenizer: StreamingMimiModel,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    try:
        if isinstance(device, str) and device.startswith("cuda"):
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_id)
        elif isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.set_device(device.index or 0)

        audio_tokenizer = audio_tokenizer.to(device=device, dtype=dtype)  # type: ignore
        audio_tokenizer.eval()
        output_queue.put(("WORKER_READY", None))

    except Exception as e:
        output_queue.put(("WORKER_ERROR", e))
        return

    stream_states: dict[int, StreamDecoderState] = {}

    while True:
        try:
            item = input_queue.get()
            if item is None:
                break

            command, stream_id, sequence_id, audio_codes = item

            match command:
                case "CREATE_STREAM":
                    stream_states[stream_id] = (None, None, None)
                case "DESTROY_STREAM":
                    if stream_id in stream_states:
                        del stream_states[stream_id]

                case "DECODE_AUDIO":
                    decoder_past_key_values, conv_padding_cache, conv_transpose_padding_cache = stream_states[stream_id]
                    audio_codes = audio_codes.to(device=device)

                    outputs = audio_tokenizer.decode(
                        audio_codes.transpose(1, 2),
                        decoder_past_key_values=decoder_past_key_values,
                        conv1d_padding_cache=conv_padding_cache,
                        convtranspose1d_padding_cache=conv_transpose_padding_cache,
                        use_streaming=True,
                        return_dict=True,
                    )
                    assert isinstance(outputs, StreamingMimiDecoderOutput)
                    assert (audio_values := outputs.audio_values) is not None
                    assert isinstance(outputs.decoder_past_key_values, Cache)
                    assert isinstance(outputs.conv1d_padding_cache, MimiConv1dPaddingCache)
                    assert isinstance(outputs.convtranspose1d_padding_cache, MimiConvTranspose1dPaddingCache)

                    audio = audio_values.view(audio_values.shape[0], audio_values.shape[2])

                    stream_states[stream_id] = (
                        outputs.decoder_past_key_values,
                        outputs.conv1d_padding_cache,
                        outputs.convtranspose1d_padding_cache,
                    )
                    output_queue.put(
                        ("DECODE_AUDIO", stream_id, sequence_id, audio.float().cpu())
                    )

                case _:
                    ...

        except Exception as e:
            output_queue.put((None, None, None, e))


class ConcurrentAudioDecoder:
    def __init__(
        self,
        model: "RaonInferenceModel",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.device = device if device is not None else model.get_model().device
        self.dtype = dtype if dtype is not None else model.get_model().dtype

        self.mp_context = mp.get_context("spawn")
        self.input_queue: mp.Queue = self.mp_context.Queue()
        self.output_queue: mp.Queue = self.mp_context.Queue()
        self.process: mp.process.BaseProcess | None = None  # type: ignore
        self.stream_counter = 0
        self.sequence_counter = 0
        self.pending_sequences: dict[int, int] = {}
        self.stream_pending_counts: dict[int, int] = defaultdict(int)
        self.stream_output_queues: dict[int, queue.Queue[tuple[int, torch.Tensor]]] = defaultdict(queue.Queue)

    def start(self, timeout: float = 5.0) -> None:
        if self.process is not None and self.process.is_alive():
            raise RuntimeError("Audio decoder worker is already running.")

        process = self.mp_context.Process(
            target=audio_decoder_worker,
            kwargs={
                "audio_tokenizer": self.model.get_model().audio_tokenizer,
                "input_queue": self.input_queue,
                "output_queue": self.output_queue,
                "device": self.device,
                "dtype": self.dtype,
            },
            daemon=True,
        )
        process.start()
        self.process = process

        try:
            signal, error = self.output_queue.get(timeout=timeout)
            if signal == "WORKER_ERROR":
                process.join(timeout=1.0)
                self.process = None
                raise RuntimeError(f"Audio decoder worker failed to initialize: {error}") from None
            elif signal != "WORKER_READY":
                self.output_queue.put((signal, error))

        except queue.Empty:
            self.process = None
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            raise RuntimeError("Audio decoder worker failed to start (timeout waiting for ready signal)") from None

    def stop(self, timeout: float | None = 5.0) -> None:
        if self.process is None:
            return

        self.input_queue.put(None)

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)

        self.process = None

        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except Exception:
                break

        self.pending_sequences.clear()
        self.stream_pending_counts.clear()
        self.stream_output_queues.clear()
        self.stream_counter = 0
        self.sequence_counter = 0

    def create_stream(self) -> int:
        assert self.process is not None and self.process.is_alive()
        stream_id = self.stream_counter
        self.stream_counter += 1
        self.stream_pending_counts[stream_id] = 0
        self.input_queue.put(("CREATE_STREAM", stream_id, None, None))
        return stream_id

    def destroy_stream(self, stream_id: int) -> None:
        if self.process is None or not self.process.is_alive():
            return

        if stream_id in self.stream_pending_counts:
            del self.stream_pending_counts[stream_id]

        if stream_id in self.stream_output_queues:
            del self.stream_output_queues[stream_id]

        self.input_queue.put(("DESTROY_STREAM", stream_id, None, None))

    def push_audio_codes(self, stream_id: int, audio_codes: torch.Tensor) -> int:
        assert self.process is not None and self.process.is_alive()
        assert audio_codes.ndim == 3

        sequence_id = self.sequence_counter
        self.sequence_counter += 1
        self.pending_sequences[sequence_id] = stream_id
        self.stream_pending_counts[stream_id] += 1

        self.input_queue.put(("DECODE_AUDIO", stream_id, sequence_id, audio_codes.cpu()))
        return sequence_id

    def pull_audio(
        self,
        stream_id: int,
        block: bool = True,
        timeout: float | None = None,
    ) -> tuple[int, torch.Tensor] | None:
        try:
            while self.stream_output_queues[stream_id].empty():
                command, result_stream_id, sequence_id, audio_or_error = self.output_queue.get(block=block, timeout=timeout)
                if command == "DECODE_AUDIO":
                    assert isinstance(audio := audio_or_error, torch.Tensor)
                    self.stream_output_queues[result_stream_id].put((sequence_id, audio))
                    if sequence_id in self.pending_sequences:
                        del self.pending_sequences[sequence_id]

                    if result_stream_id in self.stream_pending_counts:
                        self.stream_pending_counts[result_stream_id] -= 1

                elif isinstance(audio_or_error, Exception):
                    raise RuntimeError(f"Audio decoder worker error: {audio_or_error}")

            return self.stream_output_queues[stream_id].get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending_count(self) -> int:
        return len(self.pending_sequences)

    def get_stream_pending_count(self, stream_id: int) -> int:
        return self.stream_pending_counts.get(stream_id, 0)

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def drain_to(
        self,
        max_pending: int,
        stream_id: int,
        timeout_per_item: float = 1.0,
    ) -> list[tuple[int, torch.Tensor]]:
        results: list[tuple[int, torch.Tensor]] = []

        while self.get_stream_pending_count(stream_id) > max_pending:
            result = self.pull_audio(stream_id=stream_id, block=True, timeout=timeout_per_item)
            if result is not None:
                results.append(result)
            else:
                break

        return results

    def __enter__(self, *args: Any, **kwargs: Any) -> "ConcurrentAudioDecoder":
        self.start()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.stop()


# ── from utils/loss.py ──


import logging
from typing import cast

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class RaonLossMixin:
    """Mixin providing loss computation methods for RaonModel.

    Expects the following attributes on the concrete class (all set by RaonModel.__init__):
        audio_lm_head, proj_code, code_predictor, output_adaptor, speaker_encoder,
        audio_loss_weight, text_loss_weight, audio_output_pad_text_loss_weight,
        epad_loss_weight, audio_end_text_loss_weight, code_predictor_grad_scale,
        num_code_groups, audio_lm_head_vocab_size, codebook_size, max_delay, delays,
        supports_audio_output, use_duplex_end_pad, is_pretrained_speaker_encoder,
        speaker_token_id.

    Expects the following method on the concrete class:
        shift_labels(labels, pad_length=1) -> torch.Tensor
    """

    def unreduced_causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-position cross-entropy loss for causal LM (no reduction).

        Args:
            logits: Language model logits. Shape: [batch_size, seq_length, vocab_size]. Dtype: float.
            labels: Ground-truth token IDs. Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Per-position loss. Shape: [batch_size, seq_length]. Dtype: float.
        """
        _, seq_length, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size).float()
        labels = self.shift_labels(labels, pad_length=seq_length - labels.shape[1] + 1).reshape(-1).to(logits.device)  # type: ignore[attr-defined]
        loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=LOSS_IGNORE_INDEX)
        return loss.reshape(-1, seq_length)

    def _compute_audio_loss(
        self,
        hidden_embeds: torch.Tensor,
        audio_output_codes: torch.Tensor | None,
        audio_output_codes_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Compute cross-entropy loss over audio codes for positions marked AUDIO_OUTPUT_PLACEHOLDER.

        Args:
            hidden_embeds: Talker hidden states. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            audio_output_codes: Ground-truth audio codes. Shape: [batch_size, num_frames, num_code_groups].
                Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            input_ids: Input token IDs used to identify audio-end prediction positions.
                Shape: [batch_size, seq_length]. Dtype: long.
            labels: Ground-truth token IDs used to identify audio frame positions.
                Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Tuple of (audio_logits, audio_loss, audio_output_mask, audio_end_mask, audio_end_loss)
            when audio positions exist, else None.
            audio_logits Shape: [num_audio_positions, num_code_groups, codebook_size + 1]. Dtype: float.
            audio_loss Shape: [num_audio_positions, num_code_groups]. Dtype: float.
            audio_output_mask Shape: [batch_size, seq_length]. Dtype: bool.
            audio_end_mask Shape: [batch_size, seq_length]. Dtype: bool.
            audio_end_loss Shape: [num_audio_end_positions]. Dtype: float.
        """
        if audio_output_codes is None:
            return None
        assert self.audio_lm_head is not None, (
            "audio_lm_head is unavailable when supports_audio_output is False."
        )  # type: ignore[attr-defined]
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]
        assert self.code_predictor is not None, "code_predictor is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]

        assert audio_output_codes_mask is not None, (
            "`audio_output_codes_mask` is required when `audio_output_codes` is provided."
        )
        shifted_labels = self.shift_labels(labels)  # type: ignore[attr-defined]
        shifted_input_ids = self.shift_labels(input_ids)  # type: ignore[attr-defined]
        audio_output_mask = shifted_labels == AUDIO_OUTPUT_PLACEHOLDER.id
        # Train AUDIO_END only for output-audio segments, not user input-audio tags.
        # Duplex: AUDIO_END appears in labels (first silence frame's [A]).
        # Pretrain/TTS: AUDIO_END appears in shifted_input_ids (token after last [A]).
        audio_end_mask = (labels == AUDIO_END.id) & (input_ids == AUDIO_OUTPUT_PLACEHOLDER.id)
        audio_end_mask |= (shifted_input_ids == AUDIO_END.id) & (input_ids == AUDIO_OUTPUT_PLACEHOLDER.id)
        if not audio_output_mask.any() and not audio_end_mask.any():
            return None

        audio_output_hidden_embeds = hidden_embeds[audio_output_mask.to(hidden_embeds.device)]
        audio_end_hidden_embeds = hidden_embeds[audio_end_mask.to(hidden_embeds.device)]
        audio_output_codes = audio_output_codes[audio_output_codes_mask.to(torch.bool)]

        # Apply delays for training if configured.
        # Moshi-style: use delayed codes as code predictor input and labels.
        if self.max_delay > 0:  # type: ignore[attr-defined]
            B = audio_output_codes.shape[0]
            assert B == 1, (
                f"Acoustic delay requires batch_size=1, got {B}. Flattened codes would leak across batch boundaries."
            )
            delayed_audio_codes = delay_audio_codes(self.delays, audio_output_codes, padding_value=0)  # type: ignore[attr-defined]
            delayed_audio_codes_labels = delay_audio_codes(
                self.delays,  # type: ignore[attr-defined]
                audio_output_codes,
                padding_value=LOSS_IGNORE_INDEX,
            )
        else:
            delayed_audio_codes = audio_output_codes
            delayed_audio_codes_labels = audio_output_codes

        audio_logits = torch.empty(
            0,
            self.num_code_groups,  # type: ignore[attr-defined]
            self.audio_lm_head_vocab_size,  # type: ignore[attr-defined]
            device=hidden_embeds.device,
            dtype=hidden_embeds.dtype,
        )
        audio_loss = torch.empty(0, self.num_code_groups, device=hidden_embeds.device, dtype=hidden_embeds.dtype)  # type: ignore[attr-defined]
        if audio_output_hidden_embeds.shape[0] > 0:
            # Truncate to shorter length when audio encoder and tokenizer frame counts
            # differ by a small amount (off-by-one from conv stride rounding).
            min_len = min(audio_output_hidden_embeds.shape[0], audio_output_codes.shape[0])
            if audio_output_hidden_embeds.shape[0] != audio_output_codes.shape[0]:
                audio_output_hidden_embeds = audio_output_hidden_embeds[:min_len]
                audio_output_codes = audio_output_codes[:min_len]
                delayed_audio_codes = delayed_audio_codes[:min_len]
                delayed_audio_codes_labels = delayed_audio_codes_labels[:min_len]
                # Keep the audio-position mask aligned with truncated audio codes.
                flat_audio_output_mask = audio_output_mask.reshape(-1)
                audio_output_indices = flat_audio_output_mask.nonzero(as_tuple=False).squeeze(-1)
                trimmed_flat_audio_output_mask = torch.zeros_like(flat_audio_output_mask, dtype=torch.bool)
                if min_len > 0:
                    trimmed_flat_audio_output_mask[audio_output_indices[:min_len]] = True
                audio_output_mask = trimmed_flat_audio_output_mask.view_as(audio_output_mask)
            audio_lm_head_logits = self.audio_lm_head(audio_output_hidden_embeds)  # type: ignore[attr-defined]

            grad_scaled_hidden_embeds = (
                self.code_predictor_grad_scale * audio_output_hidden_embeds  # type: ignore[attr-defined]
                + (1 - self.code_predictor_grad_scale) * audio_output_hidden_embeds.detach()  # type: ignore[attr-defined]
            )
            code_predictor_input_hidden_embeds = self.proj_code(grad_scaled_hidden_embeds)  # type: ignore[attr-defined]

            code_predictor_logits = self.code_predictor.parallel_forward(  # type: ignore[attr-defined]
                hidden_embeds=code_predictor_input_hidden_embeds,
                audio_codes=delayed_audio_codes,
            )
            code_predictor_logits = F.pad(
                code_predictor_logits,
                (0, 1),
                value=torch.finfo(code_predictor_logits.dtype).min,
            )

            audio_logits = torch.cat((audio_lm_head_logits[:, None], code_predictor_logits), dim=1)
            audio_loss = F.cross_entropy(
                audio_logits.reshape(-1, self.audio_lm_head_vocab_size),  # type: ignore[attr-defined]
                delayed_audio_codes_labels.reshape(-1),
                reduction="none",
                ignore_index=LOSS_IGNORE_INDEX,
            ).reshape(-1, self.num_code_groups)  # type: ignore[attr-defined]

        audio_end_loss = torch.empty(0, device=hidden_embeds.device, dtype=hidden_embeds.dtype)
        if audio_end_hidden_embeds.shape[0] > 0:
            audio_end_logits = self.audio_lm_head(audio_end_hidden_embeds)  # type: ignore[attr-defined]
            audio_end_targets = torch.full(
                (audio_end_logits.shape[0],),
                fill_value=self.codebook_size,  # type: ignore[attr-defined]
                dtype=torch.long,
                device=audio_end_logits.device,
            )
            audio_end_loss = F.cross_entropy(
                audio_end_logits,
                audio_end_targets,
                reduction="none",
            )

        return audio_logits, audio_loss, audio_output_mask, audio_end_mask, audio_end_loss

    def _dummy_audio_loss(self, hidden_embeds: torch.Tensor) -> torch.Tensor:
        """Return a zero scalar loss when no audio output positions exist, to keep gradients flowing."""
        assert self.audio_lm_head is not None, (
            "audio_lm_head is unavailable when supports_audio_output is False."
        )  # type: ignore[attr-defined]
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]
        assert self.code_predictor is not None, "code_predictor is unavailable when supports_audio_output is False."  # type: ignore[attr-defined]
        hidden_embeds = hidden_embeds[0, :1]
        audio_lm_head_logits = self.audio_lm_head(hidden_embeds)  # type: ignore[attr-defined]
        code_predictor_input_hidden_embeds = self.proj_code(hidden_embeds)  # type: ignore[attr-defined]
        dummy_audio_codes = torch.zeros(
            (1, self.num_code_groups),  # type: ignore[attr-defined]
            dtype=torch.long,
            device=code_predictor_input_hidden_embeds.device,
        )
        code_predictor_logits = self.code_predictor.parallel_forward(  # type: ignore[attr-defined]
            hidden_embeds=code_predictor_input_hidden_embeds,
            audio_codes=dummy_audio_codes,
        )
        code_predictor_logits = F.pad(code_predictor_logits, (0, 1), value=torch.finfo(code_predictor_logits.dtype).min)
        audio_logits = torch.cat((audio_lm_head_logits[:, None], code_predictor_logits), dim=1)
        result = 0 * audio_logits.sum(dim=-1)
        return result

    def _dummy_output_adaptor_loss(self) -> torch.Tensor:
        """Return a zero scalar tied to output_adaptor parameters for DDP safety."""
        if self.output_adaptor is None:  # type: ignore[attr-defined]
            return torch.tensor(0.0)
        first_param = next(self.output_adaptor.parameters(), None)  # type: ignore[attr-defined]
        if first_param is None:
            return torch.tensor(0.0)
        total = torch.zeros((), device=first_param.device, dtype=first_param.dtype)
        for parameter in self.output_adaptor.parameters():  # type: ignore[attr-defined]
            total = total + 0 * parameter.sum()
        return total

    def _dummy_speaker_loss(self) -> torch.Tensor:
        """Return a zero scalar when speaker_encoder exists but batch has no speaker positions; keeps DDP from hanging."""
        if self.speaker_encoder is None:  # type: ignore[attr-defined]
            return torch.tensor(0.0)

        first_param = next(self.speaker_encoder.parameters())  # type: ignore[attr-defined]
        if self.is_pretrained_speaker_encoder:  # type: ignore[attr-defined]
            # Minimum 24kHz duration to exceed ECAPA minimum after resample.
            dummy = torch.zeros(1, 4000, device=first_param.device, dtype=first_param.dtype)
            dummy_lengths = torch.tensor([dummy.shape[1]], device=dummy.device, dtype=torch.long)
            output = self.speaker_encoder(dummy, dummy_lengths)  # type: ignore[attr-defined]
        else:
            speaker_encoder_input_size = cast(int, self.speaker_encoder.input_size)  # type: ignore[attr-defined]
            dummy = torch.zeros(1, 1, speaker_encoder_input_size, device=first_param.device, dtype=first_param.dtype)
            dummy_mask = torch.ones(1, 1, dtype=torch.bool, device=dummy.device)
            output = self.speaker_encoder(dummy, mask=dummy_mask)  # type: ignore[attr-defined]

        return 0 * output.sum()

    def _apply_text_loss_weights(self, loss: torch.Tensor, text_labels: torch.Tensor) -> torch.Tensor:
        """Apply per-token-type loss weights to unreduced text loss.

        Args:
            loss: Per-position unreduced loss. Shape: [batch_size, seq_length]. Dtype: float.
            text_labels: Labels with AUDIO_OUTPUT_PLACEHOLDER already replaced. Shape: [batch_size, seq_length].
                Dtype: long.

        Returns:
            Weighted loss. Shape: [batch_size, seq_length]. Dtype: float.
        """
        shifted_text_labels = self.shift_labels(text_labels)  # type: ignore[attr-defined]
        assert (shifted_text_labels != AUDIO_OUTPUT_PLACEHOLDER.id).all(), (
            "text_labels must not contain AUDIO_OUTPUT_PLACEHOLDER tokens. "
            "Use get_text_labels(labels) to convert labels before passing to this method."
        )
        weighted_loss = loss.clone()

        pad_mask = shifted_text_labels == AUDIO_OUTPUT_PAD.id
        epad_mask = shifted_text_labels == AUDIO_OUTPUT_END_PAD.id
        audio_end_mask = shifted_text_labels == AUDIO_END.id
        text_mask = (
            (shifted_text_labels != AUDIO_OUTPUT_PAD.id)
            & (shifted_text_labels != AUDIO_OUTPUT_END_PAD.id)
            & (shifted_text_labels != AUDIO_END.id)
            & (shifted_text_labels != AUDIO_OUTPUT_BC.id)
            & (shifted_text_labels != LOSS_IGNORE_INDEX)
        )

        weighted_loss[text_mask] = weighted_loss[text_mask] * self.text_loss_weight  # type: ignore[attr-defined]
        weighted_loss[pad_mask] = weighted_loss[pad_mask] * self.audio_output_pad_text_loss_weight  # type: ignore[attr-defined]
        weighted_loss[audio_end_mask] = weighted_loss[audio_end_mask] * self.audio_end_text_loss_weight  # type: ignore[attr-defined]

        if self.use_duplex_end_pad:  # type: ignore[attr-defined]
            weighted_loss[epad_mask] = weighted_loss[epad_mask] * self.epad_loss_weight  # type: ignore[attr-defined]

        if self.use_sil_token:  # type: ignore[attr-defined]
            sil_mask = shifted_text_labels == DUPLEX_SIL.id
            weighted_loss[sil_mask] = weighted_loss[sil_mask] * self.sil_loss_weight  # type: ignore[attr-defined]

        if getattr(self, "use_backchannel_token", False):
            bc_mask = shifted_text_labels == AUDIO_OUTPUT_BC.id
            weighted_loss[bc_mask] = weighted_loss[bc_mask] * self.bc_loss_weight  # type: ignore[attr-defined]

        return weighted_loss

    def _combine_losses(
        self,
        text_loss: torch.Tensor | None,
        audio_loss: torch.Tensor | None,
        audio_output_mask: torch.Tensor | None,
        audio_end_loss: torch.Tensor | None,
        audio_end_mask: torch.Tensor | None,
        text_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Combine weighted text loss and audio loss into a single unreduced loss tensor.

        Args:
            text_loss: Per-position text loss. Shape: [batch_size, seq_length]. Dtype: float.
            audio_loss: Per-position-per-code-group audio loss. Shape: [num_audio_positions, num_code_groups].
                Dtype: float.
            audio_output_mask: Boolean mask for positions with audio. Shape: [batch_size, seq_length]. Dtype: bool.
            audio_end_loss: Per-position audio end loss. Shape: [num_audio_end_positions]. Dtype: float.
            audio_end_mask: Boolean mask for positions predicting AUDIO_END. Shape: [batch_size, seq_length].
                Dtype: bool.
            text_labels: Labels with placeholders replaced. Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Combined per-position loss. Shape: [batch_size, seq_length]. Dtype: float.
        """
        shifted_text_labels = self.shift_labels(text_labels)  # type: ignore[attr-defined]
        if text_loss is None:
            assert audio_loss is not None, "audio_loss is required when text_loss is None."
            loss = torch.zeros(shifted_text_labels.shape, dtype=audio_loss.dtype, device=audio_loss.device)
        else:
            loss = self._apply_text_loss_weights(text_loss, text_labels)

        if audio_loss is not None:
            if audio_output_mask is None:
                loss += audio_loss.sum()
                logger.warning("WARNING from `RaonModel._combine_losses`: `audio_loss` given but not `audio_output_mask`.")
            else:
                num_audio_positions = int(audio_output_mask.sum().item())
                num_audio_loss_positions = int(audio_loss.shape[0])
                if num_audio_positions != num_audio_loss_positions:
                    min_len = min(num_audio_positions, num_audio_loss_positions)
                    logger.warning(
                        "WARNING from `RaonModel._combine_losses`: audio_output_mask/audio_loss length mismatch "
                        "(mask=%d, loss=%d). Clipping to %d.",
                        num_audio_positions,
                        num_audio_loss_positions,
                        min_len,
                    )
                    flat_audio_output_mask = audio_output_mask.reshape(-1)
                    audio_output_indices = flat_audio_output_mask.nonzero(as_tuple=False).squeeze(-1)
                    trimmed_flat_audio_output_mask = torch.zeros_like(flat_audio_output_mask, dtype=torch.bool)
                    if min_len > 0:
                        trimmed_flat_audio_output_mask[audio_output_indices[:min_len]] = True
                    audio_output_mask = trimmed_flat_audio_output_mask.view_as(audio_output_mask)
                    audio_loss = audio_loss[:min_len]
                self.audio_loss_weight = self.audio_loss_weight.to(loss.device)  # type: ignore[attr-defined]
                loss[audio_output_mask] += (self.audio_loss_weight * audio_loss).sum(dim=1)  # type: ignore[attr-defined]
        if audio_end_loss is not None and audio_end_loss.numel() > 0:
            if audio_end_mask is None:
                loss += audio_end_loss.sum()
                logger.warning("WARNING from `RaonModel._combine_losses`: `audio_end_loss` given but not `audio_end_mask`.")
            else:
                self.audio_loss_weight = self.audio_loss_weight.to(loss.device)  # type: ignore[attr-defined]
                loss[audio_end_mask] += self.audio_loss_weight[0] * audio_end_loss  # type: ignore[attr-defined]

        return loss

    def ddp_safe_loss(
        self,
        text_loss: torch.Tensor,
        text_labels: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        hidden_embeds: torch.Tensor,
        audio_output_codes: torch.Tensor | None,
        audio_output_codes_mask: torch.Tensor | None,
        speaker_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute audio loss, combine with text loss, and add DDP-safe dummy losses.

        Args:
            text_loss: Per-position unreduced text loss.
                Shape: [batch_size, seq_length]. Dtype: float.
            text_labels: Text-only labels (AUDIO_OUTPUT_PLACEHOLDER replaced).
                Shape: [batch_size, seq_length]. Dtype: long.
            input_ids: Input token IDs used to identify audio-end prediction positions.
                Shape: [batch_size, seq_length]. Dtype: long.
            labels: Ground-truth token IDs used to identify audio frame positions.
                Shape: [batch_size, seq_length]. Dtype: long.
            hidden_embeds: Talker hidden states.
                Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            audio_output_codes: Ground-truth audio codes.
                Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask.
                Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Optional speaker conditioning.
                Shape: [batch_size, num_frames, feature_dim]. Dtype: float.

        Returns:
            Tuple of (loss, audio_loss, audio_logits).
            loss: Combined per-position loss with DDP-safe speaker term.
                Shape: [batch_size, seq_length]. Dtype: float.
            audio_loss: Per-position-per-code-group audio loss, or None when audio output is disabled.
                Shape: [num_audio_positions, num_code_groups]. Dtype: float.
            audio_logits: Audio logits if audio positions exist, else None.
                Shape: [num_audio_positions, num_code_groups, codebook_size]. Dtype: float.
        """
        audio_logits: torch.Tensor | None = None
        audio_loss: torch.Tensor | None = None
        audio_output_mask: torch.Tensor | None = None
        audio_end_loss: torch.Tensor | None = None
        audio_end_mask: torch.Tensor | None = None
        dummy_audio_loss: torch.Tensor | None = None
        dummy_output_adaptor_loss: torch.Tensor | None = None
        audio_outputs = self._compute_audio_loss(
            hidden_embeds=hidden_embeds,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            input_ids=input_ids,
            labels=labels,
        )
        if audio_outputs is not None:
            audio_logits, audio_loss, audio_output_mask, audio_end_mask, audio_end_loss = audio_outputs
        if self.supports_audio_output and (audio_loss is None or audio_loss.numel() == 0):  # type: ignore[attr-defined]
            dummy_audio_loss = self._dummy_audio_loss(hidden_embeds=hidden_embeds).sum()
            if audio_outputs is None:
                logger.warning("WARNING from `ddp_safe_loss`: using `_dummy_audio_loss`.")
        if self.supports_audio_output and (audio_output_mask is None or not audio_output_mask.any()):  # type: ignore[attr-defined]
            dummy_output_adaptor_loss = self._dummy_output_adaptor_loss()

        loss = self._combine_losses(
            text_loss=text_loss,
            audio_loss=audio_loss,
            audio_output_mask=audio_output_mask,
            audio_end_loss=audio_end_loss,
            audio_end_mask=audio_end_mask,
            text_labels=text_labels,
        )
        if dummy_audio_loss is not None:
            loss = loss + dummy_audio_loss
        if dummy_output_adaptor_loss is not None:
            loss = loss + dummy_output_adaptor_loss

        # Ensure speaker_encoder parameters always participate in backward pass
        # to prevent DDP hangs when batches have no speaker_token_id positions.
        if speaker_embeds is not None:
            loss = loss + 0 * speaker_embeds.sum()
        elif self.speaker_encoder is not None:  # type: ignore[attr-defined]
            loss = loss + self._dummy_speaker_loss()

        return loss, audio_loss, audio_logits


# ── from models/wrapper.py ──


import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import trange
from transformers import (
    LogitsProcessorList,
    StaticCache,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type definitions (from inference_utils.py)
# ---------------------------------------------------------------------------

# Set of audio special token IDs used to distinguish text tokens from audio tokens
_AUDIO_SPECIAL_TOKEN_IDS = {
    AUDIO_OUTPUT_PLACEHOLDER.id,
    AUDIO_INPUT_PLACEHOLDER.id,
    AUDIO_OUTPUT_PAD.id,
    AUDIO_OUTPUT_END_PAD.id,
}


class RaonLosses(TypedDict):
    """Training loss container: combined loss, text loss, and per-codebook audio loss.

    loss: Scalar combined loss. Dtype: float.
    text_loss: Scalar text LM loss. Dtype: float.
    audio_loss: Per-codebook audio loss. Shape: [num_code_groups]. Dtype: float.
    """

    loss: torch.Tensor
    text_loss: torch.Tensor
    audio_loss: torch.Tensor


AudioInputEncoderCache = tuple[StaticCache, StaticMimiConv1dPaddingCache | None] | AuTStreamingState | VoxtralStreamingState


@dataclass
class RaonDecodingState:
    """Mutable state for full-duplex streaming decoding.

    Tracks sequences, attention masks, audio codes, KV cache, encoder cache,
    decoder stream ID, sampling config, and acoustic-delay state.
    """

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    audio_codes: torch.Tensor
    audio_codes_mask: torch.Tensor
    past_key_values: Any  # KV cache from the text model (DynamicCache or similar)
    audio_input_encoder_cache: AudioInputEncoderCache
    audio_decoder_stream_id: int
    do_sample: bool
    logits_processor: LogitsProcessorList
    num_code_groups: int = 8
    # For acoustic delay processing: stores semantic codes from previous frame
    semantic_buffer: torch.Tensor | None = None
    # Penalty to subtract from eos logit (higher = longer responses)
    eos_penalty: float = 0.0
    # Penalty to subtract from SIL logit (higher = less silence).
    sil_penalty: float = 0.0
    # Penalty to subtract from BC logit in SIL phase (positive = suppress, negative = boost).
    bc_penalty: float = 0.0
    # Mealy state machine for logit masking
    machine_state: DuplexMachineState | None = None
    # number of frames remaining where SIL must be forced (listen-first warmup)
    forced_sil_remaining: int = 0

    def _reset(self) -> None:
        """Reset the decoding state for reuse."""
        device = self.sequences.device
        self.sequences = torch.zeros(1, 0, dtype=torch.long, device=device)
        self.attention_mask = torch.zeros(1, 0, dtype=torch.long, device=device)
        self.audio_codes = torch.zeros(1, 0, self.num_code_groups, dtype=torch.long, device=device)
        self.audio_codes_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        self.machine_state = None
        self.forced_sil_remaining = 0

        if self.semantic_buffer is not None:
            self.semantic_buffer = None

        if not isinstance(self.audio_input_encoder_cache, (AuTStreamingState, VoxtralStreamingState)):
            if self.audio_input_encoder_cache[0] is not None:
                self.audio_input_encoder_cache[0].reset()

            if self.audio_input_encoder_cache[1] is not None:
                self.audio_input_encoder_cache[1].reset()


# ---------------------------------------------------------------------------
# Output type definitions (from inference.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sampling helpers (from inference_utils.py)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def apply_repetition_aware_sampling(
    sampled_ids: torch.Tensor,
    logits: torch.Tensor,
    audio_codes: torch.Tensor,
    window_size: int,
    repetition_threshold: float,
    skip_frames: int = 0,
) -> torch.Tensor:
    """Resample overly repetitive first-code predictions."""
    result_ids = sampled_ids.clone()
    first_group_codes = audio_codes[:, skip_frames:, 0]

    for batch_index in range(sampled_ids.shape[0]):
        sampled_token = sampled_ids[batch_index, 0].item()
        codes_seq = first_group_codes[batch_index]
        window = codes_seq[max(0, codes_seq.shape[0] - window_size) :]
        if window.numel() == 0:
            continue

        repetition_ratio = (window == sampled_token).sum().item() / window.numel()
        if repetition_ratio > repetition_threshold:
            probs = F.softmax(logits[batch_index], dim=-1, dtype=torch.float32)
            probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            result_ids[batch_index, 0] = torch.multinomial(probs, num_samples=1)[0]

    return result_ids


def make_audio_code_sampler(
    sequences: torch.Tensor,
    logits_processor: LogitsProcessorList,
    audio_codes: torch.Tensor,
    ras_enabled: bool,
    ras_window_size: int,
    ras_repetition_threshold: float,
    ras_skip_frames: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the sampler used for the first generated audio code."""
    def sample_audio_code(logits: torch.Tensor) -> torch.Tensor:
        processed_logits = (
            logits_processor(input_ids=sequences, scores=logits)  # type: ignore[arg-type]
            if len(logits_processor) > 0
            else logits
        )
        probs = F.softmax(processed_logits, dim=-1, dtype=torch.float32)
        probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
        sampled_ids = torch.multinomial(probs, num_samples=1)
        if ras_enabled and audio_codes.shape[1] > 0:
            sampled_ids = apply_repetition_aware_sampling(
                sampled_ids=sampled_ids,
                logits=logits,
                audio_codes=audio_codes,
                window_size=ras_window_size,
                repetition_threshold=ras_repetition_threshold,
                skip_frames=ras_skip_frames,
            )
        return sampled_ids

    return sample_audio_code


# ---------------------------------------------------------------------------
# Inference model (from inference.py)
# ---------------------------------------------------------------------------


class GenerateOutput(TypedDict):
    sequences: torch.Tensor
    audio_codes: torch.Tensor | None
    audio_codes_mask: torch.Tensor | None
    audio: torch.Tensor | None
    audio_lengths: torch.Tensor | None


class RaonInferenceModel(ABC):
    """Abstract base class for offline raon inference."""

    vocab_size: int
    codebook_size: int
    num_code_groups: int
    sampling_rate: int
    frame_rate: float
    tokenizer: Any | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert hasattr(self, "vocab_size"), "Model must have vocab_size attribute."
        assert hasattr(self, "codebook_size"), "Model must have codebook_size attribute."
        assert hasattr(self, "num_code_groups"), "Model must have num_code_groups attribute."
        assert hasattr(self, "sampling_rate"), "Model must have sampling_rate attribute."
        assert hasattr(self, "frame_rate"), "Model must have frame_rate attribute."
        self.concurrent_audio_decoder: ConcurrentAudioDecoder | None = None

    @abstractmethod
    def inference_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
        use_cache: bool | None = False,
        past_key_values: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def tokenize_audio(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: StaticMimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioTokenizerOutput: ...

    @abstractmethod
    def get_audio_input_embeds(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioEncoderOutput: ...

    @abstractmethod
    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        ...

    @abstractmethod
    def decode_audio(
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        use_streaming: bool | None = None,
    ) -> AudioDecoderOutput: ...

    @abstractmethod
    def generate_audio_codes(
        self,
        talker_last_hidden_state: torch.Tensor,
        first_code_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        allow_audio_end: bool = True,
    ) -> torch.Tensor: ...

    @abstractmethod
    def get_model(self) -> RaonModel: ...

    @abstractmethod
    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: Any | None = None,
    ) -> Any: ...

    @abstractmethod
    def free_past_key_values(self, past_key_values: Any) -> None: ...

    def start_concurrent_audio_decoder(self, timeout: float = 5.0) -> None:
        if self.concurrent_audio_decoder is None:
            self.concurrent_audio_decoder = ConcurrentAudioDecoder(self)

        if not self.concurrent_audio_decoder.is_running:
            self.concurrent_audio_decoder.start(timeout=timeout)

    def stop_concurrent_audio_decoder(self, timeout: float | None = 5.0) -> None:
        """Stop the background audio decoder worker and release resources."""
        if self.concurrent_audio_decoder is not None:
            self.concurrent_audio_decoder.stop(timeout=timeout)
            self.concurrent_audio_decoder = None

    def create_audio_decoder_stream(self) -> int:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        return self.concurrent_audio_decoder.create_stream()

    def _destroy_audio_decoder_stream(self, stream_id: int) -> None:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        self.concurrent_audio_decoder.destroy_stream(stream_id)

    def get_silence_codes(self, device: torch.device) -> torch.Tensor:
        """Return cached silence audio codes for keeping the decoder warm during SIL frames."""
        if not hasattr(self, "_silence_codes") or self._silence_codes is None:
            samples_per_frame = int(self.sampling_rate / self.frame_rate)
            silence_audio = torch.zeros(1, 1, samples_per_frame, device=device)
            silence_lengths = torch.tensor([samples_per_frame], device=device)
            with torch.no_grad():
                result = self.tokenize_audio(
                    audio=silence_audio,
                    audio_lengths=silence_lengths,
                    num_code_groups=self.num_code_groups,
                )
            self._silence_codes = result.audio_codes[0, 0].to(device)
        return self._silence_codes.to(device)

    def push_audio_codes(self, audio_codes: torch.Tensor, stream_id: int) -> None:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        assert audio_codes.ndim == 1 and audio_codes.shape[0] == self.num_code_groups, (
            f"Expected 1D audio codes with shape `[{self.num_code_groups}]` but got `{audio_codes.shape}`."
        )
        self.concurrent_audio_decoder.push_audio_codes(stream_id, audio_codes[None, None])

    def pull_audio(self, stream_id: int) -> torch.Tensor:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        results = self.concurrent_audio_decoder.drain_to(max_pending=1, stream_id=stream_id)
        assert len(results) == 1, f"Expected exactly one audio result but got `{len(results)}`."
        _, decoded_audio = results[0]
        return decoded_audio

    def _drain_audio_decoding_queue(self, stream_id: int) -> list[tuple[int, torch.Tensor]]:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        return self.concurrent_audio_decoder.drain_to(max_pending=0, stream_id=stream_id)

    def free_duplex_decoding_state(self, state: RaonDecodingState) -> None:
        self._drain_audio_decoding_queue(state.audio_decoder_stream_id)
        self._destroy_audio_decoder_stream(state.audio_decoder_stream_id)
        self.free_past_key_values(state.past_key_values)
        state._reset()

    def init_audio_encoder_cache(
        self,
        prev_cache: AudioInputEncoderCache | None = None,
    ) -> AudioInputEncoderCache:
        """Initialize or reset the audio input encoder cache for streaming.

        For Mimi encoders, returns a (StaticCache, conv_padding_cache) tuple.
        For causal AuT encoders, returns an AuTStreamingState.

        Args:
            prev_cache: Previous cache to reset and reuse. If None, creates fresh.

        Returns:
            Initialized encoder cache suitable for streaming audio input.
        """
        model = self.get_model()
        audio_encoder_config = model.config.audio_encoder_config

        if isinstance(audio_encoder_config, VoxtralRealtimeEncoderConfig):
            assert isinstance(model.audio_encoder, VoxtralWrapper)
            return model.audio_encoder.init_streaming_state()
        elif isinstance(audio_encoder_config, Qwen3OmniMoeAudioEncoderConfig):
            assert model.aut_is_causal, (
                "Duplex streaming requires a causal audio encoder. "
                "Set `aut_is_causal=True` in the model config to enable "
                "causal AuT streaming for duplex decoding."
            )
            assert isinstance(model.audio_encoder, AuTWrapper)
            return model.audio_encoder.init_streaming_state()
        else:
            if prev_cache is not None:
                assert isinstance(prev_cache, tuple), "Expected Mimi cache tuple."
                prev_cache[0].reset()
                if prev_cache[1] is not None:
                    prev_cache[1].reset()
                return prev_cache

            assert audio_encoder_config.sliding_window is not None
            past_key_values = StaticCache(
                audio_encoder_config,
                max_cache_len=audio_encoder_config.sliding_window,
            )
            return past_key_values, None

    def _streaming_tokenize_audio_with_cache(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        outputs = self.tokenize_audio(
            audio=audio,
            audio_lengths=audio_lengths,
            encoder_past_key_values=audio_encoder_cache[0],
            conv_padding_cache=audio_encoder_cache[1],
            use_streaming=True,
            num_code_groups=self.num_code_groups,
        )
        assert outputs.audio_codes is not None, "Expected `audio_codes` to be not None."
        assert outputs.audio_codes_mask is not None, "Expected `audio_codes_mask` to be not None."
        assert outputs.encoder_cache is not None, "Expected `encoder_cache` to be not None."
        assert isinstance(
            outputs.encoder_cache[0],
            StaticCache,
        ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
        assert isinstance(outputs.encoder_cache[1], StaticMimiConv1dPaddingCache), (
            f"Expected `encoder_cache[1]` to be `StaticMimiConv1dPaddingCache` "
            f"but got `{type(outputs.encoder_cache[1]).__name__}`."
        )
        return (
            outputs.audio_codes,
            outputs.audio_codes_mask,
            (outputs.encoder_cache[0], outputs.encoder_cache[1]),
        )

    def streaming_tokenize_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache | None],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        """Tokenize audio in streaming mode with encoder cache for incremental encoding.

        Args:
            audio: Raw audio frame. Shape: [batch_size, num_samples]. Dtype: float.
            audio_lengths: Per-sample lengths. Shape: [batch_size]. Dtype: long.
            audio_encoder_cache: Current encoder cache (StaticCache, conv padding cache).

        Returns:
            Tuple of (audio_codes, audio_codes_mask, updated_encoder_cache).
            audio_codes: Shape [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_codes_mask: Shape [batch_size, num_frames]. Dtype: bool.
        """
        if audio_encoder_cache[1] is None or not audio_encoder_cache[1].is_initialized:
            outputs = self.tokenize_audio(
                audio=audio,
                audio_lengths=audio_lengths,
                encoder_past_key_values=audio_encoder_cache[0],
                conv_padding_cache=None,
                use_streaming=True,
                num_code_groups=self.num_code_groups,
            )
            assert outputs.audio_codes is not None, "`audio_codes` must not be None."
            assert outputs.audio_codes_mask is not None, "`audio_codes_mask` must not be None."
            assert outputs.encoder_cache is not None, "`encoder_cache` must not be None."
            assert isinstance(
                outputs.encoder_cache[0],
                StaticCache,
            ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
            assert isinstance(
                dynamic_padding_cache := outputs.encoder_cache[1],
                MimiConv1dPaddingCache,
            ), (
                f"Expected `encoder_cache[1]` to be `MimiConv1dPaddingCache` "
                f"but got `{type(outputs.encoder_cache[1]).__name__}`."
            )

            if audio_encoder_cache[1] is None:
                static_conv_padding_cache = StaticMimiConv1dPaddingCache(
                    per_layer_padding=[int(padding) for padding in dynamic_padding_cache.per_layer_padding],
                    padding_cache=dynamic_padding_cache.padding_cache,  # type: ignore
                )
            else:
                audio_encoder_cache[1].initialize(dynamic_padding_cache.padding_cache)  # type: ignore
                static_conv_padding_cache = audio_encoder_cache[1]

            return (
                outputs.audio_codes,
                outputs.audio_codes_mask,
                (outputs.encoder_cache[0], static_conv_padding_cache),
            )
        else:
            return self._streaming_tokenize_audio_with_cache(
                audio=audio.flatten().view(audio.shape),
                audio_lengths=audio_lengths,
                audio_encoder_cache=audio_encoder_cache,  # type: ignore
            )

    def _streaming_get_audio_input_embeds_with_cache(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        outputs = self.get_audio_input_embeds(
            audio=audio,
            audio_lengths=audio_lengths,
            encoder_past_key_values=audio_encoder_cache[0],
            conv_padding_cache=audio_encoder_cache[1],
            use_streaming=True,
        )
        assert outputs.audio_embeds is not None, "Expected `audio_embeds` to be not None."
        assert outputs.audio_embeds_mask is not None, "Expected `audio_embeds_mask` to be not None."
        assert outputs.encoder_cache is not None, "Expected `encoder_cache` to be not None."
        assert isinstance(
            outputs.encoder_cache[0],
            StaticCache,
        ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
        assert isinstance(outputs.encoder_cache[1], StaticMimiConv1dPaddingCache), (
            f"Expected `encoder_cache[1]` to be `StaticMimiConv1dPaddingCache` "
            f"but got `{type(outputs.encoder_cache[1]).__name__}`."
        )
        return (
            outputs.audio_embeds,
            outputs.audio_embeds_mask,
            (outputs.encoder_cache[0], outputs.encoder_cache[1]),
        )

    def _streaming_get_audio_input_embeds(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache | None],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        if audio_encoder_cache[1] is None or not audio_encoder_cache[1].is_initialized:
            outputs = self.get_audio_input_embeds(
                audio=audio,
                audio_lengths=audio_lengths,
                encoder_past_key_values=audio_encoder_cache[0],
                conv_padding_cache=None,
                use_streaming=True,
            )
            assert outputs.audio_embeds is not None, "Expected `audio_embeds` to be not None."
            assert outputs.audio_embeds_mask is not None, "Expected `audio_embeds_mask` to be not None."
            assert outputs.encoder_cache is not None, "Expected `encoder_cache` to be not None."
            assert isinstance(
                outputs.encoder_cache[0],
                StaticCache,
            ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
            assert isinstance(
                dynamic_padding_cache := outputs.encoder_cache[1],
                MimiConv1dPaddingCache,
            ), (
                f"Expected `encoder_cache[1]` to be `MimiConv1dPaddingCache` "
                f"but got `{type(outputs.encoder_cache[1]).__name__}`."
            )

            if audio_encoder_cache[1] is None:
                static_conv_padding_cache = StaticMimiConv1dPaddingCache(
                    per_layer_padding=[int(padding) for padding in dynamic_padding_cache.per_layer_padding],
                    padding_cache=dynamic_padding_cache.padding_cache,  # type: ignore
                )
            else:
                audio_encoder_cache[1].initialize(dynamic_padding_cache.padding_cache)  # type: ignore
                static_conv_padding_cache = audio_encoder_cache[1]

            return (
                outputs.audio_embeds,
                outputs.audio_embeds_mask,
                (outputs.encoder_cache[0], static_conv_padding_cache),
            )
        else:
            return self._streaming_get_audio_input_embeds_with_cache(
                audio=audio.flatten().view(audio.shape),
                audio_lengths=audio_lengths,
                audio_encoder_cache=audio_encoder_cache,  # type: ignore
            )

    def _streaming_get_audio_input_embeds_aut(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        streaming_state: AuTStreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor, AuTStreamingState]:
        """Get audio input embeddings using AuT causal streaming encoder.

        Runs the AuT wrapper in streaming mode (one chunk at a time) and passes
        the output through the input adaptor.

        Args:
            audio: Raw audio frame. Shape: [1, num_samples]. Dtype: float.
            audio_lengths: Per-sample lengths. Shape: [1]. Dtype: long.
            streaming_state: Current AuT streaming state.

        Returns:
            Tuple of (audio_embeds, audio_embeds_mask, updated_streaming_state).
            audio_embeds: Shape [1, num_frames, hidden_size]. Dtype: float.
            audio_embeds_mask: Shape [1, num_frames]. Dtype: bool.
        """
        model = self.get_model()
        assert isinstance(model.audio_encoder, AuTWrapper)

        # Reshape audio to [batch_size, num_channels, num_samples] for the wrapper.
        if audio.ndim == 2:
            audio_3d = audio[:, None, :]
        else:
            audio_3d = audio

        audio_3d = cast_to_module_dtype(audio_3d, model.audio_encoder)

        encoder_outputs = model.audio_encoder(
            audio_3d,
            streaming_state=streaming_state,
        )

        assert encoder_outputs.embeds is not None
        audio_embeds = encoder_outputs.embeds  # [1, new_frames, output_dim]
        updated_state = encoder_outputs.streaming_state
        assert isinstance(updated_state, AuTStreamingState)

        # Build mask: all frames are valid.
        audio_embeds_mask = torch.ones(
            audio_embeds.shape[:2],
            dtype=torch.bool,
            device=audio_embeds.device,
        )

        # Pass through input adaptor.
        assert model.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."
        adaptor_outputs = model.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput)
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None  # type: ignore

        return audio_embeds, audio_embeds_mask, updated_state

    def _streaming_get_audio_input_embeds_voxtral(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        streaming_state: VoxtralStreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor, VoxtralStreamingState]:
        """Get audio input embeddings using Voxtral causal streaming encoder.

        Runs the Voxtral wrapper in streaming mode (one chunk at a time) and
        passes the output through the input adaptor.

        Args:
            audio: Raw audio frame. Shape: [1, num_samples]. Dtype: float.
            audio_lengths: Per-sample lengths. Shape: [1]. Dtype: long.
            streaming_state: Current Voxtral streaming state.

        Returns:
            Tuple of (audio_embeds, audio_embeds_mask, updated_streaming_state).
            audio_embeds: Shape [1, num_frames, hidden_size]. Dtype: float.
            audio_embeds_mask: Shape [1, num_frames]. Dtype: bool.
        """
        model = self.get_model()
        assert isinstance(model.audio_encoder, VoxtralWrapper)

        if audio.ndim == 2:
            audio_3d = audio[:, None, :]
        else:
            audio_3d = audio

        audio_3d = cast_to_module_dtype(audio_3d, model.audio_encoder)

        encoder_outputs = model.audio_encoder(
            audio_3d,
            streaming_state=streaming_state,
        )

        assert encoder_outputs.embeds is not None
        audio_embeds = encoder_outputs.embeds
        updated_state = encoder_outputs.streaming_state
        assert isinstance(updated_state, VoxtralStreamingState)

        audio_embeds_mask = torch.ones(
            audio_embeds.shape[:2],
            dtype=torch.bool,
            device=audio_embeds.device,
        )

        assert model.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."
        adaptor_outputs = model.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput)
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None  # type: ignore

        return audio_embeds, audio_embeds_mask, updated_state

    def compile_audio_modules(self, duplex: bool = True, max_sequence_length: int = 8192) -> RaonDecodingState | None:
        """torch.compile audio code predictor and optionally duplex streaming path; run warmup.

        Args:
            duplex: If True, compile duplex streaming path and run warmup; else only code predictor.
            max_sequence_length: Max sequence length for KV cache during warmup.

        Returns:
            RaonDecodingState after warmup if duplex=True; None otherwise.
        """
        model = self.get_model()
        code_predictor = model.code_predictor
        assert code_predictor is not None, "compile_audio_modules requires a model with audio output support."

        code_predictor.generate_greedy = torch.compile(
            code_predictor.generate_greedy,
            fullgraph=True,
            dynamic=False,
            backend="inductor",
            mode="max-autotune",
        )

        if isinstance(model.audio_encoder, VoxtralWrapper):
            model.audio_encoder.compile_encoder()

        if duplex:
            self._streaming_get_audio_input_embeds_with_cache = torch.compile(
                self._streaming_get_audio_input_embeds_with_cache,
                fullgraph=False,
                dynamic=None,
                backend="inductor",
                mode="default",
            )

            with torch.inference_mode():
                device = self.get_model().device
                dtype = self.get_model().dtype
                samples_per_frame = int(self.sampling_rate / self.frame_rate)

                safe_vocab = min(self.vocab_size, 1000)
                state = self.init_duplex_decoding_state(
                    sequences=torch.randint(0, safe_vocab, (1, 1), device=device),
                    attention_mask=torch.ones(1, 1, dtype=torch.long, device=device),
                    do_sample=False,
                    max_sequence_length=max_sequence_length,
                )

                is_uta_warmup = getattr(self, "sequence_mode", "tua") == "uta"
                text_warmup_pos = -2 if is_uta_warmup else -3
                for step in trange(8, desc="Warmup 1/3", mininterval=0):
                    state, _ = self.duplex_decoding_step(
                        state=state,
                        audio_input=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                    )
                    if state.sequences.shape[1] >= 3:
                        if step % 2 == 0:
                            state.sequences[0, text_warmup_pos] = 0
                        else:
                            state.sequences[0, text_warmup_pos] = AUDIO_OUTPUT_PLACEHOLDER.id

                self._drain_audio_decoding_queue(state.audio_decoder_stream_id)

                state = self.init_duplex_decoding_state(
                    sequences=torch.randint(0, safe_vocab, (1, 20), device=device),
                    attention_mask=torch.ones(1, 20, dtype=torch.long, device=device),
                    do_sample=False,
                    max_sequence_length=max_sequence_length,
                    prev_state=state,
                )
                for step in trange(8, desc="Warmup 2/3", mininterval=0):
                    state, _ = self.duplex_decoding_step(
                        state=state,
                        audio_input=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                    )
                    if state.sequences.shape[1] >= 3:
                        if step % 2 == 0:
                            state.sequences[0, text_warmup_pos] = 0
                        else:
                            state.sequences[0, text_warmup_pos] = AUDIO_OUTPUT_PLACEHOLDER.id

                self._drain_audio_decoding_queue(state.audio_decoder_stream_id)

                audio_input_encoder_cache: AudioInputEncoderCache = state.audio_input_encoder_cache
                for _ in trange(256, desc="Warmup 3/3", mininterval=0):
                    if isinstance(audio_input_encoder_cache, AuTStreamingState):
                        _, _, audio_input_encoder_cache = self._streaming_get_audio_input_embeds_aut(
                            audio=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                            audio_lengths=torch.tensor([samples_per_frame], device=device),
                            streaming_state=audio_input_encoder_cache,
                        )
                    elif isinstance(audio_input_encoder_cache, VoxtralStreamingState):
                        _, _, audio_input_encoder_cache = self._streaming_get_audio_input_embeds_voxtral(
                            audio=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                            audio_lengths=torch.tensor([samples_per_frame], device=device),
                            streaming_state=audio_input_encoder_cache,
                        )
                    else:
                        _, _, audio_input_encoder_cache = self._streaming_get_audio_input_embeds(
                            audio=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                            audio_lengths=torch.tensor([samples_per_frame], device=device),
                            audio_encoder_cache=audio_input_encoder_cache,
                        )

                self.free_duplex_decoding_state(state)
                return state

        return None

    def _compute_speaker_embeds(
        self,
        speaker_audio: torch.Tensor,
        speaker_audio_lengths: torch.Tensor | None,
    ) -> torch.Tensor:

        model = self.get_model()
        assert model.speaker_encoder is not None, "_compute_speaker_embeds requires a model with a speaker encoder."

        if speaker_audio_lengths is None:
            speaker_audio_lengths = torch.full(
                (speaker_audio.shape[0],),
                speaker_audio.shape[1],
                device=speaker_audio.device,
                dtype=torch.long,
            )

        if isinstance(model.speaker_encoder, PretrainedSpeakerEncoder):
            return model.speaker_encoder(speaker_audio, speaker_audio_lengths)

        tokenizer_output = self.tokenize_audio(
            audio=speaker_audio,
            audio_lengths=speaker_audio_lengths,
            return_mimi_features=True,
        )
        return model.speaker_encoder(
            tokenizer_output.mimi_features,
            mask=tokenizer_output.audio_codes_mask,
        )

    def _pad_audio_input(self, audio_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        sampling_rate = self.sampling_rate
        frame_rate: float = self.frame_rate
        assert (samples_per_frame := int(sampling_rate / frame_rate)) == sampling_rate / frame_rate, (
            f"Expected `sampling_rate / frame_rate` to be an integer but got `{sampling_rate / frame_rate}`."
        )

        if audio_input.shape[1] < samples_per_frame:
            audio_input = F.pad(audio_input, (0, samples_per_frame - audio_input.shape[1]))
            logger.warning(
                f"Duplex decoding uses {samples_per_frame} samples per frame, "
                f"but {audio_input.shape[1]} samples were input. "
                "The input audio has been padded accordingly."
            )
        elif audio_input.shape[1] > samples_per_frame:
            audio_input = audio_input[:, :samples_per_frame]
            logger.warning(
                f"Duplex decoding uses {samples_per_frame} samples per frame, "
                f"but {audio_input.shape[1]} samples were input. "
                "The input audio has been truncated accordingly."
            )

        audio_input_lengths = torch.tensor([audio_input.shape[1]], device=audio_input.device)
        return audio_input, audio_input_lengths

    @torch.inference_mode()
    def _update_duplex_sequences_and_generate_audio_codes(
        self,
        new_logits: torch.Tensor,
        new_last_hidden_state: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        machine_state: DuplexMachineState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DuplexMachineState | None]:
        if new_logits.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{new_logits.shape[0]}`.")

        # Text prediction is always read from the token before [A].
        user_logits = new_logits[:, -2:-1, : self.vocab_size]
        # Apply pad penalty to reduce pad probability (encourage longer responses)
        if eos_penalty > 0:
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_pad_token_id] -= eos_penalty
        if sil_penalty > 0 and getattr(self, "use_sil_token", False):
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_sil_token_id] -= sil_penalty
        # Apply BC penalty to adjust backchannel probability in SIL phase.
        # Positive values suppress BC, negative values boost BC.
        if (
            bc_penalty != 0
            and getattr(self, "use_backchannel_token", False)
            and machine_state is not None
            and machine_state.phase == DuplexPhase.SIL
        ):
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_bc_token_id] -= bc_penalty

        # Apply state-machine logit masking to enforce valid transitions.
        if machine_state is not None and self.use_duplex_end_pad:
            user_logits = self._state_manager.apply_logit_mask(user_logits, machine_state, self.vocab_size)

        if do_sample:
            user_logits = logits_processor(input_ids=sequences, scores=user_logits[:, -1])  # type: ignore
            user_probs = F.softmax(user_logits, dim=-1, dtype=torch.float32)
            user_probs = user_probs.clamp_min(torch.finfo(user_probs.dtype).tiny)
            text_or_eos_id = torch.multinomial(user_probs, num_samples=1)
        else:
            text_or_eos_id = user_logits[:, -1:].argmax(dim=-1)

        predicted_token_id = int(text_or_eos_id.item())

        # Determine current speech phase for conditional audio code generation.
        is_in_speech = machine_state is not None and machine_state.phase == DuplexPhase.SPEECH

        # Only generate audio codes when currently in speech (pre-transition).
        # For onset frames (SIL→SPEECH), fresh codes are generated after transition.
        new_audio_codes = None
        if is_in_speech:
            first_code_sampler = None
            if do_sample:
                first_code_sampler = make_audio_code_sampler(
                    sequences=sequences,
                    logits_processor=logits_processor,
                    audio_codes=audio_codes,
                    ras_enabled=False,
                    ras_window_size=40,
                    ras_repetition_threshold=0.1,
                )
            new_audio_codes = self.generate_audio_codes(
                talker_last_hidden_state=new_last_hidden_state[:, -1:],
                first_code_sampler=first_code_sampler,
                allow_audio_end=False,
            )
            new_audio_codes = new_audio_codes.clone()

        new_machine_state: DuplexMachineState | None = None
        # Use state machine for frame construction
        new_machine_state, frame_tokens, emitted_audio = self._state_manager.transition(
            machine_state, predicted_token_id, sequences.device
        )
        input_ids = torch.tensor([frame_tokens], device=sequences.device)

        # Conditionally append audio codes based on whether audio was emitted.
        if emitted_audio:
            if new_audio_codes is None:
                # Onset frame (SIL→SPEECH): generate fresh codes now.
                first_code_sampler = None
                if do_sample:
                    first_code_sampler = make_audio_code_sampler(
                        sequences=sequences,
                        logits_processor=logits_processor,
                        audio_codes=audio_codes,
                        ras_enabled=False,
                        ras_window_size=40,
                        ras_repetition_threshold=0.1,
                    )
                new_audio_codes = self.generate_audio_codes(
                    talker_last_hidden_state=new_last_hidden_state[:, -1:],
                    first_code_sampler=first_code_sampler,
                    allow_audio_end=False,
                )
                new_audio_codes = new_audio_codes.clone()
            audio_end_predicted_mask = new_audio_codes[:, 0] == self.codebook_size
            if audio_end_predicted_mask.any():
                # Clamp audio-end sentinel back into Mimi codebook range.
                new_audio_codes[audio_end_predicted_mask, 0] = 0
            audio_codes = torch.cat((audio_codes, new_audio_codes[None]), dim=1)
            audio_codes_mask = torch.cat(
                (
                    audio_codes_mask,
                    torch.tensor([[True]], device=audio_codes.device, dtype=torch.bool),
                ),
                dim=1,
            )
        # else: SIL frame — no audio codes generated or appended (matches Reference)

        sequences = torch.cat((sequences, input_ids), dim=1)
        attention_mask = F.pad(attention_mask, (0, input_ids.shape[1]), value=1)
        return input_ids, sequences, attention_mask, audio_codes, audio_codes_mask, new_machine_state

    @torch.inference_mode()
    def duplex_decoding_step(
        self,
        state: RaonDecodingState,
        audio_input: torch.Tensor,
    ) -> tuple[RaonDecodingState, torch.Tensor]:
        """Run one duplex decoding step: encode user audio, predict tokens/codes, push codes, pull waveform.

        Args:
            state: Current duplex decoding state.
            audio_input: One frame of user audio. Shape: [1, num_samples_per_frame]. Dtype: float.

        Returns:
            Tuple of (updated_state, decoded_audio).
            decoded_audio: Decoded waveform for this frame. Shape: [1, num_samples_per_frame]. Dtype: float.
        """
        if state.sequences.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{state.sequences.shape[0]}`.")

        last_token = int(state.sequences[0, -1].item())
        valid_last_tokens = {AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id}
        if last_token not in valid_last_tokens:
            raise ValueError(f"Last token must be one of `{sorted(valid_last_tokens)}` but got `{last_token}`.")

        prev_audio_codes_length = state.audio_codes.shape[1]

        audio_input, audio_input_lengths = self._pad_audio_input(audio_input=audio_input)

        # Dispatch to AuT, Voxtral, or Mimi streaming encoder based on cache type.
        audio_input_encoder_cache: AudioInputEncoderCache
        if isinstance(state.audio_input_encoder_cache, AuTStreamingState):
            audio_input_embeds, audio_input_embeds_mask, audio_input_encoder_cache = (
                self._streaming_get_audio_input_embeds_aut(
                    audio=audio_input,
                    audio_lengths=audio_input_lengths,
                    streaming_state=state.audio_input_encoder_cache,
                )
            )
        elif isinstance(state.audio_input_encoder_cache, VoxtralStreamingState):
            audio_input_embeds, audio_input_embeds_mask, audio_input_encoder_cache = (
                self._streaming_get_audio_input_embeds_voxtral(
                    audio=audio_input,
                    audio_lengths=audio_input_lengths,
                    streaming_state=state.audio_input_encoder_cache,
                )
            )
        else:
            audio_input_embeds, audio_input_embeds_mask, audio_input_encoder_cache = self._streaming_get_audio_input_embeds(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
                audio_encoder_cache=state.audio_input_encoder_cache,
            )

        # Determine num_input_tokens from state machine
        num_input_tokens = state.machine_state.num_input_tokens
        has_text_input = num_input_tokens == 3

        step_audio_codes = state.audio_codes[:, -1:] if state.audio_codes.shape[1] > 0 else None
        step_audio_codes_mask = state.audio_codes_mask[:, -1:] if state.audio_codes_mask.shape[1] > 0 else None

        full_position_ids = state.attention_mask.cumsum(dim=1) - 1
        seq_len = state.attention_mask.shape[1]
        cache_position = torch.arange(seq_len - num_input_tokens, seq_len, device=state.sequences.device)
        talker_last_hidden_state, text_logits = self.inference_forward(
            input_ids=state.sequences[:, -num_input_tokens:],
            attention_mask=None,
            position_ids=full_position_ids[:, -num_input_tokens:],
            audio_output_codes=step_audio_codes,
            audio_output_codes_mask=step_audio_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=None,
            use_cache=True,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
        )

        # Force SIL for the remaining listen-first warmup frames.
        if state.forced_sil_remaining > 0 and getattr(self, "use_sil_token", False):
            forced_logits = torch.full_like(text_logits, fill_value=-1e9)
            forced_logits[:, -2, self.duplex_sil_token_id] = 0.0
            text_logits = forced_logits

        # Standard mode (with optional EPAD support)
        new_machine_state: DuplexMachineState | None = state.machine_state
        _, sequences, attention_mask, audio_codes, audio_codes_mask, new_machine_state = (
            self._update_duplex_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=state.sequences,
                attention_mask=state.attention_mask,
                audio_codes=state.audio_codes,
                audio_codes_mask=state.audio_codes_mask,
                do_sample=state.do_sample,
                logits_processor=state.logits_processor,
                eos_penalty=state.eos_penalty,
                sil_penalty=state.sil_penalty,
                bc_penalty=state.bc_penalty,
                machine_state=state.machine_state,
            )
        )

        # Detect if current frame is SIL-no-audio using state machine
        is_current_sil_no_audio = not new_machine_state.emitted_audio

        if is_current_sil_no_audio:
            # Clear semantic buffer so post-SIL onset starts fresh (matches Reference).
            new_semantic_buffer = None
            # Push silence codes to keep decoder conv state warm (matches Reference).
            silence_codes = self.get_silence_codes(state.sequences.device)
            self.push_audio_codes(audio_codes=silence_codes, stream_id=state.audio_decoder_stream_id)
            decoded_audio = self.pull_audio(state.audio_decoder_stream_id)
            if decoded_audio.device != state.sequences.device:
                decoded_audio = decoded_audio.to(state.sequences.device)
        else:
            # Normal speech frame
            valid_trailing_tokens = {AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id}
            assert int(sequences[0, -1].item()) in valid_trailing_tokens, (
                f"Last token must be one of `{sorted(valid_trailing_tokens)}` but got `{sequences[0, -1]}`."
            )
            # Audio codes grow by 1 only when emitted_audio=True (speech frames).
            expected_codes = prev_audio_codes_length + 1
            assert audio_codes.shape[1] == expected_codes, (
                f"Expected `{expected_codes}` audio codes but got `{audio_codes.shape[1]}`."
            )

            # Handle acoustic delay if configured
            new_semantic_buffer = state.semantic_buffer
            if self.max_delay > 0:
                current_codes = audio_codes[0, -1]
                semantic_code = current_codes[0:1]
                acoustic_codes = current_codes[1:]

                if state.semantic_buffer is None:
                    new_semantic_buffer = semantic_code
                    output_codes = torch.zeros_like(current_codes)
                    output_codes[0] = semantic_code[0]
                else:
                    output_codes = torch.cat([state.semantic_buffer, acoustic_codes], dim=0)
                    new_semantic_buffer = semantic_code

                self.push_audio_codes(audio_codes=output_codes, stream_id=state.audio_decoder_stream_id)
            else:
                self.push_audio_codes(audio_codes=audio_codes[0, -1], stream_id=state.audio_decoder_stream_id)

            decoded_audio = self.pull_audio(state.audio_decoder_stream_id)

        updated_state = RaonDecodingState(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            past_key_values=state.past_key_values,
            audio_input_encoder_cache=audio_input_encoder_cache,
            audio_decoder_stream_id=state.audio_decoder_stream_id,
            do_sample=state.do_sample,
            logits_processor=state.logits_processor,
            num_code_groups=state.num_code_groups,
            semantic_buffer=new_semantic_buffer,
            eos_penalty=state.eos_penalty,
            sil_penalty=state.sil_penalty,
            bc_penalty=state.bc_penalty,
            machine_state=new_machine_state,
            forced_sil_remaining=max(0, state.forced_sil_remaining - 1),
        )

        return updated_state, decoded_audio

    def init_duplex_decoding_state(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        max_sequence_length: int = 8192,
        prev_state: RaonDecodingState | None = None,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        speaker_embeds: torch.Tensor | None = None,
        speak_first: bool = False,
    ) -> RaonDecodingState:
        """Initialize duplex decoding state and run the first frame to obtain [U][A] prompt.

        Args:
            sequences: Initial text tokens (system prompt). Shape: [1, seq_length]. Dtype: long.
            attention_mask: Mask for valid positions. Shape: [1, seq_length]. Dtype: long.
            do_sample: Whether to sample (vs. greedy).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p filtering.
            max_sequence_length: Max sequence length for KV cache.
            prev_state: Previous state to reuse caches (e.g. from warmup).
            eos_penalty: Penalty to subtract from pad/eos logit to encourage longer output.
            sil_penalty: Penalty to subtract from SIL logit.

        Returns:
            RaonDecodingState ready for duplex_decoding_step.
        """
        self.start_concurrent_audio_decoder()

        # Lazy init state manager for logit masking
        if not hasattr(self, "_state_manager"):
            self._state_manager = DuplexStateManager(
                DuplexStateConfig(
                    use_duplex_end_pad=self.use_duplex_end_pad,
                    use_sil_token=getattr(self, "use_sil_token", False),
                    no_audio_in_sil=getattr(self, "no_audio_in_sil", False),
                    sequence_mode=getattr(self, "sequence_mode", "tua"),
                    duplex_pad_token_id=self.duplex_pad_token_id,
                    duplex_end_pad_token_id=self.duplex_end_pad_token_id,
                    duplex_sil_token_id=getattr(self, "duplex_sil_token_id", -1),
                    use_backchannel_token=getattr(self, "use_backchannel_token", False),
                    duplex_bc_token_id=getattr(self, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id),
                )
            )

        if isinstance(self.get_model().config.audio_encoder_config, Qwen3OmniMoeAudioEncoderConfig):
            aut_is_causal = getattr(self.get_model(), "aut_is_causal", False)
            if not aut_is_causal:
                raise NotImplementedError(
                    "Duplex streaming decoding requires a causal audio encoder. "
                    "Set `aut_is_causal=True` in the model config to enable "
                    "causal AuT streaming for duplex decoding."
                )

        if sequences.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{sequences.shape[0]}`.")

        if self.max_delay > 1:
            raise NotImplementedError(
                f"Duplex decoding only supports acoustic_delay of 0 or 1, "
                f"got max_delay={self.max_delay}. semantic_buffer assumes single-step delay."
            )
        if self.max_delay > 0 and self.delays[0] != 0:
            raise ValueError(
                f"Semantic codebook (index 0) must have delay=0 for duplex decoding, "
                f"got delays[0]={self.delays[0]}. The semantic_buffer logic assumes "
                f"delays=[0, N, N, ..., N]."
            )

        # Auto-insert speaker token if speaker_embeds provided but not in sequences
        if speaker_embeds is not None and self.speaker_token_id is not None:
            if not (sequences == self.speaker_token_id).any():
                speaker_token = torch.full(
                    (sequences.shape[0], 1),
                    fill_value=self.speaker_token_id,
                    dtype=sequences.dtype,
                    device=sequences.device,
                )
                sequences = torch.cat((sequences, speaker_token), dim=1)
                if attention_mask is not None:
                    attention_mask = F.pad(attention_mask, (0, 1), value=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(sequences)

        _audio_ids = torch.tensor(list(_AUDIO_SPECIAL_TOKEN_IDS), device=sequences.device)
        assert not torch.isin(sequences, _audio_ids).any() and (attention_mask == 1).all(), (
            "All `sequences` must be text tokens and all `attention_mask` values must be 1. "
            f"`{sequences=}`, `{attention_mask=}`."
        )

        logits_processor = LogitsProcessorList()
        if do_sample and temperature and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
        if do_sample and top_k and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        if prev_state is not None:
            past_key_values = self.init_past_key_values(
                batch_size=1,
                max_sequence_length=max_sequence_length,
                prev_cache=prev_state.past_key_values,
            )
            audio_input_encoder_cache = self.init_audio_encoder_cache(prev_cache=prev_state.audio_input_encoder_cache)
            self._drain_audio_decoding_queue(stream_id=prev_state.audio_decoder_stream_id)
            self._destroy_audio_decoder_stream(prev_state.audio_decoder_stream_id)
        else:
            past_key_values = self.init_past_key_values(batch_size=1, max_sequence_length=max_sequence_length)
            audio_input_encoder_cache = self.init_audio_encoder_cache()

        audio_decoder_stream_id = self.create_audio_decoder_stream()

        audio_codes = torch.zeros(1, 0, self.num_code_groups, dtype=torch.long, device=sequences.device)
        audio_codes_mask = torch.zeros(1, 0, dtype=torch.bool, device=sequences.device)

        input_ids = torch.cat(
            [
                sequences,
                torch.tensor(
                    [[IM_START.id, AUDIO_START.id]],
                    device=sequences.device,
                ),
            ],
            dim=1,
        )
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

        talker_last_hidden_state, text_logits = self.inference_forward(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            speaker_embeds=speaker_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        initial_machine_state = self._state_manager.initial_state(speak_first=speak_first)

        # Force the first [U] prediction explicitly.
        forced_initial_prediction_id = self._state_manager.initial_forced_prediction_id(speak_first)
        if forced_initial_prediction_id is not None:
            forced_logits = torch.full_like(text_logits, fill_value=-1e9)
            forced_logits[:, -2, forced_initial_prediction_id] = 0.0
            text_logits = forced_logits

        _, sequences, attention_mask, audio_codes, audio_codes_mask, initial_machine_state = (
            self._update_duplex_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=input_ids,
                attention_mask=torch.ones_like(input_ids),
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                do_sample=do_sample,
                logits_processor=logits_processor,
                machine_state=initial_machine_state,
            )
        )

        # Check if audio was emitted (listen-first SIL may not emit audio).
        emitted_audio = initial_machine_state.emitted_audio if initial_machine_state is not None else True

        initial_semantic_buffer = None
        if not emitted_audio:
            # Listen-first: no audio emitted on first frame. Push silence to keep decoder warm.
            silence_codes = self.get_silence_codes(sequences.device)
            self.push_audio_codes(audio_codes=silence_codes, stream_id=audio_decoder_stream_id)
        elif self.max_delay > 0:
            # When acoustic delay is active, the first frame is a placeholder:
            # only the semantic code (CB0) is valid; acoustic codes (CB1-7) are zeros
            # because no previous frame exists to provide delayed acoustic predictions.
            # The duplex_step handler (semantic_buffer logic) will produce properly
            # aligned frames from the second step onward. This initial placeholder frame
            # is expected and acceptable — the audio decoder handles it gracefully.
            first_codes = audio_codes[0, -1]
            semantic_code = first_codes[0:1]
            output_codes = torch.zeros_like(first_codes)
            output_codes[0] = semantic_code[0]
            initial_semantic_buffer = semantic_code
            self.push_audio_codes(audio_codes=output_codes, stream_id=audio_decoder_stream_id)
        else:
            self.push_audio_codes(audio_codes=audio_codes[0, -1], stream_id=audio_decoder_stream_id)

        # Listen-first warmup: force one additional SIL frame after init.
        forced_sil_remaining = 1 if (not speak_first and getattr(self, "use_sil_token", False)) else 0

        state = RaonDecodingState(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            past_key_values=past_key_values,
            audio_input_encoder_cache=audio_input_encoder_cache,
            audio_decoder_stream_id=audio_decoder_stream_id,
            do_sample=do_sample,
            logits_processor=logits_processor,
            num_code_groups=self.num_code_groups,
            semantic_buffer=initial_semantic_buffer,
            eos_penalty=eos_penalty,
            sil_penalty=float(sil_penalty),
            bc_penalty=float(bc_penalty),
            machine_state=initial_machine_state,
            forced_sil_remaining=forced_sil_remaining,
        )
        return state

    def _extract_gt_tokens_per_frame(
        self,
        gt_input_ids: torch.Tensor,
        system_prefix_len: int,
    ) -> list[list[int]]:
        """
        Extract GT tokens for each frame from the training input_ids sequence.

        The GT sequence structure after system prefix:
        - [im_start] [audio_start] then per-frame tokens
        - Each frame ends with [A] token
        - Silence frame: [U] [A] (2 tokens)
        - EPAD frame: EPAD [U] [A] (3 tokens)
        - Text frame: text [U] [A] (3 tokens)

        Returns:
            List of token lists, one per frame. Each list contains the tokens
            to add for that frame (e.g., [text, U, A] or [U, A]).
        """
        input_ids = gt_input_ids[0].tolist() if gt_input_ids.dim() > 1 else gt_input_ids.tolist()

        frame_tokens: list[list[int]] = []
        # Skip preamble tokens after system prefix.
        # Preamble may vary ([IM_START], [IM_START, AUDIO_START], [IM_START, EPAD, AUDIO_START]).
        preamble_token_ids = {IM_START.id, AUDIO_START.id}
        position = system_prefix_len
        while position < len(input_ids) and input_ids[position] in preamble_token_ids:
            position += 1
        sequence_mode = self._state_manager._config.effective_sequence_mode

        while position < len(input_ids):
            token_id = input_ids[position]

            if token_id == AUDIO_INPUT_PLACEHOLDER.id:
                if sequence_mode == "uta" and position + 2 < len(input_ids):
                    frame_end_token = input_ids[position + 2]
                    if frame_end_token in (AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id):
                        frame_tokens.append(input_ids[position : position + 3])
                        position += 3
                        continue
                if position + 1 < len(input_ids) and input_ids[position + 1] == AUDIO_OUTPUT_PLACEHOLDER.id:
                    frame_tokens.append(input_ids[position : position + 2])
                    position += 2
                    continue
            elif sequence_mode == "tua" and position + 2 < len(input_ids):
                if input_ids[position + 1] == AUDIO_INPUT_PLACEHOLDER.id and input_ids[position + 2] in (
                    AUDIO_OUTPUT_PLACEHOLDER.id,
                    AUDIO_START.id,
                ):
                    frame_tokens.append(input_ids[position : position + 3])
                    position += 3
                    continue

            position += 1

        return frame_tokens

    def _sample_from_logits(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        force_audio_output: bool,
        force_text_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
    ) -> torch.Tensor:
        if force_audio_output:
            return torch.full((logits.shape[0], 1), AUDIO_OUTPUT_PAD.id, dtype=torch.long, device=logits.device)

        next_token_logits = logits[:, -1].clone()
        if force_text_output:
            next_token_logits[..., AUDIO_OUTPUT_PAD.id] = torch.finfo(next_token_logits.dtype).min

        if do_sample:
            processed_logits = logits_processor(input_ids=sequences, scores=next_token_logits)  # type: ignore[arg-type]
            probs = F.softmax(processed_logits, dim=-1, dtype=torch.float32)
            probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            return torch.multinomial(probs, num_samples=1)
        return next_token_logits.argmax(dim=-1, keepdim=True)

    def _update_sequences_and_generate_audio_codes(
        self,
        new_logits: torch.Tensor,
        new_last_hidden_state: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        is_complete: torch.Tensor,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        is_generating_audio: torch.Tensor | None = None,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        suppress_audio_eos: bool = False,
        ras_skip_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample next token and generate audio codes for one autoregressive step.

        Args:
            new_logits: Logits from the latest forward pass. Shape: [batch_size, 1, vocab_size].
            new_last_hidden_state: Talker hidden state. Shape: [batch_size, 1, hidden_dim].
            sequences: Current token sequence. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length].
            audio_codes: Audio code history. Shape: [batch_size, num_frames, num_code_groups].
            audio_codes_mask: Audio code mask. Shape: [batch_size, num_frames].
            is_complete: Per-sample completion flag. Shape: [batch_size].
            pad_token_id: Token ID used for padding completed sequences.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            do_sample: Whether to sample (vs. greedy).
            logits_processor: Logits processing pipeline.
            is_generating_audio: Per-sample audio generation state. Shape: [batch_size] or None.
            ras_enabled: Enable repetition-aware sampling for audio codes.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.

        Returns:
            Tuple of (sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio).
        """
        next_is_generating_audio = is_generating_audio.clone() if is_generating_audio is not None else None

        if is_generating_audio is None:
            new_ids = self._sample_from_logits(
                sequences=sequences,
                logits=new_logits,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
            )
        else:
            new_ids = torch.full((new_logits.shape[0], 1), pad_token_id, dtype=torch.long, device=new_logits.device)
            text_mode_mask = ~is_generating_audio
            if text_mode_mask.any():
                new_ids[text_mode_mask] = self._sample_from_logits(
                    sequences=sequences[text_mode_mask],
                    logits=new_logits[text_mode_mask],
                    force_audio_output=False,
                    force_text_output=True,
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                )
            if is_generating_audio.any():
                new_ids[is_generating_audio] = AUDIO_OUTPUT_PAD.id

        new_ids[is_complete, -1] = pad_token_id
        is_complete |= new_ids[:, -1] == IM_END.id

        if next_is_generating_audio is not None:
            assert is_generating_audio is not None
            start_audio_mask = (~is_complete) & (~is_generating_audio) & (new_ids[:, -1] == AUDIO_START.id)
            next_is_generating_audio[start_audio_mask] = True
            is_audio_output = (~is_complete) & is_generating_audio
        else:
            is_audio_output = (~is_complete) & (new_ids[:, -1] == AUDIO_OUTPUT_PAD.id)
            new_ids[is_audio_output, -1] = AUDIO_OUTPUT_PLACEHOLDER.id

        final_audio_output_mask = is_audio_output.clone()
        sequences_with_new_ids = torch.cat((sequences, new_ids), dim=1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        audio_codes_mask = torch.cat((audio_codes_mask, final_audio_output_mask[:, None]), dim=1)
        audio_codes = F.pad(audio_codes, (0, 0, 0, 1))

        if is_audio_output.any():
            first_code_sampler = None
            if do_sample:
                first_code_sampler = make_audio_code_sampler(
                    sequences=sequences_with_new_ids[is_audio_output],
                    logits_processor=logits_processor,
                    audio_codes=audio_codes[is_audio_output, :-1],
                    ras_enabled=ras_enabled,
                    ras_window_size=ras_window_size,
                    ras_repetition_threshold=ras_repetition_threshold,
                    ras_skip_frames=ras_skip_frames,
                )

            generated_audio_codes = self.generate_audio_codes(
                talker_last_hidden_state=new_last_hidden_state[is_audio_output, -1:],
                first_code_sampler=first_code_sampler,
            )
            generated_audio_end_mask = generated_audio_codes[:, 0] == self.codebook_size
            if suppress_audio_eos:
                generated_audio_end_mask[:] = False
            new_ids[is_audio_output, -1] = AUDIO_OUTPUT_PLACEHOLDER.id

            if generated_audio_end_mask.any():
                local_non_end_mask = ~generated_audio_end_mask
                global_non_end_mask = is_audio_output.clone()
                global_non_end_mask[is_audio_output] = local_non_end_mask
                final_audio_output_mask = global_non_end_mask
                audio_end_global_mask = is_audio_output.clone()
                audio_end_global_mask[is_audio_output] = generated_audio_end_mask
                new_ids[audio_end_global_mask, -1] = AUDIO_END.id
                if next_is_generating_audio is not None:
                    next_is_generating_audio[audio_end_global_mask] = False
                if local_non_end_mask.any():
                    audio_codes[global_non_end_mask, -1] = generated_audio_codes[local_non_end_mask]
            else:
                audio_codes[is_audio_output, -1] = generated_audio_codes

        audio_codes_mask[:, -1] = final_audio_output_mask
        sequences = torch.cat((sequences, new_ids), dim=1)
        if next_is_generating_audio is not None:
            next_is_generating_audio[is_complete] = False

        return sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, next_is_generating_audio

    def _generation_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        audio_input: torch.Tensor | None,
        audio_output: torch.Tensor | None,
        audio_input_lengths: torch.Tensor | None,
        audio_output_lengths: torch.Tensor | None,
        audio_output_codes: torch.Tensor | None,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        disable_eos_on_first_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        max_sequence_length: int,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        speaker_embeds: torch.Tensor | None = None,
        suppress_audio_eos: bool = False,
        ras_skip_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, Any]:
        """Run the prefill phase: encode inputs and generate the first output token and audio codes.

        Args:
            input_ids: Input token IDs. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length] or None.
            audio_input: Raw input audio. Shape: [batch_size, num_samples] or None.
            audio_output: Raw output audio for teacher-forced codes. Shape: [batch_size, num_samples] or None.
            audio_input_lengths: Per-sample input audio lengths. Shape: [batch_size] or None.
            audio_output_lengths: Per-sample output audio lengths. Shape: [batch_size] or None.
            audio_output_codes: Pre-computed output audio codes or None.
            pad_token_id: Token ID used for padding completed sequences.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            disable_eos_on_first_output: If True, prevent EOS on the first generated token.
            do_sample: Whether to sample (vs. greedy).
            logits_processor: Logits processing pipeline.
            max_sequence_length: Maximum sequence length for KV cache allocation.
            audio_input_embeds: Pre-computed audio input embeddings or None.
            audio_input_embeds_mask: Mask for pre-computed audio input embeddings or None.
            ras_enabled: Enable repetition-aware sampling.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.
            speaker_embeds: Speaker conditioning embeddings or None.
            suppress_audio_eos: If True, prevent audio EOS on the first generated audio frame.
            ras_skip_frames: Number of leading frames to ignore in RAS history.

        Returns:
            Tuple of (sequences, attention_mask, audio_codes, audio_codes_mask,
            is_complete, is_generating_audio, past_key_values).
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = attention_mask.cumsum(dim=1) - 1
        past_key_values = self.init_past_key_values(batch_size=input_ids.shape[0], max_sequence_length=max_sequence_length)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        # Auto-create audio_output_codes_mask when codes are provided without mask.
        audio_output_codes_mask = None
        if audio_output_codes is not None:
            audio_output_codes_mask = torch.ones(
                audio_output_codes.shape[0],
                audio_output_codes.shape[1],
                dtype=torch.bool,
                device=audio_output_codes.device,
            )
        talker_last_hidden_state, text_logits = self.inference_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_input=audio_input,
            audio_output=audio_output,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        if disable_eos_on_first_output:
            text_logits[..., IM_END.id] = torch.finfo(text_logits.dtype).min

        batch_size = input_ids.shape[0]
        sequences = input_ids
        audio_codes = torch.zeros(batch_size, 0, self.num_code_groups, dtype=torch.long, device=input_ids.device)
        audio_codes_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=input_ids.device)
        is_complete = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        if force_audio_output:
            is_generating_audio = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        else:
            is_generating_audio = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio = (
            self._update_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=sequences,
                attention_mask=attention_mask,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                is_complete=is_complete,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
                is_generating_audio=is_generating_audio,
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
                suppress_audio_eos=suppress_audio_eos,
                ras_skip_frames=ras_skip_frames,
            )
        )
        return sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio, past_key_values

    def _decoding_step(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        is_complete: torch.Tensor,
        past_key_values: Any,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        is_generating_audio: torch.Tensor | None,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        suppress_audio_eos: bool = False,
        ras_skip_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run one autoregressive decoding step using cached KV states.

        Args:
            sequences: Current token sequence. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length].
            audio_codes: Audio code history. Shape: [batch_size, num_frames, num_code_groups].
            audio_codes_mask: Audio code mask. Shape: [batch_size, num_frames].
            is_complete: Per-sample completion flag. Shape: [batch_size].
            past_key_values: Cached key-value states from previous steps.
            pad_token_id: Token ID used for padding completed sequences.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            is_generating_audio: Per-sample audio generation state or None.
            do_sample: Whether to sample (vs. greedy).
            logits_processor: Logits processing pipeline.
            ras_enabled: Enable repetition-aware sampling.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.

        Returns:
            Tuple of (sequences, attention_mask, audio_codes, audio_codes_mask,
            is_complete, is_generating_audio).
        """
        supports_audio_output = bool(getattr(self, "supports_audio_output", True))
        cache_position = attention_mask.sum(dim=1, keepdim=False) - 1
        talker_last_hidden_state, text_logits = self.inference_forward(
            input_ids=sequences[:, -1:],
            position_ids=cache_position.unsqueeze(1),
            attention_mask=attention_mask,
            audio_output_codes=audio_codes[:, -1:] if supports_audio_output else None,
            audio_output_codes_mask=audio_codes_mask[:, -1:] if supports_audio_output else None,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        return self._update_sequences_and_generate_audio_codes(
            new_logits=text_logits,
            new_last_hidden_state=talker_last_hidden_state,
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            is_complete=is_complete,
            pad_token_id=pad_token_id,
            force_audio_output=force_audio_output,
            force_text_output=force_text_output,
            do_sample=do_sample,
            logits_processor=logits_processor,
            is_generating_audio=is_generating_audio,
            ras_enabled=ras_enabled,
            ras_window_size=ras_window_size,
            ras_repetition_threshold=ras_repetition_threshold,
            suppress_audio_eos=suppress_audio_eos,
            ras_skip_frames=ras_skip_frames,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        pad_token_id: int = PAD.id,
        num_code_groups: int | None = None,
        force_audio_output: bool = False,
        force_text_output: bool = False,
        disable_eos_on_first_output: bool = True,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        disable_tqdm: bool = False,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        speaker_embeds: torch.Tensor | None = None,
        speaker_audio: torch.Tensor | None = None,
        speaker_audio_lengths: torch.Tensor | None = None,
        continuation_silence_frames: int = 0,
    ) -> GenerateOutput:
        """Run offline (non-streaming) inference: generate text and/or audio from input tokens.

        Args:
            input_ids: Input token IDs. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length] or None.
            audio_input: Raw input audio. Shape: [batch_size, num_samples] or None.
            audio_output: Raw output audio for teacher-forced codes or None.
            audio_input_lengths: Per-sample input audio lengths or None.
            audio_output_lengths: Per-sample output audio lengths or None.
            audio_output_codes: Pre-computed output audio codes or None.
            audio_input_embeds: Pre-computed audio input embeddings or None.
            audio_input_embeds_mask: Mask for pre-computed audio input embeddings or None.
            max_new_tokens: Maximum number of new tokens to generate.
            pad_token_id: Token ID used for padding completed sequences.
            num_code_groups: Number of audio codebook groups to use.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            disable_eos_on_first_output: If True, prevent EOS on the first generated token.
            do_sample: Whether to sample (vs. greedy).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p filtering.
            disable_tqdm: If True, suppress progress bar.
            ras_enabled: Enable repetition-aware sampling for audio codes.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.
            speaker_embeds: Pre-computed speaker conditioning embeddings or None.
            speaker_audio: Raw speaker reference audio for on-the-fly embedding computation or None.
            speaker_audio_lengths: Per-sample speaker audio lengths or None.
            continuation_silence_frames: Number of initial generated audio frames to
                replace with Mimi-encoded silence during TTS continuation warmup.

        Returns:
            GenerateOutput with generated sequences, audio codes, and masks.
        """
        if speaker_audio is not None and speaker_embeds is None:
            speaker_embeds = self._compute_speaker_embeds(speaker_audio, speaker_audio_lengths)

        if num_code_groups is None:
            num_code_groups = self.num_code_groups
        assert num_code_groups <= self.num_code_groups, (
            f"Expected num_code_groups <= {self.num_code_groups}, got {num_code_groups}."
        )

        if not bool(getattr(self, "supports_audio_output", True)):
            force_audio_output = False
            force_text_output = True

        logits_processor = LogitsProcessorList()
        if do_sample and temperature and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
        if do_sample and top_k and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio, past_key_values = (
            self._generation_prefill(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_input=audio_input,
                audio_output=audio_output,
                audio_input_lengths=audio_input_lengths,
                audio_output_lengths=audio_output_lengths,
                audio_output_codes=audio_output_codes,
                audio_input_embeds=audio_input_embeds,
                audio_input_embeds_mask=audio_input_embeds_mask,
                speaker_embeds=speaker_embeds,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                disable_eos_on_first_output=disable_eos_on_first_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
                max_sequence_length=8 * (1 + (input_ids.shape[1] + max_new_tokens) // 8),
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
                suppress_audio_eos=continuation_silence_frames > 0,
                ras_skip_frames=continuation_silence_frames,
            )
        )

        silence_codes = None
        generated_audio_frame_count = 0
        if continuation_silence_frames > 0:
            silence_codes = self.get_silence_codes(input_ids.device)

        if silence_codes is not None and audio_codes_mask.any():
            generated_audio_frame_count = 1
            if generated_audio_frame_count <= continuation_silence_frames:
                audio_codes[:, -1] = silence_codes

        for _ in trange(max_new_tokens - 1, disable=disable_tqdm):
            if is_complete.all():
                break
            in_silence_window = silence_codes is not None and generated_audio_frame_count < continuation_silence_frames
            sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio = self._decoding_step(
                sequences=sequences,
                attention_mask=attention_mask,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                is_complete=is_complete,
                past_key_values=past_key_values,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                is_generating_audio=is_generating_audio,
                do_sample=do_sample,
                logits_processor=logits_processor,
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
                suppress_audio_eos=in_silence_window,
                ras_skip_frames=continuation_silence_frames,
            )

            if silence_codes is not None and audio_codes_mask[:, -1].any():
                generated_audio_frame_count += 1
                if generated_audio_frame_count <= continuation_silence_frames:
                    audio_codes[audio_codes_mask[:, -1], -1] = silence_codes

        audio = None
        audio_lengths = None
        if not force_text_output and audio_codes_mask.any():
            contiguous_audio_sequences = pad_sequence(
                [seq[mask] for seq, mask in zip(audio_codes, audio_codes_mask, strict=True)],
                batch_first=True,
                padding_value=0,
            )
            # Realign delayed codes before decoding
            if self.max_delay > 0:
                contiguous_audio_sequences = undelay_audio_codes(self.delays, contiguous_audio_sequences, padding_value=0)
            # Build padding_mask so Mimi decoder trims extra samples from causal ConvTranspose1d.
            audio_lengths = (audio_codes_mask.float().sum(dim=1) * self.sampling_rate / self.frame_rate).floor().long()
            max_audio_len = int(audio_lengths.max().item())
            padding_mask = torch.arange(max_audio_len, device=audio_lengths.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
            audio = self.decode_audio(
                audio_codes=contiguous_audio_sequences,
                padding_mask=padding_mask.long(),
                use_streaming=None,
            ).audio

            if continuation_silence_frames > 0:
                trim_samples = int(continuation_silence_frames * self.sampling_rate / self.frame_rate)
                audio = audio[:, trim_samples:]
                audio_lengths = (audio_lengths - trim_samples).clamp(min=0)

        self.free_past_key_values(past_key_values)
        return {
            "sequences": sequences,
            "audio_codes": audio_codes,
            "audio_codes_mask": audio_codes_mask,
            "audio": audio,
            "audio_lengths": audio_lengths,
        }


# ── from models/raon.py ──

TEXT_MODELS: dict[str, type[PreTrainedModel]] = {
    Qwen3Config.model_type: Qwen3Model,
}


@dataclass
class RaonModelOutput(ModelOutput):
    """Output container for the training forward pass."""

    loss: torch.Tensor | None = None
    text_loss: torch.Tensor | None = None
    audio_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    text_last_hidden_state: torch.Tensor | None = None
    talker_last_hidden_state: torch.Tensor | None = None
    text_logits: torch.Tensor | None = None
    audio_logits: torch.Tensor | None = None
    router_logits: tuple[torch.Tensor, ...] | None = None
    past_key_values: Cache | None = None


class RaonModel(RaonLossMixin, PreTrainedModel, RaonInferenceModel):
    """Core raon model combining text language model with audio codec."""

    _tied_weights_keys: list[str] = []  # type: ignore
    config_class: type[RaonConfig] = RaonConfig  # type: ignore
    config: RaonConfig

    def __init__(self, config: RaonConfig) -> None:
        super().__init__(config)
        assert config.text_model_config is not None, "Config text_model_config is required."
        assert config.audio_encoder_config is not None, "Config audio_encoder_config is required."
        assert config.audio_tokenizer_config is not None, "Config audio_tokenizer_config is required."
        assert config.input_adaptor_config is not None, "Config input_adaptor_config is required."
        assert config.output_adaptor_config is not None, "Config output_adaptor_config is required."
        assert config.code_predictor_config is not None, "Config code_predictor_config is required."
        assert config.text_model_config.vocab_size is not None, "text_model_config.vocab_size is required."
        assert config.audio_tokenizer_config.codebook_size is not None, "audio_tokenizer_config.codebook_size is required."
        assert config.code_predictor_config.num_code_groups is not None, "code_predictor_config.num_code_groups is required."
        assert config.audio_tokenizer_config.sampling_rate is not None, "audio_tokenizer_config.sampling_rate is required."
        assert config.text_model_config.hidden_size is not None, "text_model_config.hidden_size is required."
        assert config.code_predictor_config.hidden_size is not None, "code_predictor_config.hidden_size is required."
        if getattr(config, "supports_audio_output", True) and getattr(config, "audio_lm_head_enabled", True):
            assert config.talker_config is not None, "talker_config is required when audio output is enabled."
            assert config.num_talker_layers > 0, "num_talker_layers must be positive when audio output is enabled."

        self.config = config
        _dtype = getattr(config, "torch_dtype", None) or torch.float32
        if isinstance(_dtype, str):
            _dtype = getattr(torch, _dtype, torch.float32)
        self.hidden_size = int(config.text_model_config.hidden_size)
        self.vocab_size = int(config.text_model_config.vocab_size)
        self.codebook_size = config.audio_tokenizer_config.codebook_size
        self.audio_lm_head_vocab_size = self.codebook_size + 1
        self.num_talker_layers = config.num_talker_layers
        self.supports_audio_input = getattr(config, "supports_audio_input", True)
        self.supports_audio_output = getattr(config, "supports_audio_output", True)
        self.num_code_groups = config.code_predictor_config.num_code_groups
        self.sampling_rate = config.audio_tokenizer_config.sampling_rate
        assert (frame_rate := config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
            "audio_tokenizer_config._frame_rate is required."
        )
        self.frame_rate = frame_rate
        self.output_losses_only = False

        # Create thinker text_model: num_hidden_layers IS the thinker count (talker is separate).
        total_layers = int(config.text_model_config.num_hidden_layers)
        num_thinker_layers = total_layers
        thinker_text_config = deepcopy(config.text_model_config)
        thinker_text_config.num_hidden_layers = num_thinker_layers
        if hasattr(thinker_text_config, "layer_types") and thinker_text_config.layer_types:
            thinker_text_config.layer_types = thinker_text_config.layer_types[:num_thinker_layers]
        self.text_model = TEXT_MODELS[thinker_text_config.model_type]._from_config(
            thinker_text_config,
            dtype=_dtype,
        )
        if self.supports_audio_input:
            # Use model_type string instead of isinstance to avoid class identity issues
            # with trust_remote_code dynamic module loading.
            ae_model_type = getattr(config.audio_encoder_config, "model_type", "")
            if ae_model_type == "mimi":
                self.audio_encoder: CausalAudioEncoder | AuTWrapper | VoxtralWrapper | None = (
                    CausalAudioEncoder._from_config(
                        config.audio_encoder_config,
                        dtype=_dtype,
                    )
                )
            elif ae_model_type == "voxtral_realtime_encoder":
                self.audio_encoder: CausalAudioEncoder | AuTWrapper | VoxtralWrapper | None = VoxtralWrapper.from_config(
                    config=config.audio_encoder_config,
                    dtype=_dtype,
                )
            else:
                assert ae_model_type in ("qwen3_omni_moe_audio_encoder", ""), (
                    f"RAON checkpoints require audio_encoder model_type 'qwen3_omni_moe_audio_encoder' or "
                    f"'voxtral_realtime_encoder', got {ae_model_type!r}."
                )
                self.audio_encoder: CausalAudioEncoder | AuTWrapper | VoxtralWrapper | None = AuTWrapper.from_config(
                    config=config.audio_encoder_config,
                    dtype=_dtype,
                )
        else:
            self.audio_encoder = None

        self.aut_is_causal = getattr(config, "aut_is_causal", False)

        if self.supports_audio_output:
            self.audio_tokenizer: StreamingMimiModel | None = StreamingMimiModel._from_config(
                config.audio_tokenizer_config,
                dtype=_dtype,
            )
        else:
            self.audio_tokenizer = None
        if self.supports_audio_input:
            self.input_adaptor: EmbeddingAdaptor | None = EmbeddingAdaptor.from_config(
                config.input_adaptor_config, dtype=_dtype
            )
        else:
            self.input_adaptor = None
        if self.supports_audio_output:
            self.output_adaptor: EmbeddingAdaptor | None = EmbeddingAdaptor.from_config(
                config.output_adaptor_config, dtype=_dtype
            )
        else:
            self.output_adaptor = None
        # Create separate talker model and thinker-to-talker projection (audio output only).
        rms_norm_eps = getattr(config.text_model_config, "rms_norm_eps", 1e-6)
        if self.supports_audio_output and config.talker_config is not None:
            resolved_talker_config = config.talker_config
            self.talker: PreTrainedModel | None = TEXT_MODELS[resolved_talker_config.model_type]._from_config(
                resolved_talker_config,
                dtype=_dtype,
            )
            # Talker only receives inputs_embeds (from thinker_to_talker_proj), never input_ids.
            self.talker.embed_tokens = None  # type: ignore
            talker_hidden_size = int(resolved_talker_config.hidden_size)
            projection_mode = getattr(config, "thinker_to_talker_projection_mode", "linear")
            projection_intermediate_size = getattr(config, "thinker_to_talker_intermediate_size", None)
            if projection_mode == "mlp" and projection_intermediate_size is None:
                projection_intermediate_size = int(resolved_talker_config.intermediate_size)
            self.thinker_to_talker_proj: ThinkerToTalkerProjection | None = ThinkerToTalkerProjection(
                thinker_hidden_size=self.hidden_size,
                talker_hidden_size=talker_hidden_size,
                intermediate_size=projection_intermediate_size,
                mode=projection_mode,
                use_norm=getattr(config, "thinker_to_talker_pre_norm", False),
                rms_norm_eps=rms_norm_eps,
            )
        else:
            self.talker = None
            self.thinker_to_talker_proj = None
            talker_hidden_size = self.hidden_size

        # Resolve accept_hidden_layer: -1 means last thinker layer.
        accept_hidden_layer = getattr(config, "accept_hidden_layer", -1)
        if accept_hidden_layer < 0:
            accept_hidden_layer = num_thinker_layers + accept_hidden_layer
        self.accept_hidden_layer = accept_hidden_layer
        self.thinker_capture_layer_index = (
            accept_hidden_layer  # Index of the thinker layer whose output feeds the talker and audio head.
        )
        self.lm_head = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            bias=False,
            dtype=_dtype,
        )
        if self.supports_audio_output:
            self.audio_lm_head: nn.Linear | None = nn.Linear(
                in_features=talker_hidden_size,
                out_features=self.audio_lm_head_vocab_size,
                bias=False,
                dtype=_dtype,
            )
            if not getattr(config, "audio_lm_head_enabled", True):
                self.audio_lm_head = None
            self.proj_code: nn.Linear | None = nn.Linear(
                in_features=talker_hidden_size,
                out_features=config.code_predictor_config.hidden_size,
                bias=config.proj_code_bias,
                dtype=_dtype,
            )
            self.code_predictor: RaonCodePredictorModel | None = RaonCodePredictorModel._from_config(
                config.code_predictor_config,
                dtype=_dtype,
            )
        else:
            self.audio_lm_head = None
            self.proj_code = None
            self.code_predictor = None

        self.accepted_thinker_hidden_states: torch.Tensor | None = None
        self.register_thinker_capture_hook()

        self.code_predictor_grad_scale = _read_loss_param(
            env_key="RAON_CODE_PREDICTOR_GRAD_SCALE",
            config=config,
            attr_name="code_predictor_grad_scale",
            default=0.1,
        )
        self.text_loss_weight = _read_loss_param(
            env_key="RAON_TEXT_LOSS_WEIGHT",
            config=config,
            attr_name="text_loss_weight",
            default=1.0,
        )
        self.audio_output_pad_text_loss_weight = _read_loss_param(
            env_key="RAON_AUDIO_OUTPUT_PAD_TEXT_LOSS_WEIGHT",
            config=config,
            attr_name="audio_output_pad_text_loss_weight",
            default=0.0,
        )
        self.epad_loss_weight = _read_loss_param(
            env_key="RAON_EPAD_LOSS_WEIGHT",
            config=config,
            attr_name="epad_loss_weight",
            default=1.0,
        )
        self.audio_end_text_loss_weight = _read_loss_param(
            env_key="RAON_AUDIO_END_TEXT_LOSS_WEIGHT",
            config=config,
            attr_name="audio_end_text_loss_weight",
            default=0.0,
        )
        self.semantic_loss_weight = _read_loss_param(
            env_key="RAON_SEMANTIC_LOSS_WEIGHT",
            config=config,
            attr_name="semantic_loss_weight",
            default=1.0,
        )
        self.speaker_dropout = _read_loss_param(
            env_key="RAON_SPEAKER_DROPOUT",
            config=config,
            attr_name="speaker_dropout",
            default=0.2,
        )
        acoustic_loss_weights = _read_acoustic_loss_weights(config=config, num_code_groups=self.num_code_groups)
        self.audio_loss_weight = torch.tensor([self.semantic_loss_weight] + acoustic_loss_weights, dtype=_dtype)

        # Speaker encoder (optional, for speaker-conditioned TTS)
        if self.supports_audio_output and config.speaker_encoder_config is not None:
            self.speaker_encoder: PretrainedSpeakerEncoder | None = PretrainedSpeakerEncoder(
                config.speaker_encoder_config,
                dtype=_dtype,
            )
            self.is_pretrained_speaker_encoder = True
            self.speaker_token_id: int | None = SPEAKER_EMBEDDING_PLACEHOLDER.id
        else:
            self.speaker_encoder = None
            self.is_pretrained_speaker_encoder = False
            self.speaker_token_id = getattr(config, "speaker_token_id", SPEAKER_EMBEDDING_PLACEHOLDER.id)

        # Defensive cleanup: when audio output is disabled, ensure all output-side
        # audio modules are hard-disabled even if something assigned them earlier.
        if not self.supports_audio_output:
            self._disable_audio_output_modules()
        if not self.supports_audio_input:
            self._disable_audio_input_modules()

        if self.config.input_adaptor_config.output_time_scale != 1:
            raise NotImplementedError("Only `output_time_scale == 1` is supported.")

        RaonInferenceModel.__init__(self)

        # Duplex runtime attributes
        self.use_duplex_end_pad = getattr(config, "use_duplex_end_pad", False)
        self.use_sil_token = getattr(config, "use_sil_token", False)
        self.no_audio_in_sil = getattr(config, "no_audio_in_sil", False)
        self.sequence_mode = getattr(config, "sequence_mode", None)
        self.duplex_pad_token_id = getattr(config, "duplex_pad_token_id", AUDIO_OUTPUT_PAD.id)
        self.duplex_end_pad_token_id = getattr(config, "duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD.id)
        self.duplex_sil_token_id = getattr(config, "duplex_sil_token_id", -1)
        self.use_backchannel_token = getattr(config, "use_backchannel_token", False)
        self.duplex_bc_token_id = getattr(config, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id)
        self.bc_loss_weight = getattr(config, "bc_loss_weight", 1.0)
        self.audio_start_token_id = getattr(config, "audio_start_token_id", AUDIO_START.id)
        self.im_start_token_id = getattr(config, "im_start_token_id", IM_START.id)
        self.audio_input_token_id = getattr(config, "audio_input_token_id", AUDIO_INPUT_PLACEHOLDER.id)
        self.audio_output_token_id = getattr(config, "audio_output_token_id", AUDIO_OUTPUT_PLACEHOLDER.id)
        self.delays = getattr(config, "delays", None) or [0] * self.num_code_groups
        self.max_delay = max(self.delays)
        self.text_vocab_size = self.vocab_size - self.codebook_size
        # Loss weights (overridable at training time via duplex_train args)
        self.text_loss_weight = getattr(config, "text_loss_weight", 1.0)
        self.sil_loss_weight = getattr(config, "sil_loss_weight", 1.0)
        self.epad_loss_weight = getattr(config, "epad_loss_weight", 0.0)
        self.semantic_loss_weight = getattr(config, "semantic_loss_weight", 1.0)
        self.acoustic_loss_weights = getattr(config, "acoustic_loss_weights", None)

    def _disable_audio_output_modules(self) -> None:
        """Force-disable all audio-output and speaker-output modules."""
        self.audio_tokenizer = None
        self.output_adaptor = None
        self.audio_lm_head = None
        self.proj_code = None
        self.code_predictor = None
        self.speaker_encoder = None
        self.is_pretrained_speaker_encoder = False
        self.speaker_token_id = None

    def _disable_audio_input_modules(self) -> None:
        """Force-disable all audio-input modules."""
        self.audio_encoder = None
        self.input_adaptor = None

    def register_thinker_capture_hook(self) -> None:
        """Register a forward hook on the accept_hidden_layer to capture unnormalized hidden states."""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            self.accepted_thinker_hidden_states = output[0] if isinstance(output, tuple) else output

        cast(list[nn.Module], self.text_model.layers)[self.accept_hidden_layer].register_forward_hook(hook)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text model input embedding layer."""
        assert isinstance(self.text_model.embed_tokens, nn.Embedding), "text_model.embed_tokens must be nn.Embedding."
        return self.text_model.embed_tokens

    def _validate_audio_output_inputs(
        self,
        audio_output: torch.Tensor | None,
        audio_output_codes: torch.Tensor | None,
        audio_output_codes_mask: torch.Tensor | None,
    ) -> None:
        """Validate that audio-output inputs are only used when audio output is supported."""
        if self.supports_audio_output:
            return
        if audio_output is not None or audio_output_codes is not None or audio_output_codes_mask is not None:
            raise ValueError(
                "Audio output is disabled (`supports_audio_output=False`), but audio-output inputs were provided."
            )

    def _validate_audio_input_inputs(
        self,
        input_ids: torch.Tensor | None,
        audio_input: torch.Tensor | None,
        audio_input_embeds: torch.Tensor | None,
        audio_input_embeds_mask: torch.Tensor | None,
    ) -> None:
        """Validate that audio-input inputs are only used when audio input is supported."""
        if self.supports_audio_input:
            return
        _ = input_ids, audio_input
        if audio_input_embeds is not None or audio_input_embeds_mask is not None:
            raise ValueError("Audio input is disabled (`supports_audio_input=False`), but audio-input inputs were provided.")

    def get_model(self) -> Self:
        """Return self as the model instance."""
        return self

    def tie_weights(
        self,
        missing_keys: set[str] | None = None,
        recompute_mapping: bool = False,
    ) -> None: ...

    @property
    def all_tied_weights_keys(self) -> dict[str, str]:
        return {}

    @staticmethod
    def _normalize_audio_dims(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio tensor to 3D [batch_size, num_channels, num_samples].

        Accepts 1D [num_samples], 2D [batch_size, num_samples], or 3D
        [batch_size, num_channels, num_samples] inputs and returns the 3D form.
        """
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[:, None]
        assert audio.ndim == 3, "Audio tensor must have 3 dimensions [batch_size, num_channels, num_samples]."
        return audio

    def get_audio_input_embeds(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        encoder_past_key_values: Cache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioEncoderOutput:
        """Encode raw audio to input embeddings via audio encoder and input adaptor.

        Args:
            audio: Raw waveform. Shape: [batch_size, num_channels, num_samples] or
                [batch_size, num_samples] or [num_samples]. Dtype: float. None to return empty output.
            audio_lengths: Valid length per sample. Shape: [batch_size]. Dtype: long.
            sampling_rate: Sample rate of input audio; resampled if different from encoder.
            num_code_groups: Number of code groups (unused, for API compatibility).
            encoder_past_key_values: Cached encoder KV for streaming.
            conv_padding_cache: Cached conv padding for streaming encoder.
            use_streaming: Whether to use streaming mode.

        Returns:
            AudioEncoderOutput with audio_embeds (Shape: [batch_size, num_frames, feature_dim].
            Dtype: float.) and audio_embeds_mask (Shape: [batch_size, num_frames]. Dtype: bool.).
        """
        if audio is None:
            return AudioEncoderOutput()
        if self.audio_encoder is None:
            raise RuntimeError("audio_encoder is unavailable when supports_audio_input is False.")
        assert self.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."

        audio = cast_to_module_dtype(audio, self.audio_encoder)
        audio = RaonModel._normalize_audio_dims(audio)

        if sampling_rate is not None and sampling_rate != self.audio_encoder.config.sampling_rate:
            assert self.audio_encoder.config.sampling_rate is not None, (
                "audio_encoder.config.sampling_rate is required for resampling."
            )
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sampling_rate,
                new_freq=self.audio_encoder.config.sampling_rate,
            )

        encoder_cache: tuple[Cache, MimiConv1dPaddingCache] | None = None
        if isinstance(self.audio_encoder, CausalAudioEncoder):
            encoder_outputs = self.audio_encoder(
                audio,
                encoder_past_key_values=encoder_past_key_values,
                padding_cache=conv_padding_cache,
                use_streaming=use_streaming,
            )
            assert isinstance(encoder_outputs, CausalAudioEncoderOutput), "encoder_outputs must be CausalAudioEncoderOutput."
            if encoder_outputs.encoder_past_key_values is not None and encoder_outputs.padding_cache is not None:
                assert isinstance(encoder_outputs.encoder_past_key_values, Cache), "encoder_past_key_values must be Cache."
                assert isinstance(encoder_outputs.padding_cache, MimiConv1dPaddingCache), (
                    "padding_cache must be MimiConv1dPaddingCache."
                )
                encoder_cache = (
                    encoder_outputs.encoder_past_key_values,
                    encoder_outputs.padding_cache,
                )
        else:
            encoder_kwargs: dict[str, Any] = {}
            if isinstance(self.audio_encoder, (AuTWrapper, VoxtralWrapper)):
                encoder_kwargs["causal"] = self.aut_is_causal
                encoder_kwargs["audio_lengths"] = audio_lengths
            encoder_outputs = self.audio_encoder(audio, **encoder_kwargs)

        assert (audio_embeds := encoder_outputs.embeds) is not None, "Encoder outputs must contain embeds."

        if audio_lengths is not None:
            indices = torch.arange(audio.shape[-1], device=audio.device)
            audio_embeds_mask = (indices[None] < audio_lengths[:, None]).long()

            assert (encoder_sampling_rate := self.config.audio_tokenizer_config.sampling_rate) is not None, (
                "audio_tokenizer_config.sampling_rate is required."
            )
            assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
                "audio_tokenizer_config._frame_rate is required."
            )
            assert (samples_per_frame := int(encoder_sampling_rate / frame_rate)) == encoder_sampling_rate / frame_rate, (
                "samples_per_frame must divide evenly."
            )
            padded_audio_mask = F.pad(
                audio_embeds_mask,
                (0, audio_embeds.shape[1] * samples_per_frame - audio_embeds_mask.shape[1]),
            )
            audio_embeds_mask = padded_audio_mask.view(-1, audio_embeds.shape[1], samples_per_frame).any(dim=-1)
        else:
            audio_embeds_mask = torch.ones(
                audio_embeds.shape[:2],
                dtype=torch.bool,
                device=audio_embeds.device,
            )

        adaptor_outputs = self.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput), "adaptor_outputs must be EmbeddingAdaptorOutput."
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None, "adaptor outputs_embeds is required."
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None, "adaptor mask is required."  # type: ignore

        return AudioEncoderOutput(
            audio_embeds=audio_embeds,
            audio_embeds_mask=audio_embeds_mask,
            encoder_cache=encoder_cache,
        )

    def tokenize_audio(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
        encoder_past_key_values: Cache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioTokenizerOutput:
        if audio is None:
            return AudioTokenizerOutput()
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is unavailable when supports_audio_output is False.")
        target_sampling_rate = (
            self.audio_encoder.config.sampling_rate
            if self.audio_encoder is not None
            else self.audio_tokenizer.config.sampling_rate
        )
        # Cast to audio_tokenizer (Mimi) dtype — the encoder dtype may differ (e.g. VoxtralWrapper).
        audio = cast_to_module_dtype(audio, self.audio_tokenizer)
        audio = RaonModel._normalize_audio_dims(audio)

        if sampling_rate is not None and sampling_rate != target_sampling_rate:
            assert target_sampling_rate is not None, "sampling_rate is required for resampling."
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sampling_rate,
                new_freq=target_sampling_rate,
            )

        audio_mask = None
        if audio_lengths is not None:
            indices = torch.arange(audio.shape[-1], device=audio.device)
            audio_mask = (indices[None] < audio_lengths[:, None]).long()

        outputs = self.audio_tokenizer.encode(
            audio,
            padding_mask=audio_mask,
            num_quantizers=num_code_groups,
            encoder_past_key_values=encoder_past_key_values,
            padding_cache=conv_padding_cache,
            use_streaming=use_streaming,
            return_dict=True,
        )
        assert isinstance(outputs, MimiEncoderOutput), "tokenizer encode output must be MimiEncoderOutput."

        encoder_cache: tuple[Cache, MimiConv1dPaddingCache] | None = None
        if outputs.encoder_past_key_values is not None and outputs.padding_cache is not None:
            assert isinstance(outputs.encoder_past_key_values, Cache), "encoder_past_key_values must be Cache."
            assert isinstance(outputs.padding_cache, MimiConv1dPaddingCache), "padding_cache must be MimiConv1dPaddingCache."
            encoder_cache = (outputs.encoder_past_key_values, outputs.padding_cache)

        assert outputs.audio_codes is not None, "tokenizer encode output must contain audio_codes."
        audio_codes = outputs.audio_codes.view(outputs.audio_codes.shape[-3:]).transpose(1, 2)
        if audio_mask is not None:
            assert (encoder_sampling_rate := self.config.audio_tokenizer_config.sampling_rate) is not None, (
                "audio_tokenizer_config.sampling_rate is required."
            )
            assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
                "audio_tokenizer_config._frame_rate is required."
            )
            assert (samples_per_frame := int(encoder_sampling_rate / frame_rate)) == encoder_sampling_rate / frame_rate, (
                "samples_per_frame must divide evenly."
            )
            padded_audio_mask = F.pad(
                audio_mask,
                (0, audio_codes.shape[1] * samples_per_frame - audio_mask.shape[1]),
            )
            audio_codes_mask = padded_audio_mask.view(-1, audio_codes.shape[1], samples_per_frame).any(dim=-1)
        else:
            audio_codes_mask = torch.ones(
                audio_codes.shape[:2],
                dtype=torch.bool,
                device=audio_codes.device,
            )

        # Optionally return mimi_features for speaker embedding
        mimi_features = None
        if return_mimi_features:
            # Get latent features by decoding the audio codes (quantizer decode)
            # Shape: [batch_size, 512, num_frames] -> transpose to [batch_size, num_frames, 512]
            mimi_features = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)

        return AudioTokenizerOutput(
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            mimi_features=mimi_features,
            encoder_cache=encoder_cache,
        )

    def tokenize_audio_segments(
        self,
        audio: torch.Tensor,
        segments: list[tuple[int, int]],
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
    ) -> AudioTokenizerOutput:
        """Tokenize speech segments independently via batched Mimi encoding.

        Each segment is encoded independently (no cross-segment conv context).
        Used in no_audio_in_sil mode where SIL frames have no [A] token.

        Args:
            audio: Raw waveform. Shape: [1, num_samples] or [1, 1, num_samples]. Dtype: float.
            segments: List of (start_sample, end_sample) tuples for utterance regions.
            num_code_groups: Number of codec groups for Mimi quantizer.
            return_mimi_features: Whether to return pre-quantization features.

        Returns:
            AudioTokenizerOutput with codes for speech frames only (concatenated).
        """
        if not segments:
            return AudioTokenizerOutput()

        if audio.ndim == 3:
            audio = audio.squeeze(1)

        assert self.config.audio_tokenizer_config.sampling_rate is not None
        assert self.config.audio_tokenizer_config._frame_rate is not None  # type: ignore
        sr = self.config.audio_tokenizer_config.sampling_rate
        fr = self.config.audio_tokenizer_config._frame_rate  # type: ignore

        seg_audios = [audio[0, s:e] for s, e in segments]
        seg_lengths = torch.tensor([s.shape[0] for s in seg_audios], device=audio.device)

        max_len = int(seg_lengths.max().item())
        batched = torch.stack([F.pad(s, (0, max_len - s.shape[0])) for s in seg_audios]).to(audio.dtype)

        batched_out = self.tokenize_audio(
            audio=batched,
            audio_lengths=seg_lengths,
            num_code_groups=num_code_groups,
            return_mimi_features=return_mimi_features,
        )
        if batched_out.audio_codes is None:
            return AudioTokenizerOutput()

        all_codes = []
        all_masks = []
        all_features = []
        for i, length in enumerate(seg_lengths):
            n_frames = math.ceil(length.item() * fr / sr)
            n_frames = min(n_frames, batched_out.audio_codes.shape[1])
            all_codes.append(batched_out.audio_codes[i : i + 1, :n_frames])
            if batched_out.audio_codes_mask is not None:
                all_masks.append(batched_out.audio_codes_mask[i : i + 1, :n_frames])
            if batched_out.mimi_features is not None:
                all_features.append(batched_out.mimi_features[i : i + 1, :n_frames])

        audio_codes = torch.cat(all_codes, dim=1)
        audio_codes_mask = torch.cat(all_masks, dim=1) if all_masks else None
        mimi_features = torch.cat(all_features, dim=1) if all_features else None

        if audio_codes_mask is None:
            audio_codes_mask = torch.ones(audio_codes.shape[:2], dtype=torch.bool, device=audio_codes.device)

        return AudioTokenizerOutput(
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            mimi_features=mimi_features,
        )

    def _get_audio_output_embeds(
        self,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.audio_tokenizer is not None, "audio_tokenizer is unavailable when supports_audio_output is False."
        assert self.output_adaptor is not None, "output_adaptor is unavailable when supports_audio_output is False."
        assert audio_codes.ndim == 3 and audio_codes_mask.ndim == 2, (
            "audio_codes must have 3 dims and audio_codes_mask must have 2 dims."
        )
        latent_features = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        adaptor_outputs = self.output_adaptor(latent_features, mask=audio_codes_mask)
        return adaptor_outputs.outputs_embeds, adaptor_outputs.mask

    def _insert_audio_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        audio_embeds_mask: torch.Tensor | None,
        audio_token_id: int,
    ) -> torch.Tensor:
        audio_mask = (input_ids == audio_token_id)[..., None].expand_as(inputs_embeds)
        audio_embeds = cast_float_inputs(audio_embeds, inputs_embeds.dtype)
        if audio_embeds_mask is not None:
            audio_embeds = audio_embeds[audio_embeds_mask]
        else:
            audio_embeds = audio_embeds.view(-1, audio_embeds.shape[-1])

        assert audio_mask.sum() == audio_embeds.numel(), (
            f"Number of masked positions must match audio_embeds element count. "
            f"audio_mask.sum()={audio_mask.sum()}, audio_embeds.numel()={audio_embeds.numel()}, "
            f"audio_token_id={audio_token_id}, input_ids_shape={input_ids.shape}."
        )
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        return inputs_embeds

    def update_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Insert audio input, audio output, and speaker embeddings into inputs_embeds at placeholder positions.

        Args:
            inputs_embeds: Base embeddings from text tokens. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            input_ids: Token IDs for locating placeholders. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input_embeds: Encoded audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_output_codes: Discrete audio output codes. Shape: [batch_size, num_frames, num_code_groups].
                Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker conditioning embeddings. Shape: [batch_size, 1, hidden_size]. Dtype: float.

        Returns:
            Updated inputs_embeds with all placeholders filled.
            Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
        """
        if audio_output_codes is not None:
            assert input_ids is not None, "`input_ids` required when `audio_output_codes` is provided."
            assert audio_output_codes_mask is not None, (
                "`audio_output_codes_mask` required when `audio_output_codes` is provided."
            )

        if audio_input_embeds is not None:
            assert input_ids is not None, "`input_ids` required when `audio_input_embeds` is provided."
            assert audio_input_embeds_mask is not None, (
                "`audio_input_embeds_mask` required when `audio_input_embeds` is provided."
            )

        if (
            input_ids is not None
            and audio_input_embeds is not None
            and audio_input_embeds_mask is not None
            and audio_input_embeds_mask.any()
        ):
            inputs_embeds = self._insert_audio_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_embeds=audio_input_embeds,
                audio_embeds_mask=audio_input_embeds_mask,
                audio_token_id=AUDIO_INPUT_PLACEHOLDER.id,
            )

        if (
            input_ids is not None
            and audio_output_codes is not None
            and audio_output_codes_mask is not None
            and audio_output_codes_mask.any()
        ):
            audio_output_embeds, audio_output_embeds_mask = self._get_audio_output_embeds(
                audio_codes=audio_output_codes,
                audio_codes_mask=audio_output_codes_mask,
            )
            inputs_embeds = self._insert_audio_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_embeds=audio_output_embeds,
                audio_embeds_mask=audio_output_embeds_mask,
                audio_token_id=AUDIO_OUTPUT_PLACEHOLDER.id,
            )

        # Insert speaker embedding at <|speaker_embedding_placeholder|> position
        if input_ids is not None and speaker_embeds is not None and self.speaker_token_id is not None:
            speaker_mask = input_ids == self.speaker_token_id
            if speaker_mask.any():
                inputs_embeds = self._insert_audio_embeds(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    audio_embeds=speaker_embeds,
                    audio_embeds_mask=None,  # Single token per sample, no mask needed
                    audio_token_id=self.speaker_token_id,
                )

        return inputs_embeds

    def shift_labels(self, labels: torch.Tensor, pad_length: int = 1) -> torch.Tensor:
        """Shift labels left by one position for causal LM and optionally pad at the end.

        Args:
            labels: Ground-truth token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            pad_length: Number of LOSS_IGNORE_INDEX values to pad at the end (default 1 for right-aligned targets).

        Returns:
            Shifted labels. Shape: [batch_size, seq_length - 1 + pad_length] when pad_length > 0,
            else [batch_size, seq_length - 1]. Dtype: long.
        """
        if pad_length == 0:
            return labels[:, 1:]
        elif pad_length > 0:
            return F.pad(labels[:, 1:], (0, pad_length), value=LOSS_IGNORE_INDEX)
        else:
            raise ValueError("`pad_length` must be nonnegative.")

    def get_text_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels by replacing AUDIO_OUTPUT_PLACEHOLDER with AUDIO_OUTPUT_PAD for text loss.

        Args:
            labels: Ground-truth token IDs including audio placeholders. Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Labels suitable for text loss (placeholders replaced with pad). Shape: [batch_size, seq_length]. Dtype: long.
        """
        text_labels = labels.clone()
        text_labels[text_labels == AUDIO_OUTPUT_PLACEHOLDER.id] = AUDIO_OUTPUT_PAD.id
        return text_labels

    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."
        return self.proj_code

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        speaker_encoder_audio: torch.Tensor | None = None,
        speaker_encoder_audio_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: StaticCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        use_speaker_embedding: bool = False,
        speaker_embeds: torch.Tensor | None = None,
        audio_output_segments: list[tuple[int, int]] | None = None,
        debug_mode: bool = False,
        debug_step: int | None = None,
        **kwargs: Any,
    ) -> RaonModelOutput:
        """Run training forward pass: embed inputs, run text model, compute text and audio loss.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Valid position mask. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw input audio. Shape: [batch_size, num_channels, num_samples] or [batch_size, num_samples].
                Dtype: float.
            audio_output: Raw output audio for tokenization. Same shapes as audio_input.
            audio_input_lengths: Valid sample lengths for audio_input. Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample lengths for audio_output. Shape: [batch_size]. Dtype: long.
            speaker_encoder_audio: Unchunked audio for pretrained speaker encoder.
                Shape: [num_speakers, num_samples]. Dtype: float.
            speaker_encoder_audio_lengths: Valid sample lengths for speaker_encoder_audio.
                Shape: [num_speakers]. Dtype: long.
            audio_output_codes: Pre-tokenized audio codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            labels: Ground-truth token IDs for loss. Shape: [batch_size, seq_length]. Dtype: long.
            position_ids: Position indices. Shape: [batch_size, seq_length]. Dtype: long.
            past_key_values: KV cache for generation.
            inputs_embeds: Pre-computed input embeddings. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            use_cache: Whether to return KV cache.
            cache_position: Position indices for static cache.
            use_speaker_embedding: Whether to compute speaker embeddings from `speaker_encoder_audio`.
            speaker_embeds: Pre-computed speaker embeddings. Shape: [num_speakers, 1, feature_dim]. Dtype: float.
            debug_mode: Enable debug forward pass.
            debug_step: Step index for debugging.
            **kwargs: Passed to text model.

        Returns:
            RaonModelOutput with loss, text_loss, audio_loss, aux_loss, hidden states, logits, and past_key_values.
        """
        self._validate_audio_output_inputs(
            audio_output=audio_output,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
        )
        self._validate_audio_input_inputs(
            input_ids=input_ids,
            audio_input=audio_input,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
        )
        speaker_encoder = self.speaker_encoder
        need_speaker_embedding = (
            audio_output is not None and use_speaker_embedding and speaker_embeds is None and speaker_encoder is not None
        )

        if self.supports_audio_output and audio_output_codes is None:
            with torch.no_grad():
                if audio_output_segments is not None and audio_output is not None:
                    # Utterance-segmented encoding: each speech segment is encoded
                    # independently so codes align with [A] positions (no SIL frames).
                    audio_output_inputs = self.tokenize_audio_segments(
                        audio=audio_output,
                        segments=audio_output_segments,
                        num_code_groups=self.num_code_groups,
                        return_mimi_features=False,
                    )
                else:
                    audio_output_inputs = self.tokenize_audio(
                        audio=audio_output,
                        audio_lengths=audio_output_lengths,
                        num_code_groups=self.num_code_groups,
                        return_mimi_features=False,
                    )
            audio_output_codes = audio_output_inputs.audio_codes
            audio_output_codes_mask = audio_output_inputs.audio_codes_mask

        if need_speaker_embedding:
            assert self.is_pretrained_speaker_encoder, (
                "Speaker embedding requires pretrained speaker encoder path "
                "with speaker_encoder_audio inputs."
            )
            if speaker_encoder_audio is not None:
                assert speaker_encoder is not None, "speaker_encoder is required when use_speaker_embedding is enabled."
                assert speaker_encoder_audio_lengths is not None, (
                    "speaker_encoder_audio_lengths is required when speaker_encoder_audio is provided."
                )
                assert speaker_encoder_audio.shape[0] == speaker_encoder_audio_lengths.shape[0], (
                    "speaker_encoder_audio and speaker_encoder_audio_lengths must have matching batch size. "
                    f"Got `{speaker_encoder_audio.shape[0]=}` and `{speaker_encoder_audio_lengths.shape[0]=}`."
                )
                speaker_embeds = speaker_encoder(speaker_encoder_audio, speaker_encoder_audio_lengths)

        # Speaker embedding dropout reduces reliance on the speaker-conditioning path.
        if self.training and speaker_embeds is not None and self.speaker_dropout > 0:
            drop_mask = torch.rand(speaker_embeds.shape[0], device=speaker_embeds.device) < self.speaker_dropout
            if drop_mask.all():
                speaker_embeds = None
            elif drop_mask.any():
                speaker_embeds = speaker_embeds.clone()
                speaker_embeds[drop_mask] = 0.0

        if self.supports_audio_input and audio_input_embeds is None and audio_input is not None:
            audio_input_outputs = self.get_audio_input_embeds(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
            )
            audio_input_embeds = audio_input_outputs.audio_embeds
            audio_input_embeds_mask = audio_input_outputs.audio_embeds_mask

        if inputs_embeds is None:
            assert input_ids is not None, "input_ids is required when inputs_embeds is None."
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
            assert inputs_embeds is not None, "get_input_embeddings must return non-None."
            inputs_embeds = self.update_inputs_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_output_codes=audio_output_codes,
                audio_output_codes_mask=audio_output_codes_mask,
                audio_input_embeds=audio_input_embeds,
                audio_input_embeds_mask=audio_input_embeds_mask,
                speaker_embeds=speaker_embeds,
            )
        else:
            assert input_ids is None and audio_output_codes is None and audio_input_embeds is None, (
                "When inputs_embeds is provided, input_ids, audio_output_codes, and audio_input_embeds must be None."
            )

        inputs_embeds = cast_to_module_dtype(inputs_embeds, self.text_model)
        text_outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        assert self.accepted_thinker_hidden_states is not None, (
            "accepted_thinker_hidden_states must be set by thinker capture hook."
        )
        assert text_outputs.last_hidden_state is not None, "text model must return last_hidden_state."

        accepted_hidden = self.accepted_thinker_hidden_states
        self.accepted_thinker_hidden_states = None

        # Text logits from thinker output (post-norm from text_model.norm, like standard LLM).
        text_logits = self.lm_head(text_outputs.last_hidden_state)

        # Talker forward: project thinker hidden → talker → audio hidden states.
        if self.talker is not None and self.thinker_to_talker_proj is not None:
            talker_input = self.thinker_to_talker_proj(accepted_hidden)
            talker_cache = getattr(self, "_talker_past_key_values", None)
            talker_outputs = self.talker(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=talker_input,
                past_key_values=talker_cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            if use_cache:
                self._talker_past_key_values = talker_outputs.past_key_values
            talker_last_hidden_state = talker_outputs.last_hidden_state
        else:
            # STT-only: no separate talker, use thinker output directly.
            talker_last_hidden_state = text_outputs.last_hidden_state

        loss = None
        text_loss = None
        audio_loss = None
        audio_logits = None
        if labels is not None:
            text_labels = self.get_text_labels(labels)
            text_loss = self.unreduced_causal_lm_loss(logits=text_logits, labels=text_labels)
            combined_loss, audio_loss, audio_logits = self.ddp_safe_loss(
                text_loss=text_loss,
                text_labels=text_labels,
                input_ids=input_ids,
                labels=labels,
                hidden_embeds=talker_last_hidden_state,
                audio_output_codes=audio_output_codes,
                audio_output_codes_mask=audio_output_codes_mask,
                speaker_embeds=speaker_embeds,
            )
            loss = combined_loss.mean()

        router_logits = getattr(text_outputs, "router_logits", None)
        aux_loss: torch.Tensor | None = None

        if self.output_losses_only:
            return RaonModelOutput(loss=loss, text_loss=text_loss, audio_loss=audio_loss)
        else:
            return RaonModelOutput(
                loss=loss,
                text_loss=text_loss,
                audio_loss=audio_loss,
                aux_loss=aux_loss,
                text_last_hidden_state=text_outputs.last_hidden_state,
                talker_last_hidden_state=talker_last_hidden_state,
                text_logits=text_logits,
                audio_logits=audio_logits,
                router_logits=router_logits,
                past_key_values=text_outputs.past_key_values,
            )

    def inference_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
        use_cache: bool | None = False,
        past_key_values: DynamicCache | StaticCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference forward pass and return talker hidden state and text logits for decoding.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Valid position mask. Shape: [batch_size, seq_length]. Dtype: long.
            position_ids: Position indices. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw input audio. Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_output: Raw output audio. Same shape as audio_input.
            audio_input_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.
            audio_output_codes: Pre-tokenized output codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker conditioning. Shape: [batch_size, num_frames, feature_dim]. Dtype: float.
            use_cache: Whether to use KV cache.
            past_key_values: KV cache for incremental decoding.
            cache_position: Explicit cache position indices. Shape: [seq_length]. Dtype: long.

        Returns:
            Tuple of (talker_last_hidden_state, text_logits). talker_last_hidden_state Shape: [batch_size, seq_length,
            hidden_size]. Dtype: float. text_logits Shape: [batch_size, seq_length, vocab_size]. Dtype: float.
        """
        outputs: RaonModelOutput = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_input=audio_input,
            audio_output=audio_output,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        assert isinstance(talker_last_hidden_state := outputs.talker_last_hidden_state, torch.Tensor), (
            "forward must return talker_last_hidden_state as a tensor."
        )
        assert isinstance(text_logits := outputs.text_logits, torch.Tensor), "forward must return text_logits as a tensor."
        return talker_last_hidden_state, text_logits

    def generate_audio_codes(
        self,
        talker_last_hidden_state: torch.Tensor,
        first_code_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        allow_audio_end: bool = True,
    ) -> torch.Tensor:
        """Generate autoregressive audio codes from the last talker hidden state.

        Args:
            talker_last_hidden_state: Talker hidden states. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            first_code_sampler: Optional callable to sample first code from logits; else argmax.
            allow_audio_end: If False, suppress AUDIO_END sampling for duplex-style decoding.

        Returns:
            Generated audio codes. Shape: [batch_size, num_generated_frames, num_code_groups]. Dtype: long.
        """
        assert self.audio_lm_head is not None, "audio_lm_head is unavailable when supports_audio_output is False."
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."
        assert self.code_predictor is not None, "code_predictor is unavailable when supports_audio_output is False."

        first_code_logits = self.audio_lm_head(talker_last_hidden_state[:, -1])

        if not allow_audio_end and first_code_logits.shape[-1] > self.codebook_size:
            # Suppress AUDIO_END token (only when audio_lm_head produces codebook_size+1 logits)
            first_code_logits = first_code_logits.clone()
            first_code_logits[..., self.codebook_size] = torch.finfo(first_code_logits.dtype).min

        if first_code_sampler is not None:
            first_code = first_code_sampler(first_code_logits)
        else:
            first_code = first_code_logits.argmax(dim=-1, keepdim=True)

        audio_end_mask = first_code[:, 0] == self.codebook_size
        safe_first_code = first_code.clamp_max(self.codebook_size - 1)

        hidden_embeds = self.proj_code(talker_last_hidden_state[:, -1:])
        inputs_embeds = torch.cat(
            (hidden_embeds, self.code_predictor.get_input_embeddings()(safe_first_code)),
            dim=1,
        )
        sequences = self.code_predictor.predict_codes(inputs_embeds=inputs_embeds)
        sequences = torch.cat((first_code, sequences.to(first_code.device)), dim=1)
        if audio_end_mask.any():
            sequences[audio_end_mask, 1:] = 0
        return sequences

    def decode_audio(
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        use_streaming: bool | None = None,
    ) -> AudioDecoderOutput:
        """Decode discrete audio codes to waveform via the audio tokenizer decoder.

        Args:
            audio_codes: Discrete audio codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            padding_mask: Mask indicating valid audio sample positions for trimming decoder output.
                Shape: [batch_size, num_samples]. Dtype: long.
            use_streaming: Whether to use the streaming Mimi decoder. ``None`` falls back to tokenizer config.

        Returns:
            AudioDecoderOutput with audio waveform (Shape: [batch_size, num_samples]. Dtype: float.).
        """
        assert self.audio_tokenizer is not None, "audio_tokenizer is unavailable when supports_audio_output is False."
        outputs = self.audio_tokenizer.decode(
            audio_codes.transpose(1, 2),
            padding_mask=padding_mask,
            use_streaming=use_streaming,
            return_dict=True,
        )
        assert isinstance(outputs, StreamingMimiDecoderOutput), "tokenizer decode output must be StreamingMimiDecoderOutput."
        assert (audio_values := outputs.audio_values) is not None, "decode output must contain audio_values."
        audio = audio_values.view(audio_values.shape[0], audio_values.shape[2])
        return AudioDecoderOutput(audio=audio)

    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: Cache | None = None,
    ) -> Cache:
        """Initialize or reset KV cache for text model incremental decoding.

        When the model has a separate talker, also initializes a
        DynamicCache for the talker model stored as ``_talker_past_key_values``.

        Args:
            batch_size: Batch size for the cache.
            max_sequence_length: Maximum sequence length to cache.
            prev_cache: Existing cache to reset and reuse; if None, creates a new StaticCache.

        Returns:
            Initialized StaticCache ready for incremental decoding.
        """
        # Initialize talker KV cache for the separate HF talker model.
        if self.talker is not None:
            self._talker_past_key_values: DynamicCache | None = DynamicCache()

        if prev_cache is not None:
            prev_cache.reset()
            return prev_cache

        return StaticCache(
            self.config.text_model_config,
            max_cache_len=max_sequence_length,
        )

    def free_past_key_values(self, past_key_values: Cache) -> None:
        """Release KV cache resources including talker KV cache."""
        # Clean up talker KV cache.
        if hasattr(self, "_talker_past_key_values"):
            self._talker_past_key_values = None

    def _set_attention_implementation(self, attn_implementation: Literal["sdpa", "flash_attention_2"]) -> None:
        """Set attention implementation for text model and code predictor."""
        self.text_model.config._attn_implementation = attn_implementation  # type: ignore
        if self.code_predictor is not None:
            self.code_predictor.config._attn_implementation = attn_implementation  # type: ignore


# Duplex config — supports checkpoints with model_type="raon_duplex"


class RaonDuplexModel(RaonModel):
    """Model alias for full-duplex checkpoints (model_type='raon_duplex')."""

    config_class = RaonDuplexConfig


# Register model types with HuggingFace AutoConfig/AutoModel

# Method 2: register so `import raon` makes AutoModel.from_pretrained() work


# ── from utils/audio_io.py ──

def load_audio(
    path: str | Path,
    target_sr: int,
    *,
    mono: bool = True,
    channel: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and resample to *target_sr*.

    Returns:
        Tuple of (waveform, sample_rate). waveform shape: [channels, samples].
    """
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        audio = torch.from_numpy(data)[None]  # [1, samples]
    else:
        audio = torch.from_numpy(data.T)  # [channels, samples]

    if channel is not None and audio.shape[0] > 1:
        audio = audio[channel : channel + 1]
    elif mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if sr != target_sr:
        import torchaudio.functional

        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)

    if dtype is not None:
        audio = audio.to(dtype=dtype)
    if device is not None:
        audio = audio.to(device=device)

    return audio, target_sr


def save_audio(
    audio: torch.Tensor | np.ndarray,
    sampling_rate: int,
    path: str | Path,
    length: int | None = None,
) -> None:
    """Save audio to a WAV file.

    Accepts a torch tensor or numpy array. Tensors are converted to float32
    numpy before writing. Multi-dimensional tensors are squeezed to 1-D.

    Args:
        audio: Waveform data — ``torch.Tensor`` or ``np.ndarray``.
        sampling_rate: Sampling rate in Hz.
        path: Destination file path.
        length: If provided, truncate to this many samples before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        if audio.ndim == 2:
            audio = audio[0]
        audio_np = audio.float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np[0]

    if length is not None:
        audio_np = audio_np[:length]

    sf.write(str(path), audio_np, sampling_rate)

_load_audio_shared = load_audio  # alias used by pipeline

_save = save_audio  # alias used by RaonPipeline.save_audio


# ── from utils/mel_features.py ──

def compute_log_mel_spectrogram(
    audio: torch.Tensor,
    window: torch.Tensor,
    mel_filters: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Compute log-mel spectrogram from audio waveform.

    Mirrors ``WhisperFeatureExtractor._torch_extract_fbank_features``:
    runs a centered STFT, projects onto mel filterbanks, log-compresses,
    and applies global max normalization so that the dynamic range is
    clipped to 8 dB below the per-batch maximum.

    Args:
        audio: Waveform tensor. Shape: [batch, samples] or [samples].
        window: STFT window tensor of size ``n_fft``.
        mel_filters: Mel filterbank matrix. Shape: [n_mels, n_fft//2+1].
        n_fft: FFT size.
        hop_length: STFT hop length.

    Returns:
        Log-mel spectrogram. Shape: [batch, n_mels, time].
    """
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters.T @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    log_spec = torch.maximum(log_spec, max_val - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


# ── from utils/processor.py ──

DEFAULT_STT_PROMPT = "Transcribe the audio into text"


DEFAULT_TTS_PROMPT = "Speak the following text"


def get_default_stt_prompt() -> str:
    """Return the default STT prompt."""
    return DEFAULT_STT_PROMPT


def get_default_tts_prompt() -> str:
    """Return the default TTS prompt."""
    return DEFAULT_TTS_PROMPT


class TextContent(TypedDict):
    """Content item carrying plain text within a multimodal message."""

    type: Literal["text"]
    text: str


class AudioContent(TypedDict):
    """Content item with an audio file path."""

    type: Literal["audio"]
    audio: str


class SpeakerContentRequired(TypedDict):
    """Content item for speaker-conditioned TTS; requires type only."""

    type: Literal["speaker"]


class SpeakerContent(SpeakerContentRequired, total=False):
    """Content item for speaker-conditioned TTS; optionally includes audio path for embedding."""

    audio: str


ContentItem = TextContent | AudioContent | SpeakerContent


class MultiModalMessage(TypedDict):
    """Chat message with role and list of multimodal content items (text, audio, speaker)."""

    role: str
    content: list[ContentItem]


class TextMessage(TypedDict):
    """Chat message with role and plain text content."""

    role: str
    content: str


Message = TextMessage | MultiModalMessage


class RaonProcessor:
    """Tokenizer, audio loader, and collator for raon model training and inference.

    Handles chat message parsing, audio loading/resampling, tokenization with
    assistant masks, and batched collation with optional device/dtype placement.
    """

    def __init__(
        self,
        model_name_or_path: str | PathLike | None = None,
        tokenizer_path: str | PathLike | None = None,
        config_path: str | PathLike | None = None,
        tokenizer: Qwen2TokenizerFast | None = None,
        config: RaonConfig | None = None,
        max_audio_seq_length: int = 192000,
    ) -> None:
        if tokenizer is not None and config is not None:
            self.tokenizer = tokenizer
            self.config = config
            update_tokenizer(self.tokenizer)
        else:
            if tokenizer_path is None:
                tokenizer_path = model_name_or_path

            if config_path is None:
                config_path = model_name_or_path

            assert tokenizer_path is not None and config_path is not None, (
                "`model_name_or_path` or `tokenizer_path` and `config_path` must be provided."
            )

            self.tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)
            update_tokenizer(self.tokenizer)
            # Use AutoConfig to support both model_type aliases ("raon", "raon_duplex").
            from transformers import AutoConfig

            self.config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        assert isinstance(self.tokenizer.pad_token, str), "Tokenizer pad_token must be a string."
        self.pad_token = self.tokenizer.pad_token
        (self.pad_token_id,) = self.tokenizer.encode(self.pad_token)

        assert self.config.audio_tokenizer_config.sampling_rate is not None, (
            "Config audio_tokenizer_config.sampling_rate must be set."
        )
        self.sampling_rate: int = self.config.audio_tokenizer_config.sampling_rate
        assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
            "Config audio_tokenizer_config frame_rate must be set."
        )
        self.frame_rate: float = frame_rate
        self.samples_per_frame = self.sampling_rate / self.frame_rate

        assert isinstance(eos_token_id := self.tokenizer.eos_token_id, int), "Tokenizer eos_token_id must be an integer."
        self.eos_token_id = eos_token_id

        # EPAD config
        self.use_duplex_end_pad: bool = getattr(self.config, "use_duplex_end_pad", False)
        self.has_speaker_encoder: bool = getattr(self.config, "speaker_encoder_config", None) is not None
        self.max_audio_seq_length = int(max_audio_seq_length)
        assert self.max_audio_seq_length > 0, "max_audio_seq_length must be positive."

        # duplex-specific attributes from config
        self.use_sil_token: bool = getattr(self.config, "use_sil_token", False)
        self.no_audio_in_sil: bool = getattr(self.config, "no_audio_in_sil", False)
        self.use_backchannel_token: bool = getattr(self.config, "use_backchannel_token", False)
        self.sequence_mode: Literal["tua", "uta"] | None = getattr(self.config, "sequence_mode", None)
        if self.sequence_mode not in (None, "tua", "uta"):
            raise ValueError(f"Unsupported sequence_mode '{self.sequence_mode}'.")
        self.duplex_pad_token_id: int = getattr(self.config, "duplex_pad_token_id", AUDIO_OUTPUT_PAD.id)
        self.duplex_end_pad_token_id: int = getattr(self.config, "duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD.id)
        self.duplex_sil_token_id: int = getattr(self.config, "duplex_sil_token_id", DUPLEX_SIL.id)
        self.duplex_bc_token_id: int = int(getattr(self.config, "duplex_bc_token_id", None) or AUDIO_OUTPUT_BC.id)
        self.speaker_token_id: int | None = getattr(self.config, "speaker_token_id", None)
        self.audio_start_token_id: int = getattr(self.config, "audio_start_token_id", AUDIO_START.id)
        self.audio_input_token_id: int = getattr(self.config, "audio_input_token_id", AUDIO_INPUT_PLACEHOLDER.id)
        self.audio_output_token_id: int = getattr(self.config, "audio_output_token_id", AUDIO_OUTPUT_PLACEHOLDER.id)
        self.im_start_token_id: int = IM_START.id
        self.text_lookahead: int = int(getattr(self.config, "text_lookahead", 0))

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | PathLike, **kwargs: Any) -> RaonProcessor:
        """Load a RaonProcessor from a pretrained model directory or HF Hub identifier.

        Args:
            model_name_or_path: Local path or HF Hub model identifier containing
                the tokenizer and model config files.
            **kwargs: Additional keyword arguments forwarded to ``__init__``.

        Returns:
            Initialized RaonProcessor with tokenizer and config loaded from the path.
        """
        return cls(model_name_or_path=model_name_or_path, **kwargs)

    def _parse_message_content(self, content: str | list[ContentItem], role: str) -> tuple[str, list[str], list[str]]:
        """Parse multimodal content items into text with audio tags and collected audio paths.

        Routes audio items to user or assistant path lists based on the message role.
        User, system, and tool roles produce input audio tags; assistant role produces
        output audio tags. Speaker items insert a speaker embedding placeholder.

        Args:
            content: Plain text string or list of typed content items (text, audio, speaker).
            role: Message role determining audio tag type and path routing.

        Returns:
            Tuple of (assembled text, user audio paths, assistant audio paths).
        """
        if isinstance(content, str):
            return content, [], []

        text_parts: list[str] = []
        user_audio_paths: list[str] = []
        assistant_audio_paths: list[str] = []

        if role in ("user", "system", "tool"):
            audio_tag = f"{AUDIO_START}{AUDIO_INPUT_PLACEHOLDER}{AUDIO_END}"
        else:
            audio_tag = f"{AUDIO_START}{AUDIO_OUTPUT_PLACEHOLDER}{AUDIO_END}"

        for item in content:
            if item["type"] == "text":
                text_parts.append(item["text"])
            elif item["type"] == "audio":
                audio_path = item["audio"]
                if audio_path:
                    if role in ("user", "system", "tool"):
                        user_audio_paths.append(audio_path)
                    else:
                        assistant_audio_paths.append(audio_path)
                    text_parts.append(audio_tag)
            elif item["type"] == "speaker":
                # Speaker token for speaker-conditioned TTS
                # The speaker embedding will be computed from audio_output during training
                text_parts.append(str(SPEAKER_EMBEDDING_PLACEHOLDER))

        return "".join(text_parts), user_audio_paths, assistant_audio_paths

    def process_messages(self, messages: list[Message]) -> tuple[list[TextMessage], list[str], list[str]]:
        """Convert a list of chat messages into text messages with collected audio paths.

        Iterates over messages, parsing multimodal content into plain text with audio
        placeholder tags, and accumulating user and assistant audio file paths. Warns
        if string content contains the pretraining audio tag format.

        Args:
            messages: List of chat messages, each with a role and text or multimodal content.

        Returns:
            Tuple of (processed text messages, all user audio paths, all assistant audio paths).
        """
        processed_messages: list[TextMessage] = []
        all_user_audio_paths: list[str] = []
        all_assistant_audio_paths: list[str] = []

        for message in messages:
            # Support both HF format (role/content) and dataset format (from/value).
            role = message.get("role") or {"human": "user", "gpt": "assistant"}.get(message.get("from", ""), "user")
            content = message.get("content") if "content" in message else message.get("value", "")

            if isinstance(content, str):
                if PRETRAINING_AUDIO_TAG in content:
                    logger.warning(
                        f"Message content contains '{PRETRAINING_AUDIO_TAG}' tag "
                        "which is the pretraining format. Use the message content list format with audio parts instead for "
                        "proper audio handling.",
                    )

                processed_messages.append({"role": role, "content": content})
            else:
                text_content, user_audio_paths, assistant_audio_paths = self._parse_message_content(content, role)
                all_user_audio_paths.extend(user_audio_paths)
                all_assistant_audio_paths.extend(assistant_audio_paths)
                processed_messages.append({"role": role, "content": text_content})

        return processed_messages, all_user_audio_paths, all_assistant_audio_paths

    def load_audio(self, audio_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load audio files, resample to the model sampling rate, and pad into a batch.

        Supports .wav files via soundfile and all other formats via torchaudio.
        Multi-channel audio is downmixed to mono. Files with different sample rates
        are resampled to ``self.sampling_rate``.

        Args:
            audio_paths: List of file paths to load. If empty, returns None.

        Returns:
            Tuple of (padded_audio, audio_lengths) where padded_audio has shape
            [num_files, max_audio_len] (dtype: float) and audio_lengths has shape
            [num_files] (dtype: long), or None if audio_paths is empty.
        """
        if not audio_paths:
            return None

        resampled_audio: list[torch.Tensor] = []

        for audio_path in audio_paths:
            if audio_path.endswith(".wav"):
                # sf.read returns [samples, channels] so dim=-1 averages across channels.
                audio_np, prev_sampling_rate = sf.read(audio_path)
                audio = torch.from_numpy(audio_np).float()
                if audio.ndim != 1:
                    audio = audio.mean(dim=-1)
            else:
                audio_np, prev_sampling_rate = sf.read(audio_path, dtype="float32")
                audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None]).float()
                if audio.ndim != 1:
                    audio = audio.mean(dim=0)

            assert audio.ndim == 1, f"Expected 1D audio after mean but got {audio.ndim=}"

            if prev_sampling_rate != self.sampling_rate:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=int(prev_sampling_rate),
                    new_freq=self.sampling_rate,
                )

            resampled_audio.append(audio)

        audio_lengths = torch.tensor([audio.shape[0] for audio in resampled_audio], dtype=torch.long)
        padded_audio = pad_sequence(resampled_audio, batch_first=True, padding_side="right", padding_value=0)

        return padded_audio, audio_lengths

    def expand_audio_padding(self, text: str, audio_code_lengths: torch.Tensor, pad_token: str) -> str:
        """Expand single audio placeholder tokens to match the required code lengths.

        Each occurrence of pad_token in text is replaced with pad_token repeated
        ``audio_code_lengths[i]`` times, using a two-phase replacement via an
        intermediate placeholder to avoid collisions.

        Args:
            text: Input text containing one pad_token per audio segment.
            audio_code_lengths: Number of code frames for each audio segment.
                Shape: [num_segments]. Dtype: long.
            pad_token: The placeholder token string to expand.

        Returns:
            Text with each pad_token occurrence expanded to the corresponding length.
        """
        assert audio_code_lengths.dtype == torch.long and audio_code_lengths.ndim == 1, (
            "The audio_code_lengths tensor must be 1D with dtype long."
        )

        # Two-phase replacement strategy:
        # Phase 1 - replace each pad_token with a run of AUDIO_PLACEHOLDER tokens
        #   (a distinct sentinel string) sized to the corresponding code length.
        #   This avoids the problem of str.replace() accidentally matching the
        #   newly-inserted text on subsequent iterations.
        # Phase 2 - swap every AUDIO_PLACEHOLDER back to pad_token in a single pass,
        #   so the final string contains pad_token repeated the correct number of times
        #   for each audio segment.
        pattern = re.escape(pad_token)
        positions = [(match.start(), match.group()) for match in re.finditer(pattern, text)]
        positions.sort(key=lambda x: x[0])

        assert len(positions) == len(audio_code_lengths), (
            f"Audio pad token count ({len(positions)}) != audio code lengths ({len(audio_code_lengths)})"
        )

        for (_, match_text), length in zip(positions, audio_code_lengths, strict=True):
            assert match_text == pad_token, "Match text must equal pad_token."
            text = text.replace(match_text, AUDIO_PLACEHOLDER * int(length.item()), 1)

        return text.replace(AUDIO_PLACEHOLDER, pad_token)

    def decode(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        input_length: int | None = None,
        output_only: bool = False,
        collapse_audio_tokens: bool = False,
        skip_special_tokens: bool = False,
        **tokenizer_decode_kwargs: Any,
    ) -> str:
        """Decode token IDs to text, optionally restricted to output or assistant response.

        Args:
            token_ids: Token IDs to decode. Shape: [batch_size, seq_length] or [seq_length].
                Dtype: long. If 2D, uses first row.
            labels: Optional labels with LOSS_IGNORE_INDEX for non-output positions.
                Shape: [batch_size, seq_length] or [seq_length]. Dtype: long. Used when
                output_only=True to extract only assistant-predicted tokens.
            input_length: Optional length of input (prompt) in tokens. Used when
                output_only=True to slice tokens from input_length onward.
            output_only: If True, decode only assistant output tokens (requires
                labels or input_length).
            collapse_audio_tokens: If True, collapse consecutive audio placeholders.
            skip_special_tokens: If True, omit special tokens from decoded text.
            **tokenizer_decode_kwargs: Passed to underlying tokenizer decode.

        Returns:
            Decoded string.
        """
        if token_ids.ndim == 2:
            token_ids = token_ids[0]

        if labels is not None and labels.ndim == 2:
            labels = labels[0]

        if output_only:
            assert labels is not None or input_length is not None, (
                "decode: `labels` or `input_length` is required when `output_only=True`."
            )
            if labels is not None:
                token_ids = token_ids[labels != LOSS_IGNORE_INDEX]
            else:
                assert input_length is not None, "input_length must be provided when labels is None."
                token_ids = token_ids[input_length:]

        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **tokenizer_decode_kwargs,
        )

        if collapse_audio_tokens:
            text = collapse_audio_placeholder_tokens(text)

        return text

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize chat-formatted text and compute assistant response labels.

        Strips empty think tags, tokenizes the text, then builds an assistant mask
        by locating ``<|im_start|>assistant`` / ``<|im_end|>`` boundaries. Tokens
        inside assistant responses get their true token ID as the label; all other
        positions receive LOSS_IGNORE_INDEX.

        Args:
            text: Chat-template-formatted text with ``<|im_start|>`` / ``<|im_end|>`` markers.

        Returns:
            Tuple of (input_ids, attention_mask, labels), each with shape [1, seq_length]
            and dtype long. attention_mask is all ones.
        """
        text = text.replace("<think></think>\n", "").replace("<think></think>", "")

        input_ids: list[int] = self.tokenizer.encode(text)
        attention_mask = [1] * len(input_ids)

        begin_assistant_ids = self.tokenizer.encode(f"{IM_START}assistant\n")
        end_turn_ids = self.tokenizer.encode(str(IM_END))

        begin_assistant_indices = [
            i
            for i in range(len(input_ids) - len(begin_assistant_ids) + 1)
            if input_ids[i : i + len(begin_assistant_ids)] == begin_assistant_ids
        ]
        end_turn_indices = [
            i for i in range(len(input_ids) - len(end_turn_ids) + 1) if input_ids[i : i + len(end_turn_ids)] == end_turn_ids
        ]

        assistant_masks: list[int] = []
        for begin_assistant_idx in begin_assistant_indices:
            begin_response_idx = begin_assistant_idx + len(begin_assistant_ids)
            if len(assistant_masks) > begin_response_idx:
                continue

            valid_end_turn_indices = [idx for idx in end_turn_indices if idx > begin_response_idx]
            if valid_end_turn_indices:
                end_response_idx = min(valid_end_turn_indices) + 1
                assert len(assistant_masks) <= begin_response_idx, (
                    f"Tokenize: Masks length exceeds begin response index. "
                    f"Got `{len(assistant_masks)=}` and `{begin_response_idx=}`."
                )
                assistant_masks.extend([0] * (begin_response_idx - len(assistant_masks)))
                assert len(assistant_masks) == begin_response_idx, (
                    f"Tokenize: Masks length does not match begin response index. "
                    f"Got `{len(assistant_masks)=}` and `{begin_response_idx=}`."
                )
                assert len(assistant_masks) <= end_response_idx, (
                    f"Tokenize: Masks length exceeds end response index. "
                    f"Got `{len(assistant_masks)=}` and `{end_response_idx=}`."
                )
                assistant_masks.extend([1] * (end_response_idx - len(assistant_masks)))

        assistant_masks.extend([0] * (len(input_ids) - len(assistant_masks)))
        assert len(assistant_masks) == len(input_ids), (
            f"Tokenize: Masks length does not match input_ids length. Got `{len(assistant_masks)=}` and `{len(input_ids)=}`."
        )

        labels = [input_ids[i] if assistant_masks[i] == 1 else LOSS_IGNORE_INDEX for i in range(len(input_ids))]

        return (
            torch.tensor([input_ids]),
            torch.tensor([attention_mask]),
            torch.tensor([labels]),
        )

    def process_single(
        self,
        messages: list[Message],
        add_generation_prompt: bool,
        audio_preprocessor: AudioPreprocessor | None,
        max_audio_chunk_length: int | None = None,
    ) -> RaonInputs:
        """Process a single conversation into RaonInputs.

        Parses messages, applies the chat template, loads and resamples audio,
        expands audio placeholders to match codec frame counts, and tokenizes
        with assistant response labels.

        Args:
            messages: List of chat messages forming one conversation.
            add_generation_prompt: If True, append generation prompt for inference.
            audio_preprocessor: Optional callback to preprocess audio tensors
                before placeholder expansion.
            max_audio_chunk_length: If set, split audio_input into chunks of at
                most this many samples (matching training-time chunking). None
                disables chunking.

        Returns:
            RaonInputs with input_ids, attention_mask, labels (each shape
            [1, seq_length]), and optional audio_input/audio_output tensors.
        """
        processed_messages, user_audio_paths, assistant_audio_paths = self.process_messages(messages)

        text = self.tokenizer.apply_chat_template(
            cast(list[dict[str, str]], processed_messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(text, str), "Chat template must return a string."

        # Load and process user audio (audio_input)
        audio_input_data = self.load_audio(user_audio_paths)
        audio_input: torch.Tensor | None = None
        audio_input_lengths: torch.Tensor | None = None

        if audio_input_data is not None:
            audio_input, audio_input_lengths = audio_input_data
            if audio_preprocessor is not None:
                audio_input, audio_input_lengths = audio_preprocessor(audio_input, audio_input_lengths)

            if audio_input_lengths is None:
                audio_input_lengths = torch.full(
                    (audio_input.shape[0],), fill_value=audio_input.shape[1], device=audio_input.device
                )

            audio_input_code_lengths = (audio_input_lengths.float() / self.samples_per_frame).ceil().long()
            text = self.expand_audio_padding(text, audio_input_code_lengths, str(AUDIO_INPUT_PLACEHOLDER))

            # Chunk audio input to match training behavior for non-causal AuT.
            if max_audio_chunk_length is not None:
                audio_input, audio_input_lengths = self._chunk_audio(
                    audio_input, audio_input_lengths, max_audio_chunk_length
                )

        # Load and process assistant audio (audio_output)
        audio_output_data = self.load_audio(assistant_audio_paths)
        audio_output: torch.Tensor | None = None
        audio_output_lengths: torch.Tensor | None = None

        if audio_output_data is not None:
            audio_output, audio_output_lengths = audio_output_data
            if audio_preprocessor is not None:
                audio_output, audio_output_lengths = audio_preprocessor(audio_output, audio_output_lengths)

            if audio_output_lengths is None:
                audio_output_lengths = torch.full(
                    (audio_output.shape[0],), fill_value=audio_output.shape[1], device=audio_output.device
                )

            audio_output_code_lengths = (audio_output_lengths.float() / self.samples_per_frame).ceil().long()
            text = self.expand_audio_padding(text, audio_output_code_lengths, str(AUDIO_OUTPUT_PLACEHOLDER))

        input_ids, attention_mask, labels = self._tokenize(text)
        has_speaker_placeholder = bool((input_ids == SPEAKER_EMBEDDING_PLACEHOLDER.id).any().item())
        if self.has_speaker_encoder and has_speaker_placeholder:
            speaker_encoder_audio, speaker_encoder_audio_lengths = self._prepare_speaker_encoder_audio(
                audio_output=audio_output,
                audio_output_lengths=audio_output_lengths,
            )
        else:
            speaker_encoder_audio, speaker_encoder_audio_lengths = None, None

        return RaonInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input=audio_input,
            audio_output=audio_output,
            speaker_encoder_audio=speaker_encoder_audio,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            speaker_encoder_audio_lengths=speaker_encoder_audio_lengths,
            labels=labels,
        )

    @staticmethod
    def _left_pad(tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        """Left-pad and stack 2D tensors into a single batch tensor.

        Args:
            tensors: List of 2D tensors, each with shape [1, seq_length_i].
            padding_value: Value used for left-padding shorter sequences.

        Returns:
            Batched tensor with shape [num_tensors, max_seq_length]. Dtype: same as input.
        """
        rows = [row for tensor in tensors for row in tensor]
        return pad_sequence(rows, batch_first=True, padding_value=padding_value, padding_side="left")

    @staticmethod
    def _optional_cat(optional_tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
        """Concatenate non-None tensors along dim 0, returning None if all are None.

        Args:
            optional_tensors: List of tensors or None values.

        Returns:
            Concatenated tensor or None if every element is None.
        """
        tensors = [tensor for tensor in optional_tensors if tensor is not None]
        if len(tensors) == 0:
            return None
        return torch.cat(tensors)

    @staticmethod
    def _optional_left_pad(optional_tensors: list[torch.Tensor | None], padding_value: int) -> torch.Tensor | None:
        """Left-pad and stack non-None 2D tensors, returning None if all are None.

        Args:
            optional_tensors: List of 2D tensors or None values.
            padding_value: Value used for left-padding shorter sequences.

        Returns:
            Batched tensor with shape [total_rows, max_seq_length] or None if all inputs are None.
        """
        tensors = [row for tensor in optional_tensors if tensor is not None for row in tensor]
        if len(tensors) == 0:
            return None
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")

    @staticmethod
    def _chunk_audio(
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        max_audio_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split padded audio waveforms into chunks of at most ``max_audio_seq_length`` samples.

        Replicates the chunking performed by
        ``SequencePackingAudioDataset.get_chunked_audio_from_batch`` so that
        inference audio tensors match the shape the model sees during training.

        Args:
            audio: Padded audio waveforms.
                Shape: [num_files, max_audio_len]. Dtype: float.
            audio_lengths: True length of each waveform in samples.
                Shape: [num_files]. Dtype: long.
            max_audio_seq_length: Maximum number of samples per chunk.

        Returns:
            Tuple of (chunked_audio, chunked_lengths) where chunked_audio has
            shape [num_chunks, max_chunk_len] (dtype: float, right-padded) and
            chunked_lengths has shape [num_chunks] (dtype: long).
        """
        chunks: list[torch.Tensor] = []
        chunk_lengths: list[int] = []

        for waveform, length in zip(audio, audio_lengths, strict=True):
            seq = waveform[: int(length.item())]
            while len(seq) > max_audio_seq_length:
                chunks.append(seq[:max_audio_seq_length])
                chunk_lengths.append(max_audio_seq_length)
                seq = seq[max_audio_seq_length:]
            if len(seq) > 0:
                chunks.append(seq)
                chunk_lengths.append(len(seq))

        padded = pad_sequence(chunks, batch_first=True, padding_side="right", padding_value=0)
        return padded, torch.tensor(chunk_lengths, dtype=torch.long)

    def _prepare_speaker_encoder_audio(
        self,
        audio_output: torch.Tensor | None,
        audio_output_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare fixed-width speaker audio from a random segment of each file.

        For each audio file, selects a random contiguous window of up to
        ``max_audio_seq_length`` samples from the valid region. This avoids
        always using the first chunk, reducing content leakage through the
        speaker encoder (which otherwise always sees the same audio the model
        generates first).

        Args:
            audio_output: Raw assistant audio. Shape: [num_segments, num_samples]. Dtype: float.
            audio_output_lengths: Valid lengths. Shape: [num_segments]. Dtype: long.

        Returns:
            Tuple of:
                - speaker_encoder_audio: Shape [num_segments, max_audio_seq_length]. Dtype: float.
                - speaker_encoder_audio_lengths: Shape [num_segments]. Dtype: long.
            Returns (None, None) if audio_output or lengths are missing.
        """
        if audio_output is None or audio_output_lengths is None:
            return None, None

        max_len = self.max_audio_seq_length
        fixed_audio = torch.zeros(
            audio_output.shape[0],
            max_len,
            device=audio_output.device,
            dtype=audio_output.dtype,
        )
        out_lengths: list[int] = []

        for idx, valid_length in enumerate(audio_output_lengths.tolist()):
            valid_length = int(min(valid_length, audio_output.shape[1]))
            if valid_length <= 0:
                out_lengths.append(0)
                continue

            if valid_length > max_len:
                # Random start within the valid region.
                start = random.randint(0, valid_length - max_len)
                fixed_audio[idx, :max_len] = audio_output[idx, start : start + max_len]
                out_lengths.append(max_len)
            else:
                fixed_audio[idx, :valid_length] = audio_output[idx, :valid_length]
                out_lengths.append(valid_length)

        return fixed_audio, torch.tensor(out_lengths, dtype=torch.long, device=audio_output.device)

    def _collate(self, batch: list[RaonInputs]) -> RaonInputs:
        """Collate a list of single-sample RaonInputs into a batched RaonInputs.

        Left-pads input_ids, attention_mask, and labels to the longest sequence.
        Audio tensors and their lengths are concatenated or left-padded as appropriate.

        Args:
            batch: List of RaonInputs, each from a single conversation.

        Returns:
            Batched RaonInputs with consistent sequence lengths across the batch.
        """
        return RaonInputs(
            input_ids=self._left_pad([item["input_ids"] for item in batch], padding_value=self.pad_token_id),
            attention_mask=self._left_pad([item["attention_mask"] for item in batch], padding_value=0),
            labels=self._left_pad([item["labels"] for item in batch], padding_value=LOSS_IGNORE_INDEX),
            audio_input=self._optional_left_pad([item["audio_input"] for item in batch], padding_value=0),
            audio_output=self._optional_left_pad([item["audio_output"] for item in batch], padding_value=0),
            speaker_encoder_audio=self._optional_cat([item.get("speaker_encoder_audio") for item in batch]),
            audio_input_lengths=self._optional_cat([item["audio_input_lengths"] for item in batch]),
            audio_output_lengths=self._optional_cat([item["audio_output_lengths"] for item in batch]),
            speaker_encoder_audio_lengths=self._optional_cat([item.get("speaker_encoder_audio_lengths") for item in batch]),
        )

    def __call__(
        self,
        messages: list[Message] | list[list[Message]],
        add_generation_prompt: bool = False,
        audio_preprocessor: AudioPreprocessor | None = None,
        force_audio_output: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_audio_chunk_length: int | None = None,
    ) -> RaonInputs:
        """Process messages into RaonInputs for training or inference.

        Accepts a single conversation or a batch of conversations. Loads and
        resamples audio, expands placeholders to match code lengths, tokenizes
        with assistant masks, and collates when batched.

        Args:
            messages: Single conversation (list of Message) or batch (list of
                list of Message). Supports TextMessage and MultiModalMessage.
            add_generation_prompt: If True, append generation prompt for inference.
            audio_preprocessor: Optional callback (audio, lengths) -> (audio, lengths)
                to preprocess audio before expansion.
            force_audio_output: If True, append an ``<|audio_start|>`` token to
                input_ids and a corresponding 1 to attention_mask, prompting
                the model to begin generating audio output.
            device: Optional device to move tensors to (input_ids, attention_mask,
                labels, audio_input, audio_output, length tensors).
            dtype: Optional dtype to cast audio tensors to.
            max_audio_chunk_length: If set, split audio_input into chunks of at
                most this many samples (matching training-time chunking). None
                disables chunking.

        Returns:
            RaonInputs with input_ids, attention_mask, labels, optional
            audio_input/audio_output and their length tensors. Shapes follow
            batch_size and seq_length; audio tensors are [num_chunks, max_chunk_len]
            when max_audio_seq_length is set, otherwise [batch_size, audio_len].
        """
        if len(messages) > 0 and isinstance(messages[0], list):
            batched_messages = cast(list[list[Message]], messages)
            batch = [
                self.process_single(
                    conversation,
                    add_generation_prompt=add_generation_prompt,
                    audio_preprocessor=audio_preprocessor,
                    max_audio_chunk_length=max_audio_chunk_length,
                )
                for conversation in batched_messages
            ]
            result = self._collate(batch)
        else:
            conversation = cast(list[Message], messages)
            result = self.process_single(
                conversation,
                add_generation_prompt=add_generation_prompt,
                audio_preprocessor=audio_preprocessor,
                max_audio_chunk_length=max_audio_chunk_length,
            )

        if force_audio_output:
            # Strip trailing <|audio_end|> and <|im_end|> tokens so the
            # appended <|audio_start|> directly follows the audio content.
            trailing_ids = {AUDIO_END.id, int(self.tokenizer.encode(str(IM_END))[0])}
            ids = result["input_ids"]
            while ids.shape[1] > 0 and int(ids[0, -1].item()) in trailing_ids:
                ids = ids[:, :-1]
                result["attention_mask"] = result["attention_mask"][:, :-1]
            result["input_ids"] = ids

            batch_size = result["input_ids"].shape[0]
            audio_start_id = torch.full(
                (batch_size, 1), AUDIO_START.id, dtype=result["input_ids"].dtype, device=result["input_ids"].device
            )
            result["input_ids"] = torch.cat([result["input_ids"], audio_start_id], dim=1)
            result["attention_mask"] = torch.cat(
                [
                    result["attention_mask"],
                    torch.ones(batch_size, 1, dtype=result["attention_mask"].dtype, device=result["attention_mask"].device),
                ],
                dim=1,
            )

        if device is not None:
            result["input_ids"] = result["input_ids"].to(device)
            result["attention_mask"] = result["attention_mask"].to(device)
            result["labels"] = result["labels"].to(device)
            if result["audio_output"] is not None:
                result["audio_output"] = result["audio_output"].to(device)
            if result["audio_output_lengths"] is not None:
                result["audio_output_lengths"] = result["audio_output_lengths"].to(device)
            if result["audio_input"] is not None:
                result["audio_input"] = result["audio_input"].to(device)
            if result["audio_input_lengths"] is not None:
                result["audio_input_lengths"] = result["audio_input_lengths"].to(device)
            speaker_audio = result.get("speaker_encoder_audio")
            speaker_audio_lengths = result.get("speaker_encoder_audio_lengths")
            if speaker_audio is not None:
                result["speaker_encoder_audio"] = speaker_audio.to(device)
            if speaker_audio_lengths is not None:
                result["speaker_encoder_audio_lengths"] = speaker_audio_lengths.to(device)

        if dtype is not None:
            if result["audio_output"] is not None:
                result["audio_output"] = result["audio_output"].to(dtype)
            if result["audio_input"] is not None:
                result["audio_input"] = result["audio_input"].to(dtype)
            speaker_audio = result.get("speaker_encoder_audio")
            if speaker_audio is not None:
                result["speaker_encoder_audio"] = speaker_audio.to(dtype)

        return result


# ── from pipeline.py — embedded inference config ──


_DEFAULT_TASK_PARAMS: dict[str, dict] = {
    "default_text": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "force_audio_output": False,
        "force_text_output": True,
    },
    "default_audio": {
        "max_new_tokens": 512,
        "temperature": 1.2,
        "force_audio_output": True,
        "force_text_output": False,
    },
    "default_text_with_audio_input": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "force_audio_output": False,
        "force_text_output": True,
        "max_audio_chunk_length": 192000,
    },
    "tts": {
        "max_new_tokens": 512,
        "temperature": 1.2,
        "force_audio_output": True,
        "force_text_output": False,
        "ras_enabled": True,
        "ras_window_size": 50,
        "ras_repetition_threshold": 0.5,
    },
    "tts_continuation": {
        "max_new_tokens": 512,
        "temperature": 1.2,
        "force_audio_output": True,
        "force_text_output": False,
        "ras_enabled": True,
        "ras_window_size": 50,
        "ras_repetition_threshold": 0.5,
    },
    "stt": {
        "max_new_tokens": 512,
        "temperature": 0.2,
        "force_audio_output": False,
        "force_text_output": True,
        "max_audio_chunk_length": 192000,
    },
    "speech-chat": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "force_audio_output": False,
        "force_text_output": True,
        "max_audio_chunk_length": 192000,
    },
    "textqa": {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "force_audio_output": False,
        "force_text_output": True,
        "max_audio_chunk_length": 192000,
    },
}

# ── from pipeline.py ──

_DEFAULT_INFERENCE_CONFIG = Path(__file__).parent.parent.parent / "config" / "infer.yaml"


_DEFAULT_DUPLEX_CONFIG = Path(__file__).parent.parent.parent / "config" / "duplex_infer.yaml"


class RaonPipeline:
    """High-level inference API for RAON speech LLM.

    Loads the model and processor once, and exposes task-specific methods for
    STT, TTS, TextQA, and Speech-Chat.

    Example::

        pipe = RaonPipeline("/path/to/model")
        text = pipe.stt("audio.wav")
        waveform, sr = pipe.tts("Hello, world!")
        RaonPipeline.save_audio((waveform, sr), "out.wav")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        config: str | None = None,
        duplex_config: str | None = None,
    ) -> None:
        """Load model, processor, and default task parameters from infer.yaml.

        Args:
            model_path: Path to the pretrained RAON model directory.
            device: Device to run inference on (e.g. ``"cuda"``, ``"cpu"``).
            dtype: Torch dtype string — one of ``"bfloat16"``, ``"float16"``, ``"float32"``.
            attn_implementation: Attention backend string (``"sdpa"``, ``"eager"``, or ``"fa"``).
            config: Optional path to ``infer.yaml``-style task defaults.
            duplex_config: Optional path to ``duplex_infer.yaml``-style defaults.
        """
        torch_dtype = DTYPE_MAP[dtype]
        self.device = device
        self.dtype = torch_dtype

        self.model: RaonModel = RaonModel.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device).eval()
        if attn_implementation == "fa":
            attn_implementation = "flash_attention_2"
        if attn_implementation not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError(
                f"Invalid attn_implementation: {attn_implementation}. "
                "Use one of: sdpa, eager, fa."
            )
        self.model._set_attention_implementation(attn_implementation)
        logger.info("Pipeline attention implementation: %s", attn_implementation)
        self.processor: RaonProcessor = RaonProcessor.from_pretrained(model_path)

        self.task_params: dict[str, dict] = _DEFAULT_TASK_PARAMS
        self.duplex_params: dict = {}

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        force_audio_output: bool = False,
        force_text_output: bool = True,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        speaker_audio: str | None = None,
        max_audio_chunk_length: int | None = 192000,
        **gen_kwargs,
    ) -> str | tuple[torch.Tensor, int]:
        """Process messages through the model and return the result.

        Args:
            messages: HF-style message list, e.g.
                ``[{"role": "user", "content": "Hello"}]`` or multimodal
                ``[{"role": "user", "content": [{"type": "audio", "audio": "path"}, {"type": "text", "text": "..."}]}]``.
            force_audio_output: If ``True``, generate audio output (for TTS).
            force_text_output: If ``True``, generate text output.
            max_new_tokens: Maximum number of new tokens to generate.
                Defaults to 512 for audio output, 1024 for text output.
            temperature: Sampling temperature.
                Defaults to 1.2 for audio output, 0.7 for text output.
            speaker_audio: Path to speaker reference audio for voice conditioning.
            max_audio_chunk_length: If set, split audio_input into chunks of at
                most this many samples (matching training-time chunking).
            **gen_kwargs: Additional keyword arguments forwarded to ``model.generate()``.

        Returns:
            ``str`` for text output, or ``(waveform_tensor, sampling_rate)`` for audio output.
        """
        has_audio_input = any(
            isinstance(m.get("content"), list) and any(p.get("type") == "audio" for p in m["content"]) for m in messages
        )
        if force_audio_output:
            default_key = "default_audio"
        elif has_audio_input:
            default_key = "default_text_with_audio_input"
        else:
            default_key = "default_text"
        defaults = self.task_params.get(default_key, {})
        if max_new_tokens is None:
            max_new_tokens = defaults.get("max_new_tokens", 1024)
        if temperature is None:
            temperature = defaults.get("temperature", 0.7)
        if force_audio_output:
            force_text_output = False
            # Prepend speaker embedding placeholder to the first user message
            # so the model conditions on the speaker reference audio.
            if speaker_audio is not None:
                speaker_token = str(SPEAKER_EMBEDDING_PLACEHOLDER)
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str) and speaker_token not in content:
                            msg["content"] = speaker_token + content
                        elif isinstance(content, list):
                            has_speaker = any(
                                isinstance(p, dict) and p.get("type") == "text" and speaker_token in p.get("text", "")
                                for p in content
                            )
                            if not has_speaker:
                                content.insert(0, {"type": "text", "text": speaker_token})
                        break

        inputs = self.processor(
            messages,
            add_generation_prompt=True,
            force_audio_output=force_audio_output,
            device=self.device,
            dtype=self.dtype,
            max_audio_chunk_length=max_audio_chunk_length,
        )

        input_length = int(inputs["attention_mask"].sum().item())

        # Load speaker audio if a path was given
        speaker_audio_tensor: torch.Tensor | None = None
        if speaker_audio is not None:
            speaker_audio_tensor = self._load_speaker_audio(speaker_audio)

        output = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            audio_input=inputs.get("audio_input"),
            audio_input_lengths=inputs.get("audio_input_lengths"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            force_audio_output=force_audio_output,
            force_text_output=force_text_output,
            speaker_audio=speaker_audio_tensor,
            **gen_kwargs,
        )

        if force_audio_output:
            audio = output["audio"]
            audio_lengths = output["audio_lengths"]
            waveform = audio[0]
            length = int(audio_lengths[0].item())
            return self._trim_last_frame(waveform[:length]), self.processor.sampling_rate

        sequences = output["sequences"]
        generated_ids = sequences[0, input_length:]
        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Task-specific convenience methods
    # ------------------------------------------------------------------

    def stt(self, audio: str, prompt: str | None = None) -> str:
        """STT: audio → text.

        Args:
            audio: Path to the input audio file.
            prompt: Optional transcription instruction. Defaults to the
                standard STT prompt.

        Returns:
            Transcribed text string.
        """
        effective_prompt = prompt if prompt is not None else get_default_stt_prompt()
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": effective_prompt},
                ],
            }
        ]
        params = self.task_params.get("stt", {})
        return self.chat(  # type: ignore[return-value]
            messages,
            force_audio_output=params.get("force_audio_output", False),
            force_text_output=params.get("force_text_output", True),
            max_new_tokens=params.get("max_new_tokens", 512),
            temperature=params.get("temperature", 0.2),
            max_audio_chunk_length=params.get("max_audio_chunk_length"),
        )

    def tts(
        self,
        text: str,
        speaker_audio: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        """TTS: text → audio.

        Args:
            text: The text to synthesize.
            speaker_audio: Optional path to a speaker reference audio file for
                voice conditioning.

        Returns:
            ``(waveform, sampling_rate)`` tuple.
        """
        speaker_token = str(SPEAKER_EMBEDDING_PLACEHOLDER) if speaker_audio is not None else ""
        prompt = get_default_tts_prompt()
        messages: list[dict] = [
            {
                "role": "user",
                "content": f"{speaker_token}{prompt}:\n{text}",
            }
        ]
        params = self.task_params.get("tts", {})
        waveform, sampling_rate = self.chat(  # type: ignore[misc]
            messages,
            force_audio_output=params.get("force_audio_output", True),
            force_text_output=params.get("force_text_output", False),
            max_new_tokens=params.get("max_new_tokens", 512),
            temperature=params.get("temperature", 1.2),
            speaker_audio=speaker_audio,
            ras_enabled=params.get("ras_enabled", False),
            ras_window_size=params.get("ras_window_size", 50),
            ras_repetition_threshold=params.get("ras_repetition_threshold", 0.5),
        )
        return waveform, sampling_rate

    def tts_continuation(
        self,
        target_text: str,
        ref_audio: str,
        ref_text: str | None = None,
        speaker_audio: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        """TTS continuation: prefill reference audio as generated output, then continue for target text.

        Constructs the sequence as if the model already generated the reference audio,
        then continues generating audio for the target text. This produces speech that
        naturally continues from the reference, preserving speaker characteristics.

        Args:
            target_text: Text to generate speech for.
            ref_audio: Path to the reference audio file.
            ref_text: Transcription of the reference audio. If ``None``, the
                reference audio is automatically transcribed via :meth:`stt`.
            speaker_audio: Optional path to a separate speaker reference audio
                for voice conditioning. If ``None``, ``ref_audio`` is used.

        Returns:
            ``(waveform, sampling_rate)`` tuple.
        """
        if ref_text is None:
            ref_text = self.stt(ref_audio)
        if speaker_audio is None:
            speaker_audio = ref_audio

        ref_audio_tensor = self._load_speaker_audio(ref_audio).to(self.dtype)
        ref_audio_lengths = torch.tensor([ref_audio_tensor.shape[1]], device=self.device)

        # Tokenize reference audio into codes (with chunking to match training).
        model_config = self.model.config if hasattr(self.model, "config") else self.model.get_model().config
        max_output_chunk = getattr(model_config, "max_audio_output_seq_length", 192000)
        ref_samples = ref_audio_tensor.shape[-1]
        with torch.no_grad():
            num_code_groups = self.model.num_code_groups
            if ref_samples <= max_output_chunk:
                ref_codes = self.model.tokenize_audio(
                    audio=ref_audio_tensor,
                    audio_lengths=ref_audio_lengths,
                    num_code_groups=num_code_groups,
                ).audio_codes
            else:
                code_chunks = []
                offset = 0
                while offset < ref_samples:
                    end = min(offset + max_output_chunk, ref_samples)
                    chunk = ref_audio_tensor[:, offset:end]
                    chunk_len = torch.tensor([end - offset], device=self.device)
                    chunk_codes = self.model.tokenize_audio(
                        audio=chunk,
                        audio_lengths=chunk_len,
                        num_code_groups=num_code_groups,
                    ).audio_codes
                    code_chunks.append(chunk_codes)
                    offset = end
                ref_codes = torch.cat(code_chunks, dim=1)
        num_ref_frames = ref_codes.shape[1]

        speaker_token = str(SPEAKER_EMBEDDING_PLACEHOLDER) if speaker_audio is not None else ""
        combined_text = f"{ref_text} {target_text}"
        prompt = get_default_tts_prompt()
        messages = [{"role": "user", "content": f"{speaker_token}{prompt}:\n{combined_text}"}]

        inputs = self.processor(
            messages,
            add_generation_prompt=True,
            device=self.device,
            dtype=self.dtype,
        )

        # Manually append <audio_start> + <audio_output_placeholder> * num_ref_frames.
        audio_prefix = torch.full(
            (1, 1 + num_ref_frames),
            AUDIO_OUTPUT_PLACEHOLDER.id,
            dtype=torch.long,
            device=self.device,
        )
        audio_prefix[0, 0] = AUDIO_START.id
        input_ids = torch.cat([inputs["input_ids"], audio_prefix], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(audio_prefix)], dim=1)

        speaker_audio_tensor = self._load_speaker_audio(speaker_audio)

        params = self.task_params.get("tts_continuation", self.task_params.get("tts", {}))
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_output=ref_audio_tensor,
                audio_output_lengths=ref_audio_lengths,
                max_new_tokens=params.get("max_new_tokens", 512),
                temperature=params.get("temperature", 1.2),
                top_k=params.get("top_k", 20),
                top_p=params.get("top_p", 0.8),
                do_sample=True,
                force_audio_output=True,
                speaker_audio=speaker_audio_tensor,
                ras_enabled=params.get("ras_enabled", False),
                ras_window_size=params.get("ras_window_size", 50),
                ras_repetition_threshold=params.get("ras_repetition_threshold", 0.5),
                continuation_silence_frames=params.get("continuation_silence_frames", 0),
            )

        audio = output["audio"]
        audio_lengths = output["audio_lengths"]
        waveform = audio[0]
        length = int(audio_lengths[0].item())
        return waveform[:length], self.processor.sampling_rate

    def speech_chat(self, audio: str) -> str:
        """Speech-Chat: audio → text.

        Args:
            audio: Path to the audio file containing the spoken question.

        Returns:
            Text answer string.
        """
        messages: list[dict] = [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio}],
            }
        ]
        params = self.task_params.get("speech-chat", {})
        return self.chat(  # type: ignore[return-value]
            messages,
            force_audio_output=params.get("force_audio_output", False),
            force_text_output=params.get("force_text_output", True),
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.7),
            max_audio_chunk_length=params.get("max_audio_chunk_length"),
        )

    def textqa(self, text: str, audio: str | None = None) -> str:
        """TextQA: text + optional audio → text.

        Args:
            text: The input text prompt or question.
            audio: Optional path to an audio file providing context.

        Returns:
            Generated text string.
        """
        if audio is not None:
            content: str | list[dict] = [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": text},
            ]
        else:
            content = text
        messages: list[dict] = [{"role": "user", "content": content}]
        params = self.task_params.get("textqa", {})
        return self.chat(  # type: ignore[return-value]
            messages,
            force_audio_output=params.get("force_audio_output", False),
            force_text_output=params.get("force_text_output", True),
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.7),
            max_audio_chunk_length=params.get("max_audio_chunk_length"),
        )

    # ------------------------------------------------------------------
    # Duplex
    # ------------------------------------------------------------------

    def load_audio(
        self,
        path: str,
        channel: int | None = None,
    ) -> torch.Tensor:
        """Load an audio file, resample to the model's sampling rate.

        For stereo duplex audio: channel 0 = assistant, channel 1 = user.

        Args:
            path: Path to the audio file.
            channel: Channel index to extract from stereo audio. ``None`` = mono mix.

        Returns:
            Audio tensor of shape ``[1, num_samples]`` on ``self.device``.
        """
        audio, _ = _load_audio_shared(
            path,
            self.processor.sampling_rate,
            mono=channel is None,
            channel=channel,
            device=self.device,
            dtype=self.dtype,
        )
        return audio

    def duplex(
        self,
        audio_input: torch.Tensor,
        output_dir: str,
        *,
        system_prompt: str | None = None,
        speak_first: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        sil_penalty: float | None = None,
        bc_penalty: float | None = None,
        speaker_embeds: torch.Tensor | None = None,
        speaker_audio: str | None = None,
        eos_penalty: float | None = None,
    ) -> dict:
        """Run full-duplex inference.

        Parameters default to values from ``config/duplex_infer.yaml``.

        Args:
            audio_input: User audio tensor of shape ``[1, num_samples]``.
            output_dir: Directory to save output files.
            system_prompt: System prompt text.
            speak_first: If True, the model speaks first.
            temperature: Sampling temperature.
            top_p: Top-p sampling threshold.
            top_k: Top-k filtering.
            sil_penalty: Penalty subtracted from SIL token logit.
            speaker_embeds: Optional precomputed speaker embeddings.
            speaker_audio: Optional speaker reference audio path. Used only
                when ``speaker_embeds`` is not provided.
            eos_penalty: EOS penalty value.

        Returns:
            Summary dict with durations and sample counts.
        """

        cfg = self.duplex_params
        if system_prompt is None:
            system_prompt = cfg.get("system_prompt", "You are engaging in real-time conversation.")
        if speak_first is None:
            speak_first = cfg.get("speak_first", False)
        if temperature is None:
            temperature = cfg.get("temperature", 0.9)
        if top_p is None:
            top_p = cfg.get("top_p", 0.95)
        if top_k is None:
            top_k = cfg.get("top_k", 66)
        if sil_penalty is None:
            sil_penalty = cfg.get("sil_penalty", 0.0)
        if bc_penalty is None:
            bc_penalty = cfg.get("bc_penalty", 0.0)
        if eos_penalty is None:
            eos_penalty = cfg.get("eos_penalty", 0.0)
        if speaker_embeds is None and speaker_audio is not None:
            speaker_audio_tensor = self._load_speaker_audio(speaker_audio)
            speaker_embeds = self.model._compute_speaker_embeds(speaker_audio_tensor, None)

        return run_duplex_inference(
            model=self.model,
            processor=self.processor,
            audio_input=audio_input,
            output_dir=Path(output_dir),
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            sil_penalty=sil_penalty,
            bc_penalty=bc_penalty,
            speaker_embeds=speaker_embeds,
            device=self.device,
            dtype=self.dtype,
            eos_penalty=eos_penalty,
            speak_first=speak_first,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def save_audio(result: tuple[torch.Tensor, int], path: str) -> None:
        """Save an audio result to a WAV file.

        Args:
            result: ``(waveform, sampling_rate)`` tuple as returned by
                :meth:`tts` or :meth:`chat`.
            path: Destination file path (e.g. ``"output.wav"``).
        """

        waveform, sampling_rate = result
        _save(waveform, sampling_rate, path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_last_frame(self, waveform: torch.Tensor) -> torch.Tensor:
        """Drop one codec frame from generated audio to avoid trailing artifacts."""
        samples_per_frame = int(self.model.sampling_rate / self.model.frame_rate)
        if waveform.shape[-1] <= samples_per_frame:
            logger.warning(
                "Generated audio too short (%d samples <= %d samples_per_frame); returning empty waveform.",
                waveform.shape[-1],
                samples_per_frame,
            )
            return waveform[:0]
        return waveform[:-samples_per_frame]

    def _load_speaker_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess a speaker reference audio file.

        Resamples to the model's sampling rate and converts to mono.

        Args:
            path: File path to the speaker audio.

        Returns:
            Float tensor of shape ``[1, num_samples]`` on ``self.device``.
        """
        audio, _ = _load_audio_shared(
            path,
            self.processor.sampling_rate,
            mono=True,
            device=self.device,
            dtype=self.dtype,
        )
        return audio


# ── from utils/duplex_prompt_catalog.py ──


import json
import random
from pathlib import Path
from typing import Any

_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "duplex" / "personas.json"

_cached_catalog: dict[str, Any] | None = None

# Embedded fallback catalog used when the JSON file is not available (e.g. Hub loading).
_EMBEDDED_CATALOG: dict[str, Any] = {
    "system_prompt_base": "You are engaging in real-time conversation.",
    "name": "Raon",
    "personas": {
        "general": "a friendly and helpful assistant",
        "game": "a game host who facilitates interactive speech games with the user",
        "scenario_movie": "a movie and entertainment guide",
        "scenario_banking": "a personal banking advisor",
        "scenario_fitness": "a fitness coach",
        "scenario_shopping": "an online shopping assistant",
        "scenario_pet": "a pet care advisor",
        "scenario_healthcare": "a healthcare scheduling assistant",
        "scenario_realestate": "a real estate advisor",
        "scenario_techsupport": "a tech support specialist",
        "scenario_carrental": "a car rental and navigation assistant",
        "scenario_event": "an event planning assistant",
        "scenario_restaurant": "a restaurant and food delivery assistant",
        "scenario_language": "a language tutor",
        "scenario_travel": "a travel planning assistant",
        "scenario_interview": "a job interview coach",
        "scenario_game_npc": "a game NPC assistant",
    },
}


def load_persona_catalog(catalog_path: str | Path | None = None) -> dict[str, Any]:
    """Load persona catalog from JSON file.

    Args:
        catalog_path: Path to catalog JSON. Defaults to raon/data/duplex/personas.json.

    Returns:
        Parsed catalog dictionary.
    """
    global _cached_catalog
    path = Path(catalog_path) if catalog_path is not None else _DEFAULT_CATALOG_PATH
    if _cached_catalog is None or catalog_path is not None:
        if path.exists():
            with open(path) as f:
                catalog = json.load(f)
        else:
            catalog = _EMBEDDED_CATALOG
        if catalog_path is None:
            _cached_catalog = catalog
        return catalog
    return _cached_catalog


DEFAULT_ASSISTANT_PERSONA = "a friendly and helpful assistant"


def build_system_prompt(
    persona: str | None = None,
    context: str | None = None,
    name: str | None = None,
    record: dict | None = None,
    catalog_path: str | Path | None = None,
    deterministic: bool = False,
) -> str:
    """Build a system prompt from persona, context, name, or a data record.

    Supports two usage patterns:

    **Direct args** (inference CLI, notebook)::

        build_system_prompt(
            persona="scenario_restaurant",
            context="Discuss restaurant menu choices and delivery preferences with a customer.",
        )

    **Record-based** (training data)::

        build_system_prompt(record={"name": "Raon", "persona": "a restaurant assistant", "context": "..."})

    When ``record`` is provided, the mode is auto-detected:

    - **context** mode (``context`` present): 50% ``"{base} {context}"``, 50% ``"{base} You are {name}, {persona}."``
    - **persona** mode (``persona`` present): ``"{base} You are {name}, {persona}."``
    - **assistant** mode (only ``name`` present): ``"{base} You are {name}, a friendly and helpful assistant."``
    - **none** mode (no fields): ``"{base}"``

    Direct args (``persona``, ``context``, ``name``) override record values.

    Args:
        persona: Persona key (resolved via catalog) or raw persona description string.
        context: Additional context sentence.
        name: Assistant name override.
        record: Data record dict with optional persona/context/name fields.
        catalog_path: Optional override for catalog file path.

    Returns:
        Formatted system prompt string.
    """
    catalog = load_persona_catalog(catalog_path)
    base = catalog.get("system_prompt_base", "You are engaging in real-time conversation.")

    # Merge record fields with direct args (direct args take precedence).
    if record is not None:
        if persona is None:
            persona = record.get("persona")
        if context is None:
            context = record.get("context")
        if name is None:
            name = record.get("name")

    if name is None:
        name = catalog.get("name", "Raon")

    # Try resolving persona key from catalog; if not found, use raw string.
    if persona is not None:
        resolved = catalog.get("personas", {}).get(persona)
        if resolved is not None:
            persona = resolved

    # Auto-detect mode based on available fields.
    if context is not None:
        # Context mode: deterministic always uses context; training randomly
        # alternates between context and persona for data diversity.
        if deterministic or random.random() < 0.5:
            return f"{base} {context}"
        else:
            p = persona if persona else DEFAULT_ASSISTANT_PERSONA
            return f"{base} You are {name}, {p}."
    elif persona is not None:
        # Persona mode.
        return f"{base} You are {name}, {persona}."
    elif record is not None and name != catalog.get("name", "Raon"):
        # Assistant mode: name was explicitly set in record.
        return f"{base} You are {name}, {DEFAULT_ASSISTANT_PERSONA}."
    elif record is not None and record.get("name") is not None:
        # Assistant mode: name field present in record.
        return f"{base} You are {name}, {DEFAULT_ASSISTANT_PERSONA}."
    elif record is not None:
        # None mode: record provided but no persona/context/name.
        return base
    else:
        # No record, no persona — base only.
        return base


# ── from duplex_generate.py ──


import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm.auto import trange



logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: str | Path) -> dict:
    """Load duplex inference config from YAML, returning the 'duplex' section."""
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw.get("duplex", {}) if raw else {}


def _resolve_metadata_audio_path(path_value: str, metadata_jsonl_path: Path | None) -> str:
    """Resolve metadata-provided audio path to an existing filesystem path when possible."""
    raw = Path(path_value).expanduser()
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [raw]
        if metadata_jsonl_path is not None:
            candidates.append((metadata_jsonl_path.parent / raw))
            candidates.append((metadata_jsonl_path.parent.parent / raw))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _extract_speaker_audio_from_metadata(metadata: dict) -> str | None:
    """Extract speaker reference audio path from metadata JSON (preferred key: ``speaker_audio``)."""
    speaker_audio = metadata.get("speaker_audio")
    if isinstance(speaker_audio, str) and speaker_audio.strip():
        return speaker_audio.strip()

    speaker_ref_audios = metadata.get("speaker_ref_audios")
    if isinstance(speaker_ref_audios, list) and speaker_ref_audios and isinstance(speaker_ref_audios[0], str):
        first = speaker_ref_audios[0].strip()
        if first:
            return first
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAON full-duplex inference on an input audio file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument(
        "--audio_input",
        type=str,
        default=None,
        help="Path to the input audio file (.wav). Use this for single-sample inference.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory scanned recursively for *.jsonl. Each non-empty line is treated as one sample.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output audio.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to duplex_infer.yaml. Defaults to config/duplex_infer.yaml relative to repo root.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature (default: from config or 0.9)."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering (default: from config or 66).")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p filtering (default: from config or 0.95).")
    parser.add_argument(
        "--sil_penalty",
        type=float,
        default=None,
        help="Penalty subtracted from SIL token logit (default: from config or 0.0).",
    )
    parser.add_argument(
        "--bc_penalty",
        type=float,
        default=None,
        help="Penalty subtracted from BC token logit (default: from config or 0.0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible duplex decoding.")
    parser.add_argument(
        "--speak_first",
        action="store_true",
        default=None,
        help="Force model to speak first. Default: listen first.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Persona key (from catalog) or raw persona description string.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Additional context sentence appended to system prompt when persona is set.",
    )
    parser.add_argument(
        "--system_prompt_style",
        type=str,
        default=None,
        choices=["base", "persona", "persona_context", "custom"],
        help="System prompt style. Default: base.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt text. Only used when --system_prompt_style=custom.",
    )
    parser.add_argument(
        "--speaker_audio",
        type=str,
        default=None,
        help="Speaker reference audio for voice conditioning (e.g., data/duplex/eval/audio/spk_ref.wav).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["fa", "sdpa", "eager"],
        help="Attention implementation (default: eager). Use `fa` for FlashAttention.",
    )
    args = parser.parse_args()
    if bool(args.audio_input) == bool(args.data_dir):
        parser.error("Provide exactly one of --audio_input or --data_dir.")
    if args.attn_implementation == "fa":
        args.attn_implementation = "flash_attention_2"

    # Auto-detect metadata JSONL alongside audio input.
    # e.g. /path/to/duplex_00.wav → /path/to/duplex_00.jsonl or /path/to/../duplex_00.jsonl
    _meta = None
    _meta_jsonl_path: Path | None = None
    if args.audio_input:
        audio_path = Path(args.audio_input)
        jsonl_candidates = [
            audio_path.with_suffix(".jsonl"),  # same dir as audio
            audio_path.parent.parent / f"{audio_path.stem}.jsonl",  # parent dir (e.g. audio/ → ../)
        ]
        for jsonl_path in jsonl_candidates:
            if jsonl_path.exists():
                with open(jsonl_path) as _f:
                    _meta = json.loads(_f.readline())
                _meta_jsonl_path = jsonl_path
                logger.info("Loaded metadata from %s", jsonl_path)
                break

    if _meta is not None:
        # CLI args take precedence over metadata values.
        if args.speak_first is None:
            args.speak_first = _meta.get("speak_first", False)
        if args.persona is None:
            args.persona = _meta.get("persona")
        if args.context is None:
            args.context = _meta.get("context")
        if args.system_prompt is None:
            args.system_prompt = _meta.get("system_prompt")
        if args.speaker_audio is None:
            metadata_speaker_audio = _extract_speaker_audio_from_metadata(_meta)
            if metadata_speaker_audio is not None:
                args.speaker_audio = _resolve_metadata_audio_path(
                    metadata_speaker_audio,
                    _meta_jsonl_path,
                )
                logger.info("Using speaker reference audio from metadata: %s", args.speaker_audio)
        args._meta_name = _meta.get("name")
    else:
        args._meta_name = None

    # Resolve config path: explicit --config > config/duplex_infer.yaml relative to repo root.
    if args.config is None:
        repo_root = Path(__file__).resolve().parents[2]
        default_config = repo_root / "config" / "duplex_infer.yaml"
        if default_config.exists():
            args.config = str(default_config)

    # Load sampling parameters from yaml config. CLI args take precedence.
    cfg = _load_yaml_config(args.config) if args.config else {}
    if args.temperature is None:
        args.temperature = float(cfg.get("temperature", 0.9))
    if args.top_p is None:
        args.top_p = float(cfg.get("top_p", 0.95))
    if args.top_k is None:
        args.top_k = int(cfg.get("top_k", 66))
    args.do_sample = bool(cfg.get("do_sample", True))
    args.eos_penalty = float(cfg.get("eos_penalty", 0.0))
    if args.sil_penalty is None:
        args.sil_penalty = float(cfg.get("sil_penalty", 0.0))
    if args.bc_penalty is None:
        args.bc_penalty = float(cfg.get("bc_penalty", 0.0))
    if args.seed is None and cfg.get("seed") is not None:
        args.seed = int(cfg["seed"])

    # CLI args take precedence over yaml config.
    if args.speak_first is None:
        args.speak_first = bool(cfg.get("speak_first", False))
    if args.system_prompt_style is None:
        args.system_prompt_style = cfg.get("system_prompt_style", "base")
    if args.persona is None:
        args.persona = cfg.get("persona") or None
    if args.context is None:
        args.context = cfg.get("context") or None

    # Build system prompt based on style.

    if args.system_prompt is not None:
        # Already set (explicit --system_prompt or from metadata).
        system_prompt_text = args.system_prompt
    elif args.system_prompt_style == "custom" and args.system_prompt:
        system_prompt_text = args.system_prompt
    else:
        system_prompt_text = build_system_prompt(
            persona=args.persona,
            context=args.context,
            name=getattr(args, "_meta_name", None),
            deterministic=True,
        )
    args.system_prompt = system_prompt_text

    return args




def _duplex_load_audio(path: str, target_sr: int, device: str, dtype: torch.dtype, channel: int | None = None) -> torch.Tensor:
    """Load an audio file, resample to *target_sr*, and return as ``[1, num_samples]``.

    For stereo duplex audio: channel 0 = assistant, channel 1 = user.
    Use ``channel=1`` for user input, ``channel=0`` for speaker embedding source.

    Args:
        path: Path to the audio file.
        target_sr: Target sampling rate in Hz.
        device: Torch device string (e.g. ``"cuda"``).
        dtype: Torch dtype for the output tensor.
        channel: Channel index to extract from stereo audio. None = mono mix.

    Returns:
        Audio tensor of shape ``[1, num_samples]`` on *device* with *dtype*.
    """
    audio, _ = _load_audio_shared(
        path,
        target_sr,
        mono=channel is None,
        channel=channel,
        device=device,
        dtype=dtype,
    )
    return audio




def _resolve_dataset_path(path_value: str | None, jsonl_path: Path) -> str | None:
    if not path_value:
        return None
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return str(raw)
    candidates = [raw, jsonl_path.parent / raw, jsonl_path.parent.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def save_stereo_audio(
    user_np: np.ndarray,
    assistant_np: np.ndarray,
    sampling_rate: int,
    output_path: Path,
) -> None:
    """Write stereo audio (L=user, R=assistant) to a ``.wav`` file.

    Pads the shorter channel with zeros so both have equal length.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(user_np), len(assistant_np))
    user_padded = np.pad(user_np, (0, max_len - len(user_np)))
    assistant_padded = np.pad(assistant_np, (0, max_len - len(assistant_np)))
    stereo = np.stack([user_padded, assistant_padded], axis=-1)  # [samples, 2]
    sf.write(str(output_path), stereo, sampling_rate)


def save_summary(summary: dict, output_path: Path) -> None:
    """Write inference summary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_duplex_inference(
    model,
    processor: RaonProcessor,
    audio_input: torch.Tensor,
    output_dir: Path,
    system_prompt: str = "",
    do_sample: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.8,
    top_k: int = 20,
    speaker_embeds: torch.Tensor | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    eos_penalty: float = 0.0,
    sil_penalty: float = 0.0,
    bc_penalty: float = 0.0,
    speak_first: bool = False,
) -> dict:
    """Run full-duplex inference on a single audio input and save results.

    Performs frame-by-frame duplex decoding, saves assistant audio, stereo mix,
    decoded text, and a JSON summary to *output_dir*.

    Args:
        model: RAON duplex model (must support ``init_duplex_decoding_state``).
        processor: Processor with tokenizer and audio config.
        audio_input: User audio tensor of shape ``[1, num_samples]``.
        output_dir: Directory to save output files.
        system_prompt: Optional system prompt text.
        do_sample: Enable sampling.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
        top_k: Top-k filtering.
        speaker_embeds: Optional precomputed speaker embeddings.
        device: Torch device string.
        dtype: Torch dtype.

    Returns:
        Summary dict with durations and sample counts.
    """
    tokenizer = processor.tokenizer
    sr = processor.sampling_rate

    # Build system prompt tokens
    system_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    if system_messages:
        inputs = processor(system_messages, add_generation_prompt=False, device=device, dtype=dtype)
        system_tokens = inputs["input_ids"]
    else:
        system_tokens = torch.zeros((1, 0), dtype=torch.long, device=device)

    logger.info("Running duplex generation ...")
    samples_per_frame = int(sr / processor.frame_rate)
    audio_input_length = audio_input.shape[-1]

    if audio_input_length < samples_per_frame:
        raise ValueError(
            f"Audio input too short ({audio_input_length} samples) for duplex decoding "
            f"(minimum {samples_per_frame} samples = 1 frame at {sr}Hz / {processor.frame_rate}fps)."
        )

    with torch.inference_mode():
        state = model.init_duplex_decoding_state(
            sequences=system_tokens,
            attention_mask=torch.ones_like(system_tokens),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            speaker_embeds=speaker_embeds,
            eos_penalty=eos_penalty,
            sil_penalty=sil_penalty,
            bc_penalty=bc_penalty,
            speak_first=speak_first,
        )

        audio_output_frames: list[torch.Tensor] = []
        # [DEBUG-LOG] Frame-level logging for text-audio delay analysis
        _prev_seq_len = int(state.sequences.shape[1])
        _frame_log_path = output_dir / "frame_log.txt"
        _frame_log_path.parent.mkdir(parents=True, exist_ok=True)
        _frame_log = open(str(_frame_log_path), "w", buffering=1)
        _text_vocab_size = int(getattr(model, "text_vocab_size", 0) or 0)
        _frame_idx = 0
        try:
            for i in trange(
                0,
                audio_input_length - samples_per_frame + 1,
                samples_per_frame,
                mininterval=0,
                desc="Duplex Generation",
            ):
                audio_input_frame = audio_input[:, i : i + samples_per_frame]
                state, audio_output_frame = model.duplex_decoding_step(state=state, audio_input=audio_input_frame)
                audio_output_frames.append(audio_output_frame)

                # [DEBUG-LOG] Extract text delta and audio RMS for this frame
                _cur_seq_len = int(state.sequences.shape[1])
                _new_tokens = state.sequences[0, _prev_seq_len:_cur_seq_len].tolist()
                _text_tokens = [
                    t
                    for t in _new_tokens
                    if t < _text_vocab_size
                    and t
                    not in {
                        AUDIO_INPUT_PLACEHOLDER.id,
                        AUDIO_OUTPUT_PLACEHOLDER.id,
                        AUDIO_OUTPUT_PAD.id,
                        AUDIO_OUTPUT_END_PAD.id,
                        AUDIO_START.id,
                        IM_START.id,
                        IM_END.id,
                    }
                ]
                _text_str = tokenizer.decode(_text_tokens, skip_special_tokens=False) if _text_tokens else ""
                _out_rms = float(audio_output_frame.float().pow(2).mean().sqrt())
                _in_rms = float(audio_input_frame.float().pow(2).mean().sqrt())
                _phase = state.machine_state.phase.name if state.machine_state is not None else "?"
                _frame_log.write(
                    f"[{_phase}] f={_frame_idx} text={repr(_text_str) if _text_str else '-'} "
                    f"out_rms={_out_rms:.4f} in_rms={_in_rms:.4f} ntok={_cur_seq_len - _prev_seq_len}\n"
                )
                _prev_seq_len = _cur_seq_len
                _frame_idx += 1
        finally:
            _frame_log.close()
            logger.info("Saved frame log -> %s", _frame_log_path)
            model.free_duplex_decoding_state(state)

    audio_output = torch.cat(audio_output_frames, dim=1)  # [1, num_output_samples]

    # -- Save outputs --
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. assistant.wav — model-generated assistant audio (mono)
    assistant_np = audio_output[0].float().cpu().numpy()
    assistant_path = output_dir / "assistant.wav"
    save_audio(assistant_np, sr, assistant_path)
    logger.info(
        "Saved assistant audio: %d samples (%.2f sec) -> %s",
        len(assistant_np),
        len(assistant_np) / sr,
        assistant_path,
    )

    # 2. user_assistant.wav — stereo: L=user, R=assistant
    user_np = audio_input[0].float().cpu().numpy()
    stereo_path = output_dir / "user_assistant.wav"
    save_stereo_audio(user_np, assistant_np, sr, stereo_path)
    max_len = max(len(user_np), len(assistant_np))
    logger.info(
        "Saved stereo audio: %d samples (%.2f sec) -> %s",
        max_len,
        max_len / sr,
        stereo_path,
    )

    # 3. output.json — summary
    summary = {
        "assistant_duration_sec": len(assistant_np) / sr,
        "user_duration_sec": len(user_np) / sr,
        "assistant_samples": len(assistant_np),
        "user_samples": len(user_np),
        "sampling_rate": sr,
    }
    json_path = output_dir / "output.json"
    save_summary(summary, json_path)
    logger.info("Saved summary -> %s", json_path)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        output_root = Path(args.output_dir)
        jsonl_files = sorted(data_dir.rglob("*.jsonl"))
        if not jsonl_files:
            raise SystemExit(f"No JSONL files found under: {data_dir}")

        processed = 0
        for jsonl_path in jsonl_files:
            with open(jsonl_path, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    audio_path = _resolve_dataset_path(record.get("audio_path"), jsonl_path)
                    if not audio_path:
                        raise SystemExit(f"Missing audio_path in {jsonl_path}:{idx + 1}")

                    sample_out = output_root / f"{jsonl_path.stem}_{idx:05d}"
                    cmd = [
                        "python",
                        "-m",
                        "raon.duplex_generate",
                        "--model_path",
                        args.model_path,
                        "--audio_input",
                        audio_path,
                        "--output_dir",
                        str(sample_out),
                        "--device",
                        args.device,
                        "--dtype",
                        args.dtype,
                    ]
                    if args.config:
                        cmd += ["--config", args.config]

                    attn_arg = "fa" if args.attn_implementation == "flash_attention_2" else args.attn_implementation
                    cmd += ["--attn_implementation", attn_arg]

                    speaker_audio = args.speaker_audio or _resolve_dataset_path(
                        record.get("speaker_audio") or ((record.get("speaker_ref_audios") or [None])[0]),
                        jsonl_path,
                    )
                    if speaker_audio:
                        cmd += ["--speaker_audio", speaker_audio]

                    if isinstance(record.get("speak_first"), bool) and record["speak_first"]:
                        cmd.append("--speak_first")
                    if isinstance(record.get("persona"), str) and record["persona"].strip():
                        cmd += ["--persona", record["persona"].strip()]
                    if isinstance(record.get("context"), str) and record["context"].strip():
                        cmd += ["--context", record["context"].strip()]
                    if isinstance(record.get("system_prompt"), str) and record["system_prompt"].strip():
                        cmd += ["--system_prompt", record["system_prompt"].strip()]
                    if args.seed is not None:
                        cmd += ["--seed", str(args.seed)]

                    logger.info("[%s:%05d] %s", jsonl_path.name, idx, " ".join(shlex.quote(part) for part in cmd))
                    subprocess.run(cmd, check=True)
                    processed += 1

        logger.info("Processed %d duplex sample(s).", processed)
        return

    torch_dtype = resolve_dtype(args.dtype)
    output_dir = Path(args.output_dir)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info("Using RNG seed %d", args.seed)

    logger.info("Loading model from %s ...", args.model_path)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch_dtype, trust_remote_code=False)
    model._set_attention_implementation(args.attn_implementation)
    logger.info("Attention implementation: %s", args.attn_implementation)
    model = model.to(args.device).eval()

    logger.info("Loading processor from %s ...", args.model_path)
    processor = RaonProcessor.from_pretrained(args.model_path)

    # Load input audio: mono → pass through, stereo → average channels (matches duplex-model inference)
    logger.info("Loading audio from %s ...", args.audio_input)
    is_stereo = sf.info(args.audio_input).channels == 2
    if is_stereo:
        logger.info("Stereo detected: averaging channels to mono")
    audio_input = load_audio(args.audio_input, processor.sampling_rate, args.device, torch_dtype)
    logger.info(
        "Audio loaded: %d samples (%.2f sec)",
        audio_input.shape[1],
        audio_input.shape[1] / processor.sampling_rate,
    )

    # Compute speaker embeddings if --speaker_audio is provided.
    # Speaker conditioning injects an ECAPA-TDNN embedding into the decoding init state
    # via the speaker placeholder token. The embedding is baked into the KV cache at init
    # and conditions all subsequent frames. Note: effectiveness depends on the checkpoint —
    # early-stage checkpoints (e.g. v1-iter10k) may not exhibit strong voice adaptation.
    speaker_embeds = None
    if args.speaker_audio and hasattr(model, "speaker_encoder") and model.speaker_encoder is not None:
        logger.info("Computing speaker embeddings from %s ...", args.speaker_audio)
        speaker_audio = load_audio(args.speaker_audio, processor.sampling_rate, args.device, torch_dtype)
        speaker_embeds = model._compute_speaker_embeds(speaker_audio, None)
        logger.info("Speaker embeddings computed: shape %s", speaker_embeds.shape)

    run_duplex_inference(
        model=model,
        processor=processor,
        audio_input=audio_input,
        output_dir=output_dir,
        system_prompt=args.system_prompt,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        speaker_embeds=speaker_embeds,
        device=args.device,
        dtype=torch_dtype,
        eos_penalty=args.eos_penalty,
        sil_penalty=args.sil_penalty,
        bc_penalty=args.bc_penalty,
        speak_first=args.speak_first,
    )

    logger.info("Done. Output saved to %s", output_dir)


if __name__ == "__main__":
    main()

