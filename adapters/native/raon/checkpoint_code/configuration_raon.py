# AUTO-GENERATED — do not edit manually. Run build_hub_files.py to regenerate.
from __future__ import annotations

from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig, MimiConfig, Qwen3Config
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTextConfig,
)

# ── from modules/embedding.py ──

class EmbeddingAdaptorConfig(PretrainedConfig):
    """Configuration for EmbeddingAdaptor.

    Controls the projection from audio encoder embeddings to LM embedding space,
    including the time-scale ratio, MLP depth, optional transformer decoder, and
    optional post-projection RMSNorm.

    Args:
        input_size: Feature dimension of the encoder output (e.g. 512 for Mimi).
        output_size: Feature dimension expected by the LM (e.g. 4096 for Qwen3-7B).
        output_time_scale: Ratio of output frames to input frames. Values >= 1
            upsample (expand time); values < 1 downsample (compress time).
            Must be a reciprocal integer in either direction.
        num_layers: Number of MLP layers (1 or 2). Ignored in transformer mode.
        hidden_size: Hidden dimension for the 2-layer MLP. Defaults to output_size.
        decoder_config: If provided, uses a lightweight Qwen3 transformer instead
            of an MLP for the adaptor projection.
        use_post_norm: If True, apply RMSNorm to the output embeddings.
        norm_eps: Epsilon for RMSNorm.
        post_norm_init_scale: If set, initialize RMSNorm weight to this value
            (useful for residual scaling at initialisation).
    """

    model_type = "embedding_adaptor"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 4096,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: dict[str, Any] | Qwen3Config | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        post_norm_init_scale: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_post_norm = use_post_norm
        self.norm_eps = norm_eps
        self.post_norm_init_scale = post_norm_init_scale

        # Parse decoder_config for transformer adaptor mode
        if isinstance(decoder_config, dict):
            decoder_config = Qwen3Config(**decoder_config)
        self.decoder_config = decoder_config


# ── from modules/speaker_encoder.py ──

class SpeakerEncoderConfig(PretrainedConfig):
    """Configuration for SpeakerEncoder: input/output sizes, attention heads, and frame window."""

    model_type = "speaker_encoder"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 4096,
        num_heads: int = 8,
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        frame_rate: float = 12.5,
        encoder_type: str = "from_scratch",
        pretrained_model_id: str | None = None,
        pretrained_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.frame_rate = frame_rate
        self.encoder_type = encoder_type
        self.pretrained_model_id = pretrained_model_id
        self.pretrained_dim = pretrained_dim


# ── from modules/voxtral_encoder.py ──

class VoxtralRealtimeEncoderConfig(PretrainedConfig):
    """Configuration for the Voxtral Realtime audio encoder.

    Stores both the encoder architecture parameters and the projector/downsample
    settings needed to reconstruct the full audio pipeline.
    """

    model_type = "voxtral_realtime_encoder"

    def __init__(
        self,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        activation_function: str = "gelu",
        num_mel_bins: int = 128,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1500,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        sliding_window: int = 750,
        head_dim: int = 64,
        downsample_factor: int = 4,
        projector_hidden_act: str = "gelu",
        projector_output_size: int | None = None,
        output_embedding_scale: float = 1.0,
        skip_projector: bool = False,
        attn_implementation: str = "eager",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.downsample_factor = downsample_factor
        self.projector_hidden_act = projector_hidden_act
        self.projector_output_size = projector_output_size
        self.output_embedding_scale = output_embedding_scale
        self.skip_projector = skip_projector
        self._attn_implementation = attn_implementation

        # Aliases expected by the encoder layers.
        self.encoder_layers = num_hidden_layers
        self.encoder_attention_heads = num_attention_heads

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "VoxtralRealtimeEncoderConfig":
        """Load config from a Voxtral Realtime checkpoint.

        Reads ``config.json`` and extracts the ``audio_config`` sub-dict along
        with top-level ``downsample_factor``, ``projector_hidden_act``, and
        ``text_config.hidden_size`` (used as ``projector_output_size``).

        Works with both the full ``voxtral_realtime`` model config and a
        standalone ``voxtral_realtime_encoder`` config.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.

        Returns:
            Populated ``VoxtralRealtimeEncoderConfig``.
        """
        import json
        import os

        from huggingface_hub import hf_hub_download

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
            )

        with open(config_path) as f:
            full_config = json.load(f)

        # If this is the full model config, extract the audio sub-config.
        if "audio_config" in full_config:
            audio_cfg = full_config["audio_config"]
            downsample_factor = full_config.get("downsample_factor", 4)
            projector_hidden_act = full_config.get("projector_hidden_act", "gelu")
            text_hidden_size = full_config.get("text_config", {}).get("hidden_size")
        else:
            # Standalone encoder config (e.g. saved by us).
            audio_cfg = full_config
            downsample_factor = audio_cfg.get("downsample_factor", 4)
            projector_hidden_act = audio_cfg.get("projector_hidden_act", "gelu")
            text_hidden_size = audio_cfg.get("projector_output_size")

        # The upstream rope_theta lives inside rope_parameters.
        rope_params = audio_cfg.get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta", audio_cfg.get("rope_theta", 10000.0))

        return cls(
            hidden_size=audio_cfg.get("hidden_size", 1280),
            intermediate_size=audio_cfg.get("intermediate_size", 5120),
            num_hidden_layers=audio_cfg.get("num_hidden_layers", 32),
            num_attention_heads=audio_cfg.get("num_attention_heads", 32),
            num_key_value_heads=audio_cfg.get("num_key_value_heads"),
            activation_function=audio_cfg.get("activation_function", "gelu"),
            num_mel_bins=audio_cfg.get("num_mel_bins", 128),
            initializer_range=audio_cfg.get("initializer_range", 0.02),
            attention_dropout=audio_cfg.get("attention_dropout", 0.0),
            hidden_act=audio_cfg.get("hidden_act", "silu"),
            max_position_embeddings=audio_cfg.get("max_position_embeddings", 1500),
            rms_norm_eps=audio_cfg.get("rms_norm_eps", 1e-5),
            rope_theta=rope_theta,
            sliding_window=audio_cfg.get("sliding_window", 750),
            head_dim=audio_cfg.get("head_dim", 64),
            downsample_factor=downsample_factor,
            projector_hidden_act=projector_hidden_act,
            projector_output_size=text_hidden_size,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Conv1d padding cache (for streaming)
# ---------------------------------------------------------------------------


# ── from models/raon.py ──

TEXT_MODEL_CONFIGS: dict[str, type[PretrainedConfig]] = {
    Qwen3Config.model_type: Qwen3Config,
}


class RaonConfig(PretrainedConfig):
    """Configuration class for RaonModel."""

    model_type = "raon"
    has_no_defaults_at_init = True
    text_model_config: PretrainedConfig = None
    audio_encoder_config: Qwen3OmniMoeAudioEncoderConfig | VoxtralRealtimeEncoderConfig = None
    audio_tokenizer_config: MimiConfig = None
    input_adaptor_config: EmbeddingAdaptorConfig = None
    output_adaptor_config: EmbeddingAdaptorConfig = None
    code_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig = None
    speaker_encoder_config: SpeakerEncoderConfig | None = None
    # Note: speaker_encoder_config is intentionally excluded from sub_configs.
    # It is optional (can be None), and transformers' _get_dtype unconditionally
    # calls sub_config.dtype on every entry, which crashes on None.
    # Deserialization from dict is handled in __init__ instead.
    sub_configs = {
        "text_model_config": PretrainedConfig,
        "audio_encoder_config": PretrainedConfig,
        "audio_tokenizer_config": PretrainedConfig,
        "input_adaptor_config": EmbeddingAdaptorConfig,
        "output_adaptor_config": EmbeddingAdaptorConfig,
        "code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig,
    }

    def __init__(
        self,
        *,
        text_model_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_encoder_config: dict[str, Any] | Qwen3OmniMoeAudioEncoderConfig | VoxtralRealtimeEncoderConfig | None = None,
        audio_tokenizer_config: dict[str, Any] | MimiConfig | None = None,
        input_adaptor_config: dict[str, Any] | EmbeddingAdaptorConfig | None = None,
        output_adaptor_config: dict[str, Any] | EmbeddingAdaptorConfig | None = None,
        code_predictor_config: dict[str, Any] | Qwen3OmniMoeTalkerCodePredictorConfig | None = None,
        speaker_encoder_config: dict[str, Any] | SpeakerEncoderConfig | None = None,
        num_talker_layers: int = 0,
        supports_audio_input: bool = True,
        supports_audio_output: bool = True,
        aut_is_causal: bool = False,
        proj_code_bias: bool = False,
        accept_hidden_layer: int = -1,
        talker_config: dict[str, Any] | PretrainedConfig | None = None,
        thinker_to_talker_pre_norm: bool = False,
        sequence_mode: str | None = None,
        use_sil_token: bool = False,
        no_audio_in_sil: bool = False,
        text_lookahead: int = 0,
        use_duplex_end_pad: bool = False,
        speaker_embedding_to_code_predictor: bool = True,
        duplex_pad_token_id: int | None = None,
        duplex_end_pad_token_id: int | None = None,
        duplex_sil_token_id: int | None = None,
        duplex_bc_token_id: int | None = None,
        use_backchannel_token: bool = False,
        bc_loss_weight: float = 1.0,
        speaker_token_id: int | None = None,
        audio_input_token_id: int | None = None,
        audio_output_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        im_start_token_id: int | None = None,
        text_loss_weight: float = 1.0,
        sil_loss_weight: float = 1.0,
        epad_loss_weight: float = 0.0,
        semantic_loss_weight: float = 1.0,
        acoustic_loss_weights: list[float] | None = None,
        audio_lm_head_enabled: bool = True,
        delays: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Ensure auto_map is always serialized for trust_remote_code Hub loading.
        if not hasattr(self, "auto_map") or not self.auto_map:
            self.auto_map = {
                "AutoConfig": "configuration_raon.RaonConfig",
                "AutoModel": "modeling_raon.RaonModel",
            }

        assert text_model_config is not None, "RaonConfig: `text_model_config` is required."
        assert audio_encoder_config is not None, "RaonConfig: `audio_encoder_config` is required."
        assert audio_tokenizer_config is not None, "RaonConfig: `audio_tokenizer_config` is required."
        assert input_adaptor_config is not None, "RaonConfig: `input_adaptor_config` is required."
        assert output_adaptor_config is not None, "RaonConfig: `output_adaptor_config` is required."
        assert code_predictor_config is not None, "RaonConfig: `code_predictor_config` is required."

        if isinstance(text_model_config, dict):
            model_type = text_model_config.get("model_type", Qwen3Config.model_type)
            text_model_config = TEXT_MODEL_CONFIGS[model_type](**text_model_config)

        # Convert sub-configs from dict or generic PretrainedConfig to specific types.
        # The generic PretrainedConfig case occurs when transformers' sub_configs mechanism
        # auto-deserializes before __init__ runs (e.g. with trust_remote_code Hub loading).
        def _to_dict(cfg: Any) -> dict[str, Any]:
            """Convert a config to dict, handling both dict and PretrainedConfig."""
            if isinstance(cfg, dict):
                return cfg
            return cfg.to_dict()

        if isinstance(audio_encoder_config, dict) or (
            isinstance(audio_encoder_config, PretrainedConfig)
            and not isinstance(audio_encoder_config, (Qwen3OmniMoeAudioEncoderConfig, VoxtralRealtimeEncoderConfig))
        ):
            d = _to_dict(audio_encoder_config)
            model_type = d.get("model_type", Qwen3OmniMoeAudioEncoderConfig.model_type)
            if model_type == Qwen3OmniMoeAudioEncoderConfig.model_type:
                audio_encoder_config = Qwen3OmniMoeAudioEncoderConfig(**d)
            elif model_type == "voxtral_realtime_encoder":
                audio_encoder_config = VoxtralRealtimeEncoderConfig(**d)
            else:
                raise ValueError(
                    f"Unsupported audio_encoder model_type: {model_type!r}. "
                    "Expected 'qwen3_omni_moe_audio_encoder' or 'voxtral_realtime_encoder'."
                )

        if isinstance(audio_tokenizer_config, dict) or (
            isinstance(audio_tokenizer_config, PretrainedConfig) and not isinstance(audio_tokenizer_config, MimiConfig)
        ):
            audio_tokenizer_config = MimiConfig(**_to_dict(audio_tokenizer_config))

        if isinstance(input_adaptor_config, dict) or (
            isinstance(input_adaptor_config, PretrainedConfig)
            and not isinstance(input_adaptor_config, EmbeddingAdaptorConfig)
        ):
            input_adaptor_config = EmbeddingAdaptorConfig(**_to_dict(input_adaptor_config))

        if isinstance(output_adaptor_config, dict) or (
            isinstance(output_adaptor_config, PretrainedConfig)
            and not isinstance(output_adaptor_config, EmbeddingAdaptorConfig)
        ):
            output_adaptor_config = EmbeddingAdaptorConfig(**_to_dict(output_adaptor_config))

        if isinstance(code_predictor_config, dict) or (
            isinstance(code_predictor_config, PretrainedConfig)
            and not isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig)
        ):
            code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**_to_dict(code_predictor_config))

        if isinstance(speaker_encoder_config, dict) or (
            isinstance(speaker_encoder_config, PretrainedConfig)
            and not isinstance(speaker_encoder_config, SpeakerEncoderConfig)
        ):
            speaker_encoder_config = SpeakerEncoderConfig(**_to_dict(speaker_encoder_config))

        if isinstance(talker_config, dict) or (
            isinstance(talker_config, PretrainedConfig) and type(talker_config) is PretrainedConfig
        ):
            d = _to_dict(talker_config) if talker_config is not None else {}
            talker_model_type = d.get("model_type", Qwen3Config.model_type)
            talker_config = TEXT_MODEL_CONFIGS[talker_model_type](**d)

        assert isinstance(
            audio_encoder_config, (Qwen3OmniMoeAudioEncoderConfig, VoxtralRealtimeEncoderConfig, MimiConfig)
        ), "audio_encoder_config must be Qwen3OmniMoeAudioEncoderConfig, VoxtralRealtimeEncoderConfig, or MimiConfig."
        assert isinstance(audio_tokenizer_config, MimiConfig), "audio_tokenizer_config must be MimiConfig."
        assert isinstance(input_adaptor_config, EmbeddingAdaptorConfig), (
            "input_adaptor_config must be EmbeddingAdaptorConfig."
        )
        assert isinstance(output_adaptor_config, EmbeddingAdaptorConfig), (
            "output_adaptor_config must be EmbeddingAdaptorConfig."
        )
        assert isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig), (
            "code_predictor_config must be Qwen3OmniMoeTalkerCodePredictorConfig."
        )
        assert isinstance(text_model_config, PretrainedConfig), "text_model_config must be PretrainedConfig."
        assert speaker_encoder_config is None or isinstance(speaker_encoder_config, SpeakerEncoderConfig), (
            "speaker_encoder_config must be None or SpeakerEncoderConfig."
        )

        self.text_model_config = text_model_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.input_adaptor_config = input_adaptor_config
        self.output_adaptor_config = output_adaptor_config
        self.code_predictor_config = code_predictor_config
        self.speaker_encoder_config = speaker_encoder_config
        self.num_talker_layers = num_talker_layers
        self.supports_audio_input = supports_audio_input
        self.supports_audio_output = supports_audio_output
        self.aut_is_causal = aut_is_causal
        self.proj_code_bias = proj_code_bias
        self.accept_hidden_layer = accept_hidden_layer
        self.talker_config = talker_config
        self.thinker_to_talker_pre_norm = thinker_to_talker_pre_norm
        self.sequence_mode = sequence_mode
        self.use_sil_token = use_sil_token
        self.no_audio_in_sil = no_audio_in_sil
        self.text_lookahead = int(text_lookahead)
        self.use_duplex_end_pad = use_duplex_end_pad
        self.speaker_embedding_to_code_predictor = speaker_embedding_to_code_predictor
        self.duplex_pad_token_id = duplex_pad_token_id
        self.duplex_end_pad_token_id = duplex_end_pad_token_id
        self.duplex_sil_token_id = duplex_sil_token_id
        self.duplex_bc_token_id = duplex_bc_token_id
        self.use_backchannel_token = use_backchannel_token
        self.bc_loss_weight = bc_loss_weight
        self.speaker_token_id = speaker_token_id
        self.audio_input_token_id = audio_input_token_id
        self.audio_output_token_id = audio_output_token_id
        self.audio_start_token_id = audio_start_token_id
        self.im_start_token_id = im_start_token_id
        self.text_loss_weight = text_loss_weight
        self.sil_loss_weight = sil_loss_weight
        self.epad_loss_weight = epad_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.acoustic_loss_weights = acoustic_loss_weights
        self.audio_lm_head_enabled = audio_lm_head_enabled
        self.delays = delays

        if supports_audio_output and audio_lm_head_enabled:
            assert talker_config is not None, "RaonConfig: `talker_config` is required when audio output is enabled."
            assert num_talker_layers > 0, "RaonConfig: `num_talker_layers` must be positive when audio output is enabled."

    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        return {}

    def to_diff_dict(self) -> dict[str, Any]:
        """Return config as a dict suitable for diffing."""
        return self.to_dict()


class RaonDuplexConfig(RaonConfig):
    """Configuration alias for full-duplex checkpoints (model_type='raon_duplex')."""

    model_type = "raon_duplex"

    def __init__(self, **kwargs: Any) -> None:
        # Duplex-specific defaults; overridden by values in config.json when present.
        kwargs.setdefault("sequence_mode", "uta")
        kwargs.setdefault("use_sil_token", True)
        kwargs.setdefault("no_audio_in_sil", False)
        kwargs.setdefault("text_lookahead", 0)
        kwargs.setdefault("use_duplex_end_pad", True)
        kwargs.setdefault("duplex_pad_token_id", 151677)
        kwargs.setdefault("duplex_end_pad_token_id", 151678)
        kwargs.setdefault("duplex_sil_token_id", 151672)
        kwargs.setdefault("duplex_bc_token_id", 151673)
        kwargs.setdefault("speaker_token_id", 151671)
        kwargs.setdefault("audio_input_token_id", 151676)
        kwargs.setdefault("audio_output_token_id", 151675)
        kwargs.setdefault("audio_start_token_id", 151669)
        kwargs.setdefault("im_start_token_id", 151644)
        # Loss weight defaults (overridable at training time via duplex_train args)
        kwargs.setdefault("text_loss_weight", 1.0)
        kwargs.setdefault("sil_loss_weight", 1.0)
        kwargs.setdefault("epad_loss_weight", 0.0)
        kwargs.setdefault("semantic_loss_weight", 1.0)
        kwargs.setdefault("acoustic_loss_weights", None)
        super().__init__(**kwargs)
        self.auto_map = {
            "AutoConfig": "configuration_raon.RaonDuplexConfig",
            "AutoModel": "modeling_raon.RaonDuplexModel",
        }


# Duplex model — same architecture, different model_type for HF registry

