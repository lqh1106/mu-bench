"""Public μ-Bench identity and compatibility constants."""

BENCHMARK_DISPLAY_NAME = "μ-Bench"
BENCHMARK_SLUG = "mu-bench"
HISTORICAL_BENCHMARK_VERSION = "v1.3.0"
BENCHMARK_VERSION = "v15.0"
RELEASE_DATE = "2026-11-13"
RELEASE_STATUS = "isolated-candidate-completion-r2-not-activated"
CASE_CATALOG_VERSION = "v15.0-tool-completion-20261113-r2"
EXECUTION_PROTOCOL_VERSION = "mu-bench-v1.3-spoken-self-introduction-adaptive-tail"
VOICE_PIPELINE_VERSION = "mu-bench-voice-v4-global-persona-cosyvoice3"
SCORING_CONTRACT_VERSION = "mu-bench-scoring-v15.0-tool-completion-20261113-r2"
RECOGNITION_SPLIT_CONTRACT_VERSION = "mu-bench-recognition-split-v1"
RUNTIME_TRANSPORT_VERSION = "mu-bench-runtime-v1.3-hybrid-control"
WEAK_COMPREHENSION_SCHEDULE_VERSION = "weak-repair-after-commit-v1"
# Prompt delivery is intentionally split by model family.  Closed-source API
# runners receive the benchmark protocol below; open-weight runners receive no
# benchmark-authored text and may use only the model's native default prompt.
CLOSED_SOURCE_SYSTEM_PROMPT_POLICY = "closed_source_benchmark_protocol_v1"
OPEN_WEIGHT_SYSTEM_PROMPT_POLICY = "model_native_default_or_none"
LEGACY_BENCHMARK_NAMES = ("τ-MU Voice", "tau-mu-voice")
LEGACY_IMPORT_PATH = "experiments.tau_multiuser_voice"
