"""Exhaustive Bayesian transcript framework for a noisy YES/NO channel."""

from .capture_spec import CaptureSpec
from .captures import (
    get_activation,
    get_answer_surface_logits,
    get_logits,
    load_activation_tensors,
    load_logit_tensors,
)
from .constants import SOFTMAX_LOG_BASE, SOFTMAX_LOG_UNIT
from .core import (
    MetricSpec,
    TokenizerBinding,
    derive_candidate_evidence,
    render_candidate_evidence_prompt,
    render_observable_prompt,
)
from .dataset import (
    AgreementTranscriptDatasetGenerator,
    CandidateEvidenceDatasetGenerator,
    TranscriptDataset,
    TranscriptDatasetGenerator,
    exact_pattern_mass,
    summarize_representation_control,
)
from .env import (
    CandidateEvidenceBayesianEnvironment,
    NoisyChannelBayesianEnvironment,
)
from .probes import XVsYPosteriorProbe
from .questions import (
    AgreementSubsetQuestion,
    CandidateEvidenceQuestion,
    FixedSubsetQuestion,
    RandomSubsetQuestion,
    agreement_pattern_text,
    agreement_patterns,
)
from .runner import (
    ExecutionConfig,
    ModelConfig,
    QwenRunner,
    SGLangMTPConfig,
    parse_model_choice,
    resolve_selector,
    select_unpadded_tokens,
)
from .utils import (
    SystemPrompt,
    answer_patterns,
    candidate_agreements,
    exact_bayesian_target,
    natural_log_ratio,
)

__all__ = [
    "SOFTMAX_LOG_BASE",
    "SOFTMAX_LOG_UNIT",
    "AgreementSubsetQuestion",
    "AgreementTranscriptDatasetGenerator",
    "CandidateEvidenceBayesianEnvironment",
    "CandidateEvidenceDatasetGenerator",
    "CandidateEvidenceQuestion",
    "CaptureSpec",
    "ExecutionConfig",
    "FixedSubsetQuestion",
    "MetricSpec",
    "ModelConfig",
    "NoisyChannelBayesianEnvironment",
    "QwenRunner",
    "RandomSubsetQuestion",
    "SGLangMTPConfig",
    "SystemPrompt",
    "TokenizerBinding",
    "TranscriptDataset",
    "TranscriptDatasetGenerator",
    "XVsYPosteriorProbe",
    "agreement_pattern_text",
    "agreement_patterns",
    "answer_patterns",
    "candidate_agreements",
    "derive_candidate_evidence",
    "exact_bayesian_target",
    "exact_pattern_mass",
    "get_activation",
    "get_answer_surface_logits",
    "get_logits",
    "load_activation_tensors",
    "load_logit_tensors",
    "natural_log_ratio",
    "parse_model_choice",
    "render_candidate_evidence_prompt",
    "render_observable_prompt",
    "resolve_selector",
    "select_unpadded_tokens",
    "summarize_representation_control",
]
