"""Reusable experiment code for the MATS mechanistic-interpretability notebooks."""

from .controlled_posterior import (
    Answer,
    ControlledExample,
    ElicitationControl,
    ExperimentConfig,
    N8Config,
    Observation,
    Probe,
    build_elicitation_controls,
    build_ladder_examples,
    build_n8_examples,
    exact_posterior,
    messages_for_elicitation,
    messages_for_probe,
    serialize_example,
)

__all__ = [
    "Answer",
    "ControlledExample",
    "ElicitationControl",
    "ExperimentConfig",
    "N8Config",
    "Observation",
    "Probe",
    "build_elicitation_controls",
    "build_ladder_examples",
    "build_n8_examples",
    "exact_posterior",
    "messages_for_elicitation",
    "messages_for_probe",
    "serialize_example",
]
