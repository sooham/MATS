from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .constants import SOFTMAX_LOG_BASE, SOFTMAX_LOG_UNIT
from .utils import fraction_text, natural_log_ratio

CallLayout = Literal["conversation", "replay_user"]

@dataclass(frozen=True)
class XVsYPosteriorProbe:
    x: int = 2
    y: int = 7
    reasoning: bool = False
    allow_same: bool = False
    call_layout: CallLayout = "conversation"
    answer_prefix: str = "ANSWER:"

    def validate(self, n: int) -> None:
        if self.x == self.y:
            raise ValueError("x and y must be distinct; allow_same only controls tie answers.")
        if self.x not in range(1, n + 1) or self.y not in range(1, n + 1):
            raise ValueError(f"x and y must both lie in 1..{n}.")
        if not isinstance(self.reasoning, bool):
            raise TypeError("reasoning must be a bool.")
        if self.call_layout not in ("conversation", "replay_user"):
            raise ValueError("call_layout must be 'conversation' or 'replay_user'.")
        if (
            not self.answer_prefix
            or self.answer_prefix != self.answer_prefix.strip()
            or "\n" in self.answer_prefix
        ):
            raise ValueError(
                "answer_prefix must be non-empty, single-line, and have no surrounding whitespace."
            )

def _posterior_target_fields(
    *,
    posterior: Mapping[int, Fraction] | None,
    probe: XVsYPosteriorProbe,
) -> dict[str, object]:
    if posterior is None:
        return {
            "posterior_exact": None,
            "posterior": None,
            "x_posterior_exact": None,
            "y_posterior_exact": None,
            "x_posterior": None,
            "y_posterior": None,
            "posterior_difference": None,
            "posterior_log_odds": None,
            "posterior_log_odds_base": SOFTMAX_LOG_BASE,
            "posterior_log_odds_unit": SOFTMAX_LOG_UNIT,
            "ground_truth_choice": None,
            "normative_comparison": None,
        }
    x_probability = posterior[probe.x]
    y_probability = posterior[probe.y]
    if x_probability > y_probability:
        comparison = "X"
    elif y_probability > x_probability:
        comparison = "Y"
    else:
        comparison = "SAME"
    ground_truth = comparison if comparison != "SAME" or probe.allow_same else None
    if x_probability > 0 and y_probability > 0:
        log_odds: float | None = natural_log_ratio(x_probability, y_probability)
    elif x_probability == y_probability:
        log_odds = 0.0
    else:
        log_odds = None
    return {
        "posterior_exact": {
            str(candidate): fraction_text(probability)
            for candidate, probability in posterior.items()
        },
        "posterior": {
            str(candidate): float(probability) for candidate, probability in posterior.items()
        },
        "x_posterior_exact": fraction_text(x_probability),
        "y_posterior_exact": fraction_text(y_probability),
        "x_posterior": float(x_probability),
        "y_posterior": float(y_probability),
        "posterior_difference": float(x_probability - y_probability),
        "posterior_log_odds": log_odds,
        "posterior_log_odds_base": SOFTMAX_LOG_BASE,
        "posterior_log_odds_unit": SOFTMAX_LOG_UNIT,
        "ground_truth_choice": ground_truth,
        "normative_comparison": comparison,
    }
