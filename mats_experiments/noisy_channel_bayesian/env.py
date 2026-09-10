from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .utils import as_fraction


@dataclass(frozen=True)
class NoisyChannelBayesianEnvironment:
    """Finite uniform domain and the reliability of each observed report."""

    n: int = 8
    k: int = 3
    r_values: int | float | str | Fraction | Sequence[int | float | str | Fraction] = Fraction(3, 4)
    control_positional_bias: bool = False

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive.")
        if self.k < 1:
            raise ValueError("k must be positive.")
        if not isinstance(self.control_positional_bias, bool):
            raise TypeError("control_positional_bias must be a bool.")
        values = self.r_values
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            resolved = tuple(as_fraction(value) for value in values)
            if len(resolved) != self.k:
                raise ValueError(f"Expected {self.k} reliability values, got {len(resolved)}.")
            shared = False
        else:
            resolved = (as_fraction(values),) * self.k
            shared = True
        object.__setattr__(self, "_reliabilities", resolved)
        object.__setattr__(self, "_shared_reliability", shared)

    @property
    def domain(self) -> tuple[int, ...]:
        return tuple(range(1, self.n + 1))

    @property
    def reliabilities(self) -> tuple[Fraction, ...]:
        return self._reliabilities  # type: ignore[attr-defined]

    @property
    def shared_reliability(self) -> bool:
        return self._shared_reliability  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CandidateEvidenceBayesianEnvironment(NoisyChannelBayesianEnvironment):
    """The same channel model presented through candidate-specific evidence relations."""
