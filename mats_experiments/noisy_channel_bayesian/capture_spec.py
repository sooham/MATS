from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CaptureSpec:
    """Tensor capture policy.

    ``logits_scope="answer_surfaces"`` stores only the resolved X/Y token logits
    inline in each result.  Full scope persists vocabulary tensors; ``logit_tokens``
    selects prompt positions, while ``every_decode_position`` switches capture to
    every generated position.  Activation ``tokens`` are selected independently.
    The special value ``"row_selected"`` reads an ``activation_token_selector``
    selector from each dataset row, enabling sparse semantically aligned capture.
    """

    logits_boundaries: tuple[str, ...] = ()
    logits_scope: Literal["full", "answer_surfaces"] = "full"
    logit_tokens: object = "last"
    streams: tuple[str, ...] = ()
    layers: object = "all"
    tokens: object = "last"
    every_decode_position: bool = False

    def __post_init__(self) -> None:
        if not set(self.logits_boundaries) <= {"answer"}:
            raise ValueError("logits_boundaries may contain only 'answer'.")
        if self.logits_scope not in {"full", "answer_surfaces"}:
            raise ValueError("logits_scope must be 'full' or 'answer_surfaces'.")
        valid_streams = {"resid_pre", "token_mixer_out", "mlp_out", "resid_post"}
        if not set(self.streams) <= valid_streams:
            raise ValueError(f"Unknown activation stream: {set(self.streams) - valid_streams}")

    @property
    def enabled(self) -> bool:
        return bool(self.logits_boundaries or self.streams or self.every_decode_position)
