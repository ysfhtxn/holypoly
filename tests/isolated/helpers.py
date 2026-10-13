"""Shared synthetic model factories; no real signals or model weights."""

from bonito.crf.model import CTC_CRF, SeqdistModel
from bonito.nn import (
    Convolution,
    LinearCRFEncoder,
    LinearUpsample,
    NamedSerial,
    Permute,
    Serial,
)
from torch import nn


def tiny_v5_basecaller() -> SeqdistModel:
    """Create the deterministic test architecture without loading model weights."""
    encoder = NamedSerial(
        {
            "conv": Serial(
                [
                    Convolution(1, 8, 3, stride=3, padding=1, activation="swish"),
                    Convolution(8, 8, 3, stride=2, padding=1, activation="swish"),
                    Convolution(8, 8, 3, stride=2, padding=1, activation="swish"),
                    Permute([0, 2, 1]),
                ]
            ),
            "transformer_encoder": nn.Identity(),
            "upsample": LinearUpsample(d_model=8, scale_factor=2),
            "crf": LinearCRFEncoder(
                insize=8,
                n_base=4,
                state_len=1,
                bias=False,
                blank_score=2.0,
                permute=[1, 0, 2],
            ),
        }
    )
    return SeqdistModel(encoder, CTC_CRF(1, ["N", "A", "C", "G", "T"]))
