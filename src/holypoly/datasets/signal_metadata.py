"""Shared raw-sample length checks for move-based dataset preparation."""


def signal_lengths_agree(
    raw_signal_length: int, dorado_signal_samples: int, move_stride: int
) -> bool:
    """Allow only Dorado's omitted final fractional move block, in raw samples.

    Both lengths use the original, untrimmed signal coordinate system. A full
    block of difference, or Dorado extending beyond POD5, is a mismatch.
    """
    return (
        raw_signal_length > 0
        and dorado_signal_samples > 0
        and move_stride > 0
        and 0 <= raw_signal_length - dorado_signal_samples < move_stride
    )
