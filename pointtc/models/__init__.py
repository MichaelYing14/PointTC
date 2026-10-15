"""Checkpoint-compatible PointTC models for 2,048-point ShapeNet inputs."""

from importlib import import_module

METHODS = ("pointtc", "ca", "ca_ra")
METHOD_NAMES = {"pointtc": "PointTC", "ca": "PointTC+CA", "ca_ra": "PointTC+CA+RA"}
CHANNELS = ("awgn", "rayleigh")
SYMBOL_SPLITS = {
    300: (200, 100),
    250: (200, 50),
    200: (160, 40),
    150: (120, 30),
    100: (80, 20),
    50: (40, 10),
}


def symbol_split(num_symbols):
    """Return the main/auxiliary branch sizes for a supported symbol budget."""
    if type(num_symbols) is not int or num_symbols not in SYMBOL_SPLITS:
        raise ValueError(
            f"num_symbols must be one of {tuple(SYMBOL_SPLITS)} as an integer; "
            f"got {num_symbols!r}."
        )
    return SYMBOL_SPLITS[num_symbols]


def build_model(method="ca_ra", channel="awgn", num_symbols=300):
    """Build a 64-QAM architecture with a supported total symbol budget.

    The model returns ``(reconstruction, chamfer_distance)``; ``ca_ra`` also
    returns the mean number of transmitted complex symbols. During training,
    ``pointtc`` takes a scalar SNR and the adaptive methods take a two-value
    SNR interval. All methods take a scalar SNR during evaluation.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}.")
    if channel not in CHANNELS:
        raise ValueError(f"Unknown channel {channel!r}; choose one of {CHANNELS}.")
    symbol_split(num_symbols)
    module = import_module(f".{method}_{channel}", __name__)
    return module.get_model(
        revise=True, bottleneck_size=num_symbols, recon_points=2048, M_QAM=64
    )


__all__ = [
    "METHODS", "METHOD_NAMES", "CHANNELS", "SYMBOL_SPLITS", "symbol_split", "build_model"
]
