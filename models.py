import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

PAD_ID = 0  # reserved
# time-series bins: [1..B]
# EOS token appended at end: B+1

@dataclass
class ChronosTokenizer:
    B: int                        # number of quantization bins
    c_min: float                  # smallest bin center
    c_max: float                  # largest bin center
    eps: float = 1e-8             # for safe division

    def _bin_centers_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform centers c_j in [c_min, c_max]; edges midpoints; outer edges at +/-inf."""
        centers = np.linspace(self.c_min, self.c_max, self.B, dtype=np.float64)
        # inner edges at midpoints; add -inf, +inf
        inner = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate(([-np.inf], inner, [np.inf]))
        return centers, edges

    def mean_scale(self, x: np.ndarray, C: int) -> Tuple[np.ndarray, float]:
        """
        Mean scaling: \tilde{x}_i = x_i / s with s = (1/C) * sum_{i=1}^C |x_i|.
        Preserves zeros. Uses eps to avoid division by 0.
        """
        assert C >= 1 and C <= len(x)
        s = np.mean(np.abs(x[:C])).astype(np.float64)
        s = max(s, self.eps)
        return (x / s).astype(np.float64), s

    def quantize(self, x_scaled: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Map real values to discrete ids in [1..B] using edges.
        Returns (ids, aux) where aux contains centers and edges for dequantization.
        """
        centers, edges = self._bin_centers_edges()
        # np.digitize returns indices in 1..len(edges)-1 for left-closed, right-open bins when right=False
        ids = np.digitize(x_scaled, edges[1:-1], right=False).astype(np.int64) + 1  # -> [1..B]
        return ids, {"centers": centers, "edges": edges}

    def dequantize(self, ids: np.ndarray, centers: Optional[np.ndarray] = None) -> np.ndarray:
        """Map ids in [1..B] back to centers c_j; PAD->np.nan; EOS ignored."""
        if centers is None:
            centers, _ = self._bin_centers_edges()
        y = np.empty_like(ids, dtype=np.float64)
        mask_pad = (ids == PAD_ID)
        mask_eos = (ids == (self.B + 1))
        mask_bins = ~(mask_pad | mask_eos)
        y[mask_pad] = np.nan
        y[mask_eos] = np.nan
        y[mask_bins] = centers[ids[mask_bins] - 1]
        return y

    def encode_series(
        self,
        x: np.ndarray,
        C: int,
        H: int,
        pad_to: Optional[int] = None,
        replace_nans_with_pad: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        x: shape (C+H,) real-valued series.
        C: number of context points
        H: number of future points
        Returns token ids including EOS; PAD used for NaNs and optional right-padding.
        """
        assert len(x) >= C + H
        x = x.astype(np.float64)
        if replace_nans_with_pad and np.isnan(x).any():
            # keep NaNs for scale computation by treating them as 0 contribution (Chronos pads missing with PAD)
            nan_mask = np.isnan(x)
            x_safe = x.copy()
            x_safe[nan_mask] = 0.0
        else:
            x_safe = x

        x_scaled, s = self.mean_scale(x_safe, C=C)
        token_ids, aux = self.quantize(x_scaled)
        eos_id = self.B + 1
        out = np.concatenate([token_ids, np.array([eos_id], dtype=np.int64)], axis=0)

        # replace places where original x was NaN with PAD
        if replace_nans_with_pad and np.isnan(x).any():
            pad_mask = np.isnan(x)
            pad_ids = np.where(pad_mask, PAD_ID, out[:-1])  # exclude EOS during masking
            out = np.concatenate([pad_ids, np.array([eos_id], dtype=np.int64)], axis=0)

        if pad_to is not None and pad_to > len(out):
            pad_len = pad_to - len(out)
            out = np.concatenate([out, np.full(pad_len, PAD_ID, dtype=np.int64)], axis=0)

        meta = {
            "scale_s": float(s),
            "B": int(self.B),
            "c_min": float(self.c_min),
            "c_max": float(self.c_max),
        }
        # store centers for exact inverse later if needed
        self._cached_centers = aux["centers"]
        return out, meta

    def decode_series(self, ids: np.ndarray, scale_s: float) -> np.ndarray:
        """
        Dequantize to centers, then invert scaling: x_hat = \tilde{x} * s.
        PAD/EOS -> NaN.
        """
        tilde = self.dequantize(ids, getattr(self, "_cached_centers", None))
        with np.errstate(invalid="ignore"):
            return tilde * scale_s

