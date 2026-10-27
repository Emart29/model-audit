"""Adapters for models that do not speak the scikit-learn protocol.

A PyTorch module returns logits, not probabilities, and has no ``predict_proba``.
Wrapping it here keeps that translation in one tested place instead of asking
every caller to write it.
"""

from __future__ import annotations

import numpy as np


class TorchClassifier:
    """Make a ``torch.nn.Module`` classifier look like a scikit-learn estimator."""

    def __init__(self, module, *, device: str = "cpu") -> None:
        self.module = module
        self.device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.module.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            logits = self.module(tensor)
            if logits.ndim == 1:
                positive = torch.sigmoid(logits)
                return torch.stack([1.0 - positive, positive], dim=1).cpu().numpy()
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
