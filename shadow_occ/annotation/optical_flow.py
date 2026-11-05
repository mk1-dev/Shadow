"""Existing optical-flow engines, kept separate from shadow evidence logic."""
from __future__ import annotations

import cv2
import numpy as np


class DenseFlow:
    def __init__(self, backend="dis", *, weights=None, device="cpu"):
        self.backend = backend
        if backend == "dis":
            if weights is not None or device != "cpu":
                raise ValueError("DIS is a CPU baseline and does not use weights")
            self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.engine.setUseSpatialPropagation(True)
        elif backend == "raft-small":
            if weights is None:
                raise ValueError("RAFT requires an explicit local torchvision raft_small state dict; no downloads")
            import torch
            from torchvision.models.optical_flow import raft_small

            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for flow but unavailable; CPU fallback is disabled")
            self.engine = raft_small(weights=None)
            self.engine.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            self.engine.to(self.device).eval()
        else:
            raise ValueError(f"Unknown optical-flow backend: {backend}")

    def __call__(self, first_bgr, second_bgr):
        if first_bgr.shape != second_bgr.shape:
            raise ValueError("Optical-flow images must have identical dimensions")
        if self.backend == "dis":
            first = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
            second = cv2.cvtColor(second_bgr, cv2.COLOR_BGR2GRAY)
            return self.engine.calc(first, second, None).astype(np.float32)
        import torch
        import torch.nn.functional as F

        height, width = first_bgr.shape[:2]
        if min(height, width) < 128:
            raise ValueError("RAFT requires height and width >= 128")

        def tensor(image):
            rgb = np.ascontiguousarray(image[..., ::-1])
            value = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)/127.5-1
            return F.pad(value, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")

        with torch.inference_mode():
            flow = self.engine(tensor(first_bgr), tensor(second_bgr))[-1]
        return flow[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
