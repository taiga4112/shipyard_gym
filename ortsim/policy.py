"""ONNX policy with optional recurrent state.

rsl_rl's recurrent exports take ``(obs, h_in[, c_in])`` and return
``(actions, h_out[, c_out])``. Every input after the first is treated as state and fed
back from the output in the same position, so GRU and LSTM exports both work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    def __init__(self, path: Path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.obs_name = inputs[0].name
        self.obs_dim = int(inputs[0].shape[-1])
        self.action_name = outputs[0].name
        self.action_dim = int(outputs[0].shape[-1])
        self.state_inputs = inputs[1:]
        self.state_outputs = [o.name for o in outputs[1 : 1 + len(self.state_inputs)]]
        if len(self.state_outputs) != len(self.state_inputs):
            raise ValueError(f"{path}: {len(self.state_inputs)} state inputs but fewer state outputs")
        self._state: list[np.ndarray] = []
        self.reset()

    @property
    def is_recurrent(self) -> bool:
        return bool(self.state_inputs)

    def reset(self) -> None:
        self._state = [
            np.zeros([d if isinstance(d, int) else 1 for d in i.shape], dtype=np.float32) for i in self.state_inputs
        ]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        feed = {self.obs_name: obs.astype(np.float32).reshape(1, -1)}
        feed.update({i.name: s for i, s in zip(self.state_inputs, self._state)})
        results = self.session.run([self.action_name, *self.state_outputs], feed)
        self._state = list(results[1:])
        return results[0].reshape(-1).astype(np.float64)
