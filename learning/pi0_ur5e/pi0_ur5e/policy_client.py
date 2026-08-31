from __future__ import annotations

import functools
import pickle
from pathlib import Path
import time
from typing import Any
import urllib.request

import numpy as np

from .transforms import clip_pi0_action


class DryRunPolicyClient:
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        return np.zeros((self.action_dim,), dtype=np.float32)


class PicklePolicyClient:
    """Minimal local policy wrapper for smoke tests.

    Real OpenPI inference should be launched from an OpenPI checkout; this class
    intentionally avoids importing OpenPI inside the robot repository.
    """

    def __init__(self, policy_path: str | Path):
        with open(policy_path, "rb") as f:
            self.policy = pickle.load(f)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if hasattr(self.policy, "act"):
            return np.asarray(self.policy.act(observation), dtype=np.float32)
        if callable(self.policy):
            return np.asarray(self.policy(observation), dtype=np.float32)
        raise TypeError("Loaded policy must be callable or expose act(observation)")


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for policy transport: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _load_transport_modules():
    try:
        import msgpack
        import websockets.sync.client
    except ImportError as exc:
        raise RuntimeError(
            "Network policy client requires msgpack and websockets. Install them on the robot computer with "
            "`pip install msgpack websockets` or `pip install -r requirements.txt`."
        ) from exc
    return msgpack, websockets.sync.client


def build_policy_observation(
    *,
    base_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    prompt: str | None = None,
    tactile_left_rgb: np.ndarray | None = None,
    tactile_right_rgb: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the raw observation expected by the pi0_ur5e OpenPI data transform."""
    obs: dict[str, Any] = {
        "base_rgb": np.asarray(base_rgb),
        "wrist_rgb": np.asarray(wrist_rgb),
        "state": np.asarray(state, dtype=np.float32),
    }
    if prompt is not None:
        obs["prompt"] = prompt
    if tactile_left_rgb is not None:
        obs["tactile_left_rgb"] = np.asarray(tactile_left_rgb)
    if tactile_right_rgb is not None:
        obs["tactile_right_rgb"] = np.asarray(tactile_right_rgb)
    return obs


class WebsocketPolicyClient:
    """Robot-side client for an OpenPI websocket policy server.

    This mirrors OpenPI's transport protocol without importing OpenPI, so the
    robot computer can stay light and only send observations over the network.
    """

    def __init__(
        self,
        host: str,
        port: int = 8000,
        *,
        api_key: str | None = None,
        reconnect: bool = True,
        retry_interval_s: float = 2.0,
        connect_timeout_s: float = 10.0,
    ):
        msgpack, websocket_client = _load_transport_modules()
        if host.startswith("ws://") or host.startswith("wss://"):
            self.uri = host if ":" in host.rsplit("/", 1)[-1] else f"{host}:{port}"
        else:
            self.uri = f"ws://{host}:{port}"
        self._packer = msgpack.Packer(default=_pack_array)
        self._unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
        self._websocket_client = websocket_client
        self._api_key = api_key
        self._reconnect = reconnect
        self._retry_interval_s = retry_interval_s
        self._connect_timeout_s = connect_timeout_s
        self._ws = None
        self._metadata: dict[str, Any] = {}
        self._connect()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _connect(self) -> None:
        deadline = time.monotonic() + self._connect_timeout_s
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        last_exc: Exception | None = None
        while True:
            try:
                kwargs = {"compression": None, "max_size": None}
                if headers is not None:
                    kwargs["additional_headers"] = headers
                try:
                    self._ws = self._websocket_client.connect(self.uri, **kwargs)
                except TypeError:
                    if headers is not None:
                        kwargs.pop("additional_headers", None)
                        kwargs["extra_headers"] = headers
                    self._ws = self._websocket_client.connect(self.uri, **kwargs)
                self._metadata = self._unpackb(self._ws.recv())
                return
            except Exception as exc:  # pragma: no cover - exact exception type depends on websockets version
                last_exc = exc
                if time.monotonic() >= deadline:
                    raise ConnectionError(f"Could not connect to OpenPI policy server at {self.uri}") from last_exc
                time.sleep(self._retry_interval_s)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            self._connect()
        try:
            self._ws.send(self._packer.pack(observation))
            response = self._ws.recv()
        except Exception:
            if not self._reconnect:
                raise
            self.close()
            self._connect()
            self._ws.send(self._packer.pack(observation))
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpackb(response)

    def action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        result = self.infer(observation)
        if "actions" not in result:
            raise KeyError(f"Policy response missing 'actions'. Keys: {sorted(result)}")
        return np.asarray(result["actions"], dtype=np.float32)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        actions = self.action_chunk(observation)
        if actions.ndim == 1:
            return actions.astype(np.float32)
        if actions.ndim >= 2 and actions.shape[0] > 0:
            return np.asarray(actions[0], dtype=np.float32)
        raise ValueError(f"Expected non-empty action array, got shape {actions.shape}")

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def reset(self) -> None:
        self.close()
        self._connect()


def check_policy_server_health(host: str, port: int = 8000, timeout_s: float = 2.0) -> bool:
    url_host = host.removeprefix("ws://").removeprefix("wss://")
    url_host = url_host.split("/", 1)[0].split(":", 1)[0]
    with urllib.request.urlopen(f"http://{url_host}:{port}/healthz", timeout=timeout_s) as response:
        return response.status == 200 and response.read().strip() == b"OK"


def safe_action(policy, observation: dict[str, Any], limits: dict[str, float]) -> np.ndarray:
    return clip_pi0_action(policy.act(observation), limits)
