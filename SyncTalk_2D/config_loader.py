import os
import json
from typing import Any, Dict, Optional
import requests


def default_avatar_configs() -> Dict[str, Any]:
    return {
        "enrique_torres": {
            "folder_name": "alin",  # Internal folder name for landmarks/assets
            "checkpoint": "./checkpoint/alin/99.pth",
            "video_path": "dataset/alin/alin.mp4",
            "crop_bbox": [596, 10, 596 + 600, 10 + 600],  # Optional - can be omitted
            "chroma_key": {
                "enabled": True,
                "target_color": "#089831",
                "default_background_url": "https://devprofjim.blob.core.windows.net/public-assets/fastavatar/replicate-prediction-zy9qkcebdnrma0cn1vmvrwm7m8.webp",
                "color_threshold": 35,
                "edge_blur": 0.08,
                "despill_factor": 0.5,
            },
        }
    }


def _fetch_remote_json(remote_url: str, timeout: float = 30.0, logger=None) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(remote_url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        if logger:
            logger.warning(f"Failed to fetch remote avatar config from {remote_url}: {e}")
        return None


def _read_local_json(path: str, logger=None) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to read local avatar config from {path}: {e}")
        return None


def load_avatar_configs(logger=None) -> Dict[str, Any]:
    """Load avatar configs from remote URL (if provided) with local fallback.
    Expects either a flat mapping of avatar name -> config, or an object with key 'avatars'.
    Synchronous implementation with optional httpx/requests usage.
    """
    remote_url = os.getenv("AVATAR_CONFIG_REMOTE_URL", "").strip()
    local_path = os.getenv("AVATAR_CONFIG_LOCAL_PATH", "./avatar_config.json").strip()

    loaded: Optional[Dict[str, Any]] = None

    if remote_url:
        loaded = _fetch_remote_json(remote_url, logger=logger)
        if loaded is not None and logger:
            logger.info(f"Loaded avatar configs from remote URL: {remote_url}")

    if loaded is None:
        loaded = _read_local_json(local_path, logger=logger)
        if loaded is not None and logger:
            logger.info(f"Loaded avatar configs from local file: {local_path}")

    if loaded is None:
        if logger:
            logger.warning("Falling back to built-in default avatar configs")
        loaded = {"avatars": default_avatar_configs()}

    # Normalize shape
    if "avatars" in loaded and isinstance(loaded["avatars"], dict):
        avatar_configs = loaded["avatars"]
    else:
        avatar_configs = loaded  # assume direct mapping

    # Ensure mapping of str -> dict
    if not isinstance(avatar_configs, dict):
        if logger:
            logger.warning("Avatar config root is not a dict; using defaults")
        return default_avatar_configs()

    return avatar_configs


