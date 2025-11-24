import threading
import time
import io
from typing import Optional, Callable, Dict, Any

from PIL import Image

from rapido_system.api.cloudflare_streamer import (
    StreamConfig,
    LivePipelineSource,
    CloudflareStreamer,
)


class CloudflareStreamingService:
    """
    High-level wrapper to stream composed PNG frames and 40ms PCM audio chunks to Cloudflare.

    Usage:
        service = CloudflareStreamingService(
            rtmps_url="rtmps://live.cloudflare.com/live",
            stream_key="",  # leave empty when auto-creating
            auto_create=True,
            cloudflare_account_id="...",
            cloudflare_api_token="...",
            cloudflare_customer_code="...",  # optional, fallback HLS
            fps=25,
            video_bitrate="2500k",
            audio_bitrate="64k",
        )
        service.start(on_playback_update=cb, on_egress_complete=cb2)

        # Every 40 ms:
        service.submit_composed_frame(pil_image, pcm_40ms_bytes)

        service.finish()  # when done producing
        service.stop()
    """

    def __init__(
        self,
        *,
        rtmps_url: str,
        stream_key: str = "",
        auto_create: bool = False,
        cloudflare_account_id: Optional[str] = None,
        cloudflare_api_token: Optional[str] = None,
        cloudflare_customer_code: Optional[str] = None,
        fps: int = 25,
        width: int = 1920,
        height: int = 1080,
        video_bitrate: str = "2500k",
        audio_bitrate: str = "64k",
    ) -> None:
        self._config = StreamConfig(
            rtmps_url=rtmps_url,
            stream_key=stream_key,
            fps=fps,
            width=width,
            height=height,
            video_bitrate=video_bitrate,
            audio_bitrate=audio_bitrate,
            auto_create_live_input=auto_create,
            cloudflare_account_id=cloudflare_account_id,
            cloudflare_api_token=cloudflare_api_token,
            cloudflare_customer_code=cloudflare_customer_code,
        )

        # Large queue to avoid drops; submit_pair blocks when full
        self._source = LivePipelineSource(self._config, max_pairs=20000)
        self._streamer = CloudflareStreamer(self._config, self._source)

        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(
        self,
        *,
        on_playback_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_egress_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Start background streaming to Cloudflare."""
        if self._running:
            return

        # Attach callbacks if provided
        if on_playback_update is not None:
            self._config.on_playback_update = on_playback_update
        if on_egress_complete is not None:
            self._config.on_egress_complete = on_egress_complete

        self._running = True

        def _run():
            try:
                self._streamer.stream()
            finally:
                self._running = False

        self._thread = threading.Thread(target=_run, name="CloudflareStreamerThread", daemon=True)
        self._thread.start()

        # Give FFmpeg a moment to initialize
        time.sleep(0.2)

    def stop(self) -> None:
        """Stop streaming and clean up."""
        try:
            self._streamer.stop()
        finally:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3)
            self._running = False

    def finish(self) -> None:
        """Signal no more input; streamer will drain and then stop when you call stop()."""
        self._source.finish()

    def submit_pair(self, png_bytes: bytes, pcm_bytes: bytes) -> None:
        """Submit a synchronized pair. pcm_bytes will be padded/trimmed to 40ms if needed."""
        self._source.submit_pair(png_bytes, pcm_bytes)

    def submit_composed_frame(self, image: Image.Image, pcm_bytes: bytes) -> None:
        """Convert PIL image to PNG bytes and submit with the given 40ms PCM bytes."""
        png_bytes = self._pil_to_png(image)
        self.submit_pair(png_bytes, pcm_bytes)

    def get_playback_url(self) -> Optional[str]:
        """Return current playback URL if known."""
        return getattr(self._config, "playback_url", None)

    def _pil_to_png(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        # Ensure RGB, then encode as PNG
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.save(buf, format="PNG")
        return buf.getvalue()


