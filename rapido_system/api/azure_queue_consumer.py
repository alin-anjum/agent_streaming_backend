import asyncio
import json
import logging
import os
from typing import Callable, Awaitable, Optional

try:
    from azure.storage.queue.aio import QueueClient
except Exception:  # pragma: no cover - library may not be installed yet
    QueueClient = None  # type: ignore


logger = logging.getLogger(__name__)


# Load from environment variables
AZURE_QUEUE_CONNECTION_STRING = os.getenv(
    "AZURE_QUEUE_CONNECTION_STRING",
    ""
)
AZURE_QUEUE_NAME = os.getenv("AZURE_QUEUE_NAME", "test-queue")


class AzureQueueConsumer:
    """
    Async consumer for Azure Storage Queue messages.

    Consumes JSON messages and forwards them to the provided async callback.
    """

    def __init__(self, process_callback: Callable[[dict], Awaitable[None]]):
        self._process_callback = process_callback
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[QueueClient] = None

    async def start(self):
        if QueueClient is None:
            logger.error("azure-storage-queue is not installed. Please install azure-storage-queue>=12.0.0")
            return
        if self._task and not self._task.done():
            return
        self._client = QueueClient.from_connection_string(
            conn_str=AZURE_QUEUE_CONNECTION_STRING,
            queue_name=AZURE_QUEUE_NAME,
        )
        self._stop_event.clear()
        self._task = asyncio.create_task(self._consume_loop(), name="AzureQueueConsumer")
        logger.info(f"AzureQueueConsumer started for queue '{AZURE_QUEUE_NAME}'")

    async def stop(self):
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
        logger.info("AzureQueueConsumer stopped")

    async def _consume_loop(self):
        assert self._client is not None
        client: QueueClient = self._client

        # Ensure queue exists (no-op if it already exists)
        try:
            await client.create_queue()
        except Exception:
            pass

        # Poll loop
        while not self._stop_event.is_set():
            try:
                # Receive up to 8 messages per poll; keep visibility for 60s while processing
                async for msg in client.receive_messages(messages_per_page=8, visibility_timeout=60):
                    try:
                        payload = self._parse_message(msg)
                        if payload is None:
                            # Delete malformed message to avoid poison-loop
                            await client.delete_message(msg.id, msg.pop_receipt)
                            continue
                        await self._process_callback(payload)
                        # Delete after successful processing
                        await client.delete_message(msg.id, msg.pop_receipt)
                    except Exception as e:
                        logger.error(f"Error processing queue message: {e}")
                        # Let message become visible again after visibility timeout
                        continue

                # Small backoff when no messages
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Azure queue receive error: {e}")
                await asyncio.sleep(2.0)

    def _parse_message(self, msg) -> Optional[dict]:
        try:
            # SDK decodes content to string by default
            body = getattr(msg, "content", None)
            if not body:
                return None
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            return json.loads(body)
        except Exception as e:
            logger.error(f"Failed to parse message body: {e}")
            return None


# Helper to wire into FastAPI lifespan
async def start_consumer_with_callback(app, process_callback: Callable[[dict], Awaitable[None]]):
    consumer = AzureQueueConsumer(process_callback)
    app.state.azure_queue_consumer = consumer
    await consumer.start()


async def stop_consumer(app):
    consumer: Optional[AzureQueueConsumer] = getattr(app.state, "azure_queue_consumer", None)
    if consumer:
        await consumer.stop()

