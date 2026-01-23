"""
Real-time data streaming infrastructure with WebSocket support and event-driven processing.
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

import websockets

from .models import MarketData

logger = logging.getLogger(__name__)


@dataclass
class StreamingEvent:
    """Represents a streaming data event."""

    event_type: str  # 'market_data', 'trade', 'quote', 'news'
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str


class StreamingDataHandler(ABC):
    """Abstract base class for handling streaming data."""

    @abstractmethod
    async def handle_event(self, event: StreamingEvent) -> None:
        """Handle a streaming data event."""
        pass


class MarketDataHandler(StreamingDataHandler):
    """Handler for market data events."""

    def __init__(self, callback: Optional[Callable[[StreamingEvent], None]] = None):
        self.callback = callback
        self.latest_data: Dict[str, MarketData] = {}

    async def handle_event(self, event: StreamingEvent) -> None:
        """Handle market data event."""
        try:
            if event.event_type == "market_data":
                # Convert to MarketData object
                market_data = MarketData(
                    symbol=event.symbol,
                    timestamp=event.timestamp,
                    open=event.data.get("open", 0.0),
                    high=event.data.get("high", 0.0),
                    low=event.data.get("low", 0.0),
                    close=event.data.get("close", 0.0),
                    volume=event.data.get("volume", 0),
                    adjusted_close=event.data.get("adjusted_close"),
                )

                self.latest_data[event.symbol] = market_data

                if self.callback:
                    self.callback(event)

                logger.debug(
                    f"Processed market data for {event.symbol}: {market_data.close}"
                )

        except Exception as e:
            logger.error(f"Error handling market data event: {e}")


class DataBuffer:
    """Buffer for streaming data with batching capabilities."""

    def __init__(self, max_size: int = 1000, flush_interval: float = 1.0):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer: List[StreamingEvent] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self.flush_callbacks: List[Callable[[List[StreamingEvent]], None]] = []

    def add_event(self, event: StreamingEvent) -> None:
        """Add an event to the buffer."""
        with self.lock:
            self.buffer.append(event)

            # Check if we need to flush
            if (
                len(self.buffer) >= self.max_size
                or time.time() - self.last_flush >= self.flush_interval
            ):
                self._flush()

    def _flush(self) -> None:
        """Flush the buffer."""
        if not self.buffer:
            return

        events_to_flush = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = time.time()

        # Call flush callbacks
        for callback in self.flush_callbacks:
            try:
                callback(events_to_flush)
            except Exception as e:
                logger.error(f"Error in flush callback: {e}")

    def add_flush_callback(
        self, callback: Callable[[List[StreamingEvent]], None]
    ) -> None:
        """Add a callback to be called when buffer is flushed."""
        self.flush_callbacks.append(callback)

    def force_flush(self) -> None:
        """Force flush the buffer."""
        with self.lock:
            self._flush()


class WebSocketStreamer:
    """WebSocket-based data streamer."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}
        self.handlers: List[StreamingDataHandler] = []
        self.buffer = DataBuffer()
        self.is_running = False
        self.websocket = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5.0

    def add_handler(self, handler: StreamingDataHandler) -> None:
        """Add a data handler."""
        self.handlers.append(handler)

    def remove_handler(self, handler: StreamingDataHandler) -> None:
        """Remove a data handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    async def start(self) -> None:
        """Start the WebSocket connection and data streaming."""
        self.is_running = True

        while self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to WebSocket: {self.url}")

                async with websockets.connect(
                    self.url,
                    extra_headers=self.headers,
                    ping_interval=20,
                    ping_timeout=10,
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_attempts = 0
                    logger.info("WebSocket connected successfully")

                    await self._listen()

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                await self._handle_reconnect()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await self._handle_reconnect()

    async def _listen(self) -> None:
        """Listen for WebSocket messages."""
        try:
            async for message in self.websocket:
                if not self.is_running:
                    break

                await self._process_message(message)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection lost during listening")
            raise
        except Exception as e:
            logger.error(f"Error during WebSocket listening: {e}")
            raise

    async def _process_message(self, message: str) -> None:
        """Process a WebSocket message."""
        try:
            data = json.loads(message)

            # Create streaming event
            event = StreamingEvent(
                event_type=data.get("type", "unknown"),
                symbol=data.get("symbol", ""),
                timestamp=datetime.fromtimestamp(data.get("timestamp", time.time())),
                data=data.get("data", {}),
                source="websocket",
            )

            # Add to buffer
            self.buffer.add_event(event)

            # Process with handlers
            for handler in self.handlers:
                await handler.handle_event(event)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_reconnect(self) -> None:
        """Handle WebSocket reconnection."""
        if not self.is_running:
            return

        self.reconnect_attempts += 1

        if self.reconnect_attempts < self.max_reconnect_attempts:
            logger.info(
                f"Reconnecting in {self.reconnect_delay} seconds (attempt {self.reconnect_attempts})"
            )
            await asyncio.sleep(self.reconnect_delay)
            self.reconnect_delay = min(
                self.reconnect_delay * 1.5, 60
            )  # Exponential backoff
        else:
            logger.error("Max reconnection attempts reached")
            self.is_running = False

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self.is_running = False

        if self.websocket:
            await self.websocket.close()

        # Flush any remaining data
        self.buffer.force_flush()

        logger.info("WebSocket streamer stopped")

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message through the WebSocket."""
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")


class EventDrivenProcessor:
    """Event-driven processor for streaming data."""

    def __init__(self, max_queue_size: int = 10000):
        self.event_queue: Queue = Queue(maxsize=max_queue_size)
        self.processors: Dict[str, List[Callable[[StreamingEvent], None]]] = {}
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.num_workers = 4

    def add_processor(
        self, event_type: str, processor: Callable[[StreamingEvent], None]
    ) -> None:
        """Add a processor for a specific event type."""
        if event_type not in self.processors:
            self.processors[event_type] = []
        self.processors[event_type].append(processor)

    def remove_processor(
        self, event_type: str, processor: Callable[[StreamingEvent], None]
    ) -> None:
        """Remove a processor for a specific event type."""
        if event_type in self.processors and processor in self.processors[event_type]:
            self.processors[event_type].remove(processor)

    def add_event(self, event: StreamingEvent) -> None:
        """Add an event to the processing queue."""
        try:
            self.event_queue.put_nowait(event)
        except Exception:
            logger.warning("Event queue is full, dropping event")

    def start(self) -> None:
        """Start the event processing workers."""
        self.is_running = True

        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, name=f"EventWorker-{i}")
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)

        logger.info(f"Started {self.num_workers} event processing workers")

    def stop(self) -> None:
        """Stop the event processing workers."""
        self.is_running = False

        # Add sentinel values to wake up workers
        for _ in range(self.num_workers):
            try:
                self.event_queue.put_nowait(None)
            except Exception:
                pass

        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)

        logger.info("Event processing workers stopped")

    def _worker(self) -> None:
        """Worker thread for processing events."""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1.0)

                if event is None:  # Sentinel value
                    break

                self._process_event(event)
                self.event_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event worker: {e}")

    def _process_event(self, event: StreamingEvent) -> None:
        """Process a single event."""
        processors = self.processors.get(event.event_type, [])

        for processor in processors:
            try:
                processor(event)
            except Exception as e:
                logger.error(f"Error in event processor: {e}")


class StreamingDataManager:
    """Main manager for streaming data infrastructure."""

    def __init__(self):
        self.streamers: Dict[str, WebSocketStreamer] = {}
        self.processor = EventDrivenProcessor()
        self.data_handlers: Dict[str, StreamingDataHandler] = {}
        self.is_running = False

    def add_streamer(
        self, name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a WebSocket streamer."""
        streamer = WebSocketStreamer(url, headers)
        self.streamers[name] = streamer

        # Add buffer flush callback to send events to processor
        streamer.buffer.add_flush_callback(self._handle_buffered_events)

        logger.info(f"Added streamer '{name}' for {url}")

    def remove_streamer(self, name: str) -> None:
        """Remove a WebSocket streamer."""
        if name in self.streamers:
            del self.streamers[name]
            logger.info(f"Removed streamer '{name}'")

    def add_data_handler(self, name: str, handler: StreamingDataHandler) -> None:
        """Add a data handler."""
        self.data_handlers[name] = handler

        # Add handler to all streamers
        for streamer in self.streamers.values():
            streamer.add_handler(handler)

        logger.info(f"Added data handler '{name}'")

    def add_event_processor(
        self, event_type: str, processor: Callable[[StreamingEvent], None]
    ) -> None:
        """Add an event processor."""
        self.processor.add_processor(event_type, processor)

    async def start(self) -> None:
        """Start all streamers and processors."""
        self.is_running = True

        # Start event processor
        self.processor.start()

        # Start all streamers
        tasks = []
        for name, streamer in self.streamers.items():
            task = asyncio.create_task(streamer.start())
            tasks.append(task)
            logger.info(f"Starting streamer '{name}'")

        # Wait for all streamers
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop all streamers and processors."""
        self.is_running = False

        # Stop all streamers
        for name, streamer in self.streamers.items():
            await streamer.stop()
            logger.info(f"Stopped streamer '{name}'")

        # Stop event processor
        self.processor.stop()

        logger.info("Streaming data manager stopped")

    def _handle_buffered_events(self, events: List[StreamingEvent]) -> None:
        """Handle events from buffer flushes."""
        for event in events:
            self.processor.add_event(event)

    def get_latest_data(self, handler_name: str) -> Dict[str, Any]:
        """Get latest data from a specific handler."""
        if handler_name in self.data_handlers:
            handler = self.data_handlers[handler_name]
            if isinstance(handler, MarketDataHandler):
                return {
                    symbol: asdict(data) for symbol, data in handler.latest_data.items()
                }
        return {}


# Example usage and utility functions
def create_market_data_processor(
    callback: Optional[Callable[[MarketData], None]] = None,
) -> Callable[[StreamingEvent], None]:
    """Create a market data processor function."""

    def processor(event: StreamingEvent) -> None:
        if event.event_type == "market_data":
            market_data = MarketData(
                symbol=event.symbol,
                timestamp=event.timestamp,
                open=event.data.get("open", 0.0),
                high=event.data.get("high", 0.0),
                low=event.data.get("low", 0.0),
                close=event.data.get("close", 0.0),
                volume=event.data.get("volume", 0),
                adjusted_close=event.data.get("adjusted_close"),
            )

            if callback:
                callback(market_data)

    return processor


def create_data_aggregator(window_size: int = 60) -> Callable[[StreamingEvent], None]:
    """Create a data aggregator that aggregates events over a time window."""

    aggregated_data: Dict[str, List[StreamingEvent]] = {}
    last_aggregation = time.time()

    def aggregator(event: StreamingEvent) -> None:
        nonlocal last_aggregation

        # Add event to aggregation
        if event.symbol not in aggregated_data:
            aggregated_data[event.symbol] = []
        aggregated_data[event.symbol].append(event)

        # Check if we should aggregate
        current_time = time.time()
        if current_time - last_aggregation >= window_size:
            # Perform aggregation
            for symbol, events in aggregated_data.items():
                if events:
                    # Simple aggregation - could be enhanced
                    logger.info(f"Aggregated {len(events)} events for {symbol}")

            # Clear aggregated data
            aggregated_data.clear()
            last_aggregation = current_time

    return aggregator
