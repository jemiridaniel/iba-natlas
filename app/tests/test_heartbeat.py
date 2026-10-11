"""Keep-alive comments on the event stream, so tunnels do not drop slow assessments."""

import time

import pytest

from backend.app.main import with_heartbeat


def slow(chunks, delay):
    for c in chunks:
        time.sleep(delay)
        yield c


def test_slow_steps_get_keep_alive_comments() -> None:
    chunks = ["event: a\ndata: {}\n\n", "event: b\ndata: {}\n\n"]
    out = list(with_heartbeat(slow(chunks, 0.25), 0.1))
    events = [c for c in out if c.startswith("event:")]
    beats = [c for c in out if c.startswith(": keep-alive")]
    assert events == ["event: a\ndata: {}\n\n", "event: b\ndata: {}\n\n"]
    assert len(beats) >= 2


def test_fast_stream_has_no_heartbeats() -> None:
    assert list(with_heartbeat(iter(["x", "y"]), 5)) == ["x", "y"]


def test_producer_error_is_raised() -> None:
    def boom():
        yield "x"
        raise RuntimeError("pipeline broke")

    with pytest.raises(RuntimeError, match="pipeline broke"):
        list(with_heartbeat(boom(), 5))
