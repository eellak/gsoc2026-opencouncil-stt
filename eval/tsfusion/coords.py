#!/usr/bin/env python3
"""The one place that knows how the three clocks relate.

Three coordinate systems are in play and mixing them silently is the failure mode
that makes a diagnostic page persuasive and false:

  ABSOLUTE   seconds from the start of the source meeting recording. Everything the
             benchmark, pyannote and the ledger talk about is in this clock.
  PAGE       seconds from the start of the 299 s extraction, which is what the
             <audio> element in the viewer counts. page = absolute - T0.
  WINDOW     seconds from the start of ONE cached benchmark window WAV. Whisper's
             `decode-rw` word timestamps and Soniox's `start_ms` are in this clock,
             one per window. absolute = window + window.start.

The two windows are contiguous but NOT adjacent: window A ends at 2096.065 and
window B starts at 2096.234, a 169 ms hole. The page audio is cut from the source
mp3 across that hole, so PAGE time is continuous where WINDOW time is not. Nothing
here papers over the hole; `SEAM` names it so the viewer can draw it.
"""
from __future__ import annotations

from dataclasses import dataclass

CITY = "vrilissia"
MEETING = "apr1_2_2026"


@dataclass(frozen=True)
class WindowSpan:
    """One benchmark window in absolute time."""
    item_id: str
    start: float          # absolute seconds
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_absolute(self, window_t: float) -> float:
        return window_t + self.start

    def contains(self, absolute_t: float) -> bool:
        return self.start <= absolute_t < self.end


# Frozen from the benchmark report (`startSec` / `durationSec`), not from the brief:
# the brief rounded 1945.951 to 1946.0 and called the hole 100 ms.
WINDOWS = (
    WindowSpan("win_vrilissia_apr1_2_2026_1945951", 1945.951, 150.114),
    WindowSpan("win_vrilissia_apr1_2_2026_2096234", 2096.234, 149.004),
)

T0 = WINDOWS[0].start                       # absolute start of the page audio
T1 = WINDOWS[-1].end                        # absolute end of the page audio
PAGE_DURATION = T1 - T0

# The hole between the two benchmark windows, in absolute seconds. Real audio, no
# reference text, no hypothesis text: a place where a boundary error is invisible to
# every measurement in this project.
SEAM = (WINDOWS[0].end, WINDOWS[1].start)

WHISPER_CHUNK = 30.0                        # Whisper's decode window, for `t mod 30`


def to_page(absolute_t: float) -> float:
    """Absolute meeting seconds -> page seconds."""
    return absolute_t - T0


def to_absolute(page_t: float) -> float:
    """Page seconds -> absolute meeting seconds."""
    return page_t + T0


def window_of(item_id: str) -> WindowSpan:
    for w in WINDOWS:
        if w.item_id == item_id:
            return w
    raise KeyError(item_id)


def window_to_absolute(item_id: str, window_t: float) -> float:
    return window_of(item_id).to_absolute(window_t)


def window_to_page(item_id: str, window_t: float) -> float:
    return to_page(window_to_absolute(item_id, window_t))


def chunk_phase(absolute_t: float, decode_origin: float = 0.0) -> float:
    """Where in Whisper's 30 s decode window this moment sits.

    Whisper's deletions are elevated at the edges of its 30 s chunks (0.0712 in the
    first 5 s, 0.0530 in the last 5 s, measured 2026-08-18). Without this number a
    chunk-edge effect reads as a speaker-handover effect.

    The phase is measured from the start of the audio the decoder was actually FED,
    not from the start of the meeting. Each benchmark window was decoded from its own
    WAV beginning at local time zero, so `decode_origin` is that window's absolute
    start. Passing 0.0 gives the meeting-clock phase, which is the WRONG quantity for
    Whisper here and is kept only so the page can show that the two differ.
    """
    return (absolute_t - decode_origin) % WHISPER_CHUNK


def whisper_phase(absolute_t: float) -> float | None:
    """Decode-window phase for the window that actually contains `absolute_t`.

    None inside the 169 ms seam, which no decoder was fed.
    """
    for w in WINDOWS:
        if w.contains(absolute_t):
            return chunk_phase(absolute_t, w.start)
    return None


def in_seam(absolute_t: float) -> bool:
    return SEAM[0] <= absolute_t < SEAM[1]
