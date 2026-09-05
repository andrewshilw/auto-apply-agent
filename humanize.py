"""Week 6 lab: humanization primitives.

Pure-Python helpers for `browser.py` — randomized pacing, a curved
multi-point mouse path, and a chunked, non-uniform typing cadence — used in
place of this codebase's previous fixed `time.sleep(...)` + instant
`fill`/`click` calls. The goal is realism and robustness (real users don't
move the mouse in a straight line at constant speed, or type at a perfectly
even rate, and a few ATS widgets in this codebase are already known to race
against too-fast synthetic input — see `browser.focus`'s docstring). This is
NOT an attempt to defeat any platform's bot-detection or anti-abuse systems:
nothing here spoofs a browser fingerprint, solves a challenge, or hides that
the session is automated.
"""

import random


def human_delay_range(base: float, jitter: float = 0.4) -> tuple[float, float]:
    """(low, high) bounds for a `base`-second pause, +/- `jitter` fraction —
    real pacing between actions is never perfectly uniform. Returns a range
    rather than sleeping itself so callers can combine it with other
    randomized work (e.g. `browser.py` sleeps this length while also
    stepping through a mouse path)."""
    low = max(0.05, base * (1 - jitter))
    high = max(low + 0.05, base * (1 + jitter))
    return low, high


def human_delay(base: float, jitter: float = 0.4) -> float:
    """A single randomized delay in seconds — see `human_delay_range`."""
    return random.uniform(*human_delay_range(base, jitter))


def mouse_path(x0: float, y0: float, x1: float, y1: float) -> list[tuple[int, int]]:
    """A short sequence of intermediate points from (x0, y0) to (x1, y1): a
    real hand doesn't move the cursor in a straight line at constant speed,
    so this adds a slight perpendicular bow (peaking mid-path, zero at both
    ends) plus small per-point jitter and an ease-out toward the target."""
    steps = random.randint(3, 6)
    dx, dy = x1 - x0, y1 - y0
    length = max(1.0, (dx**2 + dy**2) ** 0.5)
    nx, ny = -dy / length, dx / length  # unit normal to the direct line
    bow = random.uniform(-18, 18)

    points = []
    for i in range(1, steps + 1):
        t = i / steps
        eased = 1 - (1 - t) ** 2
        bow_offset = bow * 4 * t * (1 - t)
        x = x0 + dx * eased + nx * bow_offset + random.uniform(-2, 2)
        y = y0 + dy * eased + ny * bow_offset + random.uniform(-2, 2)
        points.append((round(x), round(y)))
    points[-1] = (round(x1), round(y1))  # land exactly on target regardless of jitter
    return points


def typing_chunks(text: str) -> list[tuple[str, float]]:
    """Break `text` into small runs of 1-4 characters, each paired with the
    delay to wait *after* typing it — a non-linear typing cadence instead of
    one uniform per-character rate: quick bursts for ordinary runs, a longer
    pause after spaces/punctuation (word boundaries), and an occasional
    longer hesitation, same as real typing. Chunked rather than per-character
    to bound the number of CLI round-trips `browser.type_chunks` needs for
    longer fields (e.g. an email address or full name)."""
    chunks: list[tuple[str, float]] = []
    i = 0
    while i < len(text):
        size = min(random.randint(1, 4), len(text) - i)
        chunk = text[i : i + size]
        i += size
        delay = random.uniform(0.04, 0.12) * len(chunk)
        if chunk[-1] in " ,.-@":
            delay += random.uniform(0.08, 0.2)
        if random.random() < 0.06:
            delay += random.uniform(0.15, 0.4)  # occasional hesitation
        chunks.append((chunk, delay))
    return chunks
