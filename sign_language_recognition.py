#!/usr/bin/env python3
"""
Sign Language Recognition -- ASL fingerspelling reader
======================================================

Reads American Sign Language fingerspelling (A-Z) live from a camera or
video file and builds the spelled word on screen:

* **MediaPipe Hands** tracks the hand and supplies 21 landmarks.
* A small **landmark classifier** (a geometry heuristic, NOT a trained
  neural network -- see "What this is, honestly" below) reads the
  handshape: which fingers are extended, where the thumb sits, how far
  apart the index and middle fingers are, and which way the hand
  points, and maps that to one ASL letter.
* A letter that is **held steady for ~0.8 s commits** to the running
  word strip at the bottom of the screen. A gap with no hand for
  ~1.5 s inserts a space. SPACE adds a space by hand, C clears the
  strip, Backspace deletes the last letter.

One shared pipeline serves both run modes:

* **Real mode** (``--source``) -- MediaPipe Hands landmarks from a
  video file or webcam feed, turned into hand features by geometry.
* **Simulation mode** (``--simulate``) -- a hand is rendered with
  OpenCV spelling the word "LOVE" (L, O, V, E in turn), and the SAME
  hand features are estimated from the rendered pixels only: skin
  segmentation, convex-hull fingertips, thumb position and the ring
  hole of the O. The classifier only ever sees those pixel-derived
  features; it never peeks at the scene's scripted letters (it does
  not know "an L is scheduled at 2 s").

What this is, honestly
----------------------
The classifier is a hand-written rule set over finger geometry. It
reads the clear, static handshapes well (A, B, C, D, F, G, H, I, K,
L, O, R, S, T, U, V, W, X, Y) and it is upfront about its limits:

* **J and Z need motion** (J flicks the pinky, Z draws a Z). A still
  frame cannot tell them apart, so J reads as I and Z reads as G.
* **E, M and N** differ only by exactly where the thumb hides under
  the curled fingers -- the hardest distinction in the set. Only a
  very tight tucked fist reads as E; a looser one reads as no letter
  rather than a guessed M or N.
* **Two-handed signs are not fingerspelling.** If the footage shows
  a person signing words with both hands, the reader follows the
  largest / steadiest hand and reads whatever letter its shape
  matches at each moment. The word strip will then be a sequence of
  those shapes, not the signed word. That is the honest output of
  this tool on that footage, and the README says so.
* This is **not** a trained ASL model and its letters are **not**
  verified against a labelled dataset. Anyone wanting production
  accuracy should train a classifier on labelled landmark data; this
  project is the working skeleton for that -- landmarks in, letters
  out, with the plumbing (tracking, hold-to-commit, word strip)
  already done.

How a letter is decided
-----------------------
Landmarks (MediaPipe Hands layout): wrist 0, thumb 1-4, index 5-8,
middle 9-12, ring 13-16, pinky 17-20. All measures are normalised by
the hand scale (wrist to middle-finger knuckle), so the rules work at
any distance from the camera.

* ``reach`` -- fingertip distance from the palm centre, in hand
  scales. Extended > 1.10 (or > 0.95 with the tip clearly past the
  knuckle joint), half-curled 0.80-1.10, curled < 0.80.
* ``thumb_state`` -- UP (thumb alongside the fist, A), ACROSS (thumb
  lying over the fingers, S), BETWEEN (thumb poking between index and
  middle, T), TUCKED (thumb hidden under the fingers, E/M/N) or OUT
  (thumb stuck out to the side, L/Y).
* ``spread`` -- angle between the index and middle fingers. Wide
  (> 20 deg) is V/K, narrow (< 12 deg) is U, and a flipped tip order
  is R (index and middle crossed).
* ``orientation`` -- which way the middle finger points: up, sideways
  (G/H) or down (P/Q).
* ``circle`` -- thumb tip touching the index fingertip: F when the
  other three fingers are up, O when the hand is a fist.

A letter must hold for ``--hold-seconds`` (default 0.5 s) before it
is shown as the current letter, and for ``--commit-seconds``
(default 0.8 s) before it joins the word strip -- so a hand moving
between shapes never litters the strip.

Examples
--------
    python sign_language_recognition.py --simulate
    python sign_language_recognition.py --simulate --output demo.mp4 --no-display
    python sign_language_recognition.py --source signing.mp4
    python sign_language_recognition.py --source 0
    python sign_language_recognition.py --source pick
    python sign_language_recognition.py            (opens the file picker)

Keys (preview window): Q quit, SPACE insert a space, C clear the word
strip, Backspace delete the last letter.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_WIDTH, SIM_HEIGHT = 1280, 720   # simulation canvas
SIM_FPS = 30                        # simulation frame rate

HOLD_SECONDS = 0.5      # a letter must hold this long to be "current"
COMMIT_SECONDS = 0.7    # ... and this long to join the word strip
# (0.7 s sits between a passing shape, ~0.3 s, and a genuinely held
# letter, ~0.8 s and up, on the reference footage -- see README.)
GAP_SECONDS = 1.5       # no hand this long inserts a space in the strip
SMOOTH_WINDOW = 5       # median filter over recent per-letter holds

# Reach thresholds (fingertip distance from palm centre, in hand scales).
# Tuned on real MediaPipe landmarks (Nov 2026): a fully extended
# finger measures 1.29-1.70, a curled one 0.05-0.86, so 1.10 sits in
# the gap; a half-curled (C/X) finger measures 0.80-1.10. The
# straightness cross-check below catches an extended index finger
# that measures just under 1.10 because the palm centre shifts when
# the thumb sticks out (the L handshape).
REACH_EXTENDED = 1.10
REACH_HALF = 0.80
STRAIGHT_RATIO = 1.15   # tip past the PIP joint, from the wrist
# Spread thresholds (degrees between index and middle fingers).
SPREAD_WIDE = 20.0
SPREAD_NARROW = 12.0
# Orientation thresholds (degrees of the middle finger from image-up).
ORIENT_SIDE = 45.0
ORIENT_DOWN = 120.0
# Circle threshold (thumb tip to index tip, in hand scales).
CIRCLE_DIST = 0.38

COLOUR_SKIN = (140, 180, 240)       # BGR peach (simulation hand)
COLOUR_SKIN_DARK = (110, 150, 210)
COLOUR_BG = (52, 58, 64)            # simulation background (desaturated)
COLOUR_LETTER = (0, 200, 255)       # amber (BGR)
COLOUR_WORD = (255, 255, 255)
COLOUR_HAND = (0, 255, 255)         # skeleton yellow
COLOUR_OK = (0, 190, 70)
COLOUR_NONE = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# MediaPipe Hands landmark indices.
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_between(v1, v2) -> float:
    """Angle in degrees between two 2D vectors."""
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosang))


# ---------------------------------------------------------------------------
# Hand features -- the one interface both providers speak
# ---------------------------------------------------------------------------

@dataclass
class HandFeatures:
    """Everything the letter classifier needs, in hand-local geometry.

    ``reach`` is the fingertip distance from the palm centre in hand
    scales, per finger (index, middle, ring, pinky). ``extended`` is
    reach > 1.10, or reach > 0.95 with the fingertip clearly past the
    PIP joint; ``half`` is a hooked or C-curved finger (reach 0.80
    to the extended line, not extended).
    ``thumb_extended`` is the thumb stuck out to the side (L/Y);
    ``thumb_state`` is UP / ACROSS / BETWEEN / TUCKED / OUT for a
    folded thumb. ``spread`` is the index-middle angle in degrees,
    ``crossed`` is True when the index and middle fingertips have
    swapped sides (R). ``orientation`` is the middle-finger direction
    from image-up in degrees (0 up, 90 sideways, 180 down).
    ``circle`` is True when the thumb tip touches the index fingertip
    (F/O). ``thumb_touch_middle`` is True when the thumb tip rests
    against the middle fingertip (D). ``palm`` and ``scale`` are in
    image pixels (for drawing). ``tips`` are the fingertip points
    (for drawing, when known).
    """

    reach: tuple[float, float, float, float]
    ext4: tuple[bool, bool, bool, bool]
    half: tuple[bool, bool, bool, bool]
    thumb_extended: bool
    thumb_state: str
    spread: float
    crossed: bool
    orientation: float
    circle: bool
    thumb_touch_middle: bool = False
    palm: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    tips: dict = field(default_factory=dict)

    @property
    def extended(self) -> tuple[bool, bool, bool, bool]:
        return self.ext4

    @property
    def ext5(self) -> tuple[bool, bool, bool, bool, bool]:
        """(thumb, index, middle, ring, pinky) extended flags."""
        return (self.thumb_extended,) + self.extended


def features_from_landmarks(lm, handedness: str = "") -> HandFeatures | None:
    """Hand features from 21 MediaPipe landmarks (x, y) in pixels."""
    if lm is None or len(lm) < 21:
        return None
    wrist = lm[WRIST]
    palm = tuple(sum(lm[i][k] for i in FINGER_MCPS) / 4
                 for k in (0, 1))
    scale = _dist(wrist, lm[MIDDLE_MCP])
    if scale < 10:
        return None

    # Hand-local axes: up = wrist -> middle knuckle; right = up rotated.
    ux, uy = lm[MIDDLE_MCP][0] - wrist[0], lm[MIDDLE_MCP][1] - wrist[1]
    ulen = math.hypot(ux, uy) or 1.0
    ux, uy = ux / ulen, uy / ulen
    rx, ry = -uy, ux  # "right" in the hand's own frame

    def frame(p):
        """Point p in the hand frame (r, u), in hand scales."""
        dx, dy = p[0] - palm[0], p[1] - palm[1]
        return (dx * rx + dy * ry) / scale, (dx * ux + dy * uy) / scale

    reach = tuple(_dist(lm[t], palm) / scale for t in FINGER_TIPS)
    # Extended: clearly past the palm edge, or past the PIP joint
    # with a near-full reach (catches the L index finger, whose
    # reach dips just under the line when the thumb sticks out and
    # shifts the palm centre). Half: hooked / C-curved, not extended.
    straight = tuple(
        _dist(lm[t], wrist) > STRAIGHT_RATIO * _dist(lm[p], wrist)
        for t, p in zip(FINGER_TIPS, FINGER_PIPS))
    ext4 = tuple(
        r > REACH_EXTENDED or (r > 0.95 and s)
        for r, s in zip(reach, straight))
    half = tuple((not e) and r > REACH_HALF
                 for r, e in zip(reach, ext4))

    # Thumb: tip position in the hand frame + distance to key points.
    thumb_tip = lm[THUMB_TIP]
    index_tip = lm[INDEX_TIP]
    middle_tip = lm[MIDDLE_TIP]
    tip_r, tip_u = frame(thumb_tip)
    thumb_reach = _dist(thumb_tip, palm) / scale
    thumb_ip_reach = _dist(lm[THUMB_IP], palm) / scale
    thumb_extended = thumb_reach > 1.0 and abs(tip_r) > 0.55
    circle = _dist(thumb_tip, index_tip) / scale < CIRCLE_DIST
    thumb_touch_middle = _dist(thumb_tip, middle_tip) / scale < 0.45

    # Thumb state for a folded thumb (thumb not stuck out, no circle).
    if circle:
        thumb_state = "CIRCLE"
    elif thumb_extended:
        thumb_state = "OUT"
    elif thumb_reach < 0.55 and tip_u < 0.25:
        thumb_state = "TUCKED"
    elif tip_u > 0.45 and _dist(thumb_tip, lm[INDEX_MCP]) / scale < 0.55 \
            and _dist(thumb_tip, lm[MIDDLE_MCP]) / scale < 0.55:
        thumb_state = "BETWEEN"
    elif _dist(thumb_tip, lm[MIDDLE_MCP]) / scale < 0.5 \
            or _dist(thumb_tip, lm[RING_MCP]) / scale < 0.5:
        thumb_state = "ACROSS"
    elif tip_u > 0.25 and thumb_reach > 0.65:
        thumb_state = "UP"
    else:
        thumb_state = "TUCKED"

    # Spread: angle between the index and middle finger directions.
    idx_dir = (lm[INDEX_TIP][0] - lm[INDEX_MCP][0],
               lm[INDEX_TIP][1] - lm[INDEX_MCP][1])
    mid_dir = (lm[MIDDLE_TIP][0] - lm[MIDDLE_MCP][0],
               lm[MIDDLE_TIP][1] - lm[MIDDLE_MCP][1])
    spread = _angle_between(idx_dir, mid_dir)

    # Crossed (R): the fingertip order along the hand's right axis is
    # flipped relative to the knuckle order. Only meaningful when
    # both fingers are extended -- a curled middle fingertip sits
    # anywhere and would false-trigger this.
    idx_tip_r = (lm[INDEX_TIP][0] - palm[0]) * rx \
        + (lm[INDEX_TIP][1] - palm[1]) * ry
    mid_tip_r = (lm[MIDDLE_TIP][0] - palm[0]) * rx \
        + (lm[MIDDLE_TIP][1] - palm[1]) * ry
    idx_mcp_r = (lm[INDEX_MCP][0] - palm[0]) * rx \
        + (lm[INDEX_MCP][1] - palm[1]) * ry
    mid_mcp_r = (lm[MIDDLE_MCP][0] - palm[0]) * rx \
        + (lm[MIDDLE_MCP][1] - palm[1]) * ry
    crossed = bool(ext4[0] and ext4[1]) and \
        (idx_tip_r - mid_tip_r) * (idx_mcp_r - mid_mcp_r) < 0

    # Orientation: middle-finger direction from image-up (0, -1).
    mdir = (lm[MIDDLE_TIP][0] - wrist[0], lm[MIDDLE_TIP][1] - wrist[1])
    orientation = _angle_between(mdir, (0, -1))

    tips = {name: lm[i] for name, i in (
        ("index", INDEX_TIP), ("middle", MIDDLE_TIP),
        ("ring", RING_TIP), ("pinky", PINKY_TIP),
        ("thumb", THUMB_TIP))}
    return HandFeatures(reach=reach, ext4=ext4, half=half,
                        thumb_extended=thumb_extended,
                        thumb_state=thumb_state, spread=spread,
                        crossed=crossed, orientation=orientation,
                        circle=circle,
                        thumb_touch_middle=thumb_touch_middle,
                        palm=palm, scale=scale, tips=tips)


# ---------------------------------------------------------------------------
# Letter classifier -- shared by both modes
# ---------------------------------------------------------------------------

def classify_letter(f: HandFeatures | None) -> str | None:
    """One ASL fingerspelling letter from hand features, or None.

    A rule set over finger geometry -- NOT a trained model. J and Z
    need motion, so they read as I and G; E/M/N differ only by where
    the thumb hides, so they are a best effort. See the module
    docstring for the honest limits.
    """
    if f is None:
        return None
    ext = f.extended          # (index, middle, ring, pinky)
    thumb, idx, mid, ring, pinky = f.ext5
    n_up = sum(ext)

    # Circle shapes first: thumb tip touching the index fingertip.
    if f.circle:
        if mid and ring and pinky:
            return "F"        # index+thumb ring, other three up
        if n_up == 0:
            return "O"        # fist curled into a ring
        return None

    # C shape: all four fingers half-curled, thumb out of the way.
    if all(f.half) and n_up == 0 and f.thumb_state in ("OUT", "UP"):
        return "C"

    # X: index hooked (half), everything else curled, thumb folded.
    if f.half[0] and not idx and not mid and not ring and not pinky \
            and not thumb:
        return "X"

    # Fist family -- the thumb decides.
    if n_up == 0:
        if f.thumb_state == "UP":
            return "A"
        if f.thumb_state == "ACROSS":
            return "S"
        if f.thumb_state == "BETWEEN":
            return "T"
        if f.thumb_state == "TUCKED":
            # A tight fist with the thumb tucked reads E. A looser
            # fist is genuinely ambiguous between E, M and N (they
            # differ only by where the thumb hides), so it reads as
            # no letter rather than a guessed M or N.
            if max(f.reach) < 0.62:
                return "E"
            return None
        return None

    # Four fingers up, thumb folded across the palm.
    if ext == (True, True, True, True) and not thumb:
        return "B"
    # Open hand (all five up) is not a fingerspelling letter.
    if ext == (True, True, True, True) and thumb:
        return None

    # Index only, thumb folded -- or thumb + index (L, checked first:
    # L must win over the index-only branch below).
    if ext == (True, False, False, False) and thumb:
        return "L"
    if ext == (True, False, False, False):
        if f.thumb_touch_middle:
            return "D"        # thumb resting on the middle fingertip
        if f.orientation >= ORIENT_DOWN:
            return "Q"        # index pointing down
        if f.orientation >= ORIENT_SIDE:
            return "G"        # index pointing sideways
        return None           # index up, thumb folded, no touch

    # Index + middle.
    if ext == (True, True, False, False) and thumb:
        if f.orientation >= ORIENT_DOWN:
            return "P"        # K-shape pointing down
        return "K"
    if ext == (True, True, False, False):
        if f.orientation >= ORIENT_SIDE and f.orientation < ORIENT_DOWN:
            return "H"        # index+middle pointing sideways
        if f.crossed:
            return "R"        # index and middle crossed
        if f.spread >= SPREAD_WIDE:
            return "V"
        return "U"

    # Index + middle + ring.
    if ext == (True, True, True, False):
        return "W"

    # Thumb + pinky (Y) before pinky-only (I): the pinky-only branch
    # would swallow Y otherwise.
    if ext == (False, False, False, True) and thumb:
        return "Y"

    # Pinky only (J needs a flick, so it reads as I here).
    if ext == (False, False, False, True):
        return "I"

    return None

# ---------------------------------------------------------------------------
# Sign reader -- hold-to-commit + word strip (shared by both modes)
# ---------------------------------------------------------------------------

class SignReader:
    """Turns a stream of (letter, time) into a word strip.

    A letter becomes "current" after holding for HOLD_SECONDS and
    joins the word strip after COMMIT_SECONDS of continuous hold.
    The same letter is committed only once per hold -- it must change
    (or the hand must leave) before it can be committed again. A gap
    with no hand for GAP_SECONDS inserts a space.
    """

    def __init__(self, hold_seconds=HOLD_SECONDS,
                 commit_seconds=COMMIT_SECONDS, gap_seconds=GAP_SECONDS,
                 log_path="events.log"):
        self.hold_seconds = hold_seconds
        self.commit_seconds = commit_seconds
        self.gap_seconds = gap_seconds
        self.log_path = log_path
        self.current = None          # letter currently held
        self.current_since = None    # when the hold started
        self.committed = None        # letter already committed this hold
        self.word = ""
        self.last_hand_t = None      # last time a hand was seen
        self.space_done = False
        self.counters = {
            "frames": 0, "hand_frames": 0, "letter_frames": 0,
            "letters_committed": 0,
        }
        self.committed_letters = []  # (letter, time) in order
        self.letter_counts = {}      # letter -> times committed
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._log("Sign language recognition started.")

    def _log(self, message: str, t: float | None = None):
        stamp = f"{t:6.1f}s" if t is not None else "      "
        self._log_file.write(f"[{stamp}] {message}\n")
        self._log_file.flush()

    def update(self, letter: str | None, t: float, has_hand: bool = False):
        self.counters["frames"] += 1
        if has_hand:
            self.counters["hand_frames"] += 1
            self.last_hand_t = t
            self.space_done = False
        if letter is None:
            # No letter: the hold breaks. A long gap with NO HAND at
            # all inserts a word space -- but a hand that is simply
            # unreadable for a moment (ambiguous shape) must not.
            if self.current is not None:
                self.current = None
                self.current_since = None
                self.committed = None
            if not has_hand and self.last_hand_t is not None \
                    and t - self.last_hand_t >= self.gap_seconds \
                    and self.word and not self.word.endswith(" ") \
                    and not self.space_done:
                self.word += " "
                self.space_done = True
                self._log("Word space (no hand for "
                          f"{self.gap_seconds:.1f} s)", t)
            return self.current, self.word
        if letter != self.current:
            self.current = letter
            self.current_since = t
            self.committed = None
        held = t - (self.current_since if self.current_since is not None
                    else t)
        if held >= self.hold_seconds:
            self.counters["letter_frames"] += 1
        if held >= self.commit_seconds and self.committed != letter:
            self.committed = letter
            self.word += letter
            self.counters["letters_committed"] += 1
            self.committed_letters.append((letter, t))
            self.letter_counts[letter] = \
                self.letter_counts.get(letter, 0) + 1
            self._log(f"Committed letter '{letter}' "
                      f"(held {held:.1f} s) -> word: {self.word!r}", t)
        return self.current, self.word

    def insert_space(self, t: float):
        if self.word and not self.word.endswith(" "):
            self.word += " "
            self._log("Space inserted by key", t)

    def backspace(self, t: float):
        if self.word:
            self.word = self.word[:-1]
            self._log("Backspace", t)

    def clear(self, t: float):
        self.word = ""
        self._log("Word strip cleared by key", t)

    def close(self):
        self._log(f"Stopped. Final word strip: {self.word!r}")
        self._log_file.close()


# ---------------------------------------------------------------------------
# Real mode: MediaPipe Hands (classic solutions, else Tasks API)
# ---------------------------------------------------------------------------

class MediaPipeHandsProvider:
    """Real-mode provider backed by MediaPipe Hands (lazily imported).

    Works with BOTH mediapipe APIs: the classic solutions Hands when
    present, and the current Tasks API HandLandmarker on newer builds
    where ``mp.solutions`` no longer exists (mediapipe >= 1.x).
    Returns the primary hand's landmarks in pixels plus handedness,
    and the full landmark list for drawing. The primary hand is the
    one whose wrist is closest to the previous primary wrist, so the
    reader does not flicker between two hands.
    """

    TASKS_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task")
    TASKS_MODEL_FILE = "hand_landmarker.task"

    def __init__(self, num_hands=2):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "Real mode needs the 'mediapipe' package.\n"
                "Install it with:  pip install mediapipe\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc
        self._mp = mp
        self._hands = None       # legacy solutions Hands, when present
        self._landmarker = None  # Tasks API HandLandmarker, the fallback
        self._frame_index = 0
        self._last_wrist = None

        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            self._hands = solutions.hands.Hands(
                static_image_mode=False, max_num_hands=num_hands,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            return

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError(
                "This mediapipe build has neither the classic Hands "
                "(mp.solutions) nor the Tasks API.\n"
                "Fix:  pip install --upgrade mediapipe\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc
        model_path = self._ensure_tasks_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        try:
            self._landmarker = mp_vision.HandLandmarker.create_from_options(
                options)
        except Exception as exc:
            raise RuntimeError(
                "The MediaPipe Tasks HandLandmarker failed to start "
                f"({exc}). On Linux the system OpenGL libraries may be "
                "missing: point LD_LIBRARY_PATH at the mesa libEGL / "
                "libGLESv2 libraries (see README).\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc

    def _ensure_tasks_model(self) -> str:
        """Download the HandLandmarker model next to this script."""
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, self.TASKS_MODEL_FILE)
        if not os.path.exists(path):
            print(f"Downloading hand model ({self.TASKS_MODEL_FILE}) ...")
            urllib.request.urlretrieve(self.TASKS_MODEL_URL, path)
            print("Hand model downloaded.")
        return path

    def close(self):
        """Release the landmarker explicitly (avoids a noisy
        interpreter-shutdown error from the Tasks API dispatcher)."""
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception:
            pass
        try:
            if self._hands is not None:
                self._hands.close()
        except Exception:
            pass

    def sample(self, frame: np.ndarray):
        """Return (landmarks, handedness, all_hands) for the primary hand.

        landmarks is a list of 21 (x, y) pixel points, handedness is
        "Left"/"Right"/"", all_hands is a list of landmark lists for
        drawing. Returns (None, "", []) when no hand is found.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        hands = []  # (landmarks, handedness)
        if self._hands is not None:
            res = self._hands.process(rgb)
            if res.multi_hand_landmarks:
                for i, hand_lm in enumerate(res.multi_hand_landmarks):
                    label = ""
                    if res.multi_handedness and i < len(res.multi_handedness):
                        label = res.multi_handedness[i].classification[0].label
                    hands.append((
                        [(p.x * w, p.y * h)
                         for p in hand_lm.landmark], label))
        else:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_index += 1
            result = self._landmarker.detect_for_video(
                mp_image, self._frame_index * 33)
            for i, hand_lm in enumerate(result.hand_landmarks):
                label = ""
                if i < len(result.handedness):
                    label = result.handedness[i][0].category_name
                hands.append((
                    [(p.x * w, p.y * h) for p in hand_lm], label))
        if not hands:
            self._last_wrist = None
            return None, "", []
        # Primary hand: nearest wrist to the previous primary wrist
        # (continuity), else the largest hand.
        if self._last_wrist is not None:
            best = min(hands, key=lambda hl: _dist(hl[0][WRIST],
                                                   self._last_wrist))
        else:
            def _area(hl):
                xs = [p[0] for p in hl[0]]
                ys = [p[1] for p in hl[0]]
                return (max(xs) - min(xs)) * (max(ys) - min(ys))
            best = max(hands, key=_area)
        self._last_wrist = best[0][WRIST]
        return best[0], best[1], [hl[0] for hl in hands]


# ---------------------------------------------------------------------------
# Simulation mode: rendered hand + pixel-only feature estimation
# ---------------------------------------------------------------------------

def _smoothstep(a: float, b: float, t: float) -> float:
    """Smooth 0..1 interpolation between keyframe values a and b."""
    t = max(0.0, min(1.0, t))
    t = t * t * (3 - 2 * t)
    return a + (b - a) * t


@dataclass
class SimHandPose:
    """Render parameters of the simulated hand.

    ``ext`` is the extension of (index, middle, ring, pinky): 1.0 =
    fully extended, 0.0 = curled into the palm. ``thumb`` is one of
    "OUT" (stuck out to the side, L), "UP" (alongside the fist, A),
    "ACROSS" (lying over the fingers), "TUCKED" (hidden) or "RING"
    (thumb tip meeting the index tip, O). ``spread`` widens the gap
    between index and middle (V).
    """

    ext: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    thumb: str = "TUCKED"
    spread: float = 0.0


# The word the simulation spells, and when each letter holds.
# (letter, start_s, end_s, pose)
SIM_SCRIPT = [
    ("L", 2.0, 8.0, SimHandPose(ext=(1.0, 0.0, 0.0, 0.0), thumb="OUT")),
    ("O", 9.0, 15.0, SimHandPose(ext=(0.0, 0.0, 0.0, 0.0), thumb="RING")),
    ("V", 16.0, 22.0, SimHandPose(ext=(1.0, 1.0, 0.0, 0.0), thumb="TUCKED",
                                  spread=28.0)),
    ("E", 23.0, 29.0, SimHandPose(ext=(0.0, 0.0, 0.0, 0.0), thumb="TUCKED")),
]


class SimScene:
    """Renders the hand spelling the scripted word. Knows the letters;
    the classifier never sees them -- it only ever receives rendered
    pixels."""

    PALM_C = (640, 470)     # palm centre (x, y)
    PALM_W, PALM_H = 180, 210
    MCP_Y = (395, 380, 395, 415)   # knuckle heights: index..pinky
    MCP_X_OFF = (-62, -18, 28, 68)
    FINGER_LEN = 165
    FINGER_W = 36
    FINGER_ANG = (-14, -3, 8, 20)  # rest angles from vertical (deg)

    def pose_at(self, t: float) -> SimHandPose | None:
        """Interpolated hand pose at time t (None = no hand yet)."""
        if t < SIM_SCRIPT[0][1]:
            return None
        for i, (_, start, end, pose) in enumerate(SIM_SCRIPT):
            if start <= t < end:
                return pose
            if end <= t < (SIM_SCRIPT[i + 1][1]
                           if i + 1 < len(SIM_SCRIPT) else 30.0):
                # Morph to the next letter over 1 s.
                nxt = SIM_SCRIPT[i + 1][3] if i + 1 < len(SIM_SCRIPT) \
                    else pose
                k = _smoothstep(0.0, 1.0, (t - end) / 1.0)
                ext = tuple(_smoothstep(a, b, k)
                            for a, b in zip(pose.ext, nxt.ext))
                spread = _smoothstep(pose.spread, nxt.spread, k)
                thumb = pose.thumb if k < 0.5 else nxt.thumb
                return SimHandPose(ext=ext, thumb=thumb, spread=spread)
        return SIM_SCRIPT[-1][3]

    def _capsule(self, frame, base, tip, width, colour):
        cv2.line(frame, (int(base[0]), int(base[1])),
                 (int(tip[0]), int(tip[1])), colour, width, cv2.LINE_AA)
        cv2.circle(frame, (int(base[0]), int(base[1])), width // 2,
                   colour, -1)
        cv2.circle(frame, (int(tip[0]), int(tip[1])), width // 2,
                   colour, -1)

    def render(self, t: float) -> np.ndarray:
        frame = np.full((SIM_HEIGHT, SIM_WIDTH, 3), COLOUR_BG, np.uint8)
        # Room furniture (desaturated, so the pixel estimator never
        # mistakes it for the hand): window, clock, table line.
        cv2.rectangle(frame, (70, 80), (280, 300), (70, 76, 82), -1)
        cv2.rectangle(frame, (70, 80), (280, 300), (90, 96, 102), 3)
        cv2.circle(frame, (1130, 130), 52, (70, 76, 82), -1)
        cv2.circle(frame, (1130, 130), 52, (90, 96, 102), 3)
        cv2.line(frame, (0, 620), (SIM_WIDTH, 620), (90, 96, 102), 4)
        cv2.putText(frame, "SIMULATION -- rendered hand (logic check only)",
                    (30, 40), FONT, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        pose = self.pose_at(t)
        if pose is None:
            return frame
        cx, cy = self.PALM_C

        # Fingers: capsules from each knuckle, length by extension.
        # A curled finger (extension 0) is folded into the palm and
        # draws nothing -- from the front it is hidden behind the
        # palm, so a stub would read as a false fingertip.
        tips = []
        for i in range(4):
            mcp = (cx + self.MCP_X_OFF[i], self.MCP_Y[i])
            ang = math.radians(self.FINGER_ANG[i]
                               + (pose.spread if i < 2 else 0)
                               * (1 if i == 0 else -1) * 0.5)
            length = pose.ext[i] * self.FINGER_LEN
            tip = (mcp[0] - math.sin(ang) * length,
                   mcp[1] - math.cos(ang) * length)
            if length > 8:
                self._capsule(frame, mcp, tip, self.FINGER_W, COLOUR_SKIN)
            tips.append(tip)
        # Palm: rounded ellipse over the knuckle bases.
        cv2.ellipse(frame, (cx, cy), (self.PALM_W // 2, self.PALM_H // 2),
                    0, 0, 360, COLOUR_SKIN, -1)
        # Wrist / forearm.
        cv2.rectangle(frame, (cx - 70, cy + 80), (cx + 70, SIM_HEIGHT),
                      COLOUR_SKIN, -1)
        # Thumb.
        thumb_base = (cx - 88, cy + 10)
        if pose.thumb == "OUT":
            tip = (thumb_base[0] - 145, thumb_base[1] - 45)
            self._capsule(frame, thumb_base, tip, 34, COLOUR_SKIN)
        elif pose.thumb == "UP":
            tip = (thumb_base[0] - 18, thumb_base[1] - 135)
            self._capsule(frame, thumb_base, tip, 34, COLOUR_SKIN)
        elif pose.thumb == "ACROSS":
            tip = (thumb_base[0] + 130, thumb_base[1] - 55)
            self._capsule(frame, thumb_base, tip, 32, COLOUR_SKIN)
        elif pose.thumb == "RING":
            # Thumb tip meets the index fingertip: a ring above the
            # palm, enclosing a hole of background -- the O.
            cv2.ellipse(frame, (cx - 55, 330), (72, 62), 0, 0, 360,
                        COLOUR_SKIN, 30)
        # TUCKED: no visible thumb.
        return frame


class PixelHandEstimator:
    """Simulation-mode provider: hand features from rendered pixels.

    Segments the skin-coloured hand, then measures the same features
    the landmark path measures: extended fingers from convex-hull
    fingertips, thumb position from the side-most hull point, spread
    from the two up-pointing tips, and the ring hole of the O from
    the contour hierarchy. It never sees the scripted letters.
    """

    def sample(self, frame: np.ndarray) -> HandFeatures | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, np.array([5, 60, 90]),
                           np.array([25, 255, 255]))
        skin = cv2.morphologyEx(
            skin, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        contours, hierarchy = cv2.findContours(
            skin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 8000:
            return None
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        palm = (cx, cy)
        x, y, w, h = cv2.boundingRect(cnt)
        scale = max(60.0, h * 0.42)  # wrist-to-knuckle scale estimate

        # Ring hole (O): a child contour inside the hand contour.
        circle = False
        if hierarchy is not None:
            for i, hier in enumerate(hierarchy[0]):
                if hier[3] >= 0 and cv2.contourArea(contours[i]) > 1200:
                    circle = True
                    break

        # Fingertips: hull vertices separated by deep convexity
        # defects. A bare palm top is a smooth arc (no deep defects),
        # so it reads as no fingers -- the earlier version counted
        # the palm's own curved top as fingertips. A real finger
        # leaves a deep valley on both sides of it.
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull_idx)
        hull_pts = [tuple(p[0]) for p in cv2.convexHull(cnt,
                                                       returnPoints=True)]
        valleys = []  # (hull-order index, point, depth)
        if defects is not None:
            for d in defects.reshape(-1, 4):
                s_idx, e_idx, far_idx, depth = int(d[0]), int(d[1]), \
                    int(d[2]), d[3] / 256.0
                if depth > 0.30 * scale:
                    valleys.append((far_idx, tuple(cnt[far_idx][0]),
                                    depth))
        # Hull points in contour order; split the hull at the valleys
        # and take the highest point of each segment as a fingertip
        # candidate -- but only if it stands clearly above the
        # valleys flanking it.
        # Walk the contour; a fingertip is a hull vertex that is a
        # local minimum in y between two deep valleys.
        tips_up = []
        if valleys:
            valley_pts = [v[1] for v in valleys]
            # Candidate tips: hull points above the palm centre whose
            # nearest valleys on both sides are well below them.
            for p in hull_pts:
                if p[1] >= cy - 0.35 * scale:
                    continue
                # Distance to the nearest valley on either side.
                left = [v for v in valley_pts if v[0] < p[0]]
                right = [v for v in valley_pts if v[0] > p[0]]
                # A tip needs a valley on at least one side that is
                # clearly below it (the finger's flank).
                flanks = [v for v in valley_pts
                          if abs(v[0] - p[0]) < 1.6 * scale]
                if flanks and min(v[1] for v in flanks) - p[1] > 25:
                    tips_up.append(p)
        # Cluster nearby hull points into one tip each.
        tips_up.sort(key=lambda p: p[0])
        clusters = []
        for p in tips_up:
            if clusters and abs(p[0] - clusters[-1][0]) < 28 \
                    and abs(p[1] - clusters[-1][1]) < 40:
                old = clusters[-1]
                clusters[-1] = ((old[0] + p[0]) / 2, min(old[1], p[1]))
            else:
                clusters.append((float(p[0]), float(p[1])))
        # The thumb is a protrusion on the palm's upper-left side,
        # measured straight from the hull: the palm is widest at its
        # middle, so a thumb sticking out (or up alongside the fist)
        # is a hull point clearly left of the palm's middle edge.
        # OUT (L/Y) points mostly sideways; UP (A) points mostly up.
        thumb_tip = None
        if circle:
            thumb_state = "CIRCLE"
        else:
            mid_left = [p[0] for p in hull_pts
                        if abs(p[1] - cy) < 0.2 * scale]
            up_left = [p for p in hull_pts
                       if cy - 1.3 * scale < p[1] < cy - 0.3 * scale]
            if mid_left and up_left:
                edge = min(mid_left)
                tip = min(up_left, key=lambda p: p[0])
                if edge - tip[0] > 12:
                    # Angle of tip-from-palm-edge: 0 = sideways.
                    ang = math.degrees(math.atan2(
                        cy - tip[1], edge - tip[0]))
                    if ang < 55:
                        thumb_tip = tip
                        thumb_state = "OUT"
                    else:
                        thumb_state = "UP"
                else:
                    thumb_state = "TUCKED"
            else:
                thumb_state = "TUCKED"
        # Drop the thumb tip from the up-pointing fingertips if the
        # defect pass clustered it there too.
        if thumb_tip is not None:
            clusters = [p for p in clusters
                        if _dist(p, thumb_tip) > 30]
        thumb_extended = thumb_tip is not None
        # Up-pointing fingertips, left to right: index, middle, ring,
        # pinky (right hand, palm facing the camera). A closed ring
        # (O) replaces the fingers, so its top is not a fingertip.
        up_tips = [] if circle else list(clusters)
        reach = [0.4, 0.4, 0.4, 0.4]
        n_up = min(len(up_tips), 4)
        for i in range(n_up):
            p = up_tips[i]
            reach[i] = _dist(p, palm) / scale
        # Extended / half from the same reach line the landmark path
        # uses (the rendered hand is drawn to the same proportions).
        ext4 = tuple(r > REACH_EXTENDED for r in reach)
        half = tuple((not e) and r > REACH_HALF
                     for r, e in zip(reach, ext4))
        # Spread: angle between the two leftmost up-tips from the palm.
        spread = 0.0
        if len(up_tips) >= 2:
            v1 = (up_tips[0][0] - cx, up_tips[0][1] - cy)
            v2 = (up_tips[1][0] - cx, up_tips[1][1] - cy)
            spread = _angle_between(v1, v2)
        # Orientation: the hand is rendered upright.
        orientation = 0.0
        crossed = False
        tips = {}
        for name, i in (("index", 0), ("middle", 1),
                        ("ring", 2), ("pinky", 3)):
            if i < len(up_tips):
                tips[name] = up_tips[i]
        if thumb_tip is not None:
            tips["thumb"] = thumb_tip
        return HandFeatures(reach=tuple(reach), ext4=ext4, half=half,
                            thumb_extended=thumb_extended,
                            thumb_state=thumb_state, spread=spread,
                            crossed=crossed, orientation=orientation,
                            circle=circle, palm=palm, scale=scale,
                            tips=tips)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame, all_hands, features, current, word, reader, t,
                 fps, held):
    h, w = frame.shape[:2]
    # Skeletons of every detected hand.
    for lm in all_hands:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, (int(lm[a][0]), int(lm[a][1])),
                     (int(lm[b][0]), int(lm[b][1])), COLOUR_HAND, 2,
                     cv2.LINE_AA)
        for p in lm:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, COLOUR_HAND, -1)
    if features is not None:
        x1 = int(features.palm[0] - features.scale * 1.1)
        y1 = int(features.palm[1] - features.scale * 1.6)
        x2 = int(features.palm[0] + features.scale * 1.1)
        y2 = int(features.palm[1] + features.scale * 1.1)
        cv2.rectangle(frame, (max(0, x1), max(0, y1)),
                      (min(w - 1, x2), min(h - 1, y2)), COLOUR_OK, 2)
    # Status panel.
    cv2.rectangle(frame, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.putText(frame, "SIGN LANGUAGE RECOGNITION  --  ASL fingerspelling",
                (20, 32), FONT, 0.85, COLOUR_WORD, 2, cv2.LINE_AA)
    cv2.putText(frame, f"t={t:5.1f}s   fps={fps:4.1f}   "
                       f"committed={reader.counters['letters_committed']}",
                (20, 64), FONT, 0.6, COLOUR_WORD, 2, cv2.LINE_AA)
    # Big current letter (right side).
    letter = current if current is not None else "-"
    colour = COLOUR_LETTER if current is not None else COLOUR_NONE
    cv2.putText(frame, letter, (w - 170, 190), FONT, 4.5, colour, 8,
                cv2.LINE_AA)
    cv2.putText(frame, "CURRENT LETTER", (w - 260, 220), FONT, 0.55,
                COLOUR_WORD, 2, cv2.LINE_AA)
    if held is not None:
        frac = min(1.0, held / COMMIT_SECONDS)
        bar_w = int(180 * frac)
        cv2.rectangle(frame, (w - 260, 235), (w - 80, 250), (80, 80, 80), -1)
        cv2.rectangle(frame, (w - 260, 235), (w - 260 + bar_w, 250),
                      COLOUR_LETTER, -1)
        cv2.putText(frame, "hold to commit", (w - 260, 272), FONT, 0.5,
                    COLOUR_WORD, 1, cv2.LINE_AA)
    # Word strip.
    cv2.rectangle(frame, (0, h - 84), (w, h), (0, 0, 0), -1)
    shown = word if word else "(spell a word -- hold each letter ~1 s)"
    cv2.putText(frame, shown, (20, h - 28), FONT, 1.1, COLOUR_WORD, 2,
                cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Source selection (Windows CMD file chooser, per the standing rule)
# ---------------------------------------------------------------------------

def pick_video_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ASL fingerspelling recognition: letters + word strip.")
    parser.add_argument("--source", default=None,
                        help="video file, webcam index, or 'pick' "
                             "(default: opens the file chooser)")
    parser.add_argument("--simulate", action="store_true",
                        help="rendered hand spelling LOVE instead of a camera")
    parser.add_argument("--output", default="output.mp4",
                        help="annotated video written here")
    parser.add_argument("--log", default="events.log",
                        help="timestamped event log")
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="stop after this many seconds")
    parser.add_argument("--loop", action="store_true",
                        help="loop a short video file until --max-seconds")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a preview window")
    parser.add_argument("--hold-seconds", type=float, default=HOLD_SECONDS)
    parser.add_argument("--commit-seconds", type=float, default=COMMIT_SECONDS)
    args = parser.parse_args(argv)

    if args.simulate:
        scene = SimScene()
        estimator = PixelHandEstimator()
        fps = SIM_FPS
        total_frames = int(args.max_seconds * SIM_FPS)
        cap = None
        width, height = SIM_WIDTH, SIM_HEIGHT
        provider = None
    else:
        source = args.source
        if source is None or str(source).lower() == "pick":
            picked = pick_video_file()
            if not picked:
                print("No video file chosen. Exiting.")
                return 1
            source = picked
        if str(source).isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open video source: {source}")
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        provider = MediaPipeHandsProvider()
        scene = None
        estimator = None
        total_frames = None

    reader = SignReader(hold_seconds=args.hold_seconds,
                        commit_seconds=args.commit_seconds,
                        log_path=args.log)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"Could not open video writer: {args.output}")
        return 1

    frame_index = 0
    t_start = time.time()
    try:
        while True:
            if scene is not None:
                if frame_index >= total_frames:
                    break
                t = frame_index / SIM_FPS
                frame = scene.render(t)
                features = estimator.sample(frame)
                all_hands = []
            else:
                ok, frame = cap.read()
                if not ok:
                    if args.loop and isinstance(source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            break
                    else:
                        break
                t = frame_index / fps
                if args.max_seconds and t >= args.max_seconds:
                    break
                landmarks, _label, all_hands = provider.sample(frame)
                features = features_from_landmarks(landmarks) \
                    if landmarks is not None else None
            letter = classify_letter(features)
            current, word = reader.update(letter, t,
                                          has_hand=features is not None)
            held = None
            if letter is not None and reader.current_since is not None:
                held = t - reader.current_since
            elapsed = time.time() - t_start
            run_fps = (frame_index + 1) / max(elapsed, 1e-6)
            frame = draw_overlay(frame, all_hands, features, current,
                                 word, reader, t, run_fps, held)
            writer.write(frame)
            if not args.no_display:
                cv2.imshow("Sign Language Recognition  (Q to quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key == ord(" "):
                    reader.insert_space(t)
                elif key in (ord("c"), ord("C")):
                    reader.clear(t)
                elif key == 8:  # Backspace
                    reader.backspace(t)
            frame_index += 1
    finally:
        writer.release()
        if cap is not None:
            cap.release()
        if provider is not None:
            provider.close()
        cv2.destroyAllWindows()
        reader.close()

    c = reader.counters
    duration = c["frames"] / fps
    print("\n--- Sign Language Recognition: end-of-run summary ---")
    print(f"Frames processed:      {c['frames']} "
          f"({duration:.1f} s at {fps:.0f} fps)")
    print(f"Frames with a hand:    {c['hand_frames']} "
          f"({100.0 * c['hand_frames'] / max(1, c['frames']):.1f}%)")
    print(f"Frames with a letter:  {c['letter_frames']} "
          f"({100.0 * c['letter_frames'] / max(1, c['frames']):.1f}%)")
    print(f"Letters committed:     {c['letters_committed']}")
    print(f"Word strip:            {reader.word!r}")
    for letter in sorted(reader.letter_counts):
        print(f"  '{letter}': committed "
              f"{reader.letter_counts[letter]}x")
    print(f"Annotated video:       {args.output}")
    print(f"Event log:             {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
