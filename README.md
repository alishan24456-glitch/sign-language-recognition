# Sign Language Recognition — ASL Fingerspelling Reader

Reads American Sign Language fingerspelling (A–Z) live from a camera
or video file and builds the spelled word on screen.

- **MediaPipe Hands** tracks the hand and supplies 21 landmarks.
- A small **landmark classifier** (a geometry heuristic, *not* a
  trained neural network) reads the handshape — which fingers are
  extended, where the thumb sits, how far apart the index and middle
  fingers are, and which way the hand points — and maps it to one
  ASL letter.
- A letter held steady for **0.5 s** becomes the current letter; held
  for **0.7 s** it commits to the running word strip at the bottom of
  the screen. A gap with no hand for 1.5 s inserts a space.
- Keys in the preview window: **Q** quit, **SPACE** insert a space,
  **C** clear the word strip, **Backspace** delete the last letter.

## Run it

```bash
pip install -r requirements.txt
python sign_language_recognition.py --simulate          # rendered hand
python sign_language_recognition.py --source signing.mp4
python sign_language_recognition.py --source 0          # webcam
python sign_language_recognition.py                     # file picker
```

With no arguments the script opens a file chooser (Windows CMD
friendly). Every run writes an annotated `output.mp4` and a
timestamped `events.log`. On first run in real mode the script
downloads the MediaPipe HandLandmarker model (`hand_landmarker.task`,
~8 MB) next to itself.

> **Linux note:** newer `mediapipe` builds need the system OpenGL
> libraries (`libEGL.so.1`, `libGLESv2.so.2`). If the landmarker fails
> to start with `libEGL.so.1: cannot open shared object file`, point
> `LD_LIBRARY_PATH` at your mesa libraries, e.g. an extracted
> `libegl-mesa0` / `libgles2` pair. On Windows `pip install mediapipe`
> just works.

## What the demo shows (honest note)

**Real-footage demo** (`../project-demos/sign-language-recognition-demo.mp4`,
29.5 s, 1280×720): the reader runs on free-license Pexels footage of
a person signing with both hands against a white wall (video 10373877,
`sign-language-raw.mp4`, 7.4 s looped 4× — see
`../project-demos/real-footage/raw/SOURCES.md`). Result: **885 frames,
hand in 821 (92.8%), 8 letters committed, word strip "BLBLBLBL"**
(B and L once per loop).

Read that result for what it is:

- The footage shows a person **signing words with both hands**, not
  fingerspelling a word. The reader follows the largest/steadiest
  hand and reads whatever letter its shape matches at each moment.
  **"BLBLBLBL" is not the signed word** — it is the sequence of held
  handshapes (a flat four-finger hand = B, then an index-plus-thumb
  hand = L) repeating once per loop. The strip is the tool's honest
  output on that footage, not a transcription of the signing.
- Only 108 of 885 frames (12.2%) show a letter at all: the signer
  moves continuously, and most shapes are transitional (a brief U,
  a 0.27 s hooked-index X). The 0.5 s hold / 0.7 s commit filters
  those out, so only the two genuinely held shapes commit.
- The 0.7 s commit line sits between a passing shape (~0.3 s) and a
  genuinely held letter (~0.8 s and up) on this footage.

**Simulation** (`--simulate`, internal logic check only — never the
delivered demo): a rendered hand spells **"LOVE"** — L, O, V, E held
6 s each, all four committed, word strip "LOVE". The pixel estimator
reads the rendered hand from skin segmentation, convex-hull
fingertips, thumb position and the ring hole of the O; it never sees
the scripted letters. Verified end to end in this repo's run:
900 frames, hand in 840 (93.3%), 4 letters committed.

## What this is, honestly

The classifier is a hand-written rule set over finger geometry. It
reads the clear, static handshapes (A, B, C, D, F, G, H, I, K, L, O,
R, S, T, U, V, W, X, Y) and is upfront about its limits:

- **J and Z need motion** (J flicks the pinky, Z draws a Z). A still
  frame cannot tell them apart, so J reads as I and Z reads as G.
- **E, M and N** differ only by exactly where the thumb hides under
  the curled fingers. Only a very tight tucked fist reads as E; a
  looser one reads as no letter rather than a guessed M or N.
- **Two-handed signs are not fingerspelling** — see the demo note
  above.
- This is **not** a trained ASL model and its letters are **not**
  verified against a labelled dataset. Anyone wanting production
  accuracy should train a classifier on labelled landmark data; this
  project is the working skeleton for that — landmarks in, letters
  out, with the plumbing (hand tracking, hold-to-commit, word strip)
  already done.

## How a letter is decided

Landmarks (MediaPipe Hands layout): wrist 0, thumb 1–4, index 5–8,
middle 9–12, ring 13–16, pinky 17–20. All measures are normalised by
the hand scale (wrist to middle-finger knuckle).

- **Reach** — fingertip distance from the palm centre, in hand
  scales. Extended > 1.10 (or > 0.95 with the tip clearly past the
  knuckle joint), half-curled 0.80–1.10, curled < 0.80. Thresholds
  tuned on real MediaPipe landmarks: a fully extended finger measures
  1.29–1.70, a curled one 0.05–0.86.
- **Thumb state** — UP (thumb alongside the fist, A), ACROSS (thumb
  lying over the fingers, S), BETWEEN (thumb poking between index and
  middle, T), TUCKED (thumb hidden, E) or OUT (thumb stuck out, L/Y).
- **Spread** — angle between index and middle fingers: wide (> 20°)
  is V/K, narrow (< 12°) is U; a flipped tip order is R (crossed).
- **Orientation** — middle-finger direction from image-up: sideways
  is G/H, down is P/Q.
- **Circle** — thumb tip touching the index fingertip: F when the
  other three fingers are up, O when the hand is a fist.

## Bugs found and fixed during verification

1. **The L handshape read as nothing.** The L index finger measured
   a reach of 1.09–1.21, just under the first 1.25 extended line,
   because the outstretched thumb shifts the palm centre. Fixed with
   a straightness cross-check (fingertip past the knuckle joint) and
   a 1.10 reach line, tuned on real landmark data.
2. **The simulation's palm top read as fingertips.** The first pixel
   estimator counted the palm ellipse's own curved top as three
   fingertips, so the E fist read as two extended fingers. Fixed by
   finding fingertips only at deep convexity defects (a real finger
   leaves a deep valley on both sides; a bare palm top is smooth).
3. **The simulation's O read as no letter.** The ring's top was
   counted as an extended index finger, so circle + one finger up
   matched no letter. Fixed: when a closed ring (hole) is detected,
   the ring replaces the fingers and the classifier reads O.
4. **The simulation's E read as A.** The estimator mistook the
   palm's own left edge for a thumb bump. Fixed: a thumb counts only
   as a protrusion clearly left of the palm's middle edge, with
   OUT vs UP decided by its angle (sideways vs up).
5. **Word spaces inserted while a hand was present but unreadable.**
   The gap timer ran on "no letter" instead of "no hand", so an
   ambiguous fist inserted a space. Fixed: the timer only runs when
   no hand is detected at all.
6. **MediaPipe Tasks API shutdown traceback.** The landmarker raised
   a `TypeError` in `__del__` at interpreter exit. Fixed by closing
   it explicitly (`provider.close()`).

## Files

- `sign_language_recognition.py` — the whole project
- `requirements.txt`, `.gitignore`
- Demo: `../project-demos/sign-language-recognition-demo.mp4` (+ H.264
  LinkedIn copy in `../project-demos/linkedin-h264/`) and
  `../project-demos/sign-language-recognition-events.log`

## Author

Abdul Hanan — Computer Vision & AI Developer
