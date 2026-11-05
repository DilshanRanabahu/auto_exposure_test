"""Auto exposure, sample logic v2: only contrast, exposure and brightness (gain).

Every round takes a new frame and separates the garment again (``separate_garment``). Any mask
it finds is used, whether the separator calls it valid or not, as long as it can be a garment: it
covers 4-90% of the frame (smaller is the label or print of a clipped white garment) and does not
run along more than 15% of a frame side. Otherwise it counts as contrast 0. The contrast is measured with that round's own garment and ring.

Each round looks the contrast up:

    contrast   0-10  -> exposure +3000, brightness +2
    contrast  10-20  -> exposure +3000, brightness +1
    contrast  20-40  -> exposure +2000
    contrast  40-70  -> exposure +1000
    contrast  > 70   -> STOP

then rechecks it:

    contrast went up       -> keep going
    contrast went down     -> back to the previous values; if we were going up, turn round and go
                              down with the same steps (exposure -3000, brightness -2, ...);
                              if we were already going down, STOP
    exposure and brightness at their limit -> STOP
    8 rounds done          -> STOP

At the end the best round is set. Only rounds whose segment is at least 70% of the largest one
take part: a print, a corner of the table cloth or the desk beside it is far smaller than the
garment and can have a far higher contrast (the print of a clipped white t-shirt: 5% of the frame,
contrast 212; the t-shirt itself: 28%, contrast 1). Among those:

    any round with contrast >= 15  -> the highest contrast (dark and coloured garments)
    all below 15 (white on white)  -> the most solid segment: area / convex hull area. Below 15 the
                                      contrast says nothing about the segment: on white briefs the
                                      clean, whole outline had contrast 1 and a mask broken by
                                      clipping had 14.

Its ``exposure`` and ``gain`` are written to calibration.json (atomically), and ``save_run``
writes every round's image and the best segment with its exposure, brightness and contrast to a
folder. The camera classes in capture.py watch calibration.json and keep the values, and so does
the scanner app.

All numbers are a first guess, to be tuned on the device. Run it with app.py.
"""

import json
import os
import time
from dataclasses import dataclass, field

import cv2 # type: ignore
import numpy as np # type: ignore

from garment_separation import (MAX_AREA_FRAC, MAX_PICK_SIDE_CONTACT, MIN_AREA_FRAC, MIN_CONTRAST, draw_overlay,
                                separate_garment)

EXPOSURE_MIN_US = 2000
EXPOSURE_MAX_US = 40000
BRIGHTNESS_MIN = 0
BRIGHTNESS_MAX = 20
DEFAULT_EXPOSURE_US = 20000     # the camera class default when calibration.json has none
MAX_ROUNDS = 8
MIN_SHARE_OF_LARGEST = 0.70     # pick_best: a segment smaller than this share of the largest one is a piece
SETTLE_S = 0.2                  # after a change, frames already in the pipe still carry the old exposure
DROP_FRAMES = 3                 # ...so this many new frames after the change are skipped as well

# (contrast below, exposure step us, brightness step); contrast at or above the last row stops.
CONTRAST_STEPS = [
    (10, 3000, 2),
    (20, 3000, 1),
    (40, 2000, 0),
    (70, 1000, 0),
]


@dataclass
class Round:
    n: int
    exposure: float
    brightness: float
    contrast: float
    solidity: float = 0.0       # segment area / convex hull area, 0-1
    separation: object = None   # SeparationResult of this round's frame (masks may be None)
    frame: object = None        # the processed frame the masks were found on
    action: str = ''


@dataclass
class AutoExposureResult:
    exposure: float = None
    brightness: float = None
    contrast: float = 0.0
    solidity: float = 0.0
    picked_by: str = ''         # 'contrast' or 'solidity'
    reason: str = ''
    found_garment: bool = False
    best: Round = None          # the round that was kept: its masks are the best segment
    saved: bool = False
    rounds: list = field(default_factory=list)


def segment_problem(res):
    """Why this mask is not a garment, or '' when it can be one. Contrast alone cannot tell: it only
    says how much the mask differs from its ring, not what the mask is.

    - Size outside 4-90% of the frame (the separator's own garment size). Smaller is a piece of the
      garment or something else on the table: when a white garment clips into the table, its dark
      label is all that is left, and its contrast (19 on a test frame) beat the whole garment's (4).
    - Along more than 15% of one frame side (the separator's MAX_PICK_SIDE_CONTACT). A garment lies
      in the middle of the table; at a low exposure a darker corner of the table cloth running along
      two sides came out as the object, with contrast 65 against the white t-shirt's 3.
    """
    if res.garment_mask is None:
        return 'no garment'
    area = res.stats.get('area_pct', 0.0)
    if not MIN_AREA_FRAC * 100 <= area <= MAX_AREA_FRAC * 100:
        return 'area %.0f%%: not a garment' % area
    side = res.stats.get('side_contact_pct', 0.0)
    if side > MAX_PICK_SIDE_CONTACT * 100:
        return 'on %.0f%% of a side: not a garment' % side
    return ''


def segment_solidity(res):
    """Segment area / area of its convex hull (0-1). A whole garment fills its hull except for the gaps
    between legs or sleeves; a mask broken up by clipping leaves holes and bays and scores lower."""
    mask = res.garment_mask
    if mask is None or not np.any(mask):
        return 0.0
    points = cv2.findNonZero(mask)
    hull = cv2.convexHull(points)
    hull_area = cv2.contourArea(hull)
    return float(np.count_nonzero(mask)) / hull_area if hull_area > 0 else 0.0


def area_pct(r):
    return float(r.separation.stats.get('area_pct', 0.0))


def pick_best(rounds):
    """(best round, 'contrast' or 'solidity'). See the module docstring. Rounds whose segment is
    smaller than MIN_SHARE_OF_LARGEST of the largest are pieces of the garment or of the table and
    are left out first; a whole garment's segment keeps its size from round to round (28-29% on a
    white t-shirt) while its print or a corner of the cloth was 4-14%."""
    largest = max(area_pct(r) for r in rounds)
    rounds = [r for r in rounds if area_pct(r) >= MIN_SHARE_OF_LARGEST * largest]
    if max(r.contrast for r in rounds) >= MIN_CONTRAST:
        return max(rounds, key=lambda r: r.contrast), 'contrast'
    return max(rounds, key=lambda r: r.solidity), 'solidity'


def segment_contrast(frame):
    """Separate the garment and return (contrast, SeparationResult). Every mask counts, valid or not,
    unless segment_problem says it is not a garment; then the contrast is 0."""
    res = separate_garment(frame)
    if segment_problem(res):
        return 0.0, res
    return float(res.stats.get('contrast', 0.0)), res


def contrast_step(contrast):
    """(exposure step, brightness step) for this contrast, or None to stop."""
    for below, d_exposure, d_brightness in CONTRAST_STEPS:
        if contrast < below:
            return d_exposure, d_brightness
    return None


class _Camera:
    """The camera calls the loop needs. A camera without gain control (or on its own auto) keeps its brightness."""

    def __init__(self, cam, pre):
        self.cam, self.pre = cam, pre
        self.has_brightness = hasattr(cam, 'set_gain') and getattr(cam, 'gain', None) is not None

    @property
    def exposure(self):
        return self.cam.exposure or DEFAULT_EXPOSURE_US

    @property
    def brightness(self):
        return self.cam.gain if self.has_brightness else None

    def apply(self, exposure, brightness):
        if exposure != self.exposure:
            self.cam.set_exposure(exposure)
        if self.has_brightness and brightness is not None and brightness != self.cam.gain:
            self.cam.set_gain(brightness)

    def frame(self, timeout=5.0):
        """A processed frame taken after the last change.

        The frame held when the change was made, and DROP_FRAMES more after it, are skipped: on the
        Daheng camera they still carried the old exposure. Waiting 0.2 s alone was not enough: a round
        set to 5000 us measured a 25000 us frame (contrast 106 instead of 21), stopped at once and left
        the camera dark. A new frame is a new array object (the capture thread replaces it; a webcam
        or a saved image returns a new one on every call).
        """
        time.sleep(SETTLE_S + self.exposure / 1e6)
        end = time.time() + timeout
        last, seen = None, 0
        while time.time() < end:
            raw = self.cam.get_latest_image(copy=False)
            if raw is not None and raw is not last:
                last, seen = raw, seen + 1
                if seen > DROP_FRAMES:
                    return self.pre(raw.copy())
            else:
                time.sleep(0.005)
        return None if last is None else self.pre(last.copy())


def _clamp(value, low, high):
    return max(low, min(high, value))


def save_calibration(path, exposure, brightness):
    """Write exposure (and gain) into calibration.json, keeping every other key; atomic (.tmp + os.replace)."""
    with open(path, 'r') as f:
        data = json.load(f)
    data['exposure'] = float(exposure)
    if brightness is not None:
        data['gain'] = float(brightness)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, path)


def run_auto_exposure(cam, pre=lambda f: f, calibration_file=None, log=print):
    """Run the logic on a live camera. ``pre`` turns a raw frame into the processed one
    (PreProcessor.process_image); ``calibration_file`` None does not save. ``result.best`` holds
    the kept round, with its frame and masks."""
    cam = _Camera(cam, pre)
    result = AutoExposureResult()
    rounds = result.rounds

    log('\n round | exposure_us | brightness | contrast | %-44s | action' % 'segment')
    direction = +1                  # +1 going up, -1 going down
    previous = None                 # the round the last step started from
    stop = None
    for n in range(1, MAX_ROUNDS + 1):
        frame = cam.frame()
        if frame is None:
            stop = 'no frame from the camera'
            break
        contrast, res = segment_contrast(frame)
        r = Round(n, cam.exposure, cam.brightness, contrast, segment_solidity(res), res, frame)
        rounds.append(r)
        base = r                    # the round the next step starts from

        if previous is not None and r.contrast < previous.contrast:
            cam.apply(previous.exposure, previous.brightness)
            base = previous
            if direction > 0:
                direction = -1
                r.action = 'contrast fell: back to %d us, now going down' % previous.exposure
            else:
                r.action = 'contrast fell: back to %d us, STOP' % previous.exposure
                stop = 'contrast fell going both ways'
        if stop is None:
            step = contrast_step(base.contrast)
            if step is None:
                r.action = r.action or 'STOP'
                stop = 'contrast above %d' % CONTRAST_STEPS[-1][0]
            else:
                exposure = _clamp(base.exposure + direction * step[0], EXPOSURE_MIN_US, EXPOSURE_MAX_US)
                brightness = (_clamp(base.brightness + direction * step[1], BRIGHTNESS_MIN, BRIGHTNESS_MAX)
                              if cam.has_brightness else None)
                if (exposure, brightness) == (base.exposure, base.brightness):
                    r.action = (r.action + '; ' if r.action else '') + 'at the limit, STOP'
                    stop = 'exposure and brightness at their limit'
                else:
                    if not r.action:
                        r.action = 'exposure %+d%s' % (
                            exposure - base.exposure,
                            ', brightness %+g' % (brightness - base.brightness) if cam.has_brightness else '')
                    previous = base
                    cam.apply(exposure, brightness)

        if segment_problem(res):
            segment = segment_problem(res)
        else:
            segment = 'area %.0f%%, solid %.2f' % (res.stats.get('area_pct', 0.0), r.solidity)
        log('%6d | %11d | %10s | %8.0f | %-44s | %s' % (
            r.n, r.exposure, '-' if r.brightness is None else '%g' % r.brightness,
            r.contrast, segment, r.action))
        if stop:
            break
    else:
        stop = '%d rounds done' % MAX_ROUNDS

    found = [r for r in rounds if not segment_problem(r.separation)]
    result.found_garment = bool(found)
    if not found:
        result.exposure, result.brightness = cam.exposure, cam.brightness
        result.reason = 'no garment found in any round'
        log('auto exposure: ' + result.reason)
        return result

    best, picked_by = pick_best(found)
    cam.apply(best.exposure, best.brightness)
    result.best, result.picked_by = best, picked_by
    result.exposure, result.brightness = best.exposure, best.brightness
    result.contrast, result.solidity = best.contrast, best.solidity
    result.reason = '%s; best round %d by %s' % (stop, best.n, picked_by)
    log('auto exposure: %s -> %d us, brightness %s (contrast %.0f, solidity %.2f)' % (
        result.reason, result.exposure, result.brightness, result.contrast, result.solidity))

    if calibration_file:
        save_calibration(calibration_file, result.exposure, result.brightness)
        result.saved = True
        log('saved to %s' % calibration_file)
    return result


def _label(image, lines, long_side=1100):
    """The image scaled to ``long_side`` with text lines on top."""
    scale = long_side / max(image.shape[:2])
    vis = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for i, text in enumerate(lines):
        y = 32 + 30 * i
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2, cv2.LINE_AA)
    return vis


def _round_values(r):
    return {'round': r.n, 'exposure': float(r.exposure),
            'brightness': None if r.brightness is None else float(r.brightness),
            'contrast': round(r.contrast, 1), 'solidity': round(r.solidity, 3),
            'area_pct': round(float(r.separation.stats.get('area_pct', 0.0)), 1), 'action': r.action}


def best_values(result):
    """The kept round's exposure, brightness, contrast and segment, as a dict (None: no garment)."""
    if result.best is None:
        return None
    values = _round_values(result.best)
    values.pop('action')
    values.update(best_round=values.pop('round'), cue=result.best.separation.cue,
                  picked_by=result.picked_by, reason=result.reason)
    return values


def best_image(result, rgb=False, title='BEST'):
    """The kept round's frame with its segment and its exposure, brightness and contrast written on it."""
    b = result.best
    frame = cv2.cvtColor(b.frame, cv2.COLOR_RGB2BGR) if rgb and b.frame.ndim == 3 else b.frame
    return _label(draw_overlay(frame, b.separation), [
        '%s: exposure %d us   brightness %s' % (title, b.exposure, '-' if b.brightness is None else '%g' % b.brightness),
        'contrast %.0f   solidity %.2f   area %.0f%%' % (b.contrast, b.solidity, b.separation.stats.get('area_pct', 0.0)),
        'round %d, picked by %s' % (b.n, result.picked_by)])


def save_run(result, folder, log_lines=(), rgb=False):
    """Save one run to ``folder``: round<N>_<exposure>us.jpg for every round (its frame and segment),
    BEST_round<N>.jpg and best.json (the kept round's exposure, brightness, contrast, solidity and
    area, plus every round), and log.txt. ``rgb``: the frames are RGB (Daheng), the images are BGR."""
    os.makedirs(folder, exist_ok=True)
    for r in result.rounds:
        frame = cv2.cvtColor(r.frame, cv2.COLOR_RGB2BGR) if rgb and r.frame.ndim == 3 else r.frame
        cv2.imwrite(os.path.join(folder, 'round%d_%dus.jpg' % (r.n, r.exposure)), _label(
            draw_overlay(frame, r.separation),
            ['round %d: exposure %d us   brightness %s' % (r.n, r.exposure,
                                                          '-' if r.brightness is None else '%g' % r.brightness),
             'contrast %.0f   solidity %.2f   area %.0f%%' % (r.contrast, r.solidity,
                                                              r.separation.stats.get('area_pct', 0.0)),
             '-> ' + r.action]))
    with open(os.path.join(folder, 'log.txt'), 'w') as f:
        f.write('\n'.join(log_lines) + '\n')
    data = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'best': best_values(result),
            'rounds': [_round_values(r) for r in result.rounds]}
    if result.best is None:
        data['reason'] = result.reason
    else:
        cv2.imwrite(os.path.join(folder, 'BEST_round%d.jpg' % result.best.n), best_image(result, rgb))
    with open(os.path.join(folder, 'best.json'), 'w') as f:
        json.dump(data, f, indent=4)
    return data
