"""Auto exposure: the reference image's values, and the camera frame with the garment separated.

Reference: the reference image (image.png) is segmented and measured on the garment only, and the
values are saved in calibration.json under "reference_image" (the other keys are kept):

    brightness = mean intensity of the garment (0-255)
    exposure   = log2(brightness / 118): stops from middle grey (0 = middle grey, + brighter, - darker)
    contrast   = mean of the garment - mean of the ring of table around it

The reference also keeps the camera settings to start the auto exposure from (camera_exposure_us,
camera_gain): --exposure / --gain, else the ones in calibration.json when it is measured.

Camera: camera -> PreProcessor (rotate, crop, undistort from calibration.json) -> separate_garment

Auto exposure (run_auto_exposure), the flowchart:

    camera at the reference's exposure and gain, photo, measure the garment
    |camera contrast| - |reference contrast| > 0 ?
        Yes -> exposure kept
        No  -> gain locked; while |c| < MIN_CONTRAST the exposure goes down one step (x EXPOSURE_FACTOR),
               a new photo after every step. When |c| never gets there (exposure limit, MAX_ROUNDS
               photos, or |c| falling again after a garment was found) the photo with the largest
               |c| is kept.
    A segment that is NOT VALID, or runs along more than MAX_SIDE_CONTACT_PCT of a frame side, counts
    as no garment (|c| = 0): see segment_problem.
    exposure locked; then the gain ("brightness" in the flowchart), one step at a time, a new photo
    after every step:
        i > HIGH_INTENSITY and i < MAX_INTENSITY -> set the gain, done
        i > HIGH_INTENSITY and i >= MAX_INTENSITY -> gain down one step, check again
        i <= HIGH_INTENSITY: while i < MIN_INTENSITY gain up one step; then set it
    (i = mean intensity of the garment, c = mean of the garment - mean of the ring.) At the end the
    exposure and gain are set on the camera and saved to calibration.json. A step past the gain limits,
    MAX_ROUNDS photos (both parts together) or a photo without a garment stops the gain loop and keeps
    the last good gain.

    python3 auto_exposure.py --check             # check the setup and save one segmented frame
    python3 auto_exposure.py                     # window: live view; a = auto exposure, s = save, q = quit
    python3 auto_exposure.py --once              # one frame, no window: segment it, save it and exit
    python3 auto_exposure.py --auto              # run the auto exposure once, save, set the camera, exit
    python3 auto_exposure.py --auto --no-save    # the same without writing calibration.json
    python3 auto_exposure.py --reference         # measure image.png and save it to calibration.json
    python3 auto_exposure.py --reference other.png --exposure 25000 --gain 12
                                                 # another reference image, shot at these settings
    --calibration <file>                         # another calibration.json (default: the one here)
    --out <folder>                               # where images go (default: captures/ here)

On a production PC with the camera (Linux, Daheng SDK / gxipy installed):

    sudo systemctl stop rms-app                  # the camera can only be open in one process
    cp <the device's calibration.json> .         # its crop and rotation are this camera's
    python3 auto_exposure.py --reference         # the reference goes into that copy
    python3 auto_exposure.py --check
    python3 auto_exposure.py --auto              # or the window and 'a'
    sudo systemctl start rms-app                 # when done

Working on the copy keeps the scanner app's own calibration.json untouched. Every auto exposure run
saves an image of each round and log.txt to <out>/auto_<time>/.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2 # type: ignore

from capture import PreProcessor, WebcamCapture
from garment_separation import draw_overlay, separate_garment

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(HERE, 'calibration.json')
REFERENCE_IMAGE = os.path.join(HERE, 'image.png')
REFERENCE_KEY = 'reference_image'    # the reference's key in calibration.json
MIDDLE_GREY = 118
DISPLAY_LONG_SIDE = 1100

# Auto exposure: the exposure part (flowchart "No": the garment does not stand out more than the reference)
MIN_CONTRAST = 20               # |c| below this: exposure down one step
MAX_SIDE_CONTACT_PCT = 90.0     # a segment along more of one frame side than this is the table or the room
EXPOSURE_FACTOR = 0.8           # one step: x0.8 (about 1/3 stop darker), the same at any exposure
EXPOSURE_MIN_US = 2000          # the camera's own range narrows these when it is smaller
EXPOSURE_MAX_US = 150000

# Auto exposure: the gain part (flowchart thresholds, mean intensity of the garment 0-255)
HIGH_INTENSITY = 60             # i above this (and below MAX_INTENSITY): the gain is right
MAX_INTENSITY = 250             # i at or above this: gain down
MIN_INTENSITY = 20              # i below this: gain up
GAIN_STEP = 1.0                 # dB on the Daheng camera; 1 dB = x1.12 brighter
GAIN_MIN = 0.0                  # the camera's own range narrows these when it is smaller
GAIN_MAX = 24.0
MAX_ROUNDS = 12                 # photos in one run, the first one included
DEFAULT_EXPOSURE_US = 20000     # capture.py's defaults when calibration.json has none
DEFAULT_GAIN = 10.0

# Waiting for a frame shot at new settings: the Daheng camera keeps delivering frames at the old
# ones for a while (up to 1-2 s) after a change.
SETTLE_S = 0.2                  # always wait this long first
MIN_NEW_FRAMES = 2              # and at least this many new frames
CHANGE_SEEN = 0.5               # a frame has moved this share of the expected change: it is new
FALLBACK_WAIT_S = 1.0           # change too small or clipped to see: wait this long instead
CHANGE_TIMEOUT_S = 6.0          # never wait longer than this


# =========================================================================================
# Reference
# =========================================================================================
def read_calibration(calibration_file=CALIBRATION_FILE):
    """calibration.json as a dict ({} when it does not exist)."""
    if not os.path.exists(calibration_file):
        return {}
    with open(calibration_file, 'r') as f:
        return json.load(f)


def update_calibration(updates, calibration_file=CALIBRATION_FILE):
    """Write ``updates`` into calibration.json, keeping every other key.
    Atomic: written to a .tmp file first and moved over the old one."""
    data = read_calibration(calibration_file)
    data.update(updates)
    tmp = calibration_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, calibration_file)


def measure_reference(path=REFERENCE_IMAGE, exposure_us=None, gain=None, calibration_file=CALIBRATION_FILE):
    """Segment the reference image and measure its garment. Returns (values, image, SeparationResult).

    values: image, brightness (mean of the garment), exposure (log2(brightness / 118)), contrast
    (garment - ring), the camera settings the auto exposure starts from (``exposure_us`` / ``gain``,
    else calibration.json's) and measured_at. Raises when the image cannot be read or has no garment.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError('reference image not found: %s' % path)
    res = separate_garment(image)
    garment = garment_values(image, res)                # imread gives BGR
    if garment is None:
        raise ValueError('no garment found in the reference image: %s' % '; '.join(res.problems))
    calibration = read_calibration(calibration_file)
    values = {'image': os.path.basename(path), **garment,
              'camera_exposure_us': float(exposure_us or calibration.get('exposure', DEFAULT_EXPOSURE_US)),
              'camera_gain': float(gain if gain is not None else calibration.get('gain', DEFAULT_GAIN)),
              'measured_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    return values, image, res


def save_reference(values, calibration_file=CALIBRATION_FILE):
    """Write ``values`` into calibration.json under REFERENCE_KEY, keeping every other key."""
    update_calibration({REFERENCE_KEY: values}, calibration_file)


def load_reference(calibration_file=CALIBRATION_FILE):
    """The reference values saved in calibration.json, or None when there are none yet."""
    return read_calibration(calibration_file).get(REFERENCE_KEY)


# =========================================================================================
# Camera frame -> garment
# =========================================================================================
def wait_frame(cam, timeout=5.0):
    """The camera's latest frame, waiting up to ``timeout`` seconds for the first one."""
    end = time.time() + timeout
    while time.time() < end:
        frame = cam.get_latest_image()
        if frame is not None:
            return frame
        time.sleep(0.02)
    return None


def segment(raw, pre, edge_ok):
    """(processed frame, SeparationResult) for one raw camera frame."""
    processed = pre.process_image(raw)
    return processed, separate_garment(processed, edge_ok=edge_ok)


def measure(frame, res, rgb=False):
    """(contrast c, intensity i) of a frame with its SeparationResult, or (None, None) without a garment.

    i = mean intensity of the detected garment
    c = mean intensity of the detected garment - mean intensity of the detected ring
        (negative: the garment is darker than the table around it)

    ``rgb``: the frame is RGB (Daheng camera), else BGR (cv2.imread). The grey level then comes out the
    same for the camera and the reference image.
    """
    if res.garment_mask is None or res.ring_mask is None:
        return None, None
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)
    garment_px, ring_px = gray[res.garment_mask > 0], gray[res.ring_mask > 0]
    if not garment_px.size or not ring_px.size:
        return None, None
    i = float(garment_px.mean())
    c = i - float(ring_px.mean())
    return c, i


def garment_values(frame, res, rgb=False):
    """The garment's brightness, exposure and contrast, as for the reference, or None without a garment.

        brightness = mean intensity of the garment (0-255)
        exposure   = log2(brightness / 118): stops from middle grey
        contrast   = mean of the garment - mean of the ring
    """
    c, i = measure(frame, res, rgb)
    if i is None:
        return None
    return {'brightness': round(i, 1),
            'exposure': round(math.log2(max(i, 1.0) / MIDDLE_GREY), 2),
            'contrast': round(c, 1)}


def difference(camera, reference):
    """Reference minus camera, or None when either is missing (no garment, or no reference saved yet).

    ``camera`` is garment_values() of a camera frame, ``reference`` is load_reference().

        brightness = reference brightness - camera brightness   (+: the camera's garment is darker)
        exposure   = reference exposure - camera exposure       (stops; +: the camera's garment is darker)
        contrast   = |reference contrast| - |camera contrast|   (+: the camera's garment stands out less)

    The contrast uses its size only: a dark garment on a light table and a light one on a dark table
    have opposite signs but can stand out just as much.
    """
    if camera is None or reference is None:
        return None
    return {'brightness': round(reference['brightness'] - camera['brightness'], 1),
            'exposure': round(reference['exposure'] - camera['exposure'], 2),
            'contrast': round(abs(reference['contrast']) - abs(camera['contrast']), 1)}


def format_values(values):
    """One line of garment_values for the window and the terminal."""
    if values is None:
        return 'brightness -   exposure -   contrast -'
    return 'brightness %.1f   exposure %+.2f   contrast %+.1f' % (
        values['brightness'], values['exposure'], values['contrast'])


def overlay(processed, res, cam):
    """The processed frame (RGB from the Daheng camera, shown as BGR) with the segment and the values."""
    vis = draw_overlay(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR), res)
    scale = DISPLAY_LONG_SIDE / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = ['camera: %s us, gain %s' % (cam.exposure, cam.gain)]
    if res.garment_mask is None:
        lines.append('no garment: %s' % '; '.join(res.problems[:1]))
    else:
        lines.append('%s   cue %s   area %.0f%%' % (
            'VALID' if res.valid else 'NOT VALID', res.cue, res.stats.get('area_pct', 0.0)))
        lines.append(format_values(garment_values(processed, res, rgb=True)))
        problem = segment_problem(res)
        if problem:                                     # the auto exposure counts it as no garment
            lines.append('NOT USED: ' + problem)
    for n, text in enumerate(lines):
        y = 30 + 28 * n
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
    return vis


def save(folder, raw, processed, vis):
    """Save the raw frame, the processed frame and the overlay as BGR jpgs; returns the name prefix."""
    os.makedirs(folder, exist_ok=True)
    prefix = os.path.join(folder, time.strftime('%Y%m%d_%H%M%S'))
    cv2.imwrite(prefix + '_raw.jpg', cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(prefix + '_processed.jpg', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    cv2.imwrite(prefix + '_overlay.jpg', vis)
    return prefix


# =========================================================================================
# Auto exposure
# =========================================================================================
def _level(raw):
    """Mean brightness of a raw frame (every 8th pixel), or None."""
    return None if raw is None else float(raw[::8, ::8].mean())


def exposure_limits(cam):
    """(min, max) exposure in us: EXPOSURE_MIN_US / EXPOSURE_MAX_US, narrowed to the camera's own range."""
    low, high = EXPOSURE_MIN_US, EXPOSURE_MAX_US
    try:
        rng = cam.cam.ExposureTime.get_range()          # gxipy: {'min': .., 'max': .., ...}
        low, high = max(low, float(rng['min'])), min(high, float(rng['max']))
    except Exception:
        pass
    return low, high


def gain_limits(cam):
    """(min, max) gain: GAIN_MIN / GAIN_MAX, narrowed to the camera's own range when it has one."""
    low, high = GAIN_MIN, GAIN_MAX
    try:
        rng = cam.cam.Gain.get_range()                  # gxipy: {'min': .., 'max': .., ...}
        low, high = max(low, float(rng['min'])), min(high, float(rng['max']))
    except Exception:
        pass
    return low, high


def apply_settings(cam, exposure_us, gain):
    """Set the exposure and gain on the camera. Returns what next_frame needs to see the change:
    (when, frame level before, expected brightness ratio), or None when nothing changed."""
    old_exposure, old_gain = cam.exposure or DEFAULT_EXPOSURE_US, cam.gain
    before = _level(cam.get_latest_image(copy=False))
    if exposure_us != old_exposure:
        cam.set_exposure(exposure_us)
    if gain != old_gain:
        cam.set_gain(gain)
    ratio = exposure_us / old_exposure
    if old_gain is not None:
        ratio *= 10 ** ((gain - old_gain) / 20)         # gain is in dB
    return None if ratio == 1 else (time.time(), before, ratio)


def next_frame(cam, change):
    """A raw frame shot at the settings set last (``change`` from apply_settings).

    Brightness follows exposure and gain, so a frame that has moved CHANGE_SEEN of the way from the
    level before the change towards the expected one is shot at the new settings, and the frame after
    it is returned. When the change cannot be seen (too small, or clipped at 0 or 255) the first frame
    after FALLBACK_WAIT_S is taken. A new frame is a new array (the capture thread replaces it).
    """
    start = time.time()
    time.sleep(SETTLE_S + (cam.exposure or DEFAULT_EXPOSURE_US) / 1e6)
    visible, before, expected = False, None, None
    if change is not None:
        _, before, ratio = change
        if before is not None:
            expected = min(255.0, before * ratio)
            visible = 3 < before < 250 and 3 < expected < 250 and abs(expected - before) >= 8
    last, seen, moved = None, 0, False
    while time.time() - start < CHANGE_TIMEOUT_S:
        raw = cam.get_latest_image(copy=False)
        if raw is None or raw is last:
            time.sleep(0.005)
            continue
        last, seen = raw, seen + 1
        if seen < MIN_NEW_FRAMES:
            continue
        if change is None or (visible and moved):
            return raw.copy()
        if visible:
            moved = abs(_level(raw) - before) >= CHANGE_SEEN * abs(expected - before)
        elif time.time() - change[0] >= FALLBACK_WAIT_S:
            return raw.copy()
    return None if last is None else last.copy()


def clamp(value, low, high):
    return min(high, max(low, value))


def segment_problem(res):
    """Why the auto exposure must not use this segment, or '' when it can. A segment it cannot use
    counts as "no garment" (|c| = 0), so the flow lowers the exposure.

    1. The separator calls it NOT VALID: on the test images every one of these was a wrong segment
       (a print, a corner, a small blob).
    2. It runs along more than MAX_SIDE_CONTACT_PCT of one frame side, even with a crop. On a device a
       white shirt clipped into a white sheet; only the darker left edge of the sheet was left and was
       taken as the garment (100% of the left side). Real garments that run off a side covered 47-55%.
    """
    if res is None or res.garment_mask is None:
        return 'no garment'
    if not res.valid:
        return 'not valid: %s' % '; '.join(res.problems)
    side = res.stats.get('side_contact_pct', 0.0)
    if side > MAX_SIDE_CONTACT_PCT:
        return 'along %.0f%% of a frame side: the table or the room' % side
    return ''


def contrast_size(values):
    """|c| of a photo. A photo without a garment counts as |c| = 0."""
    return 0.0 if values is None else abs(values['contrast'])


def contrast_above_reference(values, reference):
    """Flowchart: "contrast difference (w.r.t. reference) > 0 ?"

        |camera contrast| - |reference contrast| > 0

    difference() gives |reference| - |camera|, so that is difference < 0. No garment: False (No).
    """
    if values is None:
        return False
    return difference(values, reference)['contrast'] < 0


def brightness_step(i):
    """Flowchart: the brightness boxes, for the garment's mean intensity i.
    Returns (step, why): -1 gain down, +1 gain up, 0 the gain is right."""
    if i > HIGH_INTENSITY:
        if i < MAX_INTENSITY:
            return 0, 'i = %.1f is in %d-%d' % (i, HIGH_INTENSITY, MAX_INTENSITY)
        return -1, 'i >= %d: gain down' % MAX_INTENSITY
    if i < MIN_INTENSITY:
        return +1, 'i < %d: gain up' % MIN_INTENSITY
    return 0, 'i = %.1f is not below %d' % (i, MIN_INTENSITY)


class AutoExposure:
    """One auto exposure run. run() follows the flowchart from top to bottom:

        1. photo at the reference's settings
        2. contrast above the reference?   Yes -> keep the exposure
                                           No  -> lower_exposure()
        3. set_brightness()                (the gain)
        4. set the camera and save to calibration.json

    Every photo is one "round" in self.rounds (settings, values, difference, action).
    """

    def __init__(self, cam, pre, edge_ok, reference, calibration_file=CALIBRATION_FILE, out=None, log=print):
        self.cam, self.pre, self.edge_ok = cam, pre, edge_ok
        self.reference = reference
        self.calibration_file = calibration_file    # None: never save
        self.out = out                              # folder for every round's image and log.txt
        self.log = log
        self.rounds, self.lines = [], []

        # Limits (narrowed to the camera's own range when it has one)
        self.exposure_min, self.exposure_max = exposure_limits(cam)
        self.gain_min, self.gain_max = gain_limits(cam)

        # Start from the camera settings saved with the reference
        self.exposure = clamp(float(reference.get('camera_exposure_us') or DEFAULT_EXPOSURE_US),
                              self.exposure_min, self.exposure_max)
        self.gain = clamp(float(reference.get('camera_gain', DEFAULT_GAIN)), self.gain_min, self.gain_max)

    # ------------------------------------------------------------------ the flowchart
    def run(self):
        self.say('reference: %s, start at %d us, gain %g'
                 % (format_values(self.reference), self.exposure, self.gain))

        # 1. Photo at the reference's settings
        values = self.take_photo()

        # 2. Contrast difference (w.r.t. reference) > 0 ?
        if contrast_above_reference(values, self.reference):
            exposure_note = 'contrast above the reference: exposure kept'      # Yes: lock the exposure
        else:
            values, exposure_note = self.lower_exposure(values)                 # No

        # No garment in any photo: nothing to set
        if values is None:
            apply_settings(self.cam, self.exposure, self.gain)
            return self.finish('%s; camera back to the reference settings, nothing saved' % exposure_note,
                               saved=False)

        # 3. The exposure is locked; set the brightness (gain)
        gain_note = self.set_brightness(values)

        # 4. Set the camera and save
        apply_settings(self.cam, self.exposure, self.gain)
        saved = self.save_settings()
        return self.finish('%s; %s' % (exposure_note, gain_note), saved)

    # ------------------------------------------------------------------ flowchart "No"
    def lower_exposure(self, values):
        """Lock the brightness (gain); while |c| < MIN_CONTRAST lower the exposure one step.

        A photo without a garment counts as |c| = 0: a white garment on a white table clipped at 255
        is not found at all until the exposure is low enough to show its outline.
        Once a garment was found, a lower |c| means the exposure went past its best: stop there.
        If |c| never reaches MIN_CONTRAST, the photo with the largest |c| is kept.
        Returns (values of the photo kept or None, note).
        """
        start_exposure = self.exposure
        first = len(self.rounds) - 1                    # this part's photos: self.rounds[first:]
        best_size = contrast_size(values)               # the largest |c| so far (0: no garment yet)

        while contrast_size(values) < MIN_CONTRAST:
            if best_size > 0 and contrast_size(values) < best_size:
                stop = '|c| fell after it was %.1f' % best_size
                break
            new_exposure = max(self.exposure_min, round(self.exposure * EXPOSURE_FACTOR))
            if new_exposure == self.exposure:
                stop = 'exposure at its limit (%d us)' % self.exposure
                break
            if self.out_of_photos():
                stop = '%d photos done' % MAX_ROUNDS
                break
            # one step down, new photo
            why = 'no garment' if values is None else '|c| < %d' % MIN_CONTRAST
            self.rounds[-1]['action'] = why + ': exposure down'
            self.exposure = new_exposure
            values = self.take_photo()
            best_size = max(best_size, contrast_size(values))
        else:
            # |c| reached MIN_CONTRAST: lock this exposure
            return values, '|c| = %.1f is not below %d at %d us' % (contrast_size(values), MIN_CONTRAST,
                                                                     self.exposure)

        # |c| never reached MIN_CONTRAST: keep the photo with the largest |c|
        found = [r for r in self.rounds[first:] if r['values'] is not None]
        if not found:
            self.exposure = start_exposure
            return None, '%s; no garment in any photo' % stop
        best = max(found, key=lambda r: contrast_size(r['values']))
        self.exposure = best['exposure_us']
        return best['values'], '%s; |c| stayed below %d, kept round %d (|c| %.1f, %d us)' % (
            stop, MIN_CONTRAST, best['round'], contrast_size(best['values']), self.exposure)

    # ------------------------------------------------------------------ brightness (gain)
    def set_brightness(self, values):
        """Change the gain one step at a time until brightness_step() says it is right.
        A photo without a garment goes back to the last good gain. Returns a note."""
        good_gain = self.gain                           # the last gain whose photo had a garment
        while True:
            step, why = brightness_step(values['brightness'])
            if step == 0:
                return why
            new_gain = clamp(self.gain + step * GAIN_STEP, self.gain_min, self.gain_max)
            if new_gain == self.gain:
                return '%s, but the gain is at its limit (%g)' % (why, self.gain)
            if self.out_of_photos():
                return '%s, but %d photos are done' % (why, MAX_ROUNDS)
            # one step, new photo
            self.rounds[-1]['action'] = why
            self.gain = new_gain
            values = self.take_photo()
            if values is None:
                self.gain = good_gain
                return 'no garment at gain %g: back to gain %g' % (new_gain, good_gain)
            good_gain = self.gain

    # ------------------------------------------------------------------ helpers
    def take_photo(self):
        """Set the current exposure and gain, wait for a frame shot at them, find the garment and
        measure it. Records the round and returns its values (None: no garment)."""
        raw = next_frame(self.cam, apply_settings(self.cam, self.exposure, self.gain))
        values, processed, res = None, None, None
        problem = 'no frame from the camera'
        if raw is not None:
            processed, res = segment(raw, self.pre, self.edge_ok)
            problem = segment_problem(res)
            if not problem:                             # a segment it cannot use counts as no garment
                values = garment_values(processed, res, rgb=True)

        r = {'round': len(self.rounds) + 1, 'exposure_us': self.exposure, 'gain': self.gain,
             'values': values, 'difference': difference(values, self.reference), 'action': ''}
        self.rounds.append(r)
        self.say('round %2d | %6d us | gain %4.1f | %s'
                 % (r['round'], self.exposure, self.gain, format_values(values) if values else problem))
        if self.out and processed is not None:
            os.makedirs(self.out, exist_ok=True)
            name = 'round%02d_%dus_gain%g.jpg' % (r['round'], self.exposure, self.gain)
            cv2.imwrite(os.path.join(self.out, name), overlay(processed, res, self.cam))
        return values

    def out_of_photos(self):
        return len(self.rounds) >= MAX_ROUNDS

    def save_settings(self):
        """Write the exposure and gain to calibration.json. Returns True when saved."""
        if not self.calibration_file:
            return False
        update_calibration({'exposure': self.exposure, 'gain': self.gain}, self.calibration_file)
        # the camera watches calibration.json; it already has these values
        if hasattr(self.cam, 'last_config_mtime'):
            self.cam.last_config_mtime = os.path.getmtime(self.calibration_file)
        return True

    def say(self, text):
        self.lines.append(text)
        self.log(text)

    def finish(self, reason, saved):
        """Log the result, write log.txt and return the result dict."""
        if self.rounds:
            self.rounds[-1]['action'] = self.rounds[-1]['action'] or 'STOP'
        self.say('auto exposure: %s -> %d us, gain %g%s' % (reason, self.exposure, self.gain,
                                                           ', saved to calibration.json' if saved else ''))
        if self.out:
            os.makedirs(self.out, exist_ok=True)
            with open(os.path.join(self.out, 'log.txt'), 'w') as f:
                f.write('\n'.join(self.lines) + '\n')
        return {'exposure_us': self.exposure, 'gain': self.gain, 'saved': saved, 'reason': reason,
                'rounds': self.rounds}


def run_auto_exposure(cam, pre, edge_ok, reference, calibration_file=CALIBRATION_FILE, out=None, log=print):
    """Run the auto exposure once on a live camera (see AutoExposure). Returns a dict: exposure_us,
    gain, saved, reason, rounds. ``out``: a folder for every round's image and log.txt.
    ``calibration_file`` None: never save."""
    return AutoExposure(cam, pre, edge_ok, reference, calibration_file, out, log).run()


def open_camera(calibration_file):
    """(camera, PreProcessor, edge_ok) for the Daheng camera with ``calibration_file``'s crop and settings."""
    pre = PreProcessor(calibration_file=calibration_file)
    edge_ok = pre.crop_bbox is not None         # cropped to the table: a garment may run off its sides
    cam = WebcamCapture(calibration_file=calibration_file)
    if not cam.start():
        sys.exit('camera did not start (see the log above). Is the scanner app still running? '
                 '(sudo systemctl stop rms-app)')
    # The camera re-reads calibration.json on its first frame and would put the file's exposure and
    # gain back over the ones the auto exposure sets; mark the file as already read.
    if os.path.exists(calibration_file):
        cam.last_config_mtime = os.path.getmtime(calibration_file)
    return cam, pre, edge_ok


def crop_problem(pre, raw):
    """Why the calibration crop does not fit this camera's frames, or '' when it does."""
    if pre.crop_bbox is None:
        return ''
    (x0, y0), (x1, y1) = pre.crop_bbox
    h, w = raw.shape[:2]
    if max(x0, x1) > w or max(y0, y1) > h:
        return ('the crop %s is outside the %dx%d frame: calibration.json is from another camera or '
                'resolution' % (pre.crop_bbox, w, h))
    return ''


def auto_once(cam, pre, edge_ok, reference, calibration_file, out):
    """One auto exposure run with every round saved to ``out``; prints the result."""
    result = run_auto_exposure(cam, pre, edge_ok, reference, calibration_file=calibration_file, out=out)
    print('result: %d us, gain %g (%d photos)%s' % (
        result['exposure_us'], result['gain'], len(result['rounds']),
        '' if result['saved'] else ', NOT saved'))
    print('rounds saved to %s' % out)
    return result


def check_setup(calibration_file, out):
    """Check everything a device run needs, step by step, and save one segmented frame."""
    def line(ok, text):
        print('[%s] %s' % ('OK  ' if ok else 'FAIL', text))
        return ok

    line(True, 'python %s, opencv %s' % (sys.version.split()[0], cv2.__version__))
    try:
        import gxipy  # noqa: F401  (Daheng SDK)
        line(True, 'gxipy found')
    except ImportError as e:
        return line(False, 'gxipy not found (%s): install the Daheng SDK' % e)

    calibration = read_calibration(calibration_file)
    if not line(bool(calibration), 'calibration file %s' % calibration_file):
        return False
    line(True, 'camera settings in it: exposure %s us, gain %s; crop %s' % (
        calibration.get('exposure'), calibration.get('gain'), calibration.get('crop_bbox')))
    reference = calibration.get(REFERENCE_KEY)
    if reference is None:
        return line(False, 'reference missing: run  python3 auto_exposure.py --reference  first')
    line(True, 'reference: %s, start at %s us, gain %s' % (
        format_values(reference), reference.get('camera_exposure_us'), reference.get('camera_gain')))

    cam, pre, edge_ok = open_camera(calibration_file)
    try:
        raw = wait_frame(cam)
        if raw is None:
            return line(False, 'camera started but no frame came')
        line(True, 'camera started, frame %dx%d' % (raw.shape[1], raw.shape[0]))
        problem = crop_problem(pre, raw)
        if not line(not problem, problem or 'crop fits the frame'):
            return False
        processed, res = segment(raw, pre, edge_ok)
        values = garment_values(processed, res, rgb=True)
        line(values is not None, 'garment: %s, cue %s, area %.0f%%%s' % (
            'VALID' if res.valid else 'NOT VALID', res.cue, res.stats.get('area_pct', 0.0),
            '' if res.valid else ' (%s)' % '; '.join(res.problems)))
        print('       %s' % format_values(values))
        if values is not None:
            print('       flowchart: %s' % ('Yes' if contrast_above_reference(values, reference) else 'No'))
        print('       frame saved: %s_*.jpg' % save(out, raw, processed, overlay(processed, res, cam)))
    finally:
        cam.stop()
    return True


def run_camera(mode, out, calibration_file=CALIBRATION_FILE, save_result=True):
    """mode 'live': window, 'a' runs the auto exposure; 'once': one frame segmented and saved;
    'auto': one auto exposure run. ``save_result`` False: the auto exposure never writes calibration.json."""
    reference = load_reference(calibration_file)
    if mode == 'auto' and reference is None:
        sys.exit('no reference in %s: run  python3 auto_exposure.py --reference  first' % calibration_file)
    target = calibration_file if save_result else None
    cam, pre, edge_ok = open_camera(calibration_file)

    try:
        raw = wait_frame(cam)
        if raw is None:
            sys.exit('no frame from the camera')
        problem = crop_problem(pre, raw)
        if problem:
            sys.exit(problem)

        if mode == 'auto':
            auto_once(cam, pre, edge_ok, reference, target,
                      os.path.join(out, 'auto_' + time.strftime('%Y%m%d_%H%M%S')))
            return

        if mode == 'once':
            processed, res = segment(raw, pre, edge_ok)
            prefix = save(out, raw, processed, overlay(processed, res, cam))
            print('camera: %s us, gain %s' % (cam.exposure, cam.gain))
            print('valid=%s cue=%s area=%.1f%% problems=%s' % (
                res.valid, res.cue, res.stats.get('area_pct', 0.0), '; '.join(res.problems) or '-'))
            print(format_values(garment_values(processed, res, rgb=True)))
            print('saved %s_*.jpg' % prefix)
            return

        print('a = auto exposure, s = save, q = quit')
        while True:
            raw = wait_frame(cam)
            if raw is None:
                print('no frame from the camera')
                break
            processed, res = segment(raw, pre, edge_ok)
            vis = overlay(processed, res, cam)
            cv2.imshow('garment separation', vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                print('saved %s_*.jpg' % save(out, raw, processed, vis))
            if key == ord('a'):
                if reference is None:
                    print('no reference in %s: run  --reference  first' % calibration_file)
                    continue
                auto_once(cam, pre, edge_ok, reference, target,
                          os.path.join(out, 'auto_' + time.strftime('%Y%m%d_%H%M%S')))
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Reference values and the camera with the garment separated.')
    parser.add_argument('--check', action='store_true',
                        help='check the setup (SDK, calibration, reference, camera, crop) and save one frame')
    parser.add_argument('--once', action='store_true', help='one frame, no window: segment, save, exit')
    parser.add_argument('--auto', action='store_true',
                        help='run the auto exposure once from the reference, save and set it, exit')
    parser.add_argument('--no-save', action='store_true', help='the auto exposure never writes calibration.json')
    parser.add_argument('--calibration', default=CALIBRATION_FILE,
                        help='calibration.json to use (crop, settings, reference; default: the copy in this folder)')
    parser.add_argument('--out', default=os.path.join(HERE, 'captures'), help='folder for saved images')
    parser.add_argument('--reference', nargs='?', const=REFERENCE_IMAGE, metavar='IMAGE',
                        help='segment a reference image, measure its garment and save it to calibration.json '
                             '(default: image.png)')
    parser.add_argument('--exposure', type=float, help='--reference: exposure (us) the image was shot at '
                                                       '(default: calibration.json)')
    parser.add_argument('--gain', type=float, help='--reference: gain the image was shot at (default: calibration.json)')
    args = parser.parse_args()
    calibration_file = os.path.abspath(args.calibration)

    if args.reference:
        try:
            values, image, res = measure_reference(args.reference, args.exposure, args.gain, calibration_file)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))
        if not res.valid:
            print('warning: the segment is not valid (%s)' % '; '.join(res.problems))
        for key, value in values.items():
            print('%-18s %s' % (key, value))
        save_reference(values, calibration_file)
        print('saved to %s under "%s"' % (calibration_file, REFERENCE_KEY))
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, 'reference_overlay.jpg')
        cv2.imwrite(path, draw_overlay(image, res))     # imread gives BGR: no conversion
        print('segment: %s' % path)
        return
    if args.check:
        sys.exit(0 if check_setup(calibration_file, args.out) else 1)
    mode = 'auto' if args.auto else 'once' if args.once else 'live'
    run_camera(mode, args.out, calibration_file, save_result=not args.no_save)


if __name__ == '__main__':
    main()
