"""Auto exposure: the reference image's values, and the camera frame with the garment separated.

Reference: the reference image (image.png) is segmented and measured on the garment only, and the
values are saved in calibration.json under "reference_image" (the other keys are kept):

    brightness = mean intensity of the garment (0-255)
    exposure   = log2(brightness / 118): stops from middle grey (0 = middle grey, + brighter, - darker)
    contrast   = mean of the garment - mean of the ring of table around it

Camera: camera -> PreProcessor (rotate, crop, undistort from calibration.json) -> separate_garment

    python auto_exposure.py                      # window: live frame with the garment separated
    python auto_exposure.py --once               # one frame, no window: segment it, save it and exit
    python auto_exposure.py --reference          # measure image.png and save it to calibration.json
    python auto_exposure.py --reference other.png  # the same with another reference image

Keys in the window:
    s         save the raw frame, the processed frame and the overlay
    q / Esc   quit

On a device stop the scanner app first (the camera can only be open in one process):

    sudo systemctl stop rms-app
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


# =========================================================================================
# Reference
# =========================================================================================
def measure_reference(path=REFERENCE_IMAGE):
    """Segment the reference image and measure its garment. Returns (values, image, SeparationResult).

    values: image, brightness (mean of the garment), exposure (log2(brightness / 118)), contrast
    (garment - ring) and measured_at. Raises when the image cannot be read or has no garment.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError('reference image not found: %s' % path)
    res = separate_garment(image)
    garment = garment_values(image, res)                # imread gives BGR
    if garment is None:
        raise ValueError('no garment found in the reference image: %s' % '; '.join(res.problems))
    values = {'image': os.path.basename(path), **garment,
              'measured_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    return values, image, res


def save_reference(values, calibration_file=CALIBRATION_FILE):
    """Write ``values`` into calibration.json under REFERENCE_KEY, keeping every other key.
    Atomic: written to a .tmp file first and moved over the old one."""
    data = {}
    if os.path.exists(calibration_file):
        with open(calibration_file, 'r') as f:
            data = json.load(f)
    data[REFERENCE_KEY] = values
    tmp = calibration_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, calibration_file)


def load_reference(calibration_file=CALIBRATION_FILE):
    """The reference values saved in calibration.json, or None when there are none yet."""
    if not os.path.exists(calibration_file):
        return None
    with open(calibration_file, 'r') as f:
        return json.load(f).get(REFERENCE_KEY)


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
        if res.problems:
            lines.append(res.problems[0])
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


def run_camera(once, out):
    """Live window (or one frame with ``once``) of the camera with the garment separated."""
    pre = PreProcessor(calibration_file=CALIBRATION_FILE)
    edge_ok = pre.crop_bbox is not None         # cropped to the table: a garment may run off its sides
    cam = WebcamCapture(calibration_file=CALIBRATION_FILE)
    if not cam.start():
        sys.exit('camera did not start (see the log above)')

    try:
        if once:
            raw = wait_frame(cam)
            if raw is None:
                sys.exit('no frame from the camera')
            processed, res = segment(raw, pre, edge_ok)
            prefix = save(out, raw, processed, overlay(processed, res, cam))
            print('camera: %s us, gain %s' % (cam.exposure, cam.gain))
            print('valid=%s cue=%s area=%.1f%% problems=%s' % (
                res.valid, res.cue, res.stats.get('area_pct', 0.0), '; '.join(res.problems) or '-'))
            print(format_values(garment_values(processed, res, rgb=True)))
            print('saved %s_*.jpg' % prefix)
            return

        print('s = save, q = quit')
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
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Reference values and the camera with the garment separated.')
    parser.add_argument('--once', action='store_true', help='one frame, no window: segment, save, exit')
    parser.add_argument('--out', default=os.path.join(HERE, 'captures'), help='folder for saved images')
    parser.add_argument('--reference', nargs='?', const=REFERENCE_IMAGE, metavar='IMAGE',
                        help='segment a reference image, measure its garment and save it to calibration.json '
                             '(default: image.png)')
    args = parser.parse_args()

    if args.reference:
        try:
            values, image, res = measure_reference(args.reference)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))
        if not res.valid:
            print('warning: the segment is not valid (%s)' % '; '.join(res.problems))
        for key, value in values.items():
            print('%-12s %s' % (key, value))
        save_reference(values)
        print('saved to %s under "%s"' % (CALIBRATION_FILE, REFERENCE_KEY))
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, 'reference_overlay.jpg')
        cv2.imwrite(path, draw_overlay(image, res))     # imread gives BGR: no conversion
        print('segment: %s' % path)
        return
    run_camera(args.once, args.out)


if __name__ == '__main__':
    main()
