"""Auto exposure: the reference image's values, and the camera frame with the garment separated.

Reference: measured from REFERENCE_IMAGE when this file is loaded, so a new reference image
changes the values on its own. Whole image, no segmentation.

    brightness = mean grey level of the image (0-255)
    exposure   = log2(brightness / 118): stops from middle grey (0 = middle grey, + brighter, - darker)

Camera: camera -> PreProcessor (rotate, crop, undistort from calibration.json) -> separate_garment

    python auto_exposure.py                      # window: live frame with the garment separated
    python auto_exposure.py --once               # one frame, no window: segment it, save it and exit
    python auto_exposure.py --reference          # print the reference image's values
    python auto_exposure.py --reference other.png  # print another image's values

Keys in the window:
    s         save the raw frame, the processed frame and the overlay
    q / Esc   quit

On a device stop the scanner app first (the camera can only be open in one process):

    sudo systemctl stop rms-app
"""

import argparse
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
MIDDLE_GREY = 118
DISPLAY_LONG_SIDE = 1100


# =========================================================================================
# Reference
# =========================================================================================
def image_values(image):
    """(exposure, brightness) of a BGR or grey image."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    exposure = math.log2(max(brightness, 1.0) / MIDDLE_GREY)
    return exposure, brightness


def load_reference(path=REFERENCE_IMAGE):
    """(exposure, brightness) of the reference image."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError('reference image not found: %s' % path)
    return image_values(image)


# Reference (measured from image.png when this file is loaded)
REFERENCE_EXPOSURE, REFERENCE_BRIGHTNESS = load_reference()


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

def measure(frame, res):
    """(contrast c, intensity i) of a frame with its SeparationResult, or (None, None) without a garment.

    i = mean intensity of the detected garment
    c = mean intensity of the detected garment - mean intensity of the detected ring
        (negative: the garment is darker than the table around it)
    """
    if res.garment_mask is None or res.ring_mask is None:
        return None, None
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # as separate_garment
    garment_px, ring_px = gray[res.garment_mask > 0], gray[res.ring_mask > 0]
    if not garment_px.size or not ring_px.size:
        return None, None
    i = float(garment_px.mean())
    c = i - float(ring_px.mean())
    return c, i


def overlay(processed, res, cam):
    """The processed frame (RGB from the Daheng camera, shown as BGR) with the segment and the values."""
    vis = draw_overlay(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR), res)
    scale = DISPLAY_LONG_SIDE / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = ['exposure %s us   brightness %s' % (cam.exposure, cam.gain)]
    if res.garment_mask is None:
        lines.append('no garment: %s' % '; '.join(res.problems[:1]))
    else:
        c, i = measure(processed, res)
        lines.append('%s   cue %s   area %.0f%%' % (
            'VALID' if res.valid else 'NOT VALID', res.cue, res.stats.get('area_pct', 0.0)))
        if c is not None:
            lines.append('c = %.1f   i = %.1f' % (c, i))
        if res.problems:
            lines.append(res.problems[0])
    for i, text in enumerate(lines):
        y = 30 + 28 * i
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
            c, i = measure(processed, res)
            print('valid=%s cue=%s area=%.1f%% problems=%s' % (
                res.valid, res.cue, res.stats.get('area_pct', 0.0), '; '.join(res.problems) or '-'))
            print('c = %s   i = %s' % ('-' if c is None else '%.1f' % c, '-' if i is None else '%.1f' % i))
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
                        help="print an image's exposure and brightness (default: the reference image)")
    args = parser.parse_args()

    if args.reference:
        exposure, brightness = load_reference(args.reference)
        print('brightness = %.1f' % brightness)
        print('exposure   = %.2f' % exposure)
        return
    run_camera(args.once, args.out)


if __name__ == '__main__':
    main()
