"""Auto exposure (new flowchart).

Step 1: the reference.
    python3 auto_exposure_new.py --reference               # measure image.png, save to calibration.json
    python3 auto_exposure_new.py --reference other.png     # another reference image

The reference image is segmented (garment + ring of table around it) and these values are saved in
calibration.json under "reference_image" (every other key in the file is kept):

    brightness = mean intensity of the garment (0-255)
    exposure   = log2(brightness / 118): stops from middle grey (0 = middle grey, + brighter, - darker)
    contrast   = mean of the garment - mean of the ring (negative: garment darker than the table)
    camera_exposure_us, camera_gain = the camera settings in calibration.json (the auto exposure starts there)
"""

import argparse
import json
import math
import os
import time

import cv2  # type: ignore

from garment_separation import draw_overlay, separate_garment

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(HERE, 'calibration.json')
REFERENCE_IMAGE = os.path.join(HERE, 'image.png')
REFERENCE_KEY = 'reference_image'
MIDDLE_GREY = 118
DEFAULT_EXPOSURE_US = 20000     # capture.py's defaults when calibration.json has none
DEFAULT_GAIN = 10.0


def read_calibration(calibration_file):
    """calibration.json as a dict ({} when the file does not exist)."""
    if not os.path.exists(calibration_file):
        return {}
    with open(calibration_file, 'r') as f:
        return json.load(f)


def write_calibration(data, calibration_file):
    """Write the dict to calibration.json: first to a .tmp file, then moved over the old one."""
    tmp = calibration_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, calibration_file)


def garment_values(gray, garment_mask, ring_mask):
    """brightness, exposure and contrast of the garment, or None when there is no garment."""
    if garment_mask is None or ring_mask is None:
        return None
    garment_mean = float(gray[garment_mask > 0].mean())
    ring_mean = float(gray[ring_mask > 0].mean())
    return {
        'brightness': round(garment_mean, 1),
        'exposure': round(math.log2(max(garment_mean, 1.0) / MIDDLE_GREY), 2),
        'contrast': round(garment_mean - ring_mean, 1),
    }


def save_reference(image_path, calibration_file):
    """Measure the reference image and save its values in calibration.json under "reference_image"."""
    image = cv2.imread(image_path)              # BGR
    if image is None:
        print('reference image not found: %s' % image_path)
        return

    res = separate_garment(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = garment_values(gray, res.garment_mask, res.ring_mask)
    if values is None:
        print('no garment found in the reference image: %s' % '; '.join(res.problems))
        return
    if not res.valid:
        print('warning: the segment is not valid (%s)' % '; '.join(res.problems))

    calibration = read_calibration(calibration_file)
    values['image'] = os.path.basename(image_path)
    values['camera_exposure_us'] = float(calibration.get('exposure', DEFAULT_EXPOSURE_US))
    values['camera_gain'] = float(calibration.get('gain', DEFAULT_GAIN))
    values['measured_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    calibration[REFERENCE_KEY] = values
    write_calibration(calibration, calibration_file)

    for key, value in values.items():
        print('%-20s %s' % (key, value))
    print('saved to %s under "%s"' % (calibration_file, REFERENCE_KEY))

    # the segment, to check by eye that the right garment was measured
    overlay_path = os.path.join(HERE, 'captures', 'reference_overlay.jpg')
    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    cv2.imwrite(overlay_path, draw_overlay(image, res))
    print('segment: %s' % overlay_path)


def load_reference(calibration_file):
    """The reference values saved in calibration.json, or None when there are none yet."""
    return read_calibration(calibration_file).get(REFERENCE_KEY)


def main():
    parser = argparse.ArgumentParser(description='Auto exposure (new flowchart).')
    parser.add_argument('--reference', nargs='?', const=REFERENCE_IMAGE, metavar='IMAGE',
                        help='measure a reference image and save it to calibration.json (default: image.png)')
    args = parser.parse_args()

    if args.reference:
        save_reference(args.reference, CALIBRATION_FILE)


if __name__ == '__main__':
    main()
