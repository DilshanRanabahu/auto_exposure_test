"""Auto exposure (new flowchart).

    python3 auto_exposure_new.py                           # run the auto exposure on the camera, then a
                                                           # window: a = run it again, q = quit
    python3 auto_exposure_new.py --reference               # measure image.png, save to calibration.json
    python3 auto_exposure_new.py --reference other.png     # another reference image

On a production PC (Linux, Daheng SDK / gxipy installed):
    sudo systemctl stop rms-app                            # the camera can only be open in one process
    python3 auto_exposure_new.py
    sudo systemctl start rms-app

The reference:

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
import threading
import time

import cv2  # type: ignore

from capture import PreProcessor, WebcamCapture
from garment_separation import draw_overlay, separate_garment

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(HERE, 'calibration.json')
REFERENCE_IMAGE = os.path.join(HERE, 'image.png')
REFERENCE_KEY = 'reference_image'
MIDDLE_GREY = 118
DEFAULT_EXPOSURE_US = 20000     # capture.py's defaults when calibration.json has none
DEFAULT_GAIN = 10.0

# Brightness loop (fabric mean = mean intensity of the garment, 0-255)
MIN_BRIGHTNESS = 60             # fabric mean below this: garment too dark (cannot see seamlines), gain UP
MAX_BRIGHTNESS = 200            # fabric mean at or above this: overexposed, gain DOWN
MIN_CONTRAST = 6                # final check & binary search target (white garments reach 6-10 contrast against white table)
TARGET_BRIGHTNESS = 120         # fabric mean the gain is computed for when it is out of 60-200
GAIN_MIN = 2.0                  # dB on the Daheng camera
GAIN_MAX = 24.0
MAX_PHOTOS = 12                 # photos in one run, the first one included
MAX_TRIES = 2                   # computed exposure (or gain) values tried before giving up

# Low-contrast exposure search (flowchart "No")
CLIPPED_LEVEL = 200             # clipped: the brighter of fabric mean and table mean is above this
CONTRAST_MARGIN = 1.3           # the exposure is computed for C = MIN_CONTRAST x this, so the check passes
EXPOSURE_MIN_US = 2000
EXPOSURE_MAX_US = 150000
MAX_SIDE_CONTACT_PCT = 90.0     # a segment along more of one frame side than this is the table, not a garment

# Waiting for a frame shot at new settings: the Daheng camera keeps delivering frames at the old
# settings for a while (up to 1-2 s) after a change.
SETTLE_S = 0.2                  # always wait this long first
CHANGE_SEEN = 0.5               # a frame has moved this share of the expected change: it is new
FALLBACK_WAIT_S = 1.0           # change too small or clipped to see: wait this long instead
CHANGE_TIMEOUT_S = 6.0          # never wait longer than this

# Live view
SHOW_OVERLAY = True             # show segmentation overlay on startup (toggle with 'o')


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
        'contrast': round(abs(garment_mean - ring_mean), 1),
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


def camera_gray(cam, pre):
    """The camera's latest frame, prepared like the scanner app (rotate, crop, undistort), as
    (processed frame, grey frame). (None, None) when no frame comes within 5 s."""
    frame = None
    for _ in range(250):                        # up to 5 s for a frame (camera just started)
        frame = cam.get_latest_image()
        if frame is not None:
            break
        time.sleep(0.02)
    if frame is None:
        print('no frame from the camera')
        return None, None
    processed = pre.process_image(frame)
    gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)     # Daheng frames are RGB
    return processed, gray


def segment_camera(cam, pre):
    """Separate the garment in the camera's latest frame. Returns (gray, garment_mask, ring_mask);
    the masks are None when there is no garment, or the segment runs along more than
    MAX_SIDE_CONTACT_PCT of one frame side (that is the table or a sheet edge, not a garment)."""
    processed, gray = camera_gray(cam, pre)
    if gray is None:
        return None, None, None
    edge_ok = pre.crop_bbox is not None         # cropped to the table: a garment may run off its sides
    res = separate_garment(processed, edge_ok=edge_ok)
    if res.garment_mask is None or res.ring_mask is None:
        print('no garment found: %s' % '; '.join(res.problems))
        return gray, None, None
    side = res.stats.get('side_contact_pct', 0.0)
    if side > MAX_SIDE_CONTACT_PCT:
        print('segment along %.0f%% of a frame side: the table, not a garment' % side)
        return gray, None, None
    return gray, res.garment_mask, res.ring_mask


def contrast_above_reference(contrast, reference):
    """Flowchart: "Contrast above reference?"

        |camera contrast| > |reference contrast|  -> True  (Yes: lock the exposure)
        otherwise, or no garment (contrast None)  -> False (No: low-contrast exposure search)

    The size only: a dark garment on a light table has a negative contrast, a light one on a dark
    table a positive one, but both can stand out just as much.
    """
    if contrast is None:
        return False
    return abs(contrast) > abs(reference['contrast'])


def mask_means(gray, masks):
    """(fabric mean, table mean): the mean grey level inside the garment mask and inside the ring mask."""
    garment_mask, ring_mask = masks
    return float(gray[garment_mask > 0].mean()), float(gray[ring_mask > 0].mean())


def area_level(frame, pre, area):
    """Mean level of a frame (every 8th pixel). With an area (the garment and ring, True/False) only
    inside it: the table around a white garment is often clipped at 255 over most of the frame, so
    the whole frame barely changes, but the garment and the ring do."""
    if area is None:
        return float(frame[::8, ::8].mean())
    processed = pre.process_image(frame)
    return float(processed[::8, ::8][area[::8, ::8]].mean())


def set_and_wait(cam, pre, exposure, gain, masks=None):
    """Set the exposure and gain on the camera and wait until its frames are shot at them.

    Nothing to wait for when the camera already has these settings. Otherwise brightness follows
    exposure and gain, so the expected level of the new frames is known. A frame that has moved
    CHANGE_SEEN of the way towards it is shot at the new settings. The level is measured on the
    garment and ring (masks) when they are known. When the change is too small or clipped (near 0
    or 255) to see, the frames after FALLBACK_WAIT_S are taken.
    """
    if exposure == cam.exposure and gain == cam.gain:
        return                                  # the camera already shoots at these settings

    area = None
    if masks is not None:
        area = (masks[0] > 0) | (masks[1] > 0)

    old_frame = cam.get_latest_image(copy=False)
    old_exposure = cam.exposure or DEFAULT_EXPOSURE_US
    old_gain = cam.gain if cam.gain is not None else gain

    cam.set_exposure(exposure)
    cam.set_gain(gain)

    # the level the new frames should have (gain is in dB)
    before, expected, visible = None, None, False
    if old_frame is not None:
        before = area_level(old_frame, pre, area)
        ratio = (exposure / old_exposure) * 10 ** ((gain - old_gain) / 20)
        expected = min(255.0, before * ratio)
        visible = 3 < before < 250 and 3 < expected < 250 and abs(expected - before) >= 8

    start = time.time()
    time.sleep(SETTLE_S + exposure / 1e6)
    while time.time() - start < CHANGE_TIMEOUT_S:
        frame = cam.get_latest_image(copy=False)
        if frame is not None and frame is not old_frame:      # a new frame
            if visible:
                if abs(area_level(frame, pre, area) - before) >= CHANGE_SEEN * abs(expected - before):
                    break
            elif time.time() - start >= FALLBACK_WAIT_S:
                break
        time.sleep(0.01)


def take_photo(cam, pre, exposure, gain, masks=None):
    """Set the exposure and gain, wait for a frame shot at them, and measure it.

    masks None: separate the garment and the ring (the first photo). Otherwise the saved masks
    (garment, ring) are used and nothing is separated again: the garment does not move.

    Returns (contrast C, fabric mean, table mean, masks). C is None when there is no frame or no garment.
    """
    t0 = time.time()
    set_and_wait(cam, pre, exposure, gain, masks)
    t1 = time.time()

    if masks is None:
        gray, garment_mask, ring_mask = segment_camera(cam, pre)
        if garment_mask is not None:
            masks = (garment_mask, ring_mask)
    else:
        _, gray = camera_gray(cam, pre)
    if gray is None or masks is None:
        print('photo: %d us, gain %g -> no garment (wait %.2f s, measure %.2f s)'
              % (exposure, gain, t1 - t0, time.time() - t1))
        return None, None, None, masks

    fabric, table = mask_means(gray, masks)
    contrast = abs(fabric - table)
    print('photo: %d us, gain %g -> C %.1f, fabric %.1f, table %.1f (wait %.2f s, measure %.2f s)'
          % (exposure, gain, contrast, fabric, table, t1 - t0, time.time() - t1))
    return contrast, fabric, table, masks


def brightness_loop(cam, pre, exposure, gain, contrast, fabric, masks, photos):
    """Flowchart: the brightness loop, with the exposure locked. Returns (gain to set, contrast C
    measured at that gain).

        fabric mean in 60-200   -> keep the gain
        otherwise               -> compute the gain for a fabric mean of TARGET_BRIGHTNESS (brightness
                                   follows gain: +6 dB = twice as bright), take a photo, measure on
                                   the saved masks; up to MAX_TRIES times

    Final check: C below MIN_CONTRAST after a change (or no frame) -> back to the gain before it.
    """
    for _ in range(MAX_TRIES):
        if MIN_BRIGHTNESS <= fabric < MAX_BRIGHTNESS:
            break

        new_gain = gain + 20 * math.log10(TARGET_BRIGHTNESS / max(fabric, 1.0))
        new_gain = round(min(GAIN_MAX, max(GAIN_MIN, new_gain)), 1)
        print('fabric mean %.1f is outside %d-%d: gain %g -> %g'
              % (fabric, MIN_BRIGHTNESS, MAX_BRIGHTNESS, gain, new_gain))
        if new_gain == gain:
            print('gain at its limit (%g): stop' % gain)
            break
        if photos >= MAX_PHOTOS:
            print('%d photos done: stop at gain %g' % (MAX_PHOTOS, gain))
            break

        new_contrast, new_fabric, _, _ = take_photo(cam, pre, exposure, new_gain, masks)
        photos += 1
        if new_contrast is None:
            print('no frame at gain %g: back to gain %g' % (new_gain, gain))
            break
        if new_contrast < MIN_CONTRAST:
            print('final check: C %.1f < %d at gain %g, back to gain %g' % (new_contrast, MIN_CONTRAST, new_gain, gain))
            break
        gain, contrast, fabric = new_gain, new_contrast, new_fabric

    print('fabric mean %.1f: gain %g' % (fabric, gain))
    return gain, contrast


def low_contrast_search(cam, pre, exposure, gain, contrast, fabric, table, masks, photos):
    """Flowchart "No": the low-contrast exposure search, with the brightness (gain) locked.

    The masks from the first photo are kept and every new photo is measured on them. Brightness
    follows the exposure until it clips, and so does C, so the exposure is computed, not searched:

        C >= MIN_CONTRAST  -> the exposure is found
        clipped            -> exposure x 0.5 and measure again (clipped values cannot be used)
        not clipped        -> E_need = E x MIN_CONTRAST x CONTRAST_MARGIN / C   (exposure for enough C)
                              E_clip = E x CLIPPED_LEVEL / max(fabric, table)   (exposure where it clips)
                              C cannot reach MIN_CONTRAST before E_clip -> stop and flag the operator
                              otherwise take a photo at min(E_need, E_clip); MAX_TRIES at most

    No garment in the first photo: the exposure is halved and the garment separated again until one is
    found (a clipped white garment on a white table has no outline).

    Returns (exposure, contrast, fabric mean, masks, photos); exposure None means stop and flag the operator.
    """
    while masks is None:
        new_exposure = max(EXPOSURE_MIN_US, round(exposure * 0.5))
        if new_exposure == exposure or photos >= MAX_PHOTOS:
            print('no garment found down to %d us' % exposure)
            return None, None, None, None, photos
        exposure = new_exposure
        contrast, fabric, table, masks = take_photo(cam, pre, exposure, gain)
        photos += 1

    tries = 0
    while True:
        if contrast is None:
            print('no frame from the camera')
            return None, None, None, masks, photos
        if contrast >= MIN_CONTRAST:
            return exposure, contrast, fabric, masks, photos
        if photos >= MAX_PHOTOS:
            print('%d photos done' % MAX_PHOTOS)
            return None, None, None, masks, photos

        brightest = max(fabric, table, 1.0)
        if brightest > CLIPPED_LEVEL:
            new_exposure = max(EXPOSURE_MIN_US, round(exposure * 0.5))
            print('clipped (%.1f > %d): exposure %d -> %d us' % (brightest, CLIPPED_LEVEL, exposure, new_exposure))
        else:
            if tries >= MAX_TRIES:
                print('C %.1f < %d after %d computed exposures' % (contrast, MIN_CONTRAST, MAX_TRIES))
                return None, None, None, masks, photos
            tries += 1
            e_clip = exposure * CLIPPED_LEVEL / brightest
            if exposure * MIN_CONTRAST / max(contrast, 0.1) > e_clip:
                print('C %.1f cannot reach %d before clipping (at %d us C would be %.1f)'
                      % (contrast, MIN_CONTRAST, e_clip, contrast * e_clip / exposure))
                return None, None, None, masks, photos
            e_need = exposure * MIN_CONTRAST * CONTRAST_MARGIN / max(contrast, 0.1)
            new_exposure = round(min(EXPOSURE_MAX_US, max(EXPOSURE_MIN_US, min(e_need, e_clip))))
            print('computed exposure: need %d us, clips at %d us -> %d us' % (e_need, e_clip, new_exposure))

        if new_exposure == exposure:
            print('exposure at its limit (%d us)' % exposure)
            return None, None, None, masks, photos
        exposure = new_exposure
        contrast, fabric, table, _ = take_photo(cam, pre, exposure, gain, masks)
        photos += 1


def set_and_save(cam, exposure, gain):
    """Flowchart: "Set exposure and brightness". Sets them on the camera and writes them to
    calibration.json ("exposure", "gain")."""
    cam.set_exposure(exposure)
    cam.set_gain(gain)
    calibration = read_calibration(CALIBRATION_FILE)
    calibration['exposure'] = exposure
    calibration['gain'] = gain
    write_calibration(calibration, CALIBRATION_FILE)
    # the camera re-reads calibration.json when it changes; it already has these values
    cam.last_config_mtime = os.path.getmtime(CALIBRATION_FILE)
    print('set and saved: %d us, gain %g' % (exposure, gain))


def save_final_capture(cam, pre, exposure, gain):
    """Save the final frame and its overlay to captures/ after auto exposure values are set."""
    time.sleep(0.3)
    frame = cam.get_latest_image()
    if frame is None:
        return
    processed = pre.process_image(frame)
    bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

    captures_dir = os.path.join(HERE, 'captures')
    os.makedirs(captures_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    base_name = 'ae_%s_%dus_gain%g' % (ts, int(exposure), gain)

    raw_path = os.path.join(captures_dir, '%s.jpg' % base_name)
    cv2.imwrite(raw_path, bgr)

    edge_ok = pre.crop_bbox is not None
    res = separate_garment(processed, edge_ok=edge_ok)
    overlay = draw_overlay(bgr, res)
    overlay_path = os.path.join(captures_dir, '%s_overlay.jpg' % base_name)
    cv2.imwrite(overlay_path, overlay)

    print('saved image  : %s' % raw_path)
    print('saved overlay: %s' % overlay_path)


def print_result(exposure, gain, contrast, saved, seconds):
    """The values set at the end of a run, the garment's contrast C at them (None: not found) and
    the time the whole run took."""
    bar = '=' * 50
    print('')
    print(bar)
    print(' AUTO EXPOSURE RESULT')
    print(bar)
    print('  exposure : %d us' % exposure)
    print('  gain     : %g' % gain)
    print('  contrast : %s' % ('-' if contrast is None else '%.1f' % contrast))
    print('  saved    : %s' % ('yes, calibration.json' if saved else 'NO (calibration.json not changed)'))
    print('  time     : %.1f s' % seconds)
    print(bar)


def flag_operator(cam, reference, start):
    """Flowchart: "Stop and flag the operator". The camera goes back to the reference's settings and
    nothing is saved."""
    exposure = float(reference['camera_exposure_us'])
    gain = float(reference['camera_gain'])
    cam.set_exposure(exposure)
    cam.set_gain(gain)
    print('')
    print('!!! OPERATOR: no exposure gives this garment enough contrast against the table.')
    print('!!! Camera back to the reference settings, calibration.json not changed.')
    print_result(exposure, gain, None, False, time.time() - start)


def run_auto_exposure(cam, pre):
    """The flowchart from the top: photo at the reference's settings, contrast above the reference?
    Yes: lock the exposure. No: low-contrast exposure search (gain locked), then lock the exposure it
    found, or stop and flag the operator. Then the brightness loop, set and save."""
    reference = load_reference(CALIBRATION_FILE)
    if reference is None:
        print('no reference in calibration.json: run  --reference  first')
        return
    start = time.time()
    exposure = float(reference['camera_exposure_us'])
    gain = float(reference['camera_gain'])

    # capture frame at reference settings; detect garment and ring once (the masks are kept),
    # measure C, fabric mean and table mean
    contrast, fabric, table, masks = take_photo(cam, pre, exposure, gain)
    photos = 1

    # Contrast above reference?
    if contrast_above_reference(contrast, reference):
        print('Yes: exposure locked at %d us' % exposure)
    else:
        print('No: low-contrast exposure search, gain locked at %g' % gain)
        exposure, contrast, fabric, masks, photos = low_contrast_search(
            cam, pre, exposure, gain, contrast, fabric, table, masks, photos)

    # the search found no exposure: stop and flag the operator
    if exposure is None:
        flag_operator(cam, reference, start)
        return

    # exposure locked: brightness loop, set and save
    print('exposure locked at %d us' % exposure)
    gain, contrast = brightness_loop(cam, pre, exposure, gain, contrast, fabric, masks, photos)
    set_and_save(cam, exposure, gain)
    print_result(exposure, gain, contrast, True, time.time() - start)
    # the final image and overlay are only for checking by eye: saved in the background, not in the run's time
    threading.Thread(target=save_final_capture, args=(cam, pre, exposure, gain)).start()


def main():
    parser = argparse.ArgumentParser(description='Auto exposure (new flowchart).')
    parser.add_argument('--reference', nargs='?', const=REFERENCE_IMAGE, metavar='IMAGE',
                        help='measure a reference image and save it to calibration.json (default: image.png)')
    args = parser.parse_args()

    if args.reference:
        save_reference(args.reference, CALIBRATION_FILE)
        return

    # No option: run the auto exposure once on the Daheng camera
    pre = PreProcessor(calibration_file=CALIBRATION_FILE)
    cam = WebcamCapture(calibration_file=CALIBRATION_FILE)
    if not cam.start():
        print('camera did not start.')
        return
    # the camera re-reads calibration.json on its first frame; mark it as read so it does not put
    # the file's exposure and gain back over the ones the auto exposure sets
    cam.last_config_mtime = os.path.getmtime(CALIBRATION_FILE)
    try:
        run_auto_exposure(cam, pre)

        # live view: a = run the auto exposure again, o = toggle overlay, s = save current frame, q = quit
        print('a = run the auto exposure again, o = toggle overlay, s = save frame, q = quit')
        print('overlay: garment = green, ring = blue, outline = red')
        show_overlay = SHOW_OVERLAY

        # background segmentation: runs every ~1 s so the window stays smooth
        _seg_lock = threading.Lock()
        _seg_result = [None]        # [SeparationResult or None]
        _seg_stop = threading.Event()   # set: the worker stops at once (it does not sleep out its second)

        def _seg_worker():
            while not _seg_stop.is_set():
                frame = cam.get_latest_image()
                if frame is not None:
                    processed = pre.process_image(frame)
                    edge_ok = pre.crop_bbox is not None
                    res = separate_garment(processed, edge_ok=edge_ok)
                    with _seg_lock:
                        _seg_result[0] = (processed, res)
                _seg_stop.wait(1.0)     # re-segment every second

        seg_thread = threading.Thread(target=_seg_worker, daemon=True)
        seg_thread.start()

        while True:
            frame = cam.get_latest_image()
            if frame is not None:
                processed = pre.process_image(frame)
                view = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)   # Daheng frames are RGB

                # overlay the latest segmentation result on the live frame
                with _seg_lock:
                    seg = _seg_result[0]
                if show_overlay and seg is not None:
                    seg_processed, res = seg
                    # draw_overlay expects BGR; re-use its masks (they were computed at same resolution)
                    view = draw_overlay(view, res)
                    # status text
                    if res.valid:
                        label = 'C=%.1f  %s' % (res.stats.get('contrast', 0), res.cue)
                        color = (0, 220, 0)
                    elif res.garment_mask is not None:
                        label = 'NOT VALID: ' + ', '.join(res.problems[:2])
                        color = (0, 140, 255)
                    else:
                        label = 'no garment: ' + ', '.join(res.problems[:2])
                        color = (0, 0, 220)
                    cv2.putText(view, label, (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(view, label, (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,    2, cv2.LINE_AA)

                scale = 1000 / max(view.shape[:2])
                cv2.imshow('auto exposure', cv2.resize(view, None, fx=scale, fy=scale))

            key = cv2.waitKey(30) & 0xFF
            if key == ord('s') and frame is not None:
                captures_dir = os.path.join(HERE, 'captures')
                os.makedirs(captures_dir, exist_ok=True)
                ts = time.strftime('%Y%m%d_%H%M%S')
                if show_overlay:
                    save_path = os.path.join(captures_dir, 'capture_%s_overlay.jpg' % ts)
                    clean_path = os.path.join(captures_dir, 'capture_%s_clean.jpg' % ts)
                    cv2.imwrite(save_path, view)
                    clean_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(clean_path, clean_bgr)
                    print('saved frame: %s' % save_path)
                    print('saved clean: %s' % clean_path)
                else:
                    save_path = os.path.join(captures_dir, 'capture_%s.jpg' % ts)
                    clean_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, clean_bgr)
                    print('saved frame: %s' % save_path)
            if key == ord('o'):
                show_overlay = not show_overlay
                print('overlay %s' % ('ON' if show_overlay else 'OFF'))
            if key == ord('a'):
                _seg_stop.set()             # pause segmentation during the exposure run
                seg_thread.join(timeout=2.0)
                run_auto_exposure(cam, pre)
                # restart segmentation
                _seg_stop.clear()
                seg_thread = threading.Thread(target=_seg_worker, daemon=True)
                seg_thread.start()
            if key == ord('q'):
                _seg_stop.set()
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
