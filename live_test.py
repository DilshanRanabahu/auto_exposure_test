"""Live garment / table separation and contrast on the real camera, without the scanner app.

    python live_test.py --calibration calibration.json                 # Daheng camera (the device)
    python live_test.py --camera webcam --calibration calibration.json # USB webcam (development PC)
    python live_test.py --camera image --image frame.jpg               # a saved frame, no camera

The frame goes through the same steps as in the app: camera -> PreProcessor (rotate, crop,
undistort from calibration.json) -> separate_garment -> contrast. Copy calibration.json from the
device (~/configuration/calibration.json): without its crop the frame border is the room, not
the table, and the separation is far worse.

Keys in the window:
    +  /  -   exposure x1.25 / /1.25
    w         sweep: try a range of exposures on the current garment and keep the best one
    s         save the raw frame, the processed frame and the overlay to --out
    q / Esc   quit

The exposure set here is not written to calibration.json; restart the app and it is back to
the stored value.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from capture import OpenCVWebcamCapture, PreProcessor, WebcamCapture
from garment_separation import CLIP_LEVEL, draw_overlay, measure_contrast, separate_garment

EXPOSURE_STEP = 1.25
DEFAULT_EXPOSURE_US = 20000       # the camera class default when calibration.json has none
EXPOSURE_MIN_US = 100
EXPOSURE_MAX_US = 1000000
SWEEP_US = [2000, 3000, 5000, 7000, 10000, 14000, 20000, 28000, 40000]
SETTLE_S = 0.3                  # after an exposure change, frames already in the pipe still carry the old one
MAX_GARMENT_CLIPPED = 0.01      # sweep: an exposure that clips more of the garment than this is rejected
DISPLAY_LONG_SIDE = 1100


class ImageSource:
    """A saved frame dressed up as a camera, so the rest of the script does not branch."""

    def __init__(self, path):
        self.frame = cv2.imread(path)
        if self.frame is None:
            sys.exit(f'could not read {path}')
        self.exposure, self.gain = None, None

    def start(self):
        return True

    def get_latest_image(self, copy=True):
        return self.frame.copy()

    def set_exposure(self, us):
        print('image mode: exposure cannot change')

    def stop(self):
        pass


def open_camera(args):
    if args.camera == 'image':
        return ImageSource(args.image)
    if args.camera == 'webcam':
        cam = OpenCVWebcamCapture(cam_index=args.index, calibration_file=args.calibration)
    else:
        cam = WebcamCapture(calibration_file=args.calibration)
    if not cam.start():
        sys.exit('camera did not start (see the log above)')
    return cam


def wait_frame(cam, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        frame = cam.get_latest_image()
        if frame is not None:
            return frame
        time.sleep(0.02)
    return None


def fresh_frame(cam, settle=SETTLE_S):
    """A frame taken after the last exposure change, not one already in flight."""
    time.sleep(settle)
    wait_frame(cam)
    return wait_frame(cam)


def garment_clipped(frame, mask):
    """Share of the garment at 255. The mask's own edge is eroded away first: there it overlaps the
    white table, and a black polo read 1.3% clipped from its outline alone."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r = max(2, int(0.006 * max(mask.shape)))
    px = gray[cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0]
    return float(np.mean(px >= CLIP_LEVEL)) if px.size else 0.0


def sweep(cam, pre):
    """Separate once at the current exposure, then measure the contrast on the same masks at each
    exposure in SWEEP_US and keep the one with the highest contrast that does not clip the garment."""
    start_exposure = cam.exposure or DEFAULT_EXPOSURE_US
    frame = pre(fresh_frame(cam))
    res = separate_garment(frame)
    if not res.valid:
        print('sweep: the garment is not separated at this exposure (%s); change the exposure and try again'
              % '; '.join(res.problems))
        return
    print('\n exposure_us | contrast | garment clipped %')
    best = None
    for us in SWEEP_US:
        cam.set_exposure(us)
        frame = pre(fresh_frame(cam, SETTLE_S + us / 1e6))
        if frame is None:
            print('%12d | no frame' % us)
            continue
        contrast = measure_contrast(frame, res.garment_mask, res.ring_mask)
        clipped = garment_clipped(frame, res.garment_mask)
        ok = clipped <= MAX_GARMENT_CLIPPED
        print('%12d | %8.0f | %6.1f %s' % (us, contrast, 100 * clipped, '' if ok else '(clipped)'))
        if ok and (best is None or contrast > best[1]):
            best = (us, contrast)
    if best is None:
        print('sweep: every exposure clipped the garment; back to %s us' % start_exposure)
        cam.set_exposure(start_exposure)
        return
    print('best: %d us (contrast %.0f)\n' % best)
    cam.set_exposure(best[0])


def annotate(frame, res, exposure, fps):
    vis = draw_overlay(frame, res)
    scale = DISPLAY_LONG_SIDE / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = ['exposure %s us   %.1f fps' % (exposure, fps),
             '%s  cue=%s  contrast=%.0f' % ('OK' if res.valid else 'NOT VALID', res.cue,
                                            res.stats.get('contrast', 0.0))]
    lines += res.problems[:2]
    for i, text in enumerate(lines):
        y = 30 + 28 * i
        colour = (0, 220, 0) if res.valid else (0, 0, 255)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', choices=['daheng', 'webcam', 'image'], default='daheng')
    parser.add_argument('--image', help='frame to use with --camera image')
    parser.add_argument('--index', type=int, default=0, help='webcam index')
    parser.add_argument('--calibration', default='calibration.json', help='calibration.json copied from the device')
    parser.add_argument('--exposure', type=float, help='start exposure in us (default: the calibration value)')
    parser.add_argument('--out', default='captures', help='folder for frames saved with s')
    args = parser.parse_args()
    if args.camera == 'image' and not args.image:
        parser.error('--camera image needs --image')

    if os.path.exists(args.calibration):
        processor = PreProcessor(calibration_file=args.calibration)
        pre = lambda f: None if f is None else processor.process_image(f)  # noqa: E731
    else:
        print('WARNING: %s not found: no rotate/crop/undistort, so the frame border is the room' % args.calibration)
        pre = lambda f: f  # noqa: E731

    cam = open_camera(args)
    if args.exposure:
        cam.set_exposure(args.exposure)
    os.makedirs(args.out, exist_ok=True)
    fps, last = 0.0, time.time()
    try:
        while True:
            raw = wait_frame(cam)
            if raw is None:
                print('no frame from the camera')
                continue
            frame = pre(raw)
            res = separate_garment(frame)
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(1e-3, now - last)
            last = now
            cv2.imshow('auto exposure test', annotate(frame, res, cam.exposure, fps))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            current = cam.exposure or DEFAULT_EXPOSURE_US        # a webcam left on its own auto has none
            if key in (ord('+'), ord('=')):
                cam.set_exposure(min(EXPOSURE_MAX_US, int(current * EXPOSURE_STEP)))
            elif key == ord('-'):
                cam.set_exposure(max(EXPOSURE_MIN_US, int(current / EXPOSURE_STEP)))
            elif key == ord('w'):
                sweep(cam, pre)
            elif key == ord('s'):
                stamp = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(os.path.join(args.out, f'{stamp}_raw.jpg'), raw)
                cv2.imwrite(os.path.join(args.out, f'{stamp}_processed.jpg'), frame)
                cv2.imwrite(os.path.join(args.out, f'{stamp}_overlay.jpg'), draw_overlay(frame, res))
                print('saved %s_* (exposure %s us, contrast %.0f)' % (stamp, cam.exposure, res.stats.get('contrast', 0.0)))
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
