"""Auto exposure test app: live camera view with the garment / table separation, and the auto exposure logic.

    python app.py                                    # Daheng camera, calibration.json in this folder
    python app.py --camera webcam                    # USB webcam
    python app.py --camera image --image frame.jpg   # a saved frame, exposure simulated
    python app.py --once                             # run auto exposure once, no window, and exit
    python app.py --no-save                          # never write calibration.json
    python app.py --images C:/path/img --out C:/path/img/results/auto_exposure
                                                     # every image in a folder, exposure simulated

Every run is saved: one folder per garment, a subfolder per run with an image of every round,
BEST_round<N>.jpg (the best segment with its exposure, brightness and contrast written on it),
best.json and log.txt. With --images each garment is run from every --starts exposure, one subfolder
each, and the garment folder gets the best of them (BEST_<exposure>us.jpg, best.json).

On a device stop the scanner app first: the camera can only be open in one process.

    sudo systemctl stop rms-app

The frame goes through the same steps as in the scanner app: camera -> PreProcessor (rotate, crop,
undistort from calibration.json) -> separate_garment. The logic itself is in auto_exposure.py.

Keys in the window:
    a         run auto exposure (the rounds are printed in the console, saved to --out/live_<time>)
    +  /  -   exposure x1.25 / /1.25
    s         save the raw frame, the processed frame and the overlay to --out
    q / Esc   quit
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import auto_exposure
from auto_exposure import (DEFAULT_EXPOSURE_US, best_image, best_values, pick_best, run_auto_exposure,
                           save_run)
from capture import OpenCVWebcamCapture, PreProcessor, WebcamCapture
from garment_separation import draw_overlay, separate_garment

EXPOSURE_STEP = 1.25
EXPOSURE_MIN_US = 100
EXPOSURE_MAX_US = 1000000
DISPLAY_LONG_SIDE = 1100


class ImageCamera:
    """A saved frame as a camera, for testing without one. The frame is taken to be shot at
    20000 us and gain 10; other values scale its brightness (exposure linearly, gain in dB) and
    clip at 255. Only a simulation: detail that was clipped in the file does not come back."""

    BASE_EXPOSURE_US, BASE_GAIN = 20000, 10.0

    def __init__(self, path):
        frame = cv2.imread(path)
        if frame is None:
            sys.exit(f'could not read {path}')
        self.base = frame.astype(np.float32)
        self.exposure, self.gain = self.BASE_EXPOSURE_US, self.BASE_GAIN

    def start(self):
        return True

    def stop(self):
        pass

    def set_exposure(self, us):
        self.exposure = us

    def set_gain(self, value):
        self.gain = value

    def get_latest_image(self, copy=True):
        k = self.exposure / self.BASE_EXPOSURE_US * 10 ** ((self.gain - self.BASE_GAIN) / 20)
        return np.clip(self.base * k, 0, 255).astype(np.uint8)


def open_camera(args):
    if args.camera == 'image':
        return ImageCamera(args.image)
    if args.camera == 'webcam':
        cam = OpenCVWebcamCapture(cam_index=args.index, calibration_file=args.calibration)
    else:
        cam = WebcamCapture(calibration_file=args.calibration)
    if not cam.start():
        sys.exit('camera did not start (see the log above)')
    # The camera classes re-read calibration.json on their first frame and would put the file's
    # exposure back over --exposure; mark the file as already read.
    if hasattr(cam, 'last_config_mtime') and os.path.exists(args.calibration):
        cam.last_config_mtime = os.path.getmtime(args.calibration)
    return cam


def wait_frame(cam, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        frame = cam.get_latest_image()
        if frame is not None:
            return frame
        time.sleep(0.02)
    return None


def annotate(frame, res, cam, fps, to_bgr):
    vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if to_bgr else frame
    vis = draw_overlay(vis, res)
    scale = DISPLAY_LONG_SIDE / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = ['exposure %s us   brightness %s   %.1f fps' % (cam.exposure, cam.gain, fps)]
    if res.garment_mask is None:
        lines.append('no garment: %s' % '; '.join(res.problems[:1]))
    else:
        lines.append('contrast=%.0f  (%s)' % (res.stats.get('contrast', 0.0),
                                             'valid' if res.valid else 'used anyway'))
    for i, text in enumerate(lines):
        y = 30 + 28 * i
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
    return vis


IMAGE_TYPES = ('.jpg', '.jpeg', '.png', '.bmp')


def run_and_save(cam, pre, folder, calibration_file=None, rgb=False):
    """Run auto exposure, print it and save it to ``folder`` (see save_run)."""
    lines = []

    def log(message):
        print(message)
        lines.extend(str(message).splitlines())

    result = run_auto_exposure(cam, pre, calibration_file=calibration_file, log=log)
    save_run(result, folder, lines, rgb=rgb)
    print('saved to %s' % folder)
    return result


def crop_fits(processor, frame):
    """This frame is the raw size calibration.json was made for (its stage1_reference_resolution), so
    its crop applies. The crop box merely lying inside the frame is not enough: 5472x3648 frames from
    another camera took the 1638x3343 crop and kept only a strip with a third of the garment."""
    if processor is None:
        return False
    with open(processor.calibration_file) as f:
        reference = json.load(f).get('stage1_reference_resolution')
    h, w = frame.shape[:2]
    if reference and len(reference) == 2:
        return (w, h) == (int(reference[0]), int(reference[1]))
    if processor.crop_bbox is None:
        return True
    (_, _), (x1, y1) = processor.crop_bbox
    return int(x1) <= w and int(y1) <= h


def run_folder(args, processor):
    """--images: every image in the folder, from every start exposure, saved one folder per garment."""
    auto_exposure.SETTLE_S = 0                          # a saved frame needs no settling
    starts = [float(v) for v in args.starts.split(',')]
    paths = sorted(os.path.join(args.images, f) for f in os.listdir(args.images)
                   if f.lower().endswith(IMAGE_TYPES))
    summary = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        garment_dir = os.path.join(args.out, name)
        cam = ImageCamera(path)
        use_crop = crop_fits(processor, cam.base)
        pre = processor.process_image if use_crop else (lambda f: f)
        print('\n##### %s (%s)' % (name, 'calibration crop' if use_crop else 'no crop: it does not fit this frame'))
        bests = []
        for start in starts:
            cam.exposure, cam.gain = start, ImageCamera.BASE_GAIN
            print('\n=== start %d us' % start)
            result = run_and_save(cam, pre, os.path.join(garment_dir, 'start_%dus' % start))
            if result.best is not None:
                bests.append((start, result))
        if not bests:
            summary.append('%-40s | no garment found from any start' % name[:40])
            continue
        best_round, _ = pick_best([r.best for _, r in bests])
        start, result = next((s, r) for s, r in bests if r.best is best_round)
        values = dict(best_values(result), start_exposure=start)
        with open(os.path.join(garment_dir, 'best.json'), 'w') as f:
            json.dump({'best': values, 'per_start': {'%d' % s: best_values(r) for s, r in bests}}, f, indent=4)
        cv2.imwrite(os.path.join(garment_dir, 'BEST_%dus.jpg' % result.exposure),
                    best_image(result, title='BEST (start %d us)' % start))
        summary.append('%-40s | %6d us | brightness %5s | contrast %4.0f | solidity %.2f | from start %d' % (
            name[:40], result.exposure, '%g' % result.brightness, result.contrast, result.solidity, start))
    header = '%-40s | best exposure, brightness, contrast, solidity' % 'garment'
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write('\n'.join([header] + summary) + '\n')
    print('\n' + '\n'.join([header] + summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', choices=['daheng', 'webcam', 'image'], default='daheng')
    parser.add_argument('--image', help='frame to use with --camera image')
    parser.add_argument('--images', help='folder of saved frames: run every one, exposure simulated, and exit')
    parser.add_argument('--starts', default='5000,10000,20000,39594',
                        help='--images: start exposures in us, comma separated')
    parser.add_argument('--index', type=int, default=0, help='webcam index')
    parser.add_argument('--calibration', default='calibration.json', help='calibration.json copied from the device')
    parser.add_argument('--exposure', type=float, help='start exposure in us (default: the calibration value)')
    parser.add_argument('--no-save', action='store_true', help='do not write the result to calibration.json')
    parser.add_argument('--once', action='store_true', help='run auto exposure once without a window, then exit')
    parser.add_argument('--out', default='captures', help='folder for the results and for frames saved with s')
    args = parser.parse_args()
    if args.camera == 'image' and not args.image:
        parser.error('--camera image needs --image')

    processor = None
    if os.path.exists(args.calibration):
        processor = PreProcessor(calibration_file=args.calibration)
        pre = lambda f: None if f is None else processor.process_image(f)  # noqa: E731
    else:
        print('WARNING: %s not found: no rotate/crop/undistort, so the frame border is the room' % args.calibration)
        pre = lambda f: f  # noqa: E731
    if args.images:
        run_folder(args, processor)
        return
    # A simulated exposure must not end up in the device's calibration.
    save_to = None if args.no_save or args.camera == 'image' or not os.path.exists(args.calibration) \
        else args.calibration
    to_bgr = args.camera == 'daheng'          # Daheng frames are RGB; only the display is converted

    cam = open_camera(args)
    if args.exposure:
        cam.set_exposure(args.exposure)
    def auto():
        name = os.path.splitext(os.path.basename(args.image))[0] if args.camera == 'image' else 'live'
        run_and_save(cam, pre, os.path.join(args.out, '%s_%s' % (name, time.strftime('%Y%m%d_%H%M%S'))),
                     calibration_file=save_to, rgb=to_bgr)

    try:
        if args.once:
            auto()
            return
        os.makedirs(args.out, exist_ok=True)
        fps, last = 0.0, time.time()
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
            cv2.imshow('auto exposure test', annotate(frame, res, cam, fps, to_bgr))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            current = cam.exposure or DEFAULT_EXPOSURE_US        # a webcam left on its own auto has none
            if key == ord('a'):
                auto()
            elif key in (ord('+'), ord('=')):
                cam.set_exposure(min(EXPOSURE_MAX_US, int(current * EXPOSURE_STEP)))
            elif key == ord('-'):
                cam.set_exposure(max(EXPOSURE_MIN_US, int(current / EXPOSURE_STEP)))
            elif key == ord('s'):
                stamp = time.strftime('%Y%m%d_%H%M%S')
                save = (lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) if to_bgr else (lambda f: f)
                cv2.imwrite(os.path.join(args.out, f'{stamp}_raw.jpg'), save(raw))
                cv2.imwrite(os.path.join(args.out, f'{stamp}_processed.jpg'), save(frame))
                cv2.imwrite(os.path.join(args.out, f'{stamp}_overlay.jpg'), draw_overlay(save(frame), res))
                print('saved %s_* (exposure %s us, gain %s)' % (stamp, cam.exposure, cam.gain))
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
