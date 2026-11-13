"""Auto exposure test app: live camera view with the garment / table separation, and the auto exposure logic.

    python app.py                                    # Daheng camera: auto exposure first, then the live view
    python app.py --no-auto                          # live view only, no auto exposure at start
    python app.py --once                             # run auto exposure once, no window, and exit
    python app.py --no-save                          # never write calibration.json
    python app.py --images C:/path/img --out C:/path/img/results/auto_exposure
                                                     # every image in a folder, exposure simulated

Every garment is run from each --starts exposure (default 5000, 10000, 20000, 39594 us) and saved
the same way, live or from --images: <out>/<garment>/start_<us>us/ per start with an image of every
round, BEST_round<N>.jpg, best.json and log.txt; the garment folder gets the best of all starts
(BEST_<exposure>us.jpg, best.json); <out>/summary.txt gets one line per garment. Live, the garment
folder is <out>/<--garment>_<time>/ and the camera is left at the best values.

On a device stop the scanner app first: the camera can only be open in one process.

    sudo systemctl stop rms-app

The frame goes through the same steps as in the scanner app: camera -> PreProcessor (rotate, crop,
undistort from calibration.json) -> separate_garment. The logic itself is in auto_exposure.py.

With a camera, auto exposure runs once at start and the live view then shows the camera at the
best exposure and brightness it found.

Keys in the window:
    a         run auto exposure again (e.g. after changing the garment; saved to --out/live_<time>)
    +  /  -   exposure x1.25 / /1.25
    s         save the raw frame, the processed frame and the overlay to --out
    q / Esc   quit
"""

import argparse
import json
import os
import sys
import time

# OpenCV's Qt window looks for fonts in its own package and warns on every frame when there are
# none ("QFontDatabase: Cannot find font directory"); point it at the system's fonts instead.
if sys.platform.startswith('linux') and 'QT_QPA_FONTDIR' not in os.environ:
    for _font_dir in ('/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/truetype', '/usr/share/fonts'):
        if os.path.isdir(_font_dir):
            os.environ['QT_QPA_FONTDIR'] = _font_dir
            break

import cv2 # type: ignore
import numpy as np # type: ignore

import auto_exposure
from auto_exposure import (DEFAULT_EXPOSURE_US, best_image, best_values, pick_best, run_auto_exposure,
                           save_calibration, save_run)
from capture import PreProcessor, WebcamCapture
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
    cam = WebcamCapture(calibration_file=args.calibration)
    if not cam.start():
        sys.exit('camera did not start (see the log above)')
    # The camera classes re-read calibration.json on their first frame and would put the file's
    # exposure back over the start exposure just set; mark the file as already read.
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


def say(message):
    """A timestamped line in the terminal."""
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def camera_values(cam):
    return 'exposure %s us, brightness %s' % ('%g' % cam.exposure if cam.exposure else 'auto',
                                              '%g' % cam.gain if cam.gain is not None else '-')


def run_and_save(cam, pre, folder, calibration_file=None, rgb=False, start=None):
    """Run auto exposure, print it and save it to ``folder`` (see save_run)."""
    lines = []

    def log(message):
        print(message)
        lines.extend(str(message).splitlines())

    if start is None:
        say('auto exposure: starting at %s' % camera_values(cam))
    else:
        say('auto exposure: starting at exposure %g us, brightness %s'
            % (start[0], '-' if start[1] is None else '%g' % start[1]))
    result = run_auto_exposure(cam, pre, calibration_file=calibration_file, log=log, start=start)
    save_run(result, folder, lines, rgb=rgb)
    print('-' * 100)
    if result.best is None:
        say('auto exposure: NO GARMENT FOUND, camera left at %s' % camera_values(cam))
    else:
        say('auto exposure: BEST %s, contrast %.0f, solidity %.2f (round %d, picked by %s)'
            % (camera_values(cam), result.contrast, result.solidity, result.best.n, result.picked_by))
    say('auto exposure: saved to %s' % folder)
    print('-' * 100, flush=True)
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


SUMMARY_HEADER = '%-40s | best exposure, brightness, contrast, solidity | rounds, time' % 'garment'


def run_garment(cam, pre, name, garment_dir, starts, start_gain, rgb=False):
    """Run auto exposure on one garment from every start exposure, as the local tests did.

    Saved to ``garment_dir``: start_<us>us/ per start (every round's image, BEST_round<N>.jpg,
    best.json, log.txt), and the best of all starts as BEST_<us>us.jpg and best.json. The camera is
    left at the best exposure and brightness. Returns (summary line, best result or None).
    """
    t0 = time.time()
    bests, results = [], []
    for start in starts:
        print('\n=== %s: start %d us' % (name, start))
        # Set inside run_auto_exposure so its first round waits until the frames show it: set here,
        # the first round of every start measured a frame from the end of the previous start.
        result = run_and_save(cam, pre, os.path.join(garment_dir, 'start_%dus' % start), rgb=rgb,
                              start=(start, start_gain))
        results.append((start, result))
        if result.best is not None:
            bests.append((start, result))
    total = time.time() - t0
    rounds = sum(len(r.rounds) for _, r in results)
    timing = {'seconds': round(total, 1), 'rounds': rounds}
    print('=' * 100)
    print('TOTAL TIME %s: %.1f s  (%d rounds, %d starts)' % (name, total, rounds, len(results)))
    print('=' * 100)
    if not bests:
        say('%s: NO GARMENT FOUND from any start (%d rounds, %.1f s)' % (name, rounds, total))
        print('=' * 100, flush=True)
        with open(os.path.join(garment_dir, 'best.json'), 'w') as f:
            json.dump({'best': None, 'time': timing}, f, indent=4)
        return '%-40s | no garment found from any start | %2d rounds %6.1f s' % (name[:40], rounds, total), None
    best_round, _ = pick_best([r.best for _, r in bests])
    start, result = next((s, r) for s, r in bests if r.best is best_round)
    cam.set_exposure(result.exposure)
    if result.brightness is not None and hasattr(cam, 'set_gain'):
        cam.set_gain(result.brightness)
    values = dict(best_values(result), start_exposure=start)
    with open(os.path.join(garment_dir, 'best.json'), 'w') as f:
        json.dump({'best': values, 'per_start': {'%d' % s: best_values(r) for s, r in bests},
                   'time': timing}, f, indent=4)
    cv2.imwrite(os.path.join(garment_dir, 'BEST_%dus.jpg' % result.exposure),
                best_image(result, rgb=rgb, title='BEST (start %d us)' % start))
    brightness = '-' if result.brightness is None else '%g' % result.brightness
    say('%s: BEST OF ALL STARTS exposure %d us, brightness %s, contrast %.0f, solidity %.2f (from start %d us)'
        % (name, result.exposure, brightness, result.contrast, result.solidity, start))
    say('%s: %d rounds in %.1f s; saved to %s' % (name, rounds, total, garment_dir))
    print('=' * 100, flush=True)
    line = '%-40s | %6d us | brightness %5s | contrast %4.0f | solidity %.2f | from start %d | %2d rounds %6.1f s' % (
        name[:40], result.exposure, brightness, result.contrast, result.solidity, start, rounds, total)
    return line, result


def append_summary(out, line):
    """Add one garment's line to <out>/summary.txt (header written once)."""
    path = os.path.join(out, 'summary.txt')
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new:
            f.write(SUMMARY_HEADER + '\n')
        f.write(line + '\n')


def run_folder(args, processor):
    """--images: every image in the folder, from every start exposure, saved one folder per garment."""
    auto_exposure.SETTLE_S = 0                          # a saved frame needs no settling...
    auto_exposure.MIN_NEW_FRAMES = 0                    # ...and has no old frames in flight
    auto_exposure.FALLBACK_WAIT_S = 0
    starts = [float(v) for v in args.starts.split(',')]
    paths = sorted(os.path.join(args.images, f) for f in os.listdir(args.images)
                   if f.lower().endswith(IMAGE_TYPES))
    summary = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        cam = ImageCamera(path)
        use_crop = crop_fits(processor, cam.base)
        pre = processor.process_image if use_crop else (lambda f: f)
        print('\n##### %s (%s)' % (name, 'calibration crop' if use_crop else 'no crop: it does not fit this frame'))
        line, _ = run_garment(cam, pre, name, os.path.join(args.out, name), starts, ImageCamera.BASE_GAIN)
        summary.append(line)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write('\n'.join([SUMMARY_HEADER] + summary) + '\n')
    print('\n' + '\n'.join([SUMMARY_HEADER] + summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', help='folder of saved frames: run every one, exposure simulated, and exit')
    parser.add_argument('--starts', default='5000,10000,20000,39594',
                        help='start exposures in us, comma separated: auto exposure runs once from each')
    parser.add_argument('--garment', default='garment',
                        help='name for the result folders: <out>/<garment>_<time>/ per auto exposure run')
    parser.add_argument('--calibration', default='calibration.json', help='calibration.json copied from the device')
    parser.add_argument('--no-save', action='store_true', help='do not write the result to calibration.json')
    parser.add_argument('--once', action='store_true', help='run auto exposure once without a window, then exit')
    parser.add_argument('--no-auto', action='store_true', help='do not run auto exposure when the live view starts')
    parser.add_argument('--out', default='captures', help='folder for the results and for frames saved with s')
    args = parser.parse_args()

    processor = None
    if os.path.exists(args.calibration):
        processor = PreProcessor(calibration_file=args.calibration)
        pre = lambda f: None if f is None else processor.process_image(f)
    else:
        print('WARNING: %s not found: no rotate/crop/undistort, so the frame border is the room' % args.calibration)
        pre = lambda f: f 
    if args.images:
        run_folder(args, processor)
        return
    save_to = None if args.no_save or not os.path.exists(args.calibration) else args.calibration
    to_bgr = True                             # Daheng frames are RGB; only the display is converted

    if processor is not None and processor.crop_bbox:
        (x0, y0), (x1, y1) = processor.crop_bbox
        say('calibration: %s, crop %dx%d at (%d, %d)' % (args.calibration, x1 - x0, y1 - y0, x0, y0))
        # Cropped to the table: a garment larger than the crop may run off its sides. Off for --images,
        # whose uncropped test frames have the room along their sides.
        auto_exposure.EDGE_OK = True
    say('garments running off the frame sides are %s' % ('allowed (frame cropped to the table)'
                                                      if auto_exposure.EDGE_OK else 'taken for the room'))
    cam = open_camera(args)
    say('camera: started, %s' % camera_values(cam))
    if save_to is None:
        say('calibration.json will not be written (%s)' % ('--no-save' if args.no_save else 'no file'))

    starts = [float(v) for v in args.starts.split(',')]
    start_gain = cam.gain

    def auto():
        """One garment: auto exposure from every start, saved like the local tests, best value kept."""
        name = '%s_%s' % (args.garment, time.strftime('%Y%m%d_%H%M%S'))
        say('%s: auto exposure from %d start exposures (%s us)' % (name, len(starts), args.starts))
        line, result = run_garment(cam, pre, name, os.path.join(args.out, name), starts, start_gain, rgb=to_bgr)
        append_summary(args.out, line)
        if result is not None and save_to:
            save_calibration(save_to, result.exposure, result.brightness)
            # The camera classes re-read calibration.json when it changes and set its exposure again;
            # it holds the values just set, but mark it read so it cannot reset a later run's start.
            if hasattr(cam, 'last_config_mtime'):
                cam.last_config_mtime = os.path.getmtime(save_to)
            say('%s: exposure and brightness written to %s' % (name, save_to))

    try:
        if args.once:
            auto()
            return
        os.makedirs(args.out, exist_ok=True)
        if not args.no_auto:
            auto()
        say('live view: %s. Keys (click the window first): a = auto exposure again, +/- = exposure, '
            's = save frame, q = quit' % camera_values(cam))
        fps, last = 0.0, time.time()
        while True:
            raw = wait_frame(cam)
            if raw is None:
                say('no frame from the camera')
                continue
            frame = pre(raw)
            res = separate_garment(frame, edge_ok=auto_exposure.EDGE_OK)
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(1e-3, now - last)
            last = now
            cv2.imshow('auto exposure test', annotate(frame, res, cam, fps, to_bgr))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                say('quit')
                break
            current = cam.exposure or DEFAULT_EXPOSURE_US        # none until the camera has one
            if key == ord('a'):
                auto()
                say('live view: %s' % camera_values(cam))
            elif key in (ord('+'), ord('=')):
                cam.set_exposure(min(EXPOSURE_MAX_US, int(current * EXPOSURE_STEP)))
                say('manual: %s' % camera_values(cam))
            elif key == ord('-'):
                cam.set_exposure(max(EXPOSURE_MIN_US, int(current / EXPOSURE_STEP)))
                say('manual: %s' % camera_values(cam))
            elif key == ord('s'):
                stamp = time.strftime('%Y%m%d_%H%M%S')
                save = (lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) if to_bgr else (lambda f: f)
                cv2.imwrite(os.path.join(args.out, f'{stamp}_raw.jpg'), save(raw))
                cv2.imwrite(os.path.join(args.out, f'{stamp}_processed.jpg'), save(frame))
                cv2.imwrite(os.path.join(args.out, f'{stamp}_overlay.jpg'), draw_overlay(save(frame), res))
                say('saved %s/%s_raw.jpg, _processed.jpg, _overlay.jpg (%s)' % (args.out, stamp, camera_values(cam)))
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
