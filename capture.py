'''Image Pre-Processing Module'''
import argparse
import cv2
import json
import math
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import threading

import logging
get_logger = logging.getLogger

logger = get_logger(__name__)

# v4l2-ctl exists only on Linux; a module constant so tests can force either branch.
_IS_LINUX = sys.platform.startswith('linux')

class PreProcessor:
    def __init__(self, calibration_file=None):
        if calibration_file is None:
            from core.shared import paths
            calibration_file = paths.get_config_path('calibration.json')
        self.rotation_angle = 0
        self.crop_bbox = None
        self.calibration_file = calibration_file
        self.rotation_matrix = None
        self.last_config_mtime = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_resolution = None
        # Precomputed undistortion remap tables (computed once per calibration load)
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_roi = None
        self._undistort_size = None  # (w, h) the maps were built for
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration data from a JSON file if it exists."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                self.rotation_angle = calibration_data.get('rotation_angle', 0)
                self.crop_bbox = calibration_data.get('crop_bbox', None)
                self.camera_matrix = calibration_data.get('camera_matrix', None)
                self.dist_coeffs = calibration_data.get('dist_coeffs', None)
                self.calibration_resolution = calibration_data.get('calibration_resolution', None)
                self._warned_resolution = False
                self._precompute_rotation_matrix()
                # Invalidate cached undistort maps so they are rebuilt on next use
                self._undistort_map1 = None
                self._undistort_map2 = None
                self._undistort_roi = None
                self._undistort_size = None
                self.last_config_mtime = os.path.getmtime(self.calibration_file)
                logger.info(f"Calibration data loaded: rotation_angle={self.rotation_angle}, crop_bbox={self.crop_bbox}, camera_matrix={'set' if self.camera_matrix else 'none'}, dist_coeffs={'set' if self.dist_coeffs else 'none'}")
            except Exception as e:
                logger.exception(f"Failed to load calibration data: {e}")
        else:
            logger.warning("Calibration data not found. Using default settings.")

    def _reload_calibration_if_changed(self):
        """Reload calibration settings if the calibration file changes."""
        if not os.path.exists(self.calibration_file):
            return
        try:
            mtime = os.path.getmtime(self.calibration_file)
            if self.last_config_mtime is None or mtime != self.last_config_mtime:
                self._load_calibration()
        except Exception as e:
            logger.error(f"Failed to reload calibration data: {e}")


    def process_image(self, image_data, save_path=None):
        """Read and process the image from the given path."""
        try:
            image = image_data
            if image is None:
                raise RuntimeError(f"Failed to read image")
            self._reload_calibration_if_changed()

            if self.rotation_angle != 0:
                image = self.rotate_image(image, self.rotation_angle)

            if self.crop_bbox is not None:
                image = self.crop_image(image)
            
            # Finally apply undistortion using precomputed remap tables
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                try:
                    h, w = image.shape[:2]
                    # Build remap tables once per resolution (cached)
                    if self._undistort_map1 is None or self._undistort_size != (w, h):
                        camera_matrix = np.array(self.camera_matrix, dtype=np.float64)
                        dist_coeffs = np.array(self.dist_coeffs, dtype=np.float64)
                        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
                        self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
                            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
                        )
                        self._undistort_roi = roi
                        self._undistort_size = (w, h)
                        logger.info(f"[preprocess] Built undistort maps for {w}x{h}")
                    image = cv2.remap(image, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)
                    # Crop to valid ROI
                    x, y, w_roi, h_roi = self._undistort_roi
                    if w_roi > 0 and h_roi > 0:
                        image = image[y:y + h_roi, x:x + w_roi]
                    # Restore the ROI (a pixel or two smaller) to the crop size. The crop
                    # box decides that size: calibration_resolution can be saved stale by
                    # the calibration page, and resizing to a mismatched size stretches
                    # the image by a different factor on each axis.
                    target = self._processed_resolution()
                    if target is not None:
                        target_w, target_h = target
                        h_now, w_now = image.shape[:2]
                        if w_now != target_w or h_now != target_h:
                            if abs(w_now - target_w) > 0.01 * target_w or abs(h_now - target_h) > 0.01 * target_h:
                                logger.warning(f"[preprocess] processed frame is {w_now}x{h_now}, expected "
                                      f"{target_w}x{target_h}; not resizing, which would stretch it. "
                                      f"Check rotation and crop in calibration.")
                            else:
                                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    logger.exception(f"Error during undistortion: {e}")
            
            if save_path:
                self.save_image(image, save_path)
            return image
        except Exception as e:
            logger.exception(f"An error occurred during image processing: {e}")
            raise


    def _processed_resolution(self):
        """(w, h) of a processed frame: the crop box size, else calibration_resolution."""
        declared = None
        if self.calibration_resolution and len(self.calibration_resolution) == 2:
            declared = (int(self.calibration_resolution[0]), int(self.calibration_resolution[1]))
        if self.crop_bbox is None:
            return declared if declared and declared[0] > 0 and declared[1] > 0 else None

        (x0, y0), (x1, y1) = self.crop_bbox
        crop = (int(x1) - int(x0), int(y1) - int(y0))
        if declared and declared != crop and not getattr(self, '_warned_resolution', False):
            logger.warning(f"[preprocess] calibration_resolution {declared[0]}x{declared[1]} disagrees with "
                  f"the crop box {crop[0]}x{crop[1]}; using the crop box. Re-save calibration to fix the file.")
            self._warned_resolution = True
        return crop

    def crop_image(self, image):
        """Crop the image according to the crop_bbox in the calibration data."""
        try:
            if self.crop_bbox is None:
                raise ValueError("Crop bounding box is not defined.")
            x0, y0 = self.crop_bbox[0]
            x1, y1 = self.crop_bbox[1]
            x0, y0 = int(x0), int(y0)
            x1, y1 = int(x1), int(y1)
            h, w = image.shape[:2]
            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            if x0 == x1 or y0 == y1:
                raise ValueError("Invalid crop area: width or height is zero.")
            image = image[y0:y1, x0:x1]
            return image
        except Exception as e:
            logger.exception(f"Error during cropping: {e}")
            raise


    def _precompute_rotation_matrix(self):
        """Pre-compute rotation matrix for arbitrary angles."""
        angle = self.rotation_angle % 360
        if angle not in [0, 90, 180, 270] and angle != 0:
            # Matrix will be computed on first use with actual image dimensions
            self.rotation_matrix = None
        else:
            self.rotation_matrix = None

    def rotate_image(self, image, angle):
        """Rotate the image by the given angle."""
        angle = angle % 360
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            (h, w) = image.shape[:2]
            if self.rotation_matrix is None:
                center = (w // 2, h // 2)
                self.rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, self.rotation_matrix, (w, h))


    def save_image(self, image, path):
        """Save the image to the specified path."""
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        logger.info(f"Image saved to {path}")


'''Camera health and stall watchdog (shared by both capture backends)'''

# The Daheng stream occasionally stalls (every few weeks, most likely a USB or power event).
# gxipy then returns nothing and nothing raises, so _capture_loop kept serving the last good
# frame as if it were live until someone restarted the app. The watchdog below notices that
# no new frame has arrived and reopens the camera, which is what the manual restart did.
CAMERA_STATE_STARTING = 'starting'
CAMERA_STATE_OK = 'ok'
CAMERA_STATE_RESTARTING = 'restarting'
CAMERA_STATE_UNRESPONSIVE = 'unresponsive'
# Not plugged in. Told apart from a frozen camera by asking the SDK whether it still lists a
# device: restarting cannot fix a missing camera, and neither can exiting the app.
CAMERA_STATE_DISCONNECTED = 'disconnected'


def _env_number(name, default, cast=float):
    """A positive number from the environment; anything unusable falls back to the default."""
    try:
        value = cast(os.environ.get(name, '').strip() or default)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


# The camera runs at roughly 10 fps, so five seconds without a frame is a stall, not a slow
# frame. Plain constants with env overrides rather than app_settings keys: an operator has
# no reason to tune them, and a new key would need a settings migration on every device.
CAMERA_STALL_TIMEOUT_S = _env_number('CAMERA_STALL_TIMEOUT_S', 5.0)
CAMERA_RESTART_ATTEMPTS = _env_number('CAMERA_RESTART_ATTEMPTS', 3, int)
CAMERA_RESTART_BACKOFF_S = (2.0, 5.0, 10.0)   # settle time before attempt 1, 2, 3...
CAMERA_FIRST_FRAME_TIMEOUT_S = 5.0            # a reopened camera must deliver a frame within this
CAMERA_TEARDOWN_TIMEOUT_S = 10.0              # close_device() on a wedged USB device can hang
CAMERA_EXIT_GUARD_S = 600.0                   # no second watchdog exit inside this window
CAMERA_UNRESPONSIVE_RETRY_S = 60.0            # slow retry while the exit guard holds
CAMERA_DISCONNECTED_POLL_S = 2.0              # how often an unplugged camera is looked for
CAMERA_USB_RESET_SETTLE_S = 3.0               # a reset makes the device re-enumerate; let it
CAMERA_WATCHDOG_POLL_S = 1.0
CAMERA_EXIT_GUARD_FILE = 'camera_watchdog_exit.json'


DAHENG_USB_VENDOR_ID = '2ba2'
USB_SYSFS_ROOT = '/sys/bus/usb/devices'


def usb_camera_speed_mbit(vendor_id=DAHENG_USB_VENDOR_ID, sysfs_root=USB_SYSFS_ROOT):
    """Negotiated link speed of the attached Daheng camera in Mbit/s, or None.

    480 is USB 2.0 and 5000 or more is USB 3.x. The camera refuses to open on a USB 2.0 link,
    and the SDK's error does not say what speed it saw, so the journal gets this instead.
    Linux only (sysfs); anywhere else, or with no camera attached, it is None.
    """
    try:
        names = os.listdir(sysfs_root)
    except OSError:
        return None
    for name in sorted(names):
        base = os.path.join(sysfs_root, name)
        try:
            with open(os.path.join(base, 'idVendor'), 'r') as f:
                if f.read().strip().lower() != vendor_id:
                    continue
            with open(os.path.join(base, 'speed'), 'r') as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


USBDEVFS_RESET = 0x5514   # _IO('U', 20) from <linux/usbdevice_fs.h>


def usb_reset_camera(vendor_id=DAHENG_USB_VENDOR_ID, sysfs_root=USB_SYSFS_ROOT,
                     dev_root='/dev/bus/usb', ioctl=None):
    """Reset the attached Daheng camera at the USB level, like `usbreset` or a replug.

    A camera that came back on a USB 2.0 link refuses to open until it is reset. This sends
    the same USBDEVFS_RESET the `usbreset` tool does, straight from Python, so nothing needs
    installing. It needs write access to the device node, which the app already has: it
    opens the camera through libusb as its own user. Returns True when the reset was sent;
    any failure (not Linux, no camera, no permission) is logged and returns False.
    """
    try:
        names = os.listdir(sysfs_root)
    except OSError:
        return False
    node = None
    for name in sorted(names):
        base = os.path.join(sysfs_root, name)
        try:
            with open(os.path.join(base, 'idVendor'), 'r') as f:
                if f.read().strip().lower() != vendor_id:
                    continue
            with open(os.path.join(base, 'busnum'), 'r') as f:
                bus = int(f.read().strip())
            with open(os.path.join(base, 'devnum'), 'r') as f:
                dev = int(f.read().strip())
        except (OSError, ValueError):
            continue
        node = os.path.join(dev_root, f'{bus:03d}', f'{dev:03d}')
        break
    if node is None:
        logger.warning("USB reset: no Daheng camera found on the USB bus")
        return False
    try:
        if ioctl is None:
            import fcntl
            ioctl = fcntl.ioctl
        fd = os.open(node, os.O_WRONLY)
        try:
            ioctl(fd, USBDEVFS_RESET, 0)
        finally:
            os.close(fd)
        logger.info(f"USB reset sent to {node}")
        return True
    except PermissionError:
        logger.warning(f"USB reset: no permission to write {node}. "
              "Give the app's user access to the camera (a udev rule for vendor 2ba2)")
    except Exception as e:
        logger.error(f"USB reset of {node} failed: {e}")
    return False


def is_usb_speed_error(error):
    """True for the SDK's "can only be operated on an USB3.0 Port" open failure.

    It arrives wrapped as 'Access denied -1005', which sends people looking at permissions;
    the camera is found but the link is not USB 3.0.
    """
    text = str(error).lower().replace(' ', '')
    return 'usb3.0port' in text


class CameraHealthMixin:
    """Frame-age tracking, a health snapshot and the restart watchdog.

    The watchdog runs in its own thread, never inside _capture_loop: tearing the camera down
    from the thread that is reading it can deadlock on the very call that is hanging.
    """

    # Overridable per instance so tests need neither real time nor a real exit.
    _clock = staticmethod(time.monotonic)
    _sleep = staticmethod(time.sleep)
    _exit = staticmethod(os._exit)

    def _init_health(self):
        self.state = CAMERA_STATE_STARTING
        self._last_frame_ts = None       # self._clock() of the last good frame
        self._stream_started_ts = None   # when the stream was last opened; frame age starts here
        self.restart_attempt = 0
        self.restarts_total = 0
        self.last_error = None
        # What is wrong, when the SDK said so: 'usb_speed' = found but not on a USB 3.0 link,
        # which restarting cannot fix and the operator can (re-seat the cable, another port).
        self.problem = None
        self.usb_speed_mbit = None
        self._ever_ok = False            # has this process ever seen a frame
        self._intentional_stop = False
        self._created_ts = self._clock()
        self._outage_started_ts = None
        self._outage_kind = None         # 'frozen' or 'disconnected' while an outage is open
        self._next_unresponsive_retry_ts = 0.0
        self._next_probe_ts = 0.0
        self._probe_error_logged = False
        self._on_event = None
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()
        self.exit_guard_file = None      # resolved lazily from the config dir

    # -- frame bookkeeping ------------------------------------------------------------

    def _mark_frame(self):
        """Record that a good frame just arrived."""
        self._last_frame_ts = self._clock()
        if self.state == CAMERA_STATE_STARTING:
            self.state = CAMERA_STATE_OK
            self._ever_ok = True

    def _frame_age(self):
        """Seconds since the last good frame, or since the stream opened if none has come."""
        reference = self._last_frame_ts
        if reference is None or (self._stream_started_ts is not None and self._stream_started_ts > reference):
            reference = self._stream_started_ts
        return None if reference is None else self._clock() - reference

    def _frame_is_stale(self):
        """True when the cached frame is older than a stall. Never serve it as live."""
        if self._last_frame_ts is None:
            return False
        return (self._clock() - self._last_frame_ts) > CAMERA_STALL_TIMEOUT_S

    def health(self):
        """JSON-safe snapshot for /camera/health, the popup and the automation matcher."""
        if getattr(self, 'debug_mode', False):
            state = CAMERA_STATE_OK
        else:
            state = self.state
        age = self._frame_age()
        return {
            'state': state,
            'last_frame_age_s': None if age is None else round(age, 2),
            'restart_attempt': self.restart_attempt,
            'max_attempts': CAMERA_RESTART_ATTEMPTS,
            'restarts_total': self.restarts_total,
            'last_error': self.last_error,
            'problem': self.problem,
            'usb_speed_mbit': self.usb_speed_mbit,
        }

    def set_event_callback(self, callback):
        """`callback(event_type, fields)` is told about each freeze and outcome (telemetry).

        A callback rather than an import: this module must not depend on app.py or telemetry.
        """
        self._on_event = callback

    def _emit(self, event_type, **fields):
        callback = self._on_event
        if callback is None:
            return
        try:
            callback(event_type, fields)
        except Exception as e:
            logger.error(f"Event callback failed for {event_type}: {e}")

    def _log_usb_speed(self):
        """Put the camera's real link speed in the journal when it will not open."""
        speed = usb_camera_speed_mbit()
        self.usb_speed_mbit = speed
        if speed is None:
            return
        label = 'USB 2.0 or slower' if speed < 5000 else 'USB 3.x'
        logger.info(f"USB link speed of the camera: {speed:g} Mbit/s ({label})")

    def _reset_usb_camera(self):
        """Reset the camera's USB device and give it time to re-enumerate. True if it was sent."""
        before = usb_camera_speed_mbit()
        ok = usb_reset_camera()
        logger.info(f"USB reset {'sent' if ok else 'not possible'} "
              f"(link was {'unknown' if before is None else f'{before:g} Mbit/s'})")
        self._emit('camera_usb_reset', ok=ok, speed_before_mbit=before)
        if ok:
            self._sleep(CAMERA_USB_RESET_SETTLE_S)
            self._log_usb_speed()   # after: shows in the journal whether the reset fixed the link
        return ok

    def _camera_present(self):
        """Whether the SDK lists a camera: True, False, or None when that cannot be told.

        None is treated as "present": when unsure, the camera is restarted rather than
        reported unplugged to an operator whose camera is merely frozen.
        """
        return None

    # -- watchdog ---------------------------------------------------------------------

    def start_watchdog(self, on_event=None):
        """Start the stall watchdog once. Not used in debug mode, where the frame is a file."""
        if on_event is not None:
            self._on_event = on_event
        if getattr(self, 'debug_mode', False):
            return False
        thread = self._watchdog_thread
        if thread is not None and thread.is_alive():
            return True
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, name='camera-watchdog', daemon=True)
        self._watchdog_thread.start()
        logger.info(f"Watchdog started (stall timeout {CAMERA_STALL_TIMEOUT_S:g}s, "
              f"{CAMERA_RESTART_ATTEMPTS} restart attempts)")
        return True

    def stop_watchdog(self):
        self._watchdog_stop.set()

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(CAMERA_WATCHDOG_POLL_S):
            try:
                self._watchdog_tick()
            except Exception as e:
                # The watchdog is the safety net; it must outlive a bug in itself.
                logger.exception(f"Watchdog error: {e}")

    def _stall_reason(self):
        """Why the camera needs a restart right now, or None when it is healthy."""
        if self._intentional_stop or getattr(self, 'debug_mode', False):
            return None
        if not self.is_running:
            return 'not_running'      # start() failed at boot, or the stream died
        age = self._frame_age()
        if age is not None and age > CAMERA_STALL_TIMEOUT_S:
            return 'stalled'
        return None

    def _watchdog_tick(self):
        if self.state == CAMERA_STATE_DISCONNECTED:
            # Waiting for the cable. Look every couple of seconds; once the camera is listed
            # again, open it with the usual restart ladder (it may need a moment to settle).
            if self._clock() >= self._next_probe_ts:
                self._next_probe_ts = self._clock() + CAMERA_DISCONNECTED_POLL_S
                if self._camera_present() is not False:
                    self._recover('reconnected')
            return
        if self.state == CAMERA_STATE_UNRESPONSIVE:
            # Only reached while the exit guard holds: keep trying, slowly.
            if self._clock() >= self._next_unresponsive_retry_ts:
                self._next_unresponsive_retry_ts = self._clock() + CAMERA_UNRESPONSIVE_RETRY_S
                if self._restart_once():
                    self._finish_outage(attempts=0)
            return
        reason = self._stall_reason()
        if reason is not None:
            self._recover(reason)

    def _enter_disconnected(self, reason):
        """The camera is not plugged in: release it, say so, and wait for it to come back.

        No restart attempts and never an exit: neither can bring back a missing camera.
        """
        first_time = self._outage_kind != 'disconnected'
        self._outage_kind = 'disconnected'
        if self._outage_started_ts is None:
            self._outage_started_ts = self._clock()
        self._teardown_with_timeout()
        with self.frame_lock:
            self.latest_frame = None
        self.state = CAMERA_STATE_DISCONNECTED
        self.restart_attempt = 0
        self.last_error = 'camera not found'
        self.problem = None
        self._next_probe_ts = self._clock() + CAMERA_DISCONNECTED_POLL_S
        logger.warning(f"No camera found ({reason}); waiting for it to be plugged in")
        if first_time:   # a flapping cable must not send a point per flap
            self._emit('camera_disconnected', reason=reason,
                       uptime_s=round(self._clock() - self._created_ts, 1))

    def _recover(self, reason):
        """One outage: announce it, retry the restart a few times, then escalate.

        `reason` 'reconnected' continues an outage that began as a disconnect.
        """
        if reason != 'reconnected':
            age = self._frame_age()
            stalled_for = 0.0 if age is None else age
            self._outage_started_ts = self._clock()
            self._outage_kind = 'frozen'
            if self._camera_present() is False:
                self._enter_disconnected(reason)
                return
            logger.warning(f"No new frame for {stalled_for:.1f}s ({reason}); restarting the camera")
            self._emit('camera_freeze', reason=reason, stalled_for_s=round(stalled_for, 2),
                       restarts_total=self.restarts_total,
                       uptime_s=round(self._clock() - self._created_ts, 1))

        self.state = CAMERA_STATE_RESTARTING
        with self.frame_lock:
            self.latest_frame = None   # drop the stale picture; nothing may serve it as live

        for attempt in range(1, CAMERA_RESTART_ATTEMPTS + 1):
            self.restart_attempt = attempt
            delay = CAMERA_RESTART_BACKOFF_S[min(attempt, len(CAMERA_RESTART_BACKOFF_S)) - 1]
            if self._watchdog_stop.is_set():
                return
            self._sleep(delay)
            logger.info(f"Restart attempt {attempt}/{CAMERA_RESTART_ATTEMPTS}")
            if self._restart_once():
                self._finish_outage(attempts=attempt)
                return
            # start() fails the same way for a missing camera and a broken one. The handle
            # is closed by now, so the SDK's device list is trustworthy: unplugged mid-restart
            # is a disconnect, not a reason to exhaust the attempts and exit the app.
            if self._camera_present() is False:
                self._enter_disconnected('lost_during_restart')
                return

        self._give_up()

    def _finish_outage(self, attempts):
        downtime = 0.0 if self._outage_started_ts is None else self._clock() - self._outage_started_ts
        self.state = CAMERA_STATE_OK
        self._ever_ok = True
        self.restart_attempt = 0
        self.restarts_total += 1
        self.last_error = None
        self.problem = None
        self._outage_started_ts = None
        logger.info(f"Camera is back after {downtime:.1f}s")
        event = 'camera_reconnected' if self._outage_kind == 'disconnected' else 'camera_restart_ok'
        self._outage_kind = None
        self._emit(event, attempts=attempts, downtime_s=round(downtime, 1))

    def _give_up(self):
        downtime = 0.0 if self._outage_started_ts is None else self._clock() - self._outage_started_ts
        self.state = CAMERA_STATE_UNRESPONSIVE
        self.last_error = f'no frame after {CAMERA_RESTART_ATTEMPTS} restart attempts'
        # Before any exit: os._exit skips atexit, so telemetry cannot wait for cleanup.
        self._emit('camera_unresponsive', attempts=CAMERA_RESTART_ATTEMPTS, downtime_s=round(downtime, 1),
                   problem=self.problem, usb_speed_mbit=self.usb_speed_mbit)

        if self._ever_ok and self._exit_allowed():
            logger.critical("Camera unresponsive; exiting so systemd restarts the app")
            self._record_exit()
            self._exit(1)
            return
        logger.error("Camera unresponsive; not exiting "
              f"({'camera never worked in this run' if not self._ever_ok else 'the app exited for this less than 10 min ago'})"
              f", retrying every {CAMERA_UNRESPONSIVE_RETRY_S:g}s")
        self._next_unresponsive_retry_ts = self._clock() + CAMERA_UNRESPONSIVE_RETRY_S

    def _restart_once(self):
        """Fully reopen the camera. True once a fresh frame has arrived.

        A camera that came back on a USB 2.0 link will not open until its USB device is
        reset (the same as `usbreset`, found to fix it on a device). So when the SDK says
        so, reset it and open again in the same attempt, instead of burning the attempts.
        """
        self._teardown_with_timeout()
        if self._open_and_wait_for_frame():
            return True
        if self.problem == 'usb_speed' and self._reset_usb_camera():
            return self._open_and_wait_for_frame()
        return False

    def _open_and_wait_for_frame(self):
        began = self._clock()
        try:
            started = self.start()
        except Exception as e:
            logger.error(f"Restart failed: {e}")
            self.last_error = str(e)
            return False
        if not started:
            self.last_error = 'camera did not open'
            return False
        deadline = began + CAMERA_FIRST_FRAME_TIMEOUT_S
        while self._clock() < deadline:
            ts = self._last_frame_ts
            if ts is not None and ts >= began:
                return True
            self._sleep(0.1)
        self.last_error = 'camera opened but delivered no frame'
        return False

    def _teardown_with_timeout(self):
        """stop() on a worker thread: on a wedged device close_device() can block for good."""
        worker = threading.Thread(target=self._safe_stop, name='camera-teardown', daemon=True)
        worker.start()
        worker.join(CAMERA_TEARDOWN_TIMEOUT_S)
        if worker.is_alive():
            logger.warning(f"Teardown still blocked after {CAMERA_TEARDOWN_TIMEOUT_S:g}s; abandoning the old handle")
        self._abandon_handles()

    def _safe_stop(self):
        try:
            self.stop()
        except Exception as e:
            logger.error(f"Error during teardown: {e}")

    def _abandon_handles(self):
        """Forget the old device so start() builds a fresh one even if stop() never returned."""
        self.is_running = False
        self.latest_frame = None

    # -- exit guard -------------------------------------------------------------------

    def _guard_path(self):
        if self.exit_guard_file:
            return self.exit_guard_file
        from core.shared import paths
        return os.path.join(paths.get_config_dir(), CAMERA_EXIT_GUARD_FILE)

    def _exit_allowed(self):
        """False when the watchdog already exited the app recently: a camera that is really
        gone must not turn into a restart loop."""
        try:
            with open(self._guard_path(), 'r') as f:
                last = float(json.load(f).get('exited_at', 0))
            return (time.time() - last) >= CAMERA_EXIT_GUARD_S
        except Exception:
            return True

    def _record_exit(self):
        path = self._guard_path()
        try:
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'exited_at': time.time()}, f)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"Could not write the exit guard: {e}")


'''Camera Capture Module using gxipy'''
class WebcamCapture(CameraHealthMixin):
    def __init__(self, capture_interval=0.1, calibration_file=None):
        if calibration_file is None:
            from core.shared import paths
            calibration_file = paths.get_config_path('calibration.json')
        self.capture_interval = capture_interval
        self.is_running = False
        self.cam = None
        self.data_stream = None
        self.latest_frame = None
        self.device_manager = None
        self.calibration_file = calibration_file
        self.last_config_mtime = None
        self.last_config_check = 0
        self.config_check_interval = 1.0  # Check config every 1 second instead of every frame
        self.exposure = None
        self.gain = None
        self._load_calibration()        
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        # Get the absolute path for saving images
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.script_dir, ""))
        # Debug mode settings
        self.debug_mode = False
        self.debug_image_path = None
        self.debug_image = None
        self.gx = None  # gxipy module, imported on demand
        # Bumped by every start(); a capture loop from an earlier open sees the change and exits.
        self._generation = 0
        self._init_health()



    def _load_calibration(self):
        """Load calibration data from a JSON file if it exists."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                self.exposure = calibration_data.get('exposure', 20000)
                self.gain = calibration_data.get('gain', 10)
            except Exception as e:
                logger.exception(f"Failed to load calibration data: {e}")
        else:
            logger.warning("Calibration data not found. Using default settings.")

    def _check_and_update_calibration(self):
        """Check if the calibration file was modified and update camera settings."""
        # Throttle config checks to once per second
        current_time = time.time()
        if current_time - self.last_config_check < self.config_check_interval:
            return
        
        self.last_config_check = current_time
        
        try:
            mtime = os.path.getmtime(self.calibration_file)
            if self.last_config_mtime is None or mtime != self.last_config_mtime:
                with open(self.calibration_file, 'r') as f:
                    calibration_data = json.load(f)
 
                new_exposure = calibration_data.get('exposure', None)
                new_gain = calibration_data.get('gain', None)
 
                if new_exposure is not None and new_exposure != self.exposure:
                    self.exposure = new_exposure
                    self.cam.ExposureTime.set(self.exposure)
                    logger.info(f"[Live Update] Exposure changed to {self.exposure}")
 
                if new_gain is not None and new_gain != self.gain:
                    self.gain = new_gain
                    self.cam.Gain.set(self.gain)
                    logger.info(f"[Live Update] Gain changed to {self.gain}")
 
                self.last_config_mtime = mtime
        except Exception as e:
            logger.error(f"[Live Update] failed: {e}")

    def set_exposure(self, exposure_time):
        """
        Set the exposure time of the camera.
        
        Args:
            exposure_time (float): The exposure time in microseconds.
        """
        if self.cam is not None:
            try:
                self.cam.ExposureTime.set(exposure_time)
                self.exposure = exposure_time
                logger.info(f"Exposure set to {exposure_time} microseconds.")
            except Exception as e:
                logger.exception(f"Failed to set exposure: {e}")
        else:
            logger.warning("Camera is not initialized. Please start the camera first.")    

    def set_gain(self, gain_value):
        """
        Set the gain of the camera.
        
        Args:
            gain_value (float): The gain value to set.
        """
        if self.cam is not None:
            try:
                self.cam.Gain.set(gain_value)
                self.gain = gain_value
                logger.info(f"Gain set to {gain_value}.")
            except Exception as e:
                logger.exception(f"Failed to set gain: {e}")
        else:
            logger.warning("Camera is not initialized. Please start the camera first.")        

    def start(self):
        """Start and initialize the camera stream."""
        try:
            # If camera is already initialized, don't reinitialize
            if self.cam is not None and self.is_running:
                logger.warning("Camera is already running.")
                return True
            
            # Handle debug mode
            if self.debug_mode:
                logger.info("Starting in DEBUG MODE - using static image")
                if self.debug_image_path and os.path.exists(self.debug_image_path):
                    self.debug_image = cv2.imread(self.debug_image_path)
                    if self.debug_image is not None:
                        self.latest_frame = self.debug_image.copy()
                        self.is_running = True
                        logger.info(f"Debug image loaded from: {self.debug_image_path}")
                        return True
                    else:
                        logger.error(f"Failed to load debug image from: {self.debug_image_path}")
                        return False
                else:
                    logger.warning(f"Debug image not found at: {self.debug_image_path}")
                    return False

            # Import gxipy only when needed (not in debug mode)
            import gxipy as gx
            self.gx = gx

            # Create a device manager
            self.device_manager = gx.DeviceManager()

            # Update the device list
            dev_num, dev_info_list = self.device_manager.update_device_list()
            if dev_num == 0:
                logger.warning("No gxipy-compatible cameras found.")
                return False

            logger.info(f"Found {dev_num} camera(s)")
            for i in range(dev_num):
                logger.info(f"Device {i}: {dev_info_list[i]}")

            # Try to open device starting from index 1 (gxipy uses 1-based indexing)
            self.cam = None
            self.problem = None
            for index in range(1, dev_num + 1):
                try:
                    self.cam = self.device_manager.open_device_by_index(index)
                    if self.cam is not None:
                        logger.info(f"Opened camera at index {index}")
                        break
                except Exception as e:
                    logger.error(f"Failed to open camera at index {index}: {e}")
                    if is_usb_speed_error(e):
                        self.problem = 'usb_speed'
                        self.last_error = 'camera is not on a USB 3.0 link'
                        self._log_usb_speed()
                    # Check if device is already open (device not available)
                    if "device has been open" in str(e).lower() or "-1004" in str(e):
                        logger.warning("Camera device is already open in another process.")
                        return False
            
            if self.cam is None:
                logger.error("Failed to open any camera device.")
                return False

            # Apply camera settings
            if self.exposure is not None:
                try:
                    self.cam.ExposureTime.set(self.exposure)
                    logger.info(f"Exposure set to {self.exposure}")
                except Exception as e:
                    logger.error(f"Failed to set exposure: {e}")
            if self.gain is not None:
                try:
                    self.cam.Gain.set(self.gain)
                    logger.info(f"Gain set to {self.gain}")
                except Exception as e:
                    logger.error(f"Failed to set gain: {e}")

            # Set acquisition mode to continuous
            self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

            # Get data stream
            self.data_stream = self.cam.data_stream
            if self.data_stream is None or len(self.data_stream) == 0:
                logger.error("Failed to get data stream from camera")
                return False

            # Start data acquisition
            self.cam.stream_on()
            
            # Reduced delay for faster startup
            time.sleep(0.1)

            self.is_running = True
            self._intentional_stop = False
            self._stream_started_ts = self._clock()

            # Start background thread to continuously capture frames
            self._generation += 1
            self.capture_thread = threading.Thread(
                target=self._capture_loop, args=(self._generation, self.data_stream[0]), daemon=True)
            self.capture_thread.start()
            
            logger.info("Camera stream initialized and running successfully.")
            return True
            
        except Exception as e:
            logger.exception(f"Error starting camera: {e}")
            return False

    def _camera_present(self):
        """True when gxipy lists at least one camera. A fresh DeviceManager re-enumerates USB."""
        try:
            gx = self.gx
            if gx is None:
                import gxipy as gx
            count, _ = gx.DeviceManager().update_device_list()
            return count > 0
        except Exception as e:
            if not self._probe_error_logged:   # every 2 s while unplugged would flood journald
                logger.warning(f"Could not list cameras: {e}")
                self._probe_error_logged = True
            return None

    def _capture_loop(self, generation, stream):
        """Background thread to continuously capture frames from camera.

        `stream` is this open's own data stream and `generation` its start() count: after the
        watchdog reopens the camera an old loop must neither read the new stream's state nor
        publish into it, and stop() clears self.data_stream under it.
        """
        while self.is_running and generation == self._generation:
            try:
                # Get raw image with reduced timeout for faster frame drops
                raw_image = stream.get_image(timeout=100)
                if raw_image is None:
                    continue

                # Check if image acquisition was successful
                if raw_image.get_status() != self.gx.GxFrameStatusList.SUCCESS:
                    continue

                # Convert the raw image to RGB format
                rgb_image = raw_image.convert("RGB")
                if rgb_image is None:
                    continue

                # Get numpy array from the RGB image
                numpy_image = rgb_image.get_numpy_array()
                if numpy_image is None:
                    continue

                # Update latest frame with thread safety
                with self.frame_lock:
                    if generation != self._generation:
                        return
                    self.latest_frame = numpy_image
                    self._mark_frame()

            except Exception as e:
                if self.is_running and generation == self._generation:  # Only print if we're supposed to be running
                    logger.error(f"Error in capture loop: {e}")
                time.sleep(0.01)

    def get_latest_image(self, copy: bool = True):
        """Get the latest frame from the camera."""
        if not self.is_running:
            return None
        
        # In debug mode, reload the debug image on each call (to support updating the file)
        if self.debug_mode:
            if self.debug_image_path and os.path.exists(self.debug_image_path):
                try:
                    fresh_image = cv2.imread(self.debug_image_path)
                    if fresh_image is not None:
                        self.debug_image = fresh_image
                        self.latest_frame = fresh_image.copy()
                except Exception as e:
                    logger.error(f"Error reloading debug image: {e}")
            if self.latest_frame is not None:
                return self.latest_frame.copy() if copy else self.latest_frame
            return None

        # A frozen or restarting camera must read as "no frame", never as the last picture:
        # callers already turn None into "no camera feed" / "no frame available".
        if self.state in (CAMERA_STATE_RESTARTING, CAMERA_STATE_UNRESPONSIVE,
                          CAMERA_STATE_DISCONNECTED) or self._frame_is_stale():
            return None

        # Check and update calibration periodically
        self._check_and_update_calibration()

        # Return the latest captured frame with thread safety
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy() if copy else self.latest_frame
            return None

    def stop(self):
        """Stop the camera capture and release resources."""
        self.is_running = False
        self._intentional_stop = True

        # In debug mode, just clean up and return
        if self.debug_mode:
            self.debug_image = None
            self.latest_frame = None
            logger.info("Debug mode stopped.")
            return

        # Take the handles before the calls that can block. If close_device() hangs on a
        # wedged USB device the watchdog abandons this thread and opens a new camera; this
        # call must then never clear the handles of that new one when it finally returns.
        cam = self.cam
        self.cam = None
        self.data_stream = None
        self.device_manager = None
        if cam is not None:
            try:
                # Stop data acquisition
                cam.stream_off()
                # Close device
                cam.close_device()
                logger.info("Camera released and resources have been cleaned up.")
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
        cv2.destroyAllWindows()


'''OpenCV webcam backend, used on development machines without a Daheng camera'''

# The tuning pages only ever write Daheng units into calibration.json: `gain` is the
# operator's brightness on a 5-30 scale and `exposure` is microseconds (see
# _clamp_brightness / _clamp_exposure in app.py). A UVC webcam wants neither, so this
# backend converts. Inventing a second pair of settings instead would mean the scan
# page, the exposure page and the per-style camera profiles all had to know which
# camera is attached.
BRIGHTNESS_INPUT_MIN = 5.0
BRIGHTNESS_INPUT_MAX = 30.0

# Default scale for CAP_PROP_BRIGHTNESS. UVC webcams almost always report 0-255.
DEFAULT_BRIGHTNESS_RANGE = (0.0, 255.0)

# What CAP_PROP_EXPOSURE means, per driver family, and the CAP_PROP_AUTO_EXPOSURE value
# that switches that family to manual. Most UVC drivers ignore a manual exposure while
# auto-exposure is still on, and do it silently, so the two always travel together.
EXPOSURE_MANUAL_AUTO_VALUE = {
    # V4L2 exposure_time_absolute counts 100 us units; auto_exposure is 1 manual / 3 auto.
    'v4l2': 1.0,
    # DirectShow and Media Foundation take log2(seconds); auto is 0.75, manual 0.25.
    'log2': 0.25,
    # The driver already takes microseconds; pass the operator's value straight through.
    'raw': 1.0,
}

_BACKEND_EXPOSURE_MODES = {
    'V4L2': 'v4l2',
    'V4L': 'v4l2',
    'GSTREAMER': 'v4l2',
    'DSHOW': 'log2',
    'MSMF': 'log2',
}


def resolve_exposure_mode(requested, backend_name):
    """Pick the exposure scale to use, from `app.web_cam_exposure_mode` or the backend.

    'auto' (the default) maps the OpenCV backend name; an explicit mode overrides it for
    a camera whose driver disagrees with its family.
    """
    mode = str(requested or 'auto').strip().lower()
    if mode in EXPOSURE_MANUAL_AUTO_VALUE:
        return mode
    if mode != 'auto':
        logger.warning(f"[Webcam] Unknown exposure mode '{requested}'; detecting from the backend instead.")
    detected = _BACKEND_EXPOSURE_MODES.get(str(backend_name or '').strip().upper())
    if detected:
        return detected
    # Devices run Linux, so V4L2 is the better guess than handing microseconds to a
    # driver that reads them as log2 seconds and pins the sensor wide open.
    logger.warning(f"[Webcam] Unrecognised capture backend '{backend_name}'; assuming V4L2 exposure units.")
    return 'v4l2'


def exposure_to_backend_value(exposure_us, mode):
    """Convert the stored microseconds into the unit `mode`'s driver expects, or None."""
    try:
        micros = float(exposure_us)
    except (TypeError, ValueError):
        return None
    if micros <= 0:
        return None
    if mode == 'log2':
        # UVC through DirectShow/MSMF accepts roughly -13..0; anything outside is rejected.
        return float(max(-13, min(0, int(round(math.log(micros / 1000000.0, 2))))))
    if mode == 'raw':
        return micros
    return float(max(1, int(round(micros / 100.0))))


def brightness_to_backend_value(gain_value, out_min, out_max):
    """Map the operator's 5-30 brightness linearly onto the driver's own scale."""
    try:
        value = float(gain_value)
    except (TypeError, ValueError):
        return None
    value = max(BRIGHTNESS_INPUT_MIN, min(BRIGHTNESS_INPUT_MAX, value))
    fraction = (value - BRIGHTNESS_INPUT_MIN) / (BRIGHTNESS_INPUT_MAX - BRIGHTNESS_INPUT_MIN)
    return float(out_min) + fraction * (float(out_max) - float(out_min))


def brightness_range_from_env():
    """(min, max) of the driver's brightness scale, from app_settings.

    Webcams disagree: 0-255 is usual, but 0-100 and -64..64 both exist, and an inverted
    range (min > max) is a legitimate way to flip a camera whose scale runs backwards.
    Only min == max is rejected, because it would flatten the slider to one value.
    """
    def _read(name, fallback):
        raw = os.environ.get(name, '').strip()
        if not raw:
            return float(fallback)
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(f"[Webcam] Invalid {name}={raw!r}; using {fallback}")
            return float(fallback)

    low = _read('WEB_CAM_BRIGHTNESS_MIN', DEFAULT_BRIGHTNESS_RANGE[0])
    high = _read('WEB_CAM_BRIGHTNESS_MAX', DEFAULT_BRIGHTNESS_RANGE[1])
    if low == high:
        logger.warning("[Webcam] web_cam_brightness_min equals web_cam_brightness_max; using the 0-255 default.")
        return DEFAULT_BRIGHTNESS_RANGE
    return (low, high)


class OpenCVWebcamCapture(CameraHealthMixin):
    """Frame source backed by cv2.VideoCapture, selected by `app.web_cam`.

    It mirrors WebcamCapture's contract -- start/stop/get_latest_image, set_exposure /
    set_gain, the debug-image mode and the once-per-second calibration watch -- so the
    live preview, the camera exposure page and the per-style camera profiles behave the
    same on a development machine as they do on a Daheng device. Before this class the
    webcam branch was a five-method stub with no settings at all, and every brightness
    or exposure edit was written to calibration.json and then silently ignored.
    """

    def __init__(self, cam_index=0, calibration_file=None, exposure_mode=None, brightness_range=None):
        if calibration_file is None:
            from core.shared import paths
            calibration_file = paths.get_config_path('calibration.json')
        self.cam_index = int(cam_index)
        self.cap = None
        self.is_running = False
        self.calibration_file = calibration_file
        self.last_config_mtime = None
        self.last_config_check = 0
        self.config_check_interval = 1.0  # Same throttle as the Daheng backend
        self.exposure = None
        self.gain = None
        # None = never set by the operator: the driver keeps its own focus behaviour.
        self.autofocus = None
        self.exposure_mode = exposure_mode or os.environ.get('WEB_CAM_EXPOSURE_MODE', 'auto')
        # Resolved lazily: the backend name is only known once the capture is open.
        self._resolved_exposure_mode = None
        self.brightness_range = brightness_range or brightness_range_from_env()
        # Which property actually carried brightness; webcams rarely support both.
        self._brightness_prop = None
        self.latest_frame = None
        # Debug mode settings (set by app.py after the backend is chosen)
        self.debug_mode = False
        self.debug_image_path = None
        self.debug_image = None
        # Health is reported (for /camera/health and the popup) but there is no watchdog on
        # this development backend: it reads on demand, so there is no stream to stall.
        self._init_health()
        self._load_calibration()

    def start_watchdog(self, on_event=None):
        """No watchdog here (see __init__); accepts the call so app.py need not branch."""
        return False

    def _load_calibration(self):
        """Load the stored exposure/gain if calibration.json exists.

        Left as None when the file has no values: a fresh development setup is better
        off on the webcam's own auto-exposure than forced to a guessed manual one.
        """
        if not os.path.exists(self.calibration_file):
            logger.warning("[Webcam] Calibration data not found. Leaving the camera on its own settings.")
            return
        try:
            with open(self.calibration_file, 'r') as f:
                calibration_data = json.load(f)
            self.exposure = calibration_data.get('exposure', None)
            self.gain = calibration_data.get('gain', None)
            stored_autofocus = calibration_data.get('autofocus', None)
            self.autofocus = stored_autofocus if isinstance(stored_autofocus, bool) else None
        except Exception as e:
            logger.exception(f"[Webcam] Failed to load calibration data: {e}")

    def _backend_name(self):
        try:
            return self.cap.getBackendName() if self.cap is not None else ''
        except Exception:
            return ''

    def _exposure_scale(self):
        """The resolved exposure scale; worked out once, after the capture is open."""
        if self._resolved_exposure_mode is None:
            self._resolved_exposure_mode = resolve_exposure_mode(self.exposure_mode, self._backend_name())
        return self._resolved_exposure_mode

    def _apply_exposure(self, exposure_us):
        """Push a microsecond exposure to the driver. Returns whether it was accepted."""
        if self.cap is None or exposure_us is None:
            return False
        mode = self._exposure_scale()
        value = exposure_to_backend_value(exposure_us, mode)
        if value is None:
            logger.warning(f"[Webcam] Ignoring invalid exposure {exposure_us!r}")
            return False
        try:
            # Always before CAP_PROP_EXPOSURE: a driver still in auto accepts the write
            # and then keeps metering on its own, which looks like the setting doing nothing.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_MANUAL_AUTO_VALUE[mode])
            accepted = bool(self.cap.set(cv2.CAP_PROP_EXPOSURE, value))
            if accepted:
                logger.info(f"[Webcam] Exposure set to {exposure_us} us ({value} in {mode} units)")
            else:
                logger.warning(f"[Webcam] Camera rejected exposure {exposure_us} us ({value} in {mode} units)")
            return accepted
        except Exception as e:
            logger.error(f"[Webcam] Failed to set exposure: {e}")
            return False

    def _apply_brightness(self, gain_value):
        """Push the operator's brightness to the driver. Returns whether it was accepted."""
        if self.cap is None or gain_value is None:
            return False
        out_min, out_max = self.brightness_range
        value = brightness_to_backend_value(gain_value, out_min, out_max)
        if value is None:
            logger.warning(f"[Webcam] Ignoring invalid brightness {gain_value!r}")
            return False
        # A webcam splits what the Daheng calls gain across two controls and usually
        # exposes only one. Try the one the page is named after first, then the other,
        # and remember which took so later edits do not probe again.
        candidates = [('brightness', cv2.CAP_PROP_BRIGHTNESS), ('gain', cv2.CAP_PROP_GAIN)]
        if self._brightness_prop is not None:
            candidates = [c for c in candidates if c[0] == self._brightness_prop]
        for name, prop in candidates:
            try:
                if self.cap.set(prop, value):
                    self._brightness_prop = name
                    logger.info(f"[Webcam] Brightness set to {gain_value} (CAP_PROP_{name.upper()}={value:.1f})")
                    return True
            except Exception as e:
                logger.error(f"[Webcam] Failed to set {name}: {e}")
        logger.warning(f"[Webcam] Camera rejected brightness {gain_value}; it may have no manual control.")
        return False

    def _apply_autofocus_v4l2_ctl(self, enabled):
        """Fallback for a driver OpenCV cannot drive: set the V4L2 control with v4l2-ctl.

        The same control the C920 guides use (`v4l2-ctl --set-ctrl=focus_auto=0`). Its name
        differs across kernels, so each known name is tried until one is accepted.
        """
        if not _IS_LINUX or shutil.which('v4l2-ctl') is None:
            return False
        for control in ('focus_automatic_continuous', 'focus_auto'):
            try:
                result = subprocess.run(
                    ['v4l2-ctl', '-d', f'/dev/video{self.cam_index}',
                     f'--set-ctrl={control}={1 if enabled else 0}'],
                    capture_output=True, timeout=5,
                )
            except Exception as e:
                logger.error(f"[Webcam] v4l2-ctl failed for {control}: {e}")
                return False
            if result.returncode == 0:
                logger.info(f"[Webcam] Autofocus {'on' if enabled else 'off'} via v4l2-ctl ({control})")
                return True
        return False

    def _apply_autofocus(self, enabled):
        """Turn the camera's continuous autofocus on or off. Returns whether it was accepted."""
        if self.cap is None or not isinstance(enabled, bool):
            return False
        try:
            if self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if enabled else 0.0):
                logger.info(f"[Webcam] Autofocus {'on' if enabled else 'off'}")
                return True
        except Exception as e:
            logger.error(f"[Webcam] Failed to set autofocus: {e}")
        if self._apply_autofocus_v4l2_ctl(enabled):
            return True
        logger.warning(f"[Webcam] Camera rejected autofocus={enabled}; it may have no autofocus control.")
        return False

    def set_autofocus(self, enabled):
        """Turn autofocus on or off. Off freezes focus where it last settled."""
        if self.cap is None:
            logger.warning("[Webcam] Camera is not initialized. Please start the camera first.")
            return
        if self._apply_autofocus(enabled):
            self.autofocus = enabled

    def set_exposure(self, exposure_time):
        """Set the exposure time in microseconds -- the unit calibration.json stores."""
        if self.cap is None:
            logger.warning("[Webcam] Camera is not initialized. Please start the camera first.")
            return
        if self._apply_exposure(exposure_time):
            self.exposure = exposure_time

    def set_gain(self, gain_value):
        """Set the operator's brightness (5-30), mapped onto the webcam's own scale."""
        if self.cap is None:
            logger.warning("[Webcam] Camera is not initialized. Please start the camera first.")
            return
        if self._apply_brightness(gain_value):
            self.gain = gain_value

    def _check_and_update_calibration(self):
        """Apply `exposure`/`gain` edits from calibration.json to the live camera.

        The same once-per-second mtime watch the Daheng backend uses: the camera exposure
        page and the per-style camera profiles both take effect by writing that file.
        """
        current_time = time.time()
        if current_time - self.last_config_check < self.config_check_interval:
            return
        self.last_config_check = current_time

        if not os.path.exists(self.calibration_file):
            return

        try:
            mtime = os.path.getmtime(self.calibration_file)
            if self.last_config_mtime is not None and mtime == self.last_config_mtime:
                return

            with open(self.calibration_file, 'r') as f:
                calibration_data = json.load(f)

            new_exposure = calibration_data.get('exposure', None)
            new_gain = calibration_data.get('gain', None)

            if new_exposure is not None and new_exposure != self.exposure:
                self.exposure = new_exposure
                self._apply_exposure(new_exposure)

            if new_gain is not None and new_gain != self.gain:
                self.gain = new_gain
                self._apply_brightness(new_gain)

            new_autofocus = calibration_data.get('autofocus', None)
            if isinstance(new_autofocus, bool) and new_autofocus != self.autofocus:
                self.autofocus = new_autofocus
                self._apply_autofocus(new_autofocus)

            self.last_config_mtime = mtime
        except Exception as e:
            logger.error(f"[Webcam] Live update error: {e}")

    def start(self):
        """Open the webcam and apply the stored exposure/brightness."""
        if self.cap is not None and self.is_running:
            logger.warning("[Webcam] Camera is already running.")
            return True

        if self.debug_mode:
            logger.info("[Webcam] Starting in DEBUG MODE - using static image")
            if self.debug_image_path and os.path.exists(self.debug_image_path):
                self.debug_image = cv2.imread(self.debug_image_path)
                if self.debug_image is not None:
                    self.latest_frame = self.debug_image.copy()
                    self.is_running = True
                    logger.info(f"[Webcam] Debug image loaded from: {self.debug_image_path}")
                    return True
                logger.error(f"[Webcam] Failed to load debug image from: {self.debug_image_path}")
                return False
            logger.warning(f"[Webcam] Debug image not found at: {self.debug_image_path}")
            return False

        try:
            self.cap = cv2.VideoCapture(self.cam_index)
            self.is_running = bool(self.cap.isOpened())
        except Exception as e:
            logger.exception(f"[Webcam] Error opening OpenCV webcam at index {self.cam_index}: {e}")
            self.cap = None
            self.is_running = False
            return False

        if not self.is_running:
            logger.error(f"[Webcam] Failed to open OpenCV webcam at index {self.cam_index}")
            self.cap = None
            return False

        self.state = CAMERA_STATE_OK
        logger.info(f"[Webcam] Opened OpenCV webcam at index {self.cam_index} "
              f"(backend={self._backend_name() or 'unknown'}, exposure units={self._exposure_scale()})")
        self._apply_exposure(self.exposure)
        self._apply_brightness(self.gain)
        # Only once the operator has chosen: an unset value leaves the driver's own focus alone.
        if self.autofocus is not None:
            self._apply_autofocus(self.autofocus)
        return True

    def get_latest_image(self, copy: bool = True):
        """Grab the current frame, applying any pending calibration edits first."""
        if not self.is_running:
            return None

        # In debug mode, reload the debug image on each call (to support updating the file)
        if self.debug_mode:
            if self.debug_image_path and os.path.exists(self.debug_image_path):
                try:
                    fresh_image = cv2.imread(self.debug_image_path)
                    if fresh_image is not None:
                        self.debug_image = fresh_image
                        self.latest_frame = fresh_image.copy()
                except Exception as e:
                    logger.error(f"[Webcam] Error reloading debug image: {e}")
            if self.latest_frame is not None:
                return self.latest_frame.copy() if copy else self.latest_frame
            return None

        # Brightness and exposure edits reach the camera here, so the live preview shows
        # them while the operator drags the slider.
        self._check_and_update_calibration()

        if self.cap is None:
            return None
        try:
            ret, frame = self.cap.read()
        except Exception as e:
            logger.error(f"[Webcam] Error reading frame: {e}")
            return None
        if not ret or frame is None:
            return None
        self.latest_frame = frame
        self._mark_frame()
        return frame.copy() if copy else frame

    def stop(self):
        """Release the webcam."""
        self.is_running = False

        if self.debug_mode:
            self.debug_image = None
            self.latest_frame = None
            logger.info("[Webcam] Debug mode stopped.")
            return

        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("[Webcam] Released OpenCV webcam.")
            except Exception as e:
                logger.error(f"[Webcam] Error releasing webcam: {e}")
        self.cap = None
        self.latest_frame = None


if __name__ == '__main__':

    # Create an instance of WebcamCapture
    camera_capture = WebcamCapture(capture_interval=0.1)

    # Start the camera stream
    if camera_capture.start():
        frame_count = 0
        start_time = time.time()
        fps = 0
        fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
        
        try:
            while True:
                # Get the latest frame
                frame = camera_capture.get_latest_image()
                
                if frame is not None:
                    frame_count += 1
                    
                    # Resize frame to 640x480 using faster interpolation
                    display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                    
                    # Calculate FPS
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time >= fps_update_interval:
                        fps = frame_count / elapsed_time
                        frame_count = 0
                        start_time = current_time
                    
                    # Add FPS text to frame
                    cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display the live image
                    cv2.imshow('Live Frame', display_frame)
                    
                    # Handle key presses (q to quit) - cv2.waitKey(1) already introduces minimal delay
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            pass
        finally:
            camera_capture.stop()
            cv2.destroyAllWindows()
    else:
        logger.error("Failed to start camera stream.")