# CLAUDE.md: auto exposure test bench

Context for a new Claude Code session in this folder. It was written at the end of a long session in the
`scanner-app` repo (`C:\Users\prave\Desktop\Thullex\scanner-app`), which this folder was copied out of so the
feature can be tried on the real camera **without running the scanner app**.

## Working with the user

- The user (Dilshan) writes in Singlish (Sinhala typed in English letters). **Reply in Sinhala script**,
  conversational. Keep code, file names, commands and technical terms in English.
- **Do not write or change code until the user asks for it.** Explanations, tests on images and proposals
  are fine. Ask before larger changes.
- Never add Claude co-author or "Generated with Claude" lines to commits or PRs.
- Run tests on images and save the results under `C:\Users\prave\Desktop\Thullex\img\results\<name>\`.
- Risindu (a colleague) reviews this work; his feedback is pasted into the chat.

## The goal

Auto exposure for the garment scanner (`rms-app`: a Flask app on customer-site Linux PCs with a Daheng
USB camera looking down at a white table; YOLO/RF-DETR models measure garments from the frames).

Exposure must be judged on **the garment** and **a ring of table just around it**, not on the whole frame:
the white table fills most of the picture, so a whole-frame average exposes for the table and washes a
light garment into it.

1. **Done:** separate the garment from the table with a rule-based OpenCV approach (deliberately **no
   segmentation model**: it runs inside the exposure loop on badly exposed frames and must not depend on
   which garment models are loaded), plus a buffer ring around it that represents the table.
2. **Done:** contrast = intensity difference between garment and ring.
3. **Next, not written yet:** the exposure loop itself (see "Next steps").

## Files here

| File | What it is |
|---|---|
| `garment_separation.py` | Copy of `scanner-app/core/camera/auto_exposure/garment_separation.py`. Only change: `core.shared.logger` replaced by `logging`. |
| `capture.py` | Copy of `scanner-app/core/camera/capture.py`, same logger swap. Its `core.shared.paths` imports run only when no `calibration_file` is given; `live_test.py` always gives one. |
| `live_test.py` | Live test: camera -> `PreProcessor` -> `separate_garment` -> contrast, in a window. |
| `calibration.json` | Not here yet: copy it from the device (`~/configuration/calibration.json`). |

**The scanner-app repo is the source of truth.** The two copies drift if either side changes: port a
change made here back into `core/camera/auto_exposure/garment_separation.py` (keep the repo's logger import),
and recopy after changing the repo.

### live_test.py

```bash
sudo systemctl stop rms-app        # on a device: the camera can only be open in one process
python3 live_test.py --calibration calibration.json                  # Daheng camera
python3 live_test.py --camera webcam --calibration calibration.json  # USB webcam
python3 live_test.py --camera image --image frame.jpg                # saved frame, no camera
```

Keys: `+`/`-` exposure x/÷1.25 · `w` sweep · `s` save raw/processed/overlay to `captures/` · `q` quit.
The sweep separates once at the current exposure (must be OK), then sets each of `SWEEP_US`
(2000-40000 us), waits `SETTLE_S` + exposure time, measures contrast and garment clipping on the **same masks**,
and keeps the exposure with the highest contrast whose garment is ≤1% at 255. Nothing is written to
`calibration.json`.

Only the image mode and the sweep logic (with a fake camera) were run; **it has not been run on a real
camera yet**. Daheng frames come out of `capture.py` as RGB arrays (`convert("RGB")`), the app uses them as-is.

## How the separation works (`separate_garment(image, px_per_mm=None)`)

Works on a copy scaled to ~1000 px on the long side (`WORK_LONG_SIDE`, `scale = min(1, 1000/long_side)`;
never upscales); masks are resized back (INTER_NEAREST) to the input size.

1. **Table model** `fit_table_surface`: a robust 2nd-order polynomial (`1, x, y, x², xy, y²`) of the
   table brightness, fitted to `_border_band` (6% of each side, `BORDER_BAND`). The first fit uses only
   samples near the band's median (median/MAD), then 3 refits with 3-sigma rejection. Returns the surface
   and `sigma` (table noise). Lights make the table brighter in the middle, so a flat threshold fails.
2. **Cue A** (brightness): `|pixel - surface| > max(k*sigma, floor)`. Band trusted (`sigma <= 15`):
   k=3.0, floor=8 (Risindu's values). Not trusted: k=5.0, floor=12 (the low pair caused false OKs on raw frames).
3. **Cascade**: A, then **A-otsu** (only if sigma > 15: Otsu two-class split, the class covering more of
   the band is table; `strip_edge_bands` cuts a room strip along a side with a robust line fit; ring must lie
   ≥80% on the table class), then **A+B+C** (B = texture energy, C = Canny closed-edge fill, for white on
   white). The first attempt with no problems wins.
4. **`select_garment`**: connected components. Tiers: inner (not touching the border) ≥1.5% of the frame
   > inner <1.5% (a piece of a white shirt beats a mouse) > border-touching (rejected if it covers >15% of one
   side: the room). Nearby pieces merged, small close (a large one filled the armpits), holes filled from
   contours (never a corner flood fill: it turned the table into garment once).
5. **`_evaluate`** checks: area 4-90%, border contact ≤25% and one side ≤50%, ring on table ≥80% (Otsu
   path), garment ≤50% clipped when the band is not trusted, **contrast ≥ 15** (`MIN_CONTRAST`). Any
   failure goes to `res.problems` and makes `res.valid` False.
6. **Ring** = `dilate(garment, gap+width) - dilate(garment, gap)`; gap 5 mm / width 30 mm with `px_per_mm`,
   else 0.6% / 3% of the long side. The gap keeps the soft edge and shadow out.

**Contrast** (`measure_contrast(image, garment_mask, ring_mask) -> float`):
`|median(ring) - median(garment)|` in grey levels; medians so a print or shadow does not drag it. Inside
`_evaluate` the garment mask is first eroded by the gap (`core`) so its edge is not counted. The user asked
for **only the contrast** to be returned (Weber/Michelson and the other stats were removed on request).
It can be re-measured on new frames with the same masks.

`SeparationResult`: `garment_mask`, `ring_mask` (uint8 0/255 or None), `valid`, `cue`, `problems`,
`stats` (`area_pct`, `border_contact_pct`, `side_contact_pct`, `contrast`, `table_noise`, `table_texture`,
sometimes `ring_on_table_pct`, `note`).

## Results so far (25 images in `C:\Users\prave\Desktop\Thullex\img`)

- **16 OK**, contrast 42-191 (dark and coloured garments). Hand-checked against full-resolution medians:
  equal or within 1-3 grey levels (erosion and working size).
- **6 NOT VALID with the right outline**: white shirt on white table, contrast 0-5. That is the correct
  finding for exposure (garment and table look the same to the camera).
- **3 NOT VALID showing a shirt piece** (a6a34c88, aabd6e67, d1114aaf: 1024 px overexposed copies). On two of
  them the mask is only the collar line and the ring falls on the shirt, so their contrast means nothing;
  they are rejected anyway (area 1%, clipped).
- No false OK on any image. Most tables read 255 (clipped): the devices' exposure is already too high.
- Conclusion: an overexposed or downscaled white-on-white frame has no outline left to recover in software.
  Full-resolution (4024x3036) and lower-exposure frames are needed; a grey desk would help.
- Tried and rejected: `BORDER_BAND` 0.03 (Risindu's idea), a histogram-peak table init, removing cues B/C,
  `EDGE_GROW_PX` 4, a "partial garment" mode.
- Open question: whether to lower `MIN_CONTRAST` from 15.

## scanner-app state

Branch `auto-exposure`, three commits not pushed: `4a43a66` (separation), `a2a913e` (small-piece
preference), `6fb5098` (`measure_contrast` returning a float; CLI prints `.3g`). Working tree clean.
`tests/test_garment_separation.py` has 20 tests. Run the repo's tests with `venv\Scripts\python.exe -m pytest` (system Python lacks pymongo/telegraf);
the fast suite was 1427 passed. CLI: `python -m core.camera.auto_exposure <out_dir> <images...>`.

Camera facts (`capture.py`): `WebcamCapture` (Daheng, gxipy) reads `exposure` (µs, default 20000) and `gain`
(default 10) from `calibration.json`; `set_exposure(us)` / `set_gain(v)` set them live;
`_check_and_update_calibration` re-applies the file's values when its mtime changes (once a second).
`PreProcessor.process_image` does rotate -> crop -> undistort; feed `separate_garment` that frame, since
its border is table. `OpenCVWebcamCapture` converts µs to V4L2 (100 µs units) or DirectShow/MSMF (log2 s).

## Next steps (only when the user asks)

1. Run `live_test.py` on a device and record contrast versus exposure for dark, light and white garments.
2. The exposure loop: contrast alone is not the target. It rises with exposure until the table clips at
   255, then falls for light garments. Aim for the highest contrast with the garment unclipped (≤1% at 255),
   preferably with the table not clipped either. Change exposure before gain (gain adds noise). Drop 2-3 frames
   or wait ~100-200 ms after a change. If the garment is NOT VALID, lower the exposure and separate again.
3. Persist the chosen value by writing `exposure` to `calibration.json` atomically (`.tmp` + `os.replace`).
4. Later in scanner-app: wire it into the app, add docs (`docs/README.md`, `docs/DEVELOPER_GUIDE.md`).
