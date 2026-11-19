"""Separate the garment from the table with rule-based OpenCV filters, plus a ring of table around it.

Auto exposure meters on two regions: the garment (it must not clip) and a ring of table just
outside it (the garment edge must stay visible against it). A segmentation model is deliberately
not used: this runs inside the exposure loop on frames that may be badly exposed, and must not
depend on which garment models are loaded.

The pipeline works on a copy scaled to ~1000 px on the long side:

1. **Table model.** A 2nd-order polynomial surface is fitted to a band along the frame border,
   which is table in a normal scan (the garment lies in the middle). A single table brightness
   is not enough: the desk lighting has hotspots and darker corners, and a flat threshold cuts
   into the garment where the table is darkest.
2. **Cue A, brightness.** Pixels that differ from the table surface by more than a multiple of
   the table's own noise. This alone separates any garment darker (or brighter) than the table.
3. **Cue B, texture** and **cue C, closed edges.** A white garment on a white table can match
   it in brightness; what remains is its weave, seams and outline.
4. **Cascade.** Cue A alone is tried first and kept when the result is plausible and has real
   contrast against its ring. When it fails, cue A is tried again with a two-class (Otsu) split
   of the whole frame in place of the band model (needed when the band is not just table, and
   when uneven light joins the garment to a darker part of the table). Only then are B and C added. All three together from the
   start bloated the mask: the texture and edge cues also fire on creases in a table cloth, which
   spread a trouser mask across the table on a test frame where cue A alone was exact.
5. **Pick the garment.** The largest object, preferring one that does not touch the frame
   border (a cable or the room beyond a loose crop does), plus large pieces lying close to it
   (a print or a clipped highlight can cut a garment in two), with holes filled (a white
   pocket lining or print inside a dark garment is still garment).
6. **Ring.** ``dilate(garment, gap + width) - dilate(garment, gap)``. The gap keeps the
   garment's soft edge and its shadow out of the table statistics.

Holes are filled from the outer contours, never by flooding from a corner pixel: a tight crop
puts the garment in the corner, and a flood from there turned the whole table into "garment".

Read-only and free of app.py, the camera and the files on disk.
"""

from dataclasses import dataclass, field

import cv2 # type: ignore
import numpy as np # type: ignore

import logging
get_logger = logging.getLogger

logger = get_logger(__name__)

WORK_LONG_SIDE = 1000           # px; everything below is tuned at this size

BORDER_BAND = 0.06              # fraction of each side used to learn the table
MAX_TABLE_NOISE = 15.0          # band noise above this: the band is not just table, try the two-class split
MIN_RING_ON_TABLE = 0.80        # two-class split: at least this much of the ring must be table class
# Cue A thresholds depend on whether the band model can be trusted. With a clean band (noise <= MAX_TABLE_NOISE)
# the lower pair catches a garment only 8-11 grey levels off the table, and recovered whole white briefs a
# threshold of 12 left as fragments. With a noisy band (a raw frame whose band crosses the room) the lower
# pair took an overexposed patch of white sheet as the garment, scored contrast 130-170 against the dark room
# in its ring and passed every check, so the stricter pair stays there.
K_INTENSITY = 3.0               # cue A, trusted band: multiples of the table's noise
MIN_INTENSITY_DIFF = 8          # cue A, trusted band: never below this many grey levels (a clipped table has no noise)
K_INTENSITY_NOISY = 5.0         # cue A, band not just table
MIN_INTENSITY_DIFF_NOISY = 12
MAX_GARMENT_CLIPPED = 0.50      # band not trusted: a 'garment' mostly at 255 is an overexposed patch of table
K_TEXTURE = 3.0                 # cue B: multiples of the table's texture
TEXTURE_SPREAD_PX = 12          # cue B: how far the texture averaging spreads past an edge, eroded back
EDGE_GROW_PX = 3                # cue C: edges are grown this much so a broken outline still closes

MERGE_DIST_FRAC = 0.02          # merge pieces within this fraction of the long side
DETACH_FRAC = 0.015             # two-class path: bridges narrower than twice this are cut before picking
MIN_PIECE_FRAC = 0.002          # ignore pieces smaller than this fraction of the frame
MIN_CANDIDATE_FRAC = 0.015      # select_garment: an object smaller than this is never the garment (tags, pens, mice)
MAX_PICK_SIDE_CONTACT = 0.15    # select_garment: an object along more of one frame side than this is the room
MIN_AREA_FRAC = 0.04            # a garment covers at least this much of the frame...
MAX_AREA_FRAC = 0.90            # ...and at most this much (the matcher uses the same 4-90 %)
MIN_CONTRAST = 6                # grey levels between garment and ring for a result to count as separated (white garments reach 6-10)
MAX_BORDER_CONTACT = 0.25       # a garment covering more of the frame's edge than this has run into the table
MAX_SIDE_CONTACT = 0.50         # ...or more of any one side than this has joined something lying along it

# Ring geometry. In mm when the caller knows the scale (px_per_mm), else as a fraction of the long side.
RING_GAP_MM = 5.0
RING_WIDTH_MM = 30.0
RING_GAP_FRAC = 0.006
RING_WIDTH_FRAC = 0.03

CLIP_LEVEL = 250                # grey level counted as clipped in the statistics


@dataclass
class SeparationResult:
    """Garment and ring masks at the input image's resolution (uint8, 0/255), or None when nothing was found.

    ``valid`` is False when the masks should not steer the exposure: no garment, an implausible
    area, or a garment that cannot be told from the table (``problems`` says which). Low contrast
    is reported even when the outline itself is right, because for exposure that is the finding.
    """
    garment_mask: np.ndarray = None
    ring_mask: np.ndarray = None
    valid: bool = False
    cue: str = ''
    problems: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _ellipse(radius):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _border_band(h, w, frac):
    band = np.zeros((h, w), bool)
    bh, bw = max(2, int(h * frac)), max(2, int(w * frac))
    band[:bh, :] = band[-bh:, :] = True
    band[:, :bw] = band[:, -bw:] = True
    return band


def fit_table_surface(gray, sample_mask):
    """Robust 2nd-order polynomial fit of the table brightness to the sampled pixels.

    Returns (surface, sigma): the fitted brightness everywhere and the table's noise (robust std).
    Samples off the fit by more than 3 sigma are dropped and the fit repeated, so a garment that
    reaches into the border band does not drag the table model towards it.

    The first fit already starts without the samples far from the band's median. A least-squares
    fit over everything bends towards a garment that covers a quarter of the band (a tight crop
    with the garment in a corner); the noise then came out at ~38 grey levels and cue A found
    nothing at all. The median is still table as long as the garment covers less than half the band.

    Starting from the band's histogram peak instead was tried for raw (uncropped) frames, whose band
    also crosses the room: it modelled the table better, but then cue A marked the room as well and
    it joined white garments into one blob, losing outlines the texture fallback had found. The
    separator expects the preprocessed frame, whose border is table.
    """
    h, w = gray.shape
    ys, xs = np.nonzero(sample_mask)
    step = max(1, len(xs) // 20000)
    xs, ys = xs[::step], ys[::step]
    values = gray[ys, xs].astype(np.float64)

    def design(x, y):
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)

    a = design(xs / w - 0.5, ys / h - 0.5)
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    keep = np.abs(values - median) < max(4 * mad, MIN_INTENSITY_DIFF)
    sigma = 1.0
    coef = np.zeros(6)
    for _ in range(3):
        if np.count_nonzero(keep) < 6:
            break
        coef, *_ = np.linalg.lstsq(a[keep], values[keep], rcond=None)
        residual = values - a @ coef
        kept = residual[keep]
        sigma = 1.4826 * float(np.median(np.abs(kept - np.median(kept)))) + 1e-6
        keep = np.abs(residual) < 3 * sigma
    gx, gy = np.meshgrid(np.arange(w) / w - 0.5, np.arange(h) / h - 0.5)
    surface = (design(gx.ravel(), gy.ravel()) @ coef).reshape(h, w)
    return surface, max(sigma, 1.0)


def two_class_objects(gray, band):
    """Objects when the border band cannot be trusted as table: split the whole frame in two by Otsu.

    The table is the class that covers more of the border band, everything else is an object. On
    a raw frame of a black t-shirt on a light desk, the band crossed the dark room strip, a laptop,
    a bottle, an overexposed patch and the garment's own sleeves, and only ~25% of it was table.
    The band model then put the table at 122 grey levels (table ~180, shirt ~34) with a noise of
    63, so cue A's threshold was above 255 and it found nothing, while the frame's histogram still
    had two clear peaks. The overexposed patch lands in the table's class here, where a fitted
    surface would call it an object.

    The split can also be wrong: on a white t-shirt with a black print on a white sheet, it put
    the print and the room in one class and a patch of sheet in the other, and that patch passed
    as the garment. A real garment's ring lies on the table class, so ``_evaluate`` requires it.

    Returns (objects, table_class) as uint8 masks.
    """
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = gray > threshold
    table_is_bright = np.mean(bright[band]) >= 0.5
    objects = ~bright if table_is_bright else bright
    return strip_edge_bands(objects.astype(np.uint8) * 255), (~objects).astype(np.uint8) * 255


def _strip_top_band(objects):
    """Remove an object lying along the top edge: per column, cut its typical thickness there."""
    h, w = objects.shape
    # Run length from the top edge in every column (0 where the column starts on table).
    not_filled = objects == 0
    runs = np.where(not_filled.any(axis=0), not_filled.argmax(axis=0), h)
    # The band's thickness is a straight line across the frame: the far edge of the desk is straight,
    # seen at an angle. It is fitted robustly to the columns' runs. Where the garment joins the band
    # (a collar against the room strip) the run is much longer; those columns are outliers to the
    # line, and only the band's part of them is cut. Two windowed estimates failed first: a median
    # over 240 px read a collar touching the strip for 200 px as the strip and cut a column out of
    # the shirt, and a low percentile over a wide window ran short where the strip thickens and
    # left a layer joining the shirt to the room.
    x = np.arange(w, dtype=np.float64)
    covered = runs > 0
    keep = covered & (runs <= 2 * np.median(runs[covered]))
    coef = np.array([0.0, float(np.median(runs[covered]))])
    for _ in range(4):
        if np.count_nonzero(keep) < 2:
            break
        coef = np.polyfit(x[keep], runs[keep].astype(np.float64), 1)
        residual = runs - np.polyval(coef, x)
        spread = 1.4826 * float(np.median(np.abs(residual[keep]))) + 1.0
        keep = covered & (np.abs(residual) < 3 * spread)
    band = np.clip(np.polyval(coef, x) + 3 * spread, 0, h)
    cut = np.minimum(runs, np.ceil(band).astype(int) + 3)
    out = objects.copy()
    rows = np.arange(h)[:, None]
    out[(rows < cut[None, :]) & (runs[None, :] > 0)] = 0
    return out


def strip_edge_bands(objects):
    """Cut objects that lie along most of one side of the frame (the room strip beyond a desk edge).

    Only the two-class split needs this: it marks everything dark, and on a raw frame a dark strip
    of room along the top joined a black t-shirt at its collar. Each side covered for more than
    MAX_SIDE_CONTACT is trimmed; the frame is rotated so one routine handles all four.
    """
    out = objects
    for k in range(4):                                  # top, then each side in turn rotated to the top
        rotated = np.rot90(out, k)
        if np.mean(rotated[0] > 0) > MAX_SIDE_CONTACT:
            rotated = _strip_top_band(np.ascontiguousarray(rotated))
            out = np.ascontiguousarray(np.rot90(rotated, -k))
    return out


def texture_energy(gray):
    """Local average of fine detail: weave and seams score high, a smooth table low."""
    g = gray.astype(np.float32)
    detail = g - cv2.GaussianBlur(g, (0, 0), 3)        # drop the lighting, keep the detail
    return cv2.GaussianBlur(np.abs(detail), (0, 0), 6)


def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def closed_edge_regions(gray):
    """Cue C: regions enclosed by an edge loop. Thresholds follow the frame's median brightness."""
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(5, 0.066 * med)), int(max(15, 0.2 * med)))
    edges = cv2.dilate(edges, _ellipse(EDGE_GROW_PX))
    return cv2.erode(fill_holes(edges), _ellipse(EDGE_GROW_PX))


def select_garment(objects, long_side, edge_ok=False):
    """Largest object (preferring one off the frame border) plus large pieces close to it, holes filled.

    The order of preference is: objects off the border of at least MIN_CANDIDATE_FRAC, then smaller
    objects off the border, then objects touching it. The middle tier is for a white shirt on a white
    sheet whose outline came out only as small pieces (collar line, print): once the size floor removed
    them, a mouse whose cable ran off the frame edge was the only object left and was picked. A piece
    of the shirt is the better wrong answer, and still fails the area check in ``_evaluate``.

    ``edge_ok``: the frame is cropped to the table, so nothing along its sides can be the room. Objects
    along the sides are kept and compete by size with the rest: a long-sleeved shirt larger than the
    crop ran off three sides and was dropped as the room, leaving its seam lines as the garment.

    Returns (mask or None, note).
    """
    h, w = objects.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(objects, 8)
    min_area = MIN_PIECE_FRAC * h * w
    candidates = []
    small_inner = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if area < MIN_CANDIDATE_FRAC * h * w:
            if not touches:
                small_inner.append((area, i, False))
            continue
        if touches and not edge_ok:
            # A tight crop clips a trouser waist or hem along a short stretch of one side (11% on a
            # test frame); the room or the desk's far edge runs along most of it. Without this, once
            # the size floor above removed a white shirt's print fragments, the dark room strip along
            # the top of a raw frame was the only object left and passed as the garment.
            comp = labels == i
            side = max(np.mean(comp[0]), np.mean(comp[-1]), np.mean(comp[:, 0]), np.mean(comp[:, -1]))
            if side > MAX_PICK_SIDE_CONTACT:
                continue
        candidates.append((area, i, touches))
    if not candidates and not small_inner:
        return None, 'no object found on the table'

    inner = (candidates if edge_ok else [c for c in candidates if not c[2]]) or small_inner
    _, pick, _ = max(inner or candidates)
    note = '' if inner else 'garment touches the frame border'
    garment = (labels == pick).astype(np.uint8) * 255

    near = cv2.dilate(garment, _ellipse(max(3, MERGE_DIST_FRAC * long_side)))
    for _, i, touches in candidates:
        if i != pick and (edge_ok or not touches) and np.any(near[labels == i]):
            garment[labels == i] = 255
    # A small close only: a large one bridged the gap under a sleeve and filled the armpits.
    garment = cv2.morphologyEx(garment, cv2.MORPH_CLOSE, _ellipse(3))
    return fill_holes(garment), note


def _ring_geometry(long_side, px_per_mm):
    if px_per_mm and px_per_mm > 0:
        return max(2, int(round(RING_GAP_MM * px_per_mm))), max(4, int(round(RING_WIDTH_MM * px_per_mm)))
    return max(2, int(RING_GAP_FRAC * long_side)), max(4, int(RING_WIDTH_FRAC * long_side))


def _evaluate(objects, gray, long_side, px_per_mm, detach=False, table_class=None, trusted=True, edge_ok=False):
    """Turn one cue's object mask into (garment, ring, problems, stats) at working size.

    ``detach`` picks the garment on an eroded copy and grows it back inside the objects, so a
    narrow bridge to something on the frame edge does not make the two one object. The two-class
    split needs it: it marks everything dark, and a black t-shirt's shoulder touched the dark
    room strip along the top of a raw frame, which put the strip in the garment (63% of the frame,
    and still valid). Off by default, because it would cut a thong's thin straps.

    ``table_class`` (two-class path) is the mask the ring must mostly lie on. ``trusted`` is False when
    the band model is not just table; a mostly clipped garment is then rejected. Not when it is trusted:
    a white garment overexposed on a dark desk is real, and auto exposure needs its mask to fix it.

    ``edge_ok`` (frame cropped to the table): a garment along the frame sides is not a problem.
    """
    objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, _ellipse(2))
    objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, _ellipse(2), iterations=2)

    if detach:
        radius = max(2, int(DETACH_FRAC * long_side))
        core_pick, note = select_garment(cv2.erode(objects, _ellipse(radius)), long_side, edge_ok)
        garment = None
        if core_pick is not None:
            garment = fill_holes(cv2.bitwise_and(cv2.dilate(core_pick, _ellipse(radius)), objects))
    else:
        garment, note = select_garment(objects, long_side, edge_ok)
    if garment is None:
        return None, None, [note], {}

    problems = []
    area_frac = np.count_nonzero(garment) / garment.size
    stats = {'area_pct': 100.0 * area_frac}
    if not MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC:
        problems.append('garment area %.0f%% is outside %d-%d%%' % (100 * area_frac, 100 * MIN_AREA_FRAC,
                                                                  100 * MAX_AREA_FRAC))
    # A tight crop may clip a trouser waist or a hem, a short stretch of the frame edge. A mask
    # along much of the edge has leaked into the table instead: on a raw frame the texture
    # fallback ran along two whole sides and still passed the contrast check at 16 grey levels.
    # One side mostly covered is the same leak along a single edge: a black t-shirt joined to the
    # dark room strip across the top of a raw frame covered 68% of the top side but only 22% of the
    # whole edge. A tight crop clipped a trouser hem along 11% of one side.
    sides = (garment[0, :], garment[-1, :], garment[:, 0], garment[:, -1])
    border_contact = float(np.mean(np.concatenate(sides) > 0))
    side_contact = max(float(np.mean(s > 0)) for s in sides)
    stats['border_contact_pct'] = 100.0 * border_contact
    stats['side_contact_pct'] = 100.0 * side_contact
    if edge_ok:
        pass
    elif border_contact > MAX_BORDER_CONTACT:
        problems.append('garment covers %.0f%% of the frame edge' % (100 * border_contact))
    elif side_contact > MAX_SIDE_CONTACT:
        problems.append('garment covers %.0f%% of one side of the frame' % (100 * side_contact))

    gap, width = _ring_geometry(long_side, px_per_mm)
    ring = cv2.subtract(cv2.dilate(garment, _ellipse(gap + width)), cv2.dilate(garment, _ellipse(gap)))
    core = cv2.erode(garment, _ellipse(gap))           # keep the garment's own soft edge out of its statistics

    contrast = stats['contrast'] = measure_contrast(gray, core, ring)
    if table_class is not None and np.any(ring > 0):
        ring_on_table = float(np.mean(table_class[ring > 0] > 0))
        stats['ring_on_table_pct'] = 100.0 * ring_on_table
        if ring_on_table < MIN_RING_ON_TABLE:
            problems.append('only %.0f%% of the ring lies on the table' % (100 * ring_on_table))
    if not trusted and np.any(core > 0):
        clipped = float(np.mean(gray[core > 0] >= CLIP_LEVEL))
        if clipped > MAX_GARMENT_CLIPPED:
            problems.append('garment is %.0f%% clipped: an overexposed patch, not a garment' % (100 * clipped))
    if contrast < MIN_CONTRAST:
        problems.append('garment and table differ by only %.0f grey levels' % contrast)
    if note:
        stats['note'] = note                            # not a problem: a tight crop clips a trouser waist
    return garment, ring, problems, stats


def measure_contrast(image, garment_mask, ring_mask):
    """Grey-level difference between the garment and the table ring around it (0-255).

    Medians, so a print on the garment or a shadow on the table does not drag it. The masks must
    match the image's size; the exposure loop can separate once and then call this on every new
    frame with the same masks. 0.0 when either region is empty.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    garment_px, ring_px = gray[garment_mask > 0], gray[ring_mask > 0]
    if not garment_px.size or not ring_px.size:
        return 0.0
    return abs(float(np.median(ring_px)) - float(np.median(garment_px)))


def separate_garment(image, px_per_mm=None, edge_ok=False):
    """Find the garment on the table and the ring of table around it.

    ``image`` is a BGR or grayscale frame, normally the preprocessed (cropped, undistorted) one so
    that the border band is table and not the room. ``px_per_mm`` is the frame's scale at the
    desk; when given, the ring gap and width are in millimetres, otherwise a fraction of the frame.
    ``edge_ok``: the frame is cropped to the table only, so a garment may run off its sides (a
    garment larger than the crop); off, anything along most of a side is taken for the room.
    """
    if image is None or getattr(image, 'size', 0) == 0:
        return SeparationResult(problems=['no image'])

    gray_full = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_h, full_w = gray_full.shape
    scale = min(1.0, WORK_LONG_SIDE / max(full_h, full_w))
    gray = cv2.resize(gray_full, (max(1, int(full_w * scale)), max(1, int(full_h * scale))), interpolation=cv2.INTER_AREA)
    work_px_per_mm = px_per_mm * scale if px_per_mm else None
    h, w = gray.shape
    long_side = max(h, w)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    band = _border_band(h, w, BORDER_BAND)

    surface, sigma = fit_table_surface(smooth, band)
    trusted = sigma <= MAX_TABLE_NOISE
    k, floor = (K_INTENSITY, MIN_INTENSITY_DIFF) if trusted else (K_INTENSITY_NOISY, MIN_INTENSITY_DIFF_NOISY)
    cue_a = (np.abs(smooth.astype(np.float64) - surface) > max(k * sigma, floor))
    cue_a = cue_a.astype(np.uint8) * 255

    texture = texture_energy(smooth)
    table_texture = float(np.median(texture[band])) + 0.5

    def cues_abc():
        cue_b = (texture > K_TEXTURE * table_texture).astype(np.uint8) * 255
        cue_b = cv2.erode(cue_b, _ellipse(TEXTURE_SPREAD_PX))
        cue_c = closed_edge_regions(smooth)
        return 'A+B+C', cv2.bitwise_or(cv2.bitwise_or(cue_a, cue_b), cue_c), None

    # The two-class split is tried whenever cue A fails, not only when the band is noisy: a dark brief
    # on a clean band (noise 2.5) joined a darker strip of table along the right side under cue A's
    # low threshold and ran off the frame edge, while the split found it alone. Each attempt is built
    # only when the one before it failed (the auto exposure loop separates every round, and B and C
    # are the slow ones).
    attempts = (lambda: ('A', cue_a, None), lambda: ('A-otsu', *two_class_objects(smooth, band)), cues_abc)
    for attempt in attempts:
        cue, objects, table_class = attempt()
        garment, ring, problems, stats = _evaluate(objects, gray, long_side, work_px_per_mm,
                                                   detach=table_class is not None, table_class=table_class,
                                                   trusted=trusted, edge_ok=edge_ok)
        if not problems:
            break
    stats.update(table_noise=sigma, table_texture=table_texture)

    if garment is None:
        logger.debug("No garment separated from the table: %s", '; '.join(problems))
        return SeparationResult(valid=False, cue=cue, problems=problems, stats=stats)

    if scale < 1.0:
        garment = cv2.resize(garment, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        ring = cv2.resize(ring, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    logger.debug("Garment separated with cue %s (%s)%s", cue,
                 ', '.join('%s=%.1f' % (k, v) for k, v in stats.items() if isinstance(v, float)),
                 '; problems: ' + '; '.join(problems) if problems else '')
    return SeparationResult(garment_mask=garment, ring_mask=ring, valid=not problems, cue=cue,
                            problems=problems, stats=stats)


def draw_overlay(image, result):
    """The frame with the garment tinted green, the ring blue and the garment outline red (for debugging)."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if result.garment_mask is None:
        return out
    tint = out.copy()
    tint[result.ring_mask > 0] = (255, 160, 0)
    tint[result.garment_mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(tint, 0.4, out, 0.6, 0)
    contours, _ = cv2.findContours(result.garment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), max(2, max(out.shape[:2]) // 500))
    return out
