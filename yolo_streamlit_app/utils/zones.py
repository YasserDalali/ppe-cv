"""
Splits each detected "person" box into 3 vertical bands (top/mid/bottom) and
assigns nearby PPE detections (helmet, vest, boots, ...) to whichever band
they fall into. Doubles as a false-positive filter: a "helmet" landing in
the bottom band of a person is almost certainly a misclassification.

Bands are computed from whatever bbox is actually detected -- not an assumed
full-body proportion -- so partially-visible people (e.g. only the top half
in frame) still get 3 correctly-scaled bands. If the box is too squat to
plausibly be a full body (aspect ratio check), labels fall back to generic
"zone 1/2/3" instead of implying a body region that may not be pictured,
and no item is rejected for its position.

compliance() turns this into a per-person decision: a required item counts
as worn only if it is assigned to that person and lies in its expected band.
"""

PERSON_CLASS_NAMES = {"person"}

# zone 0 = top, 1 = mid, 2 = bottom. Classes not listed here are never flagged.
DEFAULT_EXPECTED_ZONES = {
    "helmet": 0, "hard hat": 0, "hardhat": 0, "head": 0,
    "goggles": 0, "gas mask": 0, "gas_mask": 0, "mask": 0,
    "vest": 1, "safety vest": 1, "hi-vis vest": 1, "gloves": 1,
    "boots": 2, "shoes": 2, "safety boots": 2,
}

FULL_BODY_ASPECT_RATIO = 1.2  # height/width above this -> treat as a full-ish body
DEFAULT_REQUIRED = ("helmet", "vest", "boots")


def _full_body(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return w > 0 and h / w >= FULL_BODY_ASPECT_RATIO


def _zone_labels(bbox):
    if _full_body(bbox):
        return ["head/top", "torso/mid", "legs/bottom"]
    return ["zone 1 (top)", "zone 2 (mid)", "zone 3 (bottom)"]


def _zone_boundaries(bbox):
    y1, y2 = bbox[1], bbox[3]
    third = (y2 - y1) / 3
    return [y1, y1 + third, y1 + 2 * third, y2]


def _center_x(bbox):
    return (bbox[0] + bbox[2]) / 2


def _center_y(bbox):
    return (bbox[1] + bbox[3]) / 2


def _contains_x(person_bbox, item_bbox, slack=0.15):
    px1, _, px2, _ = person_bbox
    pad = (px2 - px1) * slack
    cx = _center_x(item_bbox)
    return (px1 - pad) <= cx <= (px2 + pad)


def classify_zones(detections, expected_zones=None):
    """
    detections: list of Detection objects, already temporal-filter-confirmed.

    Returns (persons, items):
      persons: [{bbox, zone_labels, boundaries}]
      items:   [{detection, person_idx, zone_idx, zone_label, mismatch}]
               person_idx/zone_idx are None if no person box matched.
    """
    expected_zones = expected_zones or DEFAULT_EXPECTED_ZONES

    persons = [
        {"bbox": d.bbox, "zone_labels": _zone_labels(d.bbox), "boundaries": _zone_boundaries(d.bbox),
         "full_body": _full_body(d.bbox)}
        for d in detections if d.cls_name.lower() in PERSON_CLASS_NAMES
    ]

    items = []
    for it in detections:
        if it.cls_name.lower() in PERSON_CLASS_NAMES:
            continue

        item_cy = _center_y(it.bbox)
        best_idx, best_score = None, -1.0
        for idx, p in enumerate(persons):
            if not _contains_x(p["bbox"], it.bbox):
                continue
            py1, py2 = p["bbox"][1], p["bbox"][3]
            if py1 <= item_cy <= py2:
                score = 1.0 / max(py2 - py1, 1)  # prefer tightest-fitting person box
                if score > best_score:
                    best_score, best_idx = score, idx

        zone_idx, zone_label, mismatch = None, None, False
        if best_idx is not None:
            bounds = persons[best_idx]["boundaries"]
            for z in range(3):
                if bounds[z] <= item_cy <= bounds[z + 1]:
                    zone_idx = z
                    zone_label = persons[best_idx]["zone_labels"][z]
                    break
            expected = expected_zones.get(it.cls_name.lower())
            mismatch = (expected is not None and zone_idx is not None and expected != zone_idx
                        and persons[best_idx]["full_body"])

        items.append({
            "detection": it, "person_idx": best_idx,
            "zone_idx": zone_idx, "zone_label": zone_label, "mismatch": mismatch,
        })

    return persons, items



def compliance(persons, items, required=DEFAULT_REQUIRED, expected_zones=None):
    """Per person, {item: True | False | None} for each required item.
    True: a detection of that class is assigned to the person and lies in its
    expected band. False: none does (missing, or e.g. a helmet carried at
    waist height). None: the box is too squat to be a whole body (cropped),
    so a missing item may simply be out of view and is not reported."""
    expected_zones = expected_zones or DEFAULT_EXPECTED_ZONES
    out = []
    for idx, p in enumerate(persons):
        worn = {it["detection"].cls_name.lower() for it in items
                if it["person_idx"] == idx and not it["mismatch"]
                and (not p["full_body"] or it["zone_idx"] == expected_zones.get(it["detection"].cls_name.lower()))}
        out.append({k: True if k in worn else (False if p["full_body"] else None) for k in required})
    return out
