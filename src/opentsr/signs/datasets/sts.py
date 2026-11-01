"""Swedish Traffic Signs (STS, Larsson and Felsberg, SCIA 2011) sequence loader.

Source: Linkoeping University CVL, Set1 and Set2, each with ``annotations.txt`` and ``Part0``
(the annotated frames, about 2,000 per set at 1280x960) plus the unannotated Part1 to Part4.
The authors re-uploaded the whole dataset to Zenodo (record 19666339, 2026-04-20,
``traffic-signs_dataset.zip``, 15.8 GB) under CC BY 4.0; the CVL page still carries the older
"no commercial use without approval" sentence, see ``docs/datasets.md`` 2.5 and
``docs/third_party.md``.

Annotation format (one line per annotated frame, verified on the files on 2026-11-01)::

    <timestamp>Image<nnnnnn>.jpg:<sign>;<sign>;...
    sign  = STATUS, x_max, y_max, x_min, y_min, TYPE, NAME
    STATUS in {VISIBLE, BLURRED, OCCLUDED, SIDE_ROAD}; a frame with no sign has nothing after
    the colon; "MISC_SIGNS" marks a frame that contains signs outside the label set.

TYPE is one of PROHIBITORY, MANDATORY, WARNING, INFORMATION, OTHER, UNREADABLE and NAME one of
the 20 names listed in ``configs/taxonomy/map_sts.yaml``. ``SIDE_ROAD`` is the dataset's
relevance attribute (the sign stands on a side road, not the travelled one) and becomes
``GTBox.relevant = False``; the other statuses are relevant.

Frame numbering: the file name is ``<unix seconds of the recording burst>Image<index>``; every
fifth raw frame of a burst is annotated. The raw frame rate is not documented, so the time of
a frame is ``timestamp + (index - 1) * frame_period_s`` with a default period of 0.4 s (about
2.5 fps, consistent with bursts of 132 frames over 58 s in Set2). Sequences are cut where two
consecutive annotated frames are more than ``gap_s`` apart. Exact timing can be recovered later
from the frame counts of Part1 to Part4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from opentsr.signs.events import GTBox

STATUSES = ("VISIBLE", "BLURRED", "OCCLUDED", "SIDE_ROAD")
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 960
_NAME_RE = re.compile(r"^(\d+)Image(\d+)\.jpg$")


@dataclass
class STSFrame:
    file: str  # "1277381619Image000001.jpg"
    burst: int  # unix seconds prefix
    number: int  # raw frame index within the burst
    boxes: list[GTBox] = field(default_factory=list)
    misc_signs: bool = False
    index: int = 0  # position among the annotated frames of the set (set by load_annotations)
    time_s: float = 0.0  # approximate time, see module docstring
    sequence_id: str = ""


def parse_line(line: str) -> STSFrame:
    name, _, rest = line.strip().partition(":")
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected STS file name {name!r}")
    fr = STSFrame(file=name, burst=int(m.group(1)), number=int(m.group(2)))
    for chunk in rest.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == "MISC_SIGNS":
            fr.misc_signs = True
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 7:
            raise ValueError(f"unexpected STS sign record {chunk!r} in {name}")
        status, xa, ya, xb, yb, sign_type, sign_name = parts
        # Set2 contains one coordinate typed as "478.4376l0" (letter l for digit 1); fix such typos.
        xa, ya, xb, yb = (v.replace("l", "1") for v in (xa, ya, xb, yb))
        if status not in STATUSES:
            raise ValueError(f"unexpected STS status {status!r} in {name}")
        x1, x2 = sorted((float(xa), float(xb)))
        y1, y2 = sorted((float(ya), float(yb)))
        fr.boxes.append(GTBox(frame_index=0, box=(x1, y1, x2, y2), fine=sign_name, coarse=sign_type,
                              relevant=(status != "SIDE_ROAD"), status=status))
    return fr


def load_annotations(path: str | Path, frame_period_s: float = 0.4, gap_s: float = 10.0,
                     set_name: str | None = None) -> list[STSFrame]:
    """Parse ``annotations.txt``; frames come back in time order with ``index``, ``time_s`` and
    ``sequence_id`` (``<set>/<first burst of the sequence>``) filled in."""
    set_name = set_name or Path(path).resolve().parent.name
    frames = [parse_line(line) for line in Path(path).read_text().splitlines() if line.strip()]
    frames.sort(key=lambda f: (f.burst, f.number))
    prev_t, seq = None, ""
    for i, f in enumerate(frames):
        f.index = i
        f.time_s = f.burst + (f.number - 1) * frame_period_s
        if prev_t is None or f.time_s - prev_t > gap_s:
            seq = f"{set_name}/{f.burst}"
        f.sequence_id = seq
        for b in f.boxes:
            b.frame_index = i
        prev_t = f.time_s
    return frames


def sequences(frames: list[STSFrame]) -> dict[str, list[STSFrame]]:
    out: dict[str, list[STSFrame]] = {}
    for f in frames:
        out.setdefault(f.sequence_id, []).append(f)
    return out
