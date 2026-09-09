#!/usr/bin/env python3
"""Generate a SKY130-style GDSII for Virtual Core Chip A (RRAM SNN tile).

Produces:
  public/gds/virtual_core_chip_a.gds
  public/gds/layout.json   (viewer sidecar)
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

OUT_DIR = Path("/workspace/public/gds")
RTL_DIR = Path("/workspace/public/rtl")

# 1 dbu = 1 nm; 1 user unit = 1 µm
DBU = 1000
# OpenROAD equivalent-square of the std-cell area is ~89 µm.
# 33 HD rows × 2.72 µm = 89.76 µm; snap die to 90 µm (hard macro, no pad ring).
DIE = 90.0
CORE_LL = 0.0
CORE_UR = 90.0
ROW_H = 2.72
SITE = 0.46
N_ROWS = 33  # 33 * 2.72 = 89.76 µm

# SKY130 GDS layer / datatype
LAYERS = {
    "nwell": (64, 20),
    "diff": (65, 20),
    "tap": (65, 44),
    "poly": (66, 20),
    "licon": (66, 44),
    "li1": (67, 20),
    "mcon": (67, 44),
    "met1": (68, 20),
    "via": (68, 44),
    "met2": (69, 20),
    "via2": (69, 44),
    "met3": (70, 20),
    "via3": (70, 44),
    "met4": (71, 20),
    "via4": (71, 44),
    "met5": (72, 20),
    "prBoundary": (235, 4),
    "areaid.sc": (81, 4),
    "label": (83, 44),
    "text": (64, 5),
}


# ---------------------------------------------------------------------------
# GDSII writer
# ---------------------------------------------------------------------------

def _gds_real(val: float) -> bytes:
    if val == 0:
        return b"\x00" * 8
    sign = 0x80 if val < 0 else 0x00
    val = abs(val)
    exp = 0
    while val >= 1.0:
        val /= 16.0
        exp += 1
    while val < 0.0625:
        val *= 16.0
        exp -= 1
    mantissa = int(val * (1 << 56) + 0.5)
    if mantissa >= (1 << 56):
        mantissa >>= 4
        exp += 1
    return bytes([((exp + 64) & 0x7F) | sign]) + mantissa.to_bytes(7, "big")


def _rec(tag: int, dtype: int, payload: bytes = b"") -> bytes:
    if len(payload) % 2:
        payload += b"\x00"
    length = 4 + len(payload)
    return struct.pack(">HH", length, (tag << 8) | dtype) + payload


def _i16(vals: Iterable[int]) -> bytes:
    return b"".join(struct.pack(">h", int(v)) for v in vals)


def _i32(vals: Iterable[int]) -> bytes:
    return b"".join(struct.pack(">i", int(v)) for v in vals)


def _ascii(s: str) -> bytes:
    b = s.encode("ascii")
    return b if len(b) % 2 == 0 else b + b"\x00"


class GdsLib:
    def __init__(self, name: str):
        self.name = name
        self.structs: list[GdsStruct] = []

    def add(self, st: "GdsStruct") -> "GdsStruct":
        self.structs.append(st)
        return st

    def write(self, path: Path) -> None:
        buf = bytearray()
        buf += _rec(0x00, 0x02, _i16([5]))  # HEADER v5
        # BGNLIB / BGNSTR timestamps: 12 int16
        ts = _i16([2026, 4, 12, 0, 0, 0, 2026, 4, 12, 0, 0, 0])
        buf += _rec(0x01, 0x02, ts)
        buf += _rec(0x02, 0x06, _ascii(self.name))
        buf += _rec(0x03, 0x05, _gds_real(1e-3) + _gds_real(1e-9))
        for st in self.structs:
            buf += st.bytes(ts)
        buf += _rec(0x04, 0x00)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf)


class GdsStruct:
    def __init__(self, name: str):
        self.name = name
        self.elems: list[bytes] = []

    def boundary(self, layer: str, xy_um: list[tuple[float, float]], datatype: int | None = None) -> None:
        ln, dt = LAYERS[layer]
        if datatype is not None:
            dt = datatype
        pts = []
        for x, y in xy_um:
            pts.extend((round(x * DBU), round(y * DBU)))
        # close
        if xy_um[0] != xy_um[-1]:
            pts.extend((round(xy_um[0][0] * DBU), round(xy_um[0][1] * DBU)))
        self.elems.append(
            _rec(0x08, 0x00)
            + _rec(0x0D, 0x02, _i16([ln]))
            + _rec(0x0E, 0x02, _i16([dt]))
            + _rec(0x10, 0x03, _i32(pts))
            + _rec(0x11, 0x00)
        )

    def rect(self, layer: str, x0: float, y0: float, x1: float, y1: float) -> None:
        self.boundary(layer, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def path(self, layer: str, pts_um: list[tuple[float, float]], width_um: float) -> None:
        ln, dt = LAYERS[layer]
        pts = []
        for x, y in pts_um:
            pts.extend((round(x * DBU), round(y * DBU)))
        self.elems.append(
            _rec(0x09, 0x00)
            + _rec(0x0D, 0x02, _i16([ln]))
            + _rec(0x0E, 0x02, _i16([dt]))
            + _rec(0x0F, 0x03, _i32([round(width_um * DBU)]))
            + _rec(0x10, 0x03, _i32(pts))
            + _rec(0x11, 0x00)
        )

    def sref(self, cell: str, x: float, y: float, reflect: bool = False) -> None:
        payload = _rec(0x0A, 0x00) + _rec(0x12, 0x06, _ascii(cell))
        if reflect:
            # bit 15 = reflection about x-axis
            payload += _rec(0x1A, 0x01, struct.pack(">H", 0x8000))
        payload += _rec(0x10, 0x03, _i32([round(x * DBU), round(y * DBU)]))
        payload += _rec(0x11, 0x00)
        self.elems.append(payload)

    def text(self, layer: str, x: float, y: float, s: str, mag: float = 0.4) -> None:
        ln, dt = LAYERS[layer]
        self.elems.append(
            _rec(0x0C, 0x00)
            + _rec(0x0D, 0x02, _i16([ln]))
            + _rec(0x16, 0x02, _i16([dt]))
            + _rec(0x17, 0x01, struct.pack(">H", 0x0005))  # font 0, center
            + _rec(0x1B, 0x05, _gds_real(mag))
            + _rec(0x10, 0x03, _i32([round(x * DBU), round(y * DBU)]))
            + _rec(0x19, 0x06, _ascii(s[:32]))
            + _rec(0x11, 0x00)
        )

    def bytes(self, ts: bytes) -> bytes:
        out = _rec(0x05, 0x02, ts) + _rec(0x06, 0x06, _ascii(self.name))
        for e in self.elems:
            out += e
        out += _rec(0x07, 0x00)
        return out


# ---------------------------------------------------------------------------
# Standard-cell masters (simplified SKY130 HD geometry)
# ---------------------------------------------------------------------------

@dataclass
class CellKind:
    name: str
    sites: int
    kind: str  # inv, nand, dff, tap, decap, buf, mux, xor, aoi, fill, clkbuf

    @property
    def width(self) -> float:
        return self.sites * SITE


def build_cell_master(kind: CellKind) -> GdsStruct:
    st = GdsStruct(kind.name)
    w, h = kind.width, ROW_H
    rail = 0.48
    st.rect("prBoundary", 0, 0, w, h)
    st.rect("areaid.sc", 0, 0, w, h)

    # nwell occupies upper ~55%
    st.rect("nwell", -0.12, h * 0.42, w + 0.12, h + 0.18)

    # power rails
    st.rect("met1", 0, h - rail, w, h)
    st.rect("li1", 0, h - rail, w, h)
    st.rect("met1", 0, 0, w, rail)
    st.rect("li1", 0, 0, w, rail)

    # substrate / nwell taps at ends
    tap_w = min(0.29, w * 0.35)
    st.rect("tap", 0.06, 0.06, 0.06 + tap_w, rail - 0.06)
    st.rect("tap", 0.06, h - rail + 0.06, 0.06 + tap_w, h - 0.06)

    if kind.kind == "tap":
        st.rect("tap", 0.08, 0.08, w - 0.08, rail - 0.08)
        st.rect("tap", 0.08, h - rail + 0.08, w - 0.08, h - 0.08)
        st.rect("diff", 0.10, rail + 0.1, w - 0.10, h - rail - 0.1)
        return st

    if kind.kind == "fill":
        st.rect("li1", 0.08, rail + 0.08, w - 0.08, h - rail - 0.08)
        return st

    if kind.kind == "decap":
        for i in range(kind.sites):
            x = i * SITE + 0.07
            st.rect("diff", x, rail + 0.08, x + 0.32, h * 0.48)
            st.rect("diff", x, h * 0.52, x + 0.32, h - rail - 0.08)
            st.rect("poly", x + 0.08, rail + 0.04, x + 0.22, h - rail - 0.04)
        return st

    # Active diffusion fingers
    n_fingers = max(1, kind.sites - 1)
    gate_w = 0.15
    pitch = w / (n_fingers + 1)
    for i in range(n_fingers):
        gx = pitch * (i + 1) - gate_w / 2
        # nmos (bottom) / pmos (top)
        st.rect("diff", gx - 0.12, rail + 0.10, gx + gate_w + 0.12, h * 0.42)
        st.rect("diff", gx - 0.12, h * 0.50, gx + gate_w + 0.12, h - rail - 0.10)
        st.rect("poly", gx, rail + 0.04, gx + gate_w, h - rail - 0.04)
        # contacts
        st.rect("licon", gx - 0.08, rail + 0.18, gx - 0.02, rail + 0.32)
        st.rect("licon", gx + gate_w + 0.02, rail + 0.18, gx + gate_w + 0.08, rail + 0.32)

    # local interconnect
    st.rect("li1", 0.08, h * 0.22, w - 0.08, h * 0.34)
    st.rect("li1", 0.08, h * 0.62, w - 0.08, h * 0.74)

    if kind.kind == "dff":
        # extra poly / li to look sequential
        for k in (0.22, 0.48, 0.74):
            st.rect("poly", w * k, rail + 0.06, w * k + 0.15, h - rail - 0.06)
        st.rect("met1", 0.15, h * 0.44, w - 0.15, h * 0.56)
        st.rect("mcon", w * 0.3, h * 0.46, w * 0.3 + 0.14, h * 0.54)
        st.rect("mcon", w * 0.7, h * 0.46, w * 0.7 + 0.14, h * 0.54)
        # clock pin stub
        st.rect("met1", 0.05, h * 0.46, 0.22, h * 0.54)

    if kind.kind in ("mux", "xor", "aoi"):
        st.rect("met1", 0.12, h * 0.36, w - 0.12, h * 0.42)
        st.rect("li1", 0.10, h * 0.48, w - 0.10, h * 0.58)

    return st


KINDS: list[CellKind] = [
    CellKind("sky130_fd_sc_hd__fill_1", 1, "fill"),
    CellKind("sky130_fd_sc_hd__tapvpwrvgnd_1", 1, "tap"),
    CellKind("sky130_fd_sc_hd__decap_4", 4, "decap"),
    CellKind("sky130_fd_sc_hd__inv_2", 2, "inv"),
    CellKind("sky130_fd_sc_hd__nand2_1", 3, "nand"),
    CellKind("sky130_fd_sc_hd__nor2_1", 3, "nor"),
    CellKind("sky130_fd_sc_hd__buf_2", 3, "buf"),
    CellKind("sky130_fd_sc_hd__clkbuf_4", 6, "clkbuf"),
    CellKind("sky130_fd_sc_hd__mux2_1", 6, "mux"),
    CellKind("sky130_fd_sc_hd__xor2_1", 7, "xor"),
    CellKind("sky130_fd_sc_hd__aoi21_1", 4, "aoi"),
    CellKind("sky130_fd_sc_hd__oai21_1", 4, "aoi"),
    CellKind("sky130_fd_sc_hd__dfxtp_1", 8, "dff"),
    CellKind("sky130_fd_sc_hd__dfrtp_1", 10, "dff"),
]


KIND_BY_NAME = {k.name: k for k in KINDS}


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

@dataclass
class Inst:
    kind: CellKind
    x: float
    y: float
    row: int
    flipped: bool
    block: str


def region_for(row: int, x: float) -> str:
    """Map a placement coordinate to a functional block (row bands)."""
    del x
    if row >= 28:
        return "bist_scan"
    if row >= 15:
        return "correction"
    return "lif_neuron"


def cell_bag() -> list[tuple[str, str]]:
    """Cell mix for Chip A: 8× (16-bit Vmem + 16-bit C) + combo.

    Every sequential element in the RTL is `always @(posedge clk or negedge rst_n)`,
    so the bag uses async-reset dfrtp_1 (not dfxtp_1).
      8 × 16 Vmem  + 8 spike_out  = 136  (LIF)
      8 × 16 C                    = 128  (correction)
      scan_idx ($clog2(256) = 8)  =   8  (scan)
    """
    bag: list[tuple[str, str]] = []
    bag += [("sky130_fd_sc_hd__dfrtp_1", "lif_neuron")] * 136
    bag += [("sky130_fd_sc_hd__dfrtp_1", "correction")] * 128
    bag += [("sky130_fd_sc_hd__dfrtp_1", "bist_scan")] * 8
    bag += [("sky130_fd_sc_hd__inv_2", "lif_neuron")] * 90
    bag += [("sky130_fd_sc_hd__inv_2", "correction")] * 90
    bag += [("sky130_fd_sc_hd__inv_2", "bist_scan")] * 24
    bag += [("sky130_fd_sc_hd__nand2_1", "lif_neuron")] * 80
    bag += [("sky130_fd_sc_hd__nand2_1", "correction")] * 80
    bag += [("sky130_fd_sc_hd__nand2_1", "bist_scan")] * 16
    bag += [("sky130_fd_sc_hd__nor2_1", "lif_neuron")] * 32
    bag += [("sky130_fd_sc_hd__nor2_1", "correction")] * 32
    bag += [("sky130_fd_sc_hd__buf_2", "lif_neuron")] * 32
    bag += [("sky130_fd_sc_hd__buf_2", "correction")] * 32
    bag += [("sky130_fd_sc_hd__buf_2", "bist_scan")] * 16
    bag += [("sky130_fd_sc_hd__clkbuf_4", "bist_scan")] * 8
    bag += [("sky130_fd_sc_hd__clkbuf_4", "correction")] * 8
    bag += [("sky130_fd_sc_hd__clkbuf_4", "lif_neuron")] * 8
    bag += [("sky130_fd_sc_hd__mux2_1", "lif_neuron")] * 40
    bag += [("sky130_fd_sc_hd__mux2_1", "correction")] * 16
    bag += [("sky130_fd_sc_hd__xor2_1", "lif_neuron")] * 32
    bag += [("sky130_fd_sc_hd__xor2_1", "correction")] * 32
    bag += [("sky130_fd_sc_hd__aoi21_1", "correction")] * 32
    bag += [("sky130_fd_sc_hd__aoi21_1", "lif_neuron")] * 32
    bag += [("sky130_fd_sc_hd__oai21_1", "correction")] * 20
    bag += [("sky130_fd_sc_hd__oai21_1", "lif_neuron")] * 20
    bag += [("sky130_fd_sc_hd__decap_4", "lif_neuron")] * 24
    bag += [("sky130_fd_sc_hd__decap_4", "correction")] * 24
    bag += [("sky130_fd_sc_hd__decap_4", "bist_scan")] * 8
    bag += [("sky130_fd_sc_hd__fill_1", "correction")] * 16
    bag += [("sky130_fd_sc_hd__fill_1", "lif_neuron")] * 12
    bag += [("sky130_fd_sc_hd__tapvpwrvgnd_1", "tap")] * 60
    return bag


def _band(block: str) -> range:
    if block == "lif_neuron":
        return range(0, 15)
    if block == "correction":
        return range(15, 28)
    if block == "bist_scan":
        return range(28, 33)
    return range(0, N_ROWS)


def place_rows(rng: random.Random) -> list[Inst]:
    """Pack each functional block into its own row band on the 90 µm die."""
    row0 = CORE_LL + ((CORE_UR - CORE_LL) - N_ROWS * ROW_H) / 2
    insts: list[Inst] = []
    occupied = {r: CORE_LL for r in range(N_ROWS)}

    by_block: dict[str, list[str]] = {
        "lif_neuron": [],
        "correction": [],
        "bist_scan": [],
        "tap": [],
    }
    for name, block in cell_bag():
        by_block.setdefault(block, []).append(name)
    for block in by_block:
        rng.shuffle(by_block[block])
        by_block[block].sort(key=lambda n: (0 if "dfrtp" in n or "dfxtp" in n else 1, rng.random()))

    def place_name(name: str, block: str, rows: range) -> bool:
        k = KIND_BY_NAME[name]
        row_list = list(rows)
        # round-robin from the least-filled row in the band
        row_list.sort(key=lambda r: occupied[r])
        for r in row_list:
            x = occupied[r]
            if x + k.width <= CORE_UR + 1e-9:
                y = row0 + r * ROW_H
                insts.append(Inst(k, x, y, r, r % 2 == 1, "tap" if block == "tap" else block))
                occupied[r] = x + k.width
                return True
        return False

    for block in ("lif_neuron", "correction", "bist_scan"):
        for name in by_block[block]:
            place_name(name, block, _band(block))
    for name in by_block.get("tap", []):
        place_name(name, "tap", range(0, N_ROWS))

    extras = (
        ["sky130_fd_sc_hd__fill_1"] * 40
        + ["sky130_fd_sc_hd__tapvpwrvgnd_1"] * 30
        + ["sky130_fd_sc_hd__decap_4"] * 8
    )
    ei = 0
    for r in range(N_ROWS):
        while ei < len(extras):
            k = KIND_BY_NAME[extras[ei]]
            x = occupied[r]
            if x + k.width <= CORE_UR + 1e-9:
                y = row0 + r * ROW_H
                insts.append(Inst(k, x, y, r, r % 2 == 1, region_for(r, x)))
                occupied[r] = x + k.width
                ei += 1
            else:
                break
    return insts


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def power_stripes() -> list[dict]:
    """M4 vertical + M5 horizontal power mesh."""
    stripes = []
    # M5 horizontal (VPWR / VGND alternating)
    y = CORE_LL + 3
    i = 0
    while y < CORE_UR - 2:
        net = "VPWR" if i % 2 == 0 else "VGND"
        stripes.append(
            {"layer": "met5", "x0": CORE_LL, "y0": y, "x1": CORE_UR, "y1": y + 1.2, "net": net}
        )
        y += 10.0
        i += 1
    # M4 vertical
    x = CORE_LL + 4
    i = 0
    while x < CORE_UR - 2:
        net = "VPWR" if i % 2 == 0 else "VGND"
        stripes.append(
            {"layer": "met4", "x0": x, "y0": CORE_LL, "x1": x + 1.0, "y1": CORE_UR, "net": net}
        )
        x += 10.0
        i += 1
    return stripes


def clock_tree() -> list[dict]:
    """Simple H-tree on met3."""
    routes = []
    cx = (CORE_LL + CORE_UR) / 2
    cy = (CORE_LL + CORE_UR) / 2
    # trunk from left pin
    routes.append({"layer": "met3", "w": 0.28, "pts": [[0.4, cy], [CORE_LL + 1.2, cy], [cx, cy]]})
    # H splits
    def split(x0, y0, x1, y1, depth):
        if depth == 0:
            return
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        routes.append({"layer": "met3", "w": max(0.14, 0.32 - 0.05 * (3 - depth)), "pts": [[x0, my], [x1, my]]})
        routes.append({"layer": "met3", "w": max(0.14, 0.28 - 0.05 * (3 - depth)), "pts": [[mx, y0], [mx, y1]]})
        if depth > 1:
            split(x0, y0, mx, my, depth - 1)
            split(mx, y0, x1, my, depth - 1)
            split(x0, my, mx, y1, depth - 1)
            split(mx, my, x1, y1, depth - 1)

    split(CORE_LL + 2, CORE_LL + 2, CORE_UR - 2, CORE_UR - 2, 3)
    return routes


def signal_routes(rng: random.Random, insts: list[Inst]) -> list[dict]:
    routes = []
    dffs = [i for i in insts if i.kind.kind == "dff"]
    combos = [i for i in insts if i.kind.kind in ("nand", "inv", "xor", "aoi", "mux")]
    rng.shuffle(dffs)
    rng.shuffle(combos)
    n = min(420, len(dffs), len(combos))
    for a, b in zip(dffs[:n], combos[:n]):
        x0 = a.x + a.kind.width * 0.5
        y0 = a.y + ROW_H * (0.7 if not a.flipped else 0.3)
        x1 = b.x + b.kind.width * 0.5
        y1 = b.y + ROW_H * (0.7 if not b.flipped else 0.3)
        # 2-bend Manhattan
        if rng.random() < 0.5:
            pts = [[x0, y0], [x1, y0], [x1, y1]]
            layer = "met2"
        else:
            pts = [[x0, y0], [x0, y1], [x1, y1]]
            layer = "met2" if rng.random() < 0.6 else "met3"
        routes.append({"layer": layer, "w": 0.14, "pts": pts})
    return routes


def make_pins() -> list[dict]:
    pins = []
    left = ["clk", "rst_n", "corr_en", "cfg_en", "cfg_si", "cfg_so"]
    for i, n in enumerate(left):
        y = 8 + i * 12
        pins.append({"name": n, "x": 0.3, "y": y, "side": "left", "w": 2.4, "h": 0.9})
    for i in range(3):
        y = 8 + (6 + i) * 12
        pins.append({"name": f"leak_sh[{i}]", "x": 0.3, "y": y, "side": "left", "w": 2.4, "h": 0.9})
    for i in range(8):
        x = 6.0 + i * 10.4
        pins.append({"name": f"icol{i}[5:0]", "x": x, "y": 0.3, "side": "bottom", "w": 2.4, "h": 2.0})
    for i in range(6):
        x = 6.0 + i * 6.0
        pins.append({"name": f"i_ref[{i}]", "x": x, "y": DIE - 2.7, "side": "top", "w": 0.9, "h": 2.4})
    for i in range(8):
        x = 48.0 + i * 4.8
        pins.append({"name": f"spike[{i}]", "x": x, "y": DIE - 2.7, "side": "top", "w": 0.9, "h": 2.4})
    for i in range(16):
        y = 4.0 + i * 5.2
        if y > DIE - 3.5:
            break
        pins.append({"name": f"v_th[{i}]", "x": DIE - 2.7, "y": y, "side": "right", "w": 2.4, "h": 0.8})
    return pins


def block_bboxes(insts: list[Inst]) -> list[dict]:
    row0 = CORE_LL + ((CORE_UR - CORE_LL) - N_ROWS * ROW_H) / 2
    bands = {
        "lif_neuron": (0, 15, "U1  LIF accumulator"),
        "correction": (15, 28, "U2  ref-column correction"),
        "bist_scan": (28, 33, "scan / membrane dump"),
    }
    out = []
    for k, (r0, r1, label) in bands.items():
        items = [i for i in insts if i.block == k]
        y0 = row0 + r0 * ROW_H
        y1 = row0 + r1 * ROW_H
        out.append({
            "id": k,
            "label": label,
            "bbox": [CORE_LL, round(y0, 3), CORE_UR, round(y1, 3)],
            "count": len(items),
        })
    return out


def write_rtl() -> None:
    """Keep public RTL artifacts separate from the privately maintained implementation."""
    RTL_DIR.mkdir(parents=True, exist_ok=True)
    stale = RTL_DIR / "vc_unit2_scan.v"
    if stale.exists():
        stale.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(130)
    insts = place_rows(rng)
    stripes = power_stripes()
    clk = clock_tree()
    sigs = signal_routes(rng, insts)
    pins = make_pins()
    blocks = block_bboxes(insts)

    lib = GdsLib("VIRTUAL_CORE_CHIP_A")
    for k in KINDS:
        lib.add(build_cell_master(k))

    top = GdsStruct("virtual_core_chip_a")
    # die / core
    top.rect("prBoundary", 0, 0, DIE, DIE)
    # thin seal (hard-macro outline, not a pad ring)
    top.rect("met5", 0, 0, DIE, 0.7)
    top.rect("met5", 0, DIE - 0.7, DIE, DIE)
    top.rect("met5", 0, 0, 0.7, DIE)
    top.rect("met5", DIE - 0.7, 0, DIE, DIE)
    top.rect("nwell", CORE_LL, CORE_LL, CORE_UR, CORE_UR)

    # row power rails across the full core (empty sites still show VPWR/VGND)
    row0 = CORE_LL + ((CORE_UR - CORE_LL) - N_ROWS * ROW_H) / 2
    rail = 0.48
    for r in range(N_ROWS):
        y = row0 + r * ROW_H
        top.rect("met1", CORE_LL, y, CORE_UR, y + rail)
        top.rect("met1", CORE_LL, y + ROW_H - rail, CORE_UR, y + ROW_H)
        top.rect("li1", CORE_LL, y, CORE_UR, y + rail)
        top.rect("li1", CORE_LL, y + ROW_H - rail, CORE_UR, y + ROW_H)

    for inst in insts:
        if inst.flipped:
            top.sref(inst.kind.name, inst.x, inst.y + ROW_H, reflect=True)
        else:
            top.sref(inst.kind.name, inst.x, inst.y, reflect=False)

    for s in stripes:
        top.rect(s["layer"], s["x0"], s["y0"], s["x1"], s["y1"])
        # vias at intersections onto the mesh
        if s["layer"] == "met5" and s["net"] == "VPWR":
            x = CORE_LL + 4
            while x < CORE_UR - 2:
                top.rect("via4", x + 0.2, s["y0"] + 0.3, x + 0.8, s["y0"] + 0.9)
                x += 10.0

    for r in clk + sigs:
        top.path(r["layer"], [(p[0], p[1]) for p in r["pts"]], r["w"])

    for p in pins:
        top.rect("met3", p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"])
        top.rect("met5", p["x"] - 0.1, p["y"] - 0.1, p["x"] + p["w"] + 0.1, p["y"] + p["h"] + 0.1)
        top.text("label", p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, p["name"], mag=0.35)

    top.text("text", DIE / 2, DIE - 1.15, "VIRTUAL_CORE_CHIP_A", mag=0.55)
    top.text("text", DIE / 2, 1.05, "SKY130A  HD  1.8V", mag=0.4)
    lib.add(top)

    gds_path = OUT_DIR / "virtual_core_chip_a.gds"
    lib.write(gds_path)

    # metrics
    n_dff = sum(1 for i in insts if i.kind.kind == "dff")
    n_seq_bits = n_dff  # 1-bit flops
    cell_area = sum(i.kind.width * ROW_H for i in insts)
    counts: dict[str, int] = {}
    for i in insts:
        counts[i.kind.name] = counts.get(i.kind.name, 0) + 1

    layout = {
        "name": "virtual_core_chip_a",
        "lib": "VIRTUAL_CORE_CHIP_A",
        "tech": "sky130A",
        "stdcell": "sky130_fd_sc_hd",
        "die": [0, 0, DIE, DIE],
        "core": [CORE_LL, CORE_LL, CORE_UR, CORE_UR],
        "rowHeight": ROW_H,
        "siteWidth": SITE,
        "nRows": N_ROWS,
        "kinds": [
            {"name": k.name, "sites": k.sites, "width": k.width, "kind": k.kind}
            for k in KINDS
        ],
        "cells": [
            {
                "k": i.kind.name,
                "x": round(i.x, 3),
                "y": round(i.y, 3),
                "w": round(i.kind.width, 3),
                "f": int(i.flipped),
                "b": i.block,
            }
            for i in insts
        ],
        "stripes": stripes,
        "clock": clk,
        "routes": sigs,
        "pins": pins,
        "blocks": blocks,
        "metrics": {
            "cells": len(insts),
            "dff": n_dff,
            "seqBits": n_seq_bits,
            "cellAreaUm2": round(cell_area, 1),
            "coreAreaUm2": round((CORE_UR - CORE_LL) ** 2, 1),
            "dieAreaUm2": round(DIE * DIE, 1),
            "utilization": round(cell_area / ((CORE_UR - CORE_LL) ** 2), 3),
            "power10MHzUw": 132,
            "powerMaxMw": 0,
            "fmaxMHz": 0,
            "voltage": 1.8,
            "corner": "tt 25C 1.8V (est.)",
            "clockNs": 100,
            "nColumns": 8,
            "accumBits": 16,
            "adcBits": 6,
            "gdsKind": "sky130-style visualization",
            "counts": counts,
        },
        "gdsFile": "virtual_core_chip_a.gds",
        "gdsBytes": gds_path.stat().st_size,
    }
    (OUT_DIR / "layout.json").write_text(json.dumps(layout, separators=(",", ":")))
    data_copy = Path("/workspace/src/data/layout.json")
    data_copy.write_text((OUT_DIR / "layout.json").read_text())
    write_rtl()
    print(f"GDS  {gds_path}  {gds_path.stat().st_size:,} bytes")
    print(f"JSON {(OUT_DIR / 'layout.json')}  {(OUT_DIR / 'layout.json').stat().st_size:,} bytes")
    print(f"cells={len(insts)}  dff={n_dff}  area={cell_area:.1f} um^2  util={layout['metrics']['utilization']}")


if __name__ == "__main__":
    main()
