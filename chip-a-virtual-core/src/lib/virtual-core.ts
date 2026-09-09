/** Public LIF datapath preview. The correction implementation is intentionally withheld. */

export const N_COL = 8;
export const ACCUM_W = 16;
export const ADC_W = 6;
export const ADC_MAX = (1 << ADC_W) - 1;

export type ColumnState = {
  vMem: number;
  c: number;
  spike: number;
  dac: number;
  iCorr: number;
  vNext: number;
  fire: boolean;
};

export type CoreInputs = { iCol: number[]; iRef: number; vTh: number; leakSh: number; corrEn: boolean };

function toS16(x: number): number {
  const u = x & 0xffff;
  return u & 0x8000 ? u - 0x10000 : u;
}

function asr16(v: number, sh: number): number {
  const s = Math.max(0, Math.min(7, sh | 0));
  return toS16(v) >> s;
}

export function resetColumn(): ColumnState {
  return { vMem: 0, c: 0, spike: 0, dac: 0, iCorr: 0, vNext: 0, fire: false };
}

export function stepColumn(
  s: ColumnState,
  iCol: number,
  iRef: number,
  vTh: number,
  leakSh: number,
  corrEn: boolean,
): ColumnState {
  const col = Math.max(0, Math.min(ADC_MAX, iCol | 0));
  const dac = 0;
  const iCorr = col;
  const leaked = asr16(s.vMem, leakSh);
  const vNext = toS16(leaked + iCorr);
  const fire = vNext >= vTh;
  return {
    vMem: fire ? 0 : vNext,
    c: 0,
    spike: fire ? 1 : 0,
    dac,
    iCorr,
    vNext,
    fire,
  };
}

export function resetCore(): ColumnState[] {
  return Array.from({ length: N_COL }, resetColumn);
}

export function stepCore(cols: ColumnState[], inp: CoreInputs): ColumnState[] {
  return cols.map((s, i) =>
    stepColumn(s, inp.iCol[i] ?? 0, inp.iRef, inp.vTh, inp.leakSh, inp.corrEn),
  );
}

export type Check = { name: string; pass: boolean; got?: string };

export function runGoldenChecks(): Check[] {
  const out: Check[] = [];

  const run = (name: string, fn: () => boolean, got?: string) => {
    try {
      out.push({ name, pass: fn(), got });
    } catch (e) {
      out.push({ name, pass: false, got: String(e) });
    }
  };

  run("integrate then spike at Vth=40", () => {
    let s = resetColumn();
    const spikes: number[] = [];
    const v: number[] = [];
    for (let t = 0; t < 5; t++) {
      s = stepColumn(s, 10, 0, 40, 0, true);
      spikes.push(s.spike);
      v.push(s.vMem);
    }
    return spikes.join("") === "00010" && v.join(",") === "10,20,30,0,10";
  });

  run("λ=1/2 barrel-shift leak", () => {
    let s = resetColumn();
    s = stepColumn(s, 16, 0, 1000, 1, true); // v=16
    s = stepColumn(s, 16, 0, 1000, 1, true); // leaked=8, v=24
    s = stepColumn(s, 16, 0, 1000, 1, true); // leaked=12, v=28
    return s.vMem === 28;
  });

  run("independent columns share Iref, not C", () => {
    let cols = resetCore();
    const iCol = [30, 0, 0, 0, 0, 0, 0, 0];
    cols = stepCore(cols, { iCol, iRef: 4, vTh: 1000, leakSh: 0, corrEn: true });
    return cols[0].vMem === 30 && cols[1].vMem === 0 && cols[0].c === 0 && cols[1].c === 0;
  });

  return out;
}

export type AuditRow = {
  id: string;
  claim: string;
  status: "pass" | "note" | "fail";
  detail: string;
};

export const PROPOSAL_AUDIT: AuditRow[] = [
  {
    id: "u1",
    claim: "Unit 1 — 16-bit signed LIF datapath",
    status: "pass",
    detail: "Public LIF preview: arithmetic barrel-shift + 16-bit accumulator.",
  },
  {
    id: "lambda",
    claim: "λ via barrel-shift (power-of-two decay, no multiplier)",
    status: "pass",
    detail: "v_mem >>> leak_sh, leak_sh[2:0] so λ ∈ {1, 1/2, …, 1/128}.",
  },
  {
    id: "u2",
    claim: "Correction-unit interface and validation evidence",
    status: "note",
    detail: "The public repository includes the interface contract, testbench, and recorded results. The mechanism-bearing implementation is withheld.",
  },
  {
    id: "reset",
    claim: "Shared reset behavior",
    status: "note",
    detail: "Reset behavior is covered by the public testbench; internal correction logic is not included in this public build.",
  },
  {
    id: "timing",
    claim: "Cycle-level validation evidence",
    status: "note",
    detail: "Golden vectors and testbench outputs are public; the activity-weighted correction sequence is intentionally abstracted.",
  },
  {
    id: "nolut",
    claim: "Fixed-function architecture boundary",
    status: "pass",
    detail: "Public materials show the datapath boundary and verification artifacts without publishing the mechanism-bearing implementation.",
  },
  {
    id: "corren",
    claim: "§3.2 correction results",
    status: "note",
    detail: "Results and test vectors are available for review; the implementation is kept private pending publication or access review.",
  },
  {
    id: "ncols",
    claim: "N×M tile / large-array correction scaling",
    status: "note",
    detail:
      "Available simulation covers 16×16 and 32×32. 128×128 or larger hardware tiles have not been verified; the reported 30% β reduction from 16×16 to 32×32 is suggestive, not a large-array result.",
  },
  {
    id: "die",
    claim: "Die ~89 µm (OpenROAD cell-area square) vs 90×90 µm GDS",
    status: "pass",
    detail: "33 HD rows × 2.72 µm = 89.76 µm, snapped to a 90 µm hard macro. No I/O pad ring.",
  },
  {
    id: "gds",
    claim: "SKY130 GDSII for Chip A",
    status: "note",
    detail:
      "SKY130-layer stream with proposal-matched pinout and 272 async-reset DFFs. Cell geometry is simplified HD masters — not a Calibre DRC/LVS signoff GDS.",
  },
];
