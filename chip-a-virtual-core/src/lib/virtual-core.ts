/** Bit-accurate Chip A datapath (proposal §2.3). Mirrors public/rtl. */

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

export type CoreInputs = {
  iCol: number[];
  iRef: number;
  vTh: number;
  leakSh: number;
  corrEn: boolean;
};

function toS16(x: number): number {
  const u = x & 0xffff;
  return u & 0x8000 ? u - 0x10000 : u;
}

function asr16(v: number, sh: number): number {
  const s = Math.max(0, Math.min(7, sh | 0));
  return toS16(v) >> s;
}

function sat6(c: number): number {
  if (c < 0) return 0;
  if (c > ADC_MAX) return ADC_MAX;
  return c & ADC_MAX;
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
  const ref = Math.max(0, Math.min(ADC_MAX, iRef | 0));
  const cNext = toS16(s.c + ref);
  const dac = corrEn ? sat6(cNext) : 0;
  const iCorr = col - dac;
  const leaked = asr16(s.vMem, leakSh);
  const vNext = toS16(leaked + iCorr);
  const fire = vNext >= vTh;
  return {
    vMem: fire ? 0 : vNext,
    c: fire ? 0 : cNext,
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

  run("shared reset clears C with Vmem", () => {
    let s = resetColumn();
    s = stepColumn(s, 30, 2, 40, 0, true); // v=28, c=2
    s = stepColumn(s, 30, 2, 40, 0, true); // v_next=54, fire, both 0
    return s.spike === 1 && s.vMem === 0 && s.c === 0;
  });

  run("same-cycle DAC uses C+Iref", () => {
    let s = resetColumn();
    s = stepColumn(s, 20, 5, 1000, 0, true);
    return s.dac === 5 && s.iCorr === 15 && s.vMem === 15 && s.c === 5;
  });

  run("correction subtracts growing DAC", () => {
    let s = resetColumn();
    s = stepColumn(s, 20, 5, 1000, 0, true); // v=15, c=5
    s = stepColumn(s, 20, 5, 1000, 0, true); // dac=10, v=15+10=25, c=10
    s = stepColumn(s, 20, 5, 1000, 0, true); // dac=15, v=25+5=30, c=15
    return s.vMem === 30 && s.c === 15 && s.dac === 15;
  });

  run("corr_en=0 ignores DAC (§3.2 off)", () => {
    let s = resetColumn();
    s = stepColumn(s, 20, 5, 40, 0, false);
    const t1 = s.vMem === 20 && s.dac === 0 && s.spike === 0;
    s = stepColumn(s, 20, 5, 40, 0, false);
    return t1 && s.spike === 1 && s.vMem === 0;
  });

  run("λ=1/2 barrel-shift leak", () => {
    let s = resetColumn();
    s = stepColumn(s, 16, 0, 1000, 1, true); // v=16
    s = stepColumn(s, 16, 0, 1000, 1, true); // leaked=8, v=24
    s = stepColumn(s, 16, 0, 1000, 1, true); // leaked=12, v=28
    return s.vMem === 28;
  });

  run("DAC saturates at 63", () => {
    let s = resetColumn();
    s = { ...s, c: 60 };
    s = stepColumn(s, 20, 10, 1000, 0, true);
    return s.dac === 63 && s.iCorr === 20 - 63 && s.c === 70;
  });

  run("independent columns share Iref, not C", () => {
    let cols = resetCore();
    const iCol = [30, 0, 0, 0, 0, 0, 0, 0];
    cols = stepCore(cols, { iCol, iRef: 4, vTh: 1000, leakSh: 0, corrEn: true });
    return cols[0].vMem === 26 && cols[1].vMem === -4 && cols[0].c === 4 && cols[1].c === 4;
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
    claim: "Unit 1 — 16-bit signed LIF, Vmem[t+1] = λ·Vmem[t] + Icol_corrected",
    status: "pass",
    detail: "vc_unit1.v: arithmetic barrel-shift + 16-bit accumulator.",
  },
  {
    id: "lambda",
    claim: "λ via barrel-shift (power-of-two decay, no multiplier)",
    status: "pass",
    detail: "v_mem >>> leak_sh, leak_sh[2:0] so λ ∈ {1, 1/2, …, 1/128}.",
  },
  {
    id: "u2",
    claim: "Unit 2 — 16-bit C[t+1] = C[t] + Iref_ADC[t], 6-bit DAC",
    status: "pass",
    detail: "Shared 6-bit reference ADC; per-column C because neurons reset independently.",
  },
  {
    id: "reset",
    claim: "Shared reset: spike clears Vmem and C in the same cycle",
    status: "pass",
    detail: "Combinational fire from v_next clocks both units.",
  },
  {
    id: "timing",
    claim: "Same-cycle: Iref→C→DAC→Icorr→Vmem→threshold→spike",
    status: "pass",
    detail: "DAC is combinational from C+Iref; Vmem/C update on the clock edge.",
  },
  {
    id: "nolut",
    claim: "No β LUT, no CPU, no instruction fetch",
    status: "pass",
    detail: "Fixed-function Unit 1 + Unit 2 only. The notebook LUT RTL is not this chip.",
  },
  {
    id: "corren",
    claim: "§3.2 correction on/off for membrane trajectories",
    status: "pass",
    detail: "corr_en zeros the DAC; C still tracks Iref so the window stays aligned.",
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
