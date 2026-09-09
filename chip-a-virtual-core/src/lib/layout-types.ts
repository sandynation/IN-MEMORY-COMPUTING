export type CellInst = {
  k: string;
  x: number;
  y: number;
  w: number;
  f: number;
  b: string;
};

export type KindInfo = {
  name: string;
  sites: number;
  width: number;
  kind: string;
};

export type Stripe = {
  layer: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  net: string;
};

export type Route = {
  layer: string;
  w: number;
  pts: number[][];
};

export type Pin = {
  name: string;
  x: number;
  y: number;
  side: string;
  w: number;
  h: number;
};

export type Block = {
  id: string;
  label: string;
  bbox: [number, number, number, number];
  count: number;
};

export type LayoutMetrics = {
  cells: number;
  dff: number;
  seqBits: number;
  cellAreaUm2: number;
  coreAreaUm2: number;
  dieAreaUm2: number;
  utilization: number;
  power10MHzUw: number;
  powerMaxMw: number;
  fmaxMHz: number;
  voltage: number;
  corner: string;
  clockNs: number;
  nColumns?: number;
  accumBits?: number;
  adcBits?: number;
  gdsKind?: string;
  manuscriptAreaUm2?: number;
  manuscriptCells?: number;
  counts: Record<string, number>;
};

export type LayoutData = {
  name: string;
  lib: string;
  tech: string;
  stdcell: string;
  die: [number, number, number, number];
  core: [number, number, number, number];
  rowHeight: number;
  siteWidth: number;
  nRows: number;
  kinds: KindInfo[];
  cells: CellInst[];
  stripes: Stripe[];
  clock: Route[];
  routes: Route[];
  pins: Pin[];
  blocks: Block[];
  metrics: LayoutMetrics;
  gdsFile: string;
  gdsBytes: number;
};

export const LAYER_STYLE: Record<
  string,
  { label: string; color: string; alpha: number; order: number }
> = {
  nwell: { label: "nwell", color: "#3d6b4f", alpha: 0.22, order: 0 },
  diff: { label: "diff", color: "#4ade80", alpha: 0.55, order: 1 },
  tap: { label: "tap", color: "#86efac", alpha: 0.4, order: 2 },
  poly: { label: "poly", color: "#f07167", alpha: 0.7, order: 3 },
  li1: { label: "li1", color: "#60a5fa", alpha: 0.45, order: 4 },
  met1: { label: "met1", color: "#38bdf8", alpha: 0.55, order: 5 },
  met2: { label: "met2", color: "#a78bfa", alpha: 0.5, order: 6 },
  met3: { label: "met3", color: "#c4a574", alpha: 0.32, order: 7 },
  met4: { label: "met4", color: "#fb7185", alpha: 0.45, order: 8 },
  met5: { label: "met5", color: "#e7e5e4", alpha: 0.5, order: 9 },
  cell: { label: "std cells", color: "#94a3b8", alpha: 0.35, order: 10 },
  pin: { label: "pins", color: "#f8fafc", alpha: 0.9, order: 11 },
};

export const BLOCK_TINT: Record<string, string> = {
  lif_neuron: "rgba(56, 189, 248, 0.10)",
  correction: "rgba(52, 211, 153, 0.10)",
  bist_scan: "rgba(248, 113, 113, 0.10)",
  tap: "transparent",
};

export const KIND_FILL: Record<string, string> = {
  dff: "#1e3a5f",
  inv: "#164e3b",
  nand: "#1e3a2f",
  nor: "#1e3a2f",
  buf: "#1c2e4a",
  clkbuf: "#3b2f14",
  mux: "#3b1f3a",
  xor: "#3b2a14",
  aoi: "#1a3344",
  decap: "#1a1d24",
  tap: "#243028",
  fill: "#14161c",
};
