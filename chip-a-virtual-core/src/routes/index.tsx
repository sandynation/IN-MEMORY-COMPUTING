import { createFileRoute } from "@tanstack/react-router";
import {
  BoxSelect,
  ChevronDown,
  Cpu,
  Download,
  Focus,
  Layers,
  Minus,
  Plus,
  RotateCcw,
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { DatapathPanel } from "@/components/datapath-panel";
import { DrcFlow } from "@/components/drc-flow";
import { GdsCanvas } from "@/components/gds-canvas";
import { SpecAudit } from "@/components/spec-audit";
import { Button } from "@/components/ui/button";
import layoutJson from "@/data/layout.json";
import {
  LAYER_STYLE,
  type CellInst,
  type LayoutData,
} from "@/lib/layout-types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({ component: Home });

const layout = layoutJson as unknown as LayoutData;

const LAYER_KEYS = [
  "nwell",
  "diff",
  "poly",
  "li1",
  "met1",
  "met2",
  "met3",
  "met4",
  "met5",
  "cell",
  "pin",
] as const;

type View = "floorplan" | "datapath" | "audit" | "drc";

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function Home() {
  const [view, setView] = useState<View>("floorplan");
  const [layers, setLayers] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(LAYER_KEYS.map((k) => [k, true])),
  );
  const [highlight, setHighlight] = useState<string | null>(null);
  const [selected, setSelected] = useState<CellInst | null>(null);
  const [hover, setHover] = useState<CellInst | null>(null);
  const [um, setUm] = useState<{ x: number; y: number } | null>(null);
  const [cam, setCam] = useState({ scale: 1, x: 0, y: 0 });
  const [fitToken, setFitToken] = useState(0);
  const [zoomNudge, setZoomNudge] = useState(0);
  const [mobilePanel, setMobilePanel] = useState<"layers" | "stats" | null>(null);

  const onHover = useCallback((cell: CellInst | null, pos: { x: number; y: number } | null) => {
    setHover(cell);
    setUm(pos);
  }, []);

  const mix = useMemo(() => {
    const rows = Object.entries(layout.metrics.counts)
      .map(([name, n]) => ({
        name: name.replace("sky130_fd_sc_hd__", ""),
        n,
      }))
      .sort((a, b) => b.n - a.n);
    return rows;
  }, []);

  const m = layout.metrics;
  const dieSide = layout.die[2] - layout.die[0];
  const cellSide = Math.sqrt(m.cellAreaUm2);
  const nCol = m.nColumns ?? 8;
  const accum = m.accumBits ?? 16;
  const adc = m.adcBits ?? 6;

  return (
    <main className="flex h-dvh min-h-0 flex-col bg-bg text-fg">
      <header className="flex shrink-0 items-center gap-3 border-b border-border px-3 py-2.5 sm:px-4">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-sm border border-border bg-bg-subtle">
            <Cpu className="size-4 text-accent" strokeWidth={1.75} />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold tracking-tight sm:text-base">
              Virtual Core · Chip A
            </h1>
            <p className="truncate font-mono text-[11px] text-fg-subtle">
              {layout.tech} · {nCol} col · {accum}-bit LIF · {adc}-bit Iref
            </p>
          </div>
        </div>
        <nav className="hidden items-center rounded-sm border border-border bg-bg-subtle p-0.5 md:flex">
          {(
            [
              ["floorplan", "Floorplan"],
              ["datapath", "Datapath"],
              ["audit", "Checklist"],
              ["drc", "DRC Flow"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setView(id)}
              className={cn(
                "rounded-xs px-2.5 py-1 text-xs",
                view === id ? "bg-bg-hover text-fg" : "text-fg-muted hover:text-fg",
              )}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="hidden items-center gap-2 xl:flex">
          <StatChip label="cells" value={m.cells.toLocaleString()} />
          <StatChip label="die" value={`${layout.die[2].toFixed(0)} µm`} />
        </div>
        <Button asChild size="sm" className="shrink-0">
          <a href="/gds/virtual_core_chip_a.gds" download="virtual_core_chip_a.gds">
            <Download className="size-3.5" />
            <span className="hidden sm:inline">Download GDSII</span>
            <span className="sm:hidden">GDSII</span>
          </a>
        </Button>
      </header>

      <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-border px-3 py-1.5 md:hidden">
        {(
          [
            ["floorplan", "Floorplan"],
            ["datapath", "Datapath"],
            ["audit", "Checklist"],
            ["drc", "DRC Flow"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setView(id)}
            className={cn(
              "rounded-sm px-3 py-2 text-sm",
              view === id ? "bg-bg-hover text-fg" : "text-fg-muted",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {view === "datapath" ? (
        <DatapathPanel />
      ) : view === "audit" ? (
        <SpecAudit />
      ) : view === "drc" ? (
        <DrcFlow />
      ) : (
      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-56 shrink-0 flex-col border-r border-border bg-bg-elevated lg:flex">
          <SectionTitle icon={Layers} label="Layers" />
          <div className="flex-1 overflow-y-auto px-2 pb-3">
            {LAYER_KEYS.map((key) => (
              <LayerRow
                key={key}
                id={key}
                on={layers[key]}
                onToggle={() => setLayers((s) => ({ ...s, [key]: !s[key] }))}
              />
            ))}
          </div>
          <SectionTitle icon={BoxSelect} label="Blocks" />
          <div className="px-2 pb-3">
            {layout.blocks.map((b) => (
              <button
                key={b.id}
                type="button"
                onClick={() => setHighlight((h) => (h === b.id ? null : b.id))}
                className={cn(
                  "mb-1 flex w-full items-center justify-between rounded-sm px-2 py-1.5 text-left text-xs transition-colors",
                  highlight === b.id
                    ? "bg-accent-dim text-accent"
                    : "text-fg-muted hover:bg-bg-hover hover:text-fg",
                )}
              >
                <span className="truncate">{b.label}</span>
                <span className="font-mono tabular-nums text-fg-subtle">{b.count}</span>
              </button>
            ))}
          </div>
        </aside>

        <section className="relative min-h-0 min-w-0 flex-1">
          <GdsCanvas
            layout={layout}
            layers={layers}
            highlight={highlight}
            selected={selected}
            onSelect={setSelected}
            onHover={onHover}
            onCam={setCam}
            fitToken={fitToken}
            zoomNudge={zoomNudge}
          />
          <div className="pointer-events-none absolute right-3 top-3 flex flex-col gap-1">
            <div className="pointer-events-auto flex flex-col overflow-hidden rounded-md border border-border bg-bg-elevated/90 shadow-[var(--shadow-panel)]">
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Zoom in"
                onClick={() => setZoomNudge((n) => n + 1)}
                className="rounded-none"
              >
                <Plus className="size-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Zoom out"
                onClick={() => setZoomNudge((n) => n - 1)}
                className="rounded-none border-y border-border"
              >
                <Minus className="size-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Fit die"
                onClick={() => setFitToken((n) => n + 1)}
                className="rounded-none border-b border-border"
              >
                <Focus className="size-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Reset view"
                onClick={() => {
                  setHighlight(null);
                  setSelected(null);
                  setFitToken((n) => n + 1);
                }}
                className="rounded-none"
              >
                <RotateCcw className="size-3.5" />
              </Button>
            </div>
          </div>
          <p className="pointer-events-none absolute left-3 top-3 hidden rounded-sm border border-border bg-bg-elevated/85 px-2 py-1 font-mono text-[11px] text-fg-muted sm:block">
            drag pan · wheel zoom · double-click fit
          </p>
        </section>

        <aside className="hidden w-72 shrink-0 flex-col border-l border-border bg-bg-elevated xl:flex">
          <SectionTitle label="Design" />
          <dl className="grid grid-cols-2 gap-x-3 gap-y-2 px-3 pb-3 text-xs">
            <Metric k="Die" v={`${dieSide.toFixed(0)} × ${dieSide.toFixed(0)} µm`} />
            <Metric k="Rows" v={`${layout.nRows} × ${layout.rowHeight} µm`} />
            <Metric k="Std cells" v={m.cells.toLocaleString()} />
            <Metric k="DFF" v={String(m.dff)} />
            <Metric k="Logic area" v={`${m.cellAreaUm2.toLocaleString()} µm²`} />
            <Metric k="Util." v={`${(m.utilization * 100).toFixed(1)}%`} />
            <Metric k="Columns" v={String(nCol)} />
            <Metric k="Clock" v="10 MHz nom." />
            <Metric k="Corner" v={m.corner} />
            <Metric k="GDS size" v={formatBytes(layout.gdsBytes)} />
          </dl>
          <div className="mx-3 mb-3 rounded-md border border-border bg-bg-subtle p-2.5 text-[11px] leading-relaxed text-fg-muted">
            <p className="font-medium text-fg">Proposal §2.3</p>
            <p className="mt-1">
              Chip A digital core: {nCol} columns × {accum}-bit Vmem + {accum}-bit C.
              λ = 2<sup>−k</sup> barrel-shift. {adc}-bit column ADC and reference-column DAC.
              No LUT — correction is C[t]+I<sub>ref</sub>, reset with the spike.
            </p>
          </div>
          <SectionTitle label="Selected" />
          <div className="px-3 pb-3 text-xs">
            {selected ? (
              <div className="rounded-md border border-border bg-bg-subtle p-2.5 font-mono">
                <p className="text-fg">{selected.k.replace("sky130_fd_sc_hd__", "")}</p>
                <p className="mt-1 text-fg-subtle">
                  x={selected.x.toFixed(2)}  y={selected.y.toFixed(2)}  w={selected.w.toFixed(2)} µm
                </p>
                <p className="text-fg-subtle">block {selected.b}</p>
              </div>
            ) : (
              <p className="text-fg-subtle">Click a standard cell in the layout.</p>
            )}
          </div>
          <SectionTitle label="Cell mix" />
          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
            {mix.map((row) => (
              <div key={row.name} className="flex items-center justify-between py-0.5 text-[11px]">
                <span className="truncate font-mono text-fg-muted">{row.name}</span>
                <span className="font-mono tabular-nums text-fg">{row.n}</span>
              </div>
            ))}
          </div>
          <div className="border-t border-border p-3">
            <Button asChild className="w-full">
              <a href="/gds/virtual_core_chip_a.gds" download="virtual_core_chip_a.gds">
                <Download className="size-4" />
                virtual_core_chip_a.gds
              </a>
            </Button>
            <p className="mt-2 text-[11px] leading-relaxed text-fg-subtle">
              SKY130A stream. Open in KLayout or Magic. RTL is in the same download set below.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <a className="text-[11px] text-accent underline-offset-2 hover:underline" href="/rtl/virtual_core_chip_a.v" download>
                top.v
              </a>
              <a className="text-[11px] text-accent underline-offset-2 hover:underline" href="/rtl/vc_unit1.v" download>
                unit1.v
              </a>
              <a className="text-[11px] text-accent underline-offset-2 hover:underline" href="/rtl/vc_unit2.v" download>
                unit2.v
              </a>
              <a className="text-[11px] text-accent underline-offset-2 hover:underline" href="/rtl/tb_chip_a.v" download>
                tb.v
              </a>
            </div>
          </div>
        </aside>
      </div>
      )}

      {view === "floorplan" && (
      <footer className="flex shrink-0 items-center gap-3 border-t border-border px-3 py-1.5 font-mono text-[11px] text-fg-subtle">
        <span className="tabular-nums">
          {um ? `${um.x.toFixed(2)}, ${um.y.toFixed(2)} µm` : "—"}
        </span>
        <span className="hidden sm:inline tabular-nums">{cam.scale.toFixed(2)} px/µm</span>
        <span className="hidden min-w-0 truncate md:inline">
          {hover ? hover.k.replace("sky130_fd_sc_hd__", "") : "no cell"}
        </span>
        <span className="ml-auto hidden sm:inline">{formatBytes(layout.gdsBytes)} · GDSII</span>
        <div className="ml-auto flex gap-1 sm:hidden">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setMobilePanel((p) => (p === "layers" ? null : "layers"))}
          >
            Layers
            <ChevronDown className="size-3" />
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setMobilePanel((p) => (p === "stats" ? null : "stats"))}
          >
            Specs
          </Button>
        </div>
      </footer>
      )}

      {view === "floorplan" && mobilePanel && (
        <div className="fixed inset-x-0 bottom-10 z-20 max-h-[50dvh] overflow-y-auto border-t border-border bg-bg-elevated p-3 lg:hidden">
          {mobilePanel === "layers" ? (
            <div>
              {LAYER_KEYS.map((key) => (
                <LayerRow
                  key={key}
                  id={key}
                  on={layers[key]}
                  onToggle={() => setLayers((s) => ({ ...s, [key]: !s[key] }))}
                />
              ))}
              <div className="mt-2">
                {layout.blocks.map((b) => (
                  <button
                    key={b.id}
                    type="button"
                    onClick={() => setHighlight((h) => (h === b.id ? null : b.id))}
                    className={cn(
                      "mb-1 flex w-full items-center justify-between rounded-sm px-2 py-2 text-left text-sm",
                      highlight === b.id ? "bg-accent-dim text-accent" : "text-fg-muted",
                    )}
                  >
                    {b.label}
                    <span className="font-mono">{b.count}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
            <dl className="grid grid-cols-2 gap-2 text-xs">
              <Metric k="Cells" v={m.cells.toLocaleString()} />
              <Metric k="DFF" v={String(m.dff)} />
              <Metric k="Logic" v={`${m.cellAreaUm2.toLocaleString()} µm²`} />
              <Metric k="≡ square" v={`${cellSide.toFixed(1)} µm`} />
              <Metric k="Die" v={`${dieSide.toFixed(0)} × ${dieSide.toFixed(0)} µm`} />
              <Metric k="Util." v={`${(m.utilization * 100).toFixed(1)}%`} />
            </dl>
            <p className="mt-2 text-[11px] leading-relaxed text-fg-subtle">
              {accum}-bit LIF + {accum}-bit reference-column correction, {adc}-bit ADC/DAC.
              Die {dieSide.toFixed(0)} µm.
            </p>
            </>
          )}
        </div>
      )}
    </main>
  );
}

function SectionTitle({
  icon: Icon,
  label,
}: {
  icon?: typeof Layers;
  label: string;
}) {
  return (
    <div className="flex items-center gap-2 px-3 pb-1.5 pt-3">
      {Icon ? <Icon className="size-3.5 text-fg-subtle" strokeWidth={1.75} /> : null}
      <h2 className="text-[11px] font-medium uppercase tracking-[0.14em] text-fg-subtle">{label}</h2>
    </div>
  );
}

function LayerRow({
  id,
  on,
  onToggle,
}: {
  id: string;
  on: boolean;
  onToggle: () => void;
}) {
  const style = LAYER_STYLE[id];
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-xs hover:bg-bg-hover"
    >
      <span
        className="size-2.5 shrink-0 rounded-[2px] border border-border-strong"
        style={{ background: on ? style?.color : "transparent" }}
      />
      <span className={cn("flex-1 font-mono", on ? "text-fg" : "text-fg-subtle line-through")}>
        {style?.label ?? id}
      </span>
    </button>
  );
}

function Metric({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-fg-subtle">{k}</dt>
      <dd className="font-mono tabular-nums text-fg">{v}</dd>
    </div>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border bg-bg-subtle px-2 py-1">
      <p className="text-[10px] uppercase tracking-wider text-fg-subtle">{label}</p>
      <p className="font-mono text-xs tabular-nums text-fg">{value}</p>
    </div>
  );
}
