import { Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  N_COL,
  resetColumn,
  stepColumn,
  type ColumnState,
} from "@/lib/virtual-core";
import { cn } from "@/lib/utils";

type Sample = { t: number; v: number; c: number; spike: number };
type Sim = { t: number; s: ColumnState; trace: Sample[] };

const INIT: Sim = { t: 0, s: resetColumn(), trace: [{ t: 0, v: 0, c: 0, spike: 0 }] };

export function DatapathPanel() {
  const [iCol, setICol] = useState(20);
  const [iRef, setIRef] = useState(4);
  const [vTh, setVTh] = useState(50);
  const [leakSh, setLeakSh] = useState(1);
  const [corrEn, setCorrEn] = useState(true);
  const [col, setCol] = useState(0);
  const [sim, setSim] = useState<Sim>(INIT);
  const [running, setRunning] = useState(false);
  const inputs = useRef({ iCol, iRef, vTh, leakSh, corrEn });
  inputs.current = { iCol, iRef, vTh, leakSh, corrEn };

  const reset = () => {
    setSim(INIT);
    setRunning(false);
  };

  const step = () => {
    const p = inputs.current;
    setSim((prev) => {
      const n = stepColumn(prev.s, p.iCol, p.iRef, p.vTh, p.leakSh, p.corrEn);
      const t = prev.t + 1;
      return {
        t,
        s: n,
        trace: [...prev.trace.slice(-95), { t, v: n.vMem, c: n.c, spike: n.spike }],
      };
    });
  };

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(step, 140);
    return () => window.clearInterval(id);
  }, [running]);

  const state = sim.s;
  const t = sim.t;
  const trace = sim.trace;
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto bg-bg">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-4 sm:p-6">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Public LIF datapath preview</h2>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-fg-muted">
            Standard LIF datapath model for Chip A, one column. Golden vectors, the interface
            contract, and validation evidence are public; the correction mechanism is withheld.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Knob label="Icol ADC" value={iCol} min={0} max={63} onChange={setICol} />
          <Knob label="Iref ADC" value={iRef} min={0} max={63} onChange={setIRef} />
          <Knob label="Vth" value={vTh} min={1} max={200} onChange={setVTh} />
          <Knob label="leak_sh  (λ = 2⁻ᵏ)" value={leakSh} min={0} max={7} onChange={setLeakSh} />
          <div className="rounded-md border border-border bg-bg-elevated p-3">
            <p className="text-xs text-fg-subtle">correction trace  §3.2</p>
            <button
              type="button"
              onClick={() => setCorrEn((v) => !v)}
              className={cn(
                "mt-2 h-10 w-full rounded-sm border text-sm font-medium",
                corrEn
                  ? "border-border-strong bg-accent-dim text-accent"
                  : "border-border bg-bg-subtle text-fg-muted",
              )}
            >
              {corrEn ? "public preview on" : "public preview off"}
            </button>
          </div>
          <div className="rounded-md border border-border bg-bg-elevated p-3">
            <p className="text-xs text-fg-subtle">column</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {Array.from({ length: N_COL }, (_, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setCol(i)}
                  className={cn(
                    "size-8 rounded-xs border text-xs font-mono",
                    col === i
                      ? "border-border-strong bg-bg-hover text-fg"
                      : "border-border text-fg-subtle",
                  )}
                >
                  {i}
                </button>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-fg-subtle">The public preview shows LIF behavior only.</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={step}>
            <StepForward className="size-3.5" />
            Clock
          </Button>
          <Button size="sm" variant="secondary" onClick={() => setRunning((r) => !r)}>
            {running ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            {running ? "Pause" : "Run"}
          </Button>
          <Button size="sm" variant="ghost" onClick={reset}>
            <RotateCcw className="size-3.5" />
            Reset
          </Button>
          <span className="ml-auto font-mono text-xs text-fg-subtle">t = {t}</span>
        </div>

        <Equation state={state} iCol={iCol} leakSh={leakSh} vTh={vTh} corrEn={corrEn} />

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <Stat k="Vmem" v={state.vMem} />
          <Stat k="C" v={state.c} />
          <Stat k="DAC" v={state.dac} />
          <Stat k="Icorr" v={state.iCorr} />
          <Stat k="spike" v={state.spike} hot={state.spike === 1} />
        </div>

        <TraceChart samples={trace} vTh={vTh} />
      </div>
    </div>
  );
}

function Knob({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (n: number) => void;
}) {
  return (
    <label className="rounded-md border border-border bg-bg-elevated p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs text-fg-subtle">{label}</span>
        <span className="font-mono text-sm tabular-nums text-fg">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-3 w-full accent-accent"
      />
    </label>
  );
}

function Stat({ k, v, hot }: { k: string; v: number; hot?: boolean }) {
  return (
    <div className="rounded-md border border-border bg-bg-elevated px-3 py-2.5">
      <p className="text-[11px] uppercase tracking-wider text-fg-subtle">{k}</p>
      <p className={cn("font-mono text-lg tabular-nums", hot ? "text-accent" : "text-fg")}>{v}</p>
    </div>
  );
}

function Equation({
  state,
  iCol,
  leakSh,
  vTh,
  corrEn,
}: {
  state: ColumnState;
  iCol: number;
  leakSh: number;
  vTh: number;
  corrEn: boolean;
}) {
  return (
    <div className="rounded-md border border-border bg-bg-elevated p-3 font-mono text-xs leading-relaxed text-fg-muted">
      <p>
        λ = 2<sup>−{leakSh}</sup>
        {"  "}Vmem' = (Vmem {'>>>'} {leakSh}) + ({iCol} − {corrEn ? state.dac : 0}) = {state.vNext}
      </p>
      <p className="mt-1">
        Public correction trace = withheld; DAC output is redacted in this preview
        {"   "}fire = (Vmem' ≥ {vTh}) = {state.fire ? "1 → reset" : "0"}
      </p>
    </div>
  );
}

function TraceChart({ samples, vTh }: { samples: Sample[]; vTh: number }) {
  const ref = useRef<SVGSVGElement>(null);
  const path = useMemo(() => {
    if (samples.length < 2) return "";
    const maxV = Math.max(vTh, ...samples.map((s) => Math.abs(s.v)), 1);
    const w = 640;
    const h = 140;
    const pad = 8;
    return samples
      .map((s, i) => {
        const x = pad + (i / (samples.length - 1)) * (w - pad * 2);
        const y = h - pad - ((s.v + maxV) / (2 * maxV)) * (h - pad * 2);
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [samples, vTh]);

  const spikes = samples.filter((s) => s.spike);

  return (
    <div className="rounded-md border border-border bg-bg-elevated p-3">
      <p className="mb-2 text-xs text-fg-subtle">Vmem trajectory</p>
      <svg ref={ref} viewBox="0 0 640 140" className="h-36 w-full text-accent" role="img" aria-label="Vmem trace">
        <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" />
        {spikes.map((s) => {
          const i = samples.indexOf(s);
          const x = 8 + (i / Math.max(samples.length - 1, 1)) * 624;
          return <circle key={s.t} cx={x} cy={18} r="3" className="fill-fg" />;
        })}
      </svg>
      <p className="mt-1 text-[11px] text-fg-subtle">Dots mark spike + shared reset of Vmem and C.</p>
    </div>
  );
}
