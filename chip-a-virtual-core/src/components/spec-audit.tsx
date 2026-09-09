import { useMemo } from "react";
import { PROPOSAL_AUDIT, runGoldenChecks } from "@/lib/virtual-core";
import { cn } from "@/lib/utils";

export function SpecAudit() {
  const checks = useMemo(() => runGoldenChecks(), []);
  const nPass = checks.filter((c) => c.pass).length;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto bg-bg">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 p-4 sm:p-6">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Proposal checklist</h2>
          <p className="mt-1 text-sm leading-relaxed text-fg-muted">
            Chip A against “A Virtual Core-Embedded In-Memory Computing Tile” (Pal, July 2026).
            Architecture is the digital core only — TIA, SAR ADC, and the resistor crossbar are Chip B.
          </p>
        </div>

        <div className="rounded-md border border-border bg-bg-elevated px-3 py-2.5 text-sm">
          Golden model{" "}
          <span className="font-mono tabular-nums text-fg">
            {nPass}/{checks.length}
          </span>{" "}
          vectors pass.
        </div>

        <ul className="flex flex-col gap-2">
          {PROPOSAL_AUDIT.map((row) => (
            <li key={row.id} className="rounded-md border border-border bg-bg-elevated p-3">
              <div className="flex items-start gap-2">
                <StatusPill status={row.status} />
                <div className="min-w-0 flex-1">
                  <p className="text-sm text-fg">{row.claim}</p>
                  <p className="mt-1 text-xs leading-relaxed text-fg-muted">{row.detail}</p>
                </div>
              </div>
            </li>
          ))}
        </ul>

        <div>
          <h3 className="text-[11px] font-medium uppercase tracking-[0.14em] text-fg-subtle">
            Golden vectors
          </h3>
          <ul className="mt-2 divide-y divide-border rounded-md border border-border bg-bg-elevated">
            {checks.map((c) => (
              <li key={c.name} className="flex items-start justify-between gap-3 px-3 py-2 text-xs">
                <span className="text-fg-muted">{c.name}</span>
                <span className={cn("shrink-0 font-mono", c.pass ? "text-accent" : "text-danger")}>
                  {c.pass ? "pass" : "fail"}
                </span>
              </li>
            ))}
          </ul>
        </div>

        <p className="text-xs leading-relaxed text-fg-subtle">
          The downloadable GDSII is a SKY130-layer floorplan of this RTL (90 µm hard macro, 272
          async-reset DFFs). It is not a Calibre-clean shuttle stream. Use Innovus + Calibre for
          Phase 3 signoff.
        </p>
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: "pass" | "note" | "fail" }) {
  return (
    <span
      className={cn(
        "mt-0.5 shrink-0 rounded-xs border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide",
        status === "pass" && "border-border-strong bg-accent-dim text-accent",
        status === "note" && "border-border bg-bg-subtle text-warn",
        status === "fail" && "border-border bg-bg-subtle text-danger",
      )}
    >
      {status}
    </span>
  );
}
