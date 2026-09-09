import { Check, CircleAlert, LockKeyhole } from "lucide-react";
import { cn } from "@/lib/utils";

const calibreRunset = `LAYOUT PATH    "virtual_core_chip_a.gds"
LAYOUT PRIMARY "virtual_core_chip_a"
LAYOUT SYSTEM  GDSII
DRC RESULTS DATABASE "chip_a.drc.db"
DRC SUMMARY REPORT   "chip_a.drc.rep"
INCLUDE "$PDK_CALIBRE/sky130.drc"`;

const windowsCommand = `call %USERPROFILE%\\eda\\oss-cad-suite\\start.bat
iverilog -g2012 -o tb.vvp vc_unit1.v vc_unit2.v virtual_core_chip_a.v tb_chip_a.v
vvp tb.vvp`;

const wslCommand = `wsl --install -d Ubuntu-22.04
# In Ubuntu, run the WSL/SKY130 install kit, then use OpenLane or mpw_precheck.`;

const checks = [
  { label: "RTL golden model", status: "pass", detail: "7/7 vectors pass for the archived Chip A model." },
  { label: "Magic DRC / KLayout DRC", status: "ready", detail: "Run with the official SKY130A technology and shuttle precheck deck." },
  { label: "Netgen LVS", status: "ready", detail: "Requires a real extracted netlist from a real standard-cell GDS stream." },
  { label: "Calibre nmDRC / nmLVS", status: "blocked", detail: "Requires a Siemens license and the SkyWater/ChipFoundry NDA rule deck." },
  { label: "Shuttle acceptance", status: "blocked", detail: "Requires mpw_precheck and integration into the shuttle pad/frame flow." },
] as const;

export function DrcFlow() {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto bg-bg">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-5 p-4 sm:p-6">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Verification flow</h2>
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-fg-muted">
            Chip A has an architectural golden model and a SKY130-style visualization GDS. Physical
            sign-off is a separate flow and is not reported as complete here.
          </p>
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          {checks.map((check) => (
            <div key={check.label} className="rounded-md border border-border bg-bg-elevated p-3">
              <div className="flex items-center gap-2">
                <StatusIcon status={check.status} />
                <p className="text-sm text-fg">{check.label}</p>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-fg-muted">{check.detail}</p>
            </div>
          ))}
        </div>

        <section className="rounded-md border border-border bg-bg-subtle p-3">
          <h3 className="text-sm font-medium text-fg">Open claims</h3>
          <div className="mt-2 grid gap-3 text-xs leading-relaxed text-fg-muted">
            <p>
              <span className="font-medium text-fg">Post-silicon measurement.</span> The chip has
              not been fabricated or probed. All results shown here are simulation and synthesis;
              the sneak-path correction still needs measured silicon data.
            </p>
            <p>
              <span className="font-medium text-fg">Chip B mixed-signal system.</span> The RRAM
              crossbar, column ADC, and voltage-isolation circuits are not tape-out ready. Chip A
              is the digital correction core; Chip B needs an estimated 4–6 more weeks of analog
              design work.
            </p>
            <p>
              <span className="font-medium text-fg">Large-array validation.</span> The current
              simulation covers 16×16 and 32×32. 128×128 or larger tiles have not been verified;
              the 30% β reduction from 16×16 to 32×32 is suggestive, not proof of scaling.
            </p>
          </div>
        </section>

        <section>
          <h3 className="text-[11px] font-medium uppercase tracking-[0.14em] text-fg-subtle">
            Calibre nmDRC runset template
          </h3>
          <pre className="mt-2 overflow-x-auto rounded-md border border-border bg-bg-elevated p-3 font-mono text-xs leading-relaxed text-fg-muted">
            {calibreRunset}
          </pre>
          <p className="mt-2 text-xs leading-relaxed text-fg-subtle">
            Execute with <span className="font-mono text-fg-muted">calibre -drc -hier chip_a.rs</span>{" "}
            only after the NDA deck, license, and real library GDS are available.
          </p>
        </section>

        <section className="rounded-md border border-border bg-bg-subtle p-3">
          <h3 className="text-sm font-medium text-fg">Open-tool path</h3>
          <p className="mt-1 text-xs leading-relaxed text-fg-muted">
            For chipIgnite-style acceptance, run OpenLane/OpenROAD with real SKY130 HD library
            geometry, then Magic DRC, KLayout DRC, Netgen LVS, and mpw_precheck. These checks are
            the actionable local gate; they do not convert this visualization stream into a Calibre
            sign-off layout.
          </p>
        </section>

        <div className="grid gap-2 sm:grid-cols-2">
          <section className="rounded-md border border-border bg-bg-elevated p-3">
            <h3 className="text-sm font-medium text-fg">Windows tools</h3>
            <p className="mt-1 text-xs leading-relaxed text-fg-muted">
              Install KLayout for viewing and OSS CAD Suite for Yosys/Icarus. Docker Desktop is
              needed to launch OpenLane.
            </p>
            <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs">
              <a className="text-accent underline-offset-2 hover:underline" href="https://www.klayout.org/downloads/Windows/klayout-0.30.12-win64-install.exe">KLayout installer</a>
              <a className="text-accent underline-offset-2 hover:underline" href="https://github.com/YosysHQ/oss-cad-suite-build/releases/latest">OSS CAD Suite</a>
              <a className="text-accent underline-offset-2 hover:underline" href="https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe">Docker Desktop</a>
            </div>
            <pre className="mt-3 overflow-x-auto rounded-sm border border-border bg-bg-subtle p-2 font-mono text-[11px] leading-relaxed text-fg-muted">{windowsCommand}</pre>
          </section>

          <section className="rounded-md border border-border bg-bg-elevated p-3">
            <h3 className="text-sm font-medium text-fg">WSL2 + Docker tools</h3>
            <p className="mt-1 text-xs leading-relaxed text-fg-muted">
              Magic, OpenROAD, Netgen, and mpw_precheck run in Linux containers or WSL2. Budget
              roughly 15–20 GB of disk and 16 GB RAM for the PDK and images.
            </p>
            <pre className="mt-3 overflow-x-auto rounded-sm border border-border bg-bg-subtle p-2 font-mono text-[11px] leading-relaxed text-fg-muted">{wslCommand}</pre>
          </section>
        </div>
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: "pass" | "ready" | "blocked" }) {
  const Icon = status === "pass" ? Check : status === "blocked" ? LockKeyhole : CircleAlert;
  return (
    <span
      className={cn(
        "flex size-5 items-center justify-center rounded-full border",
        status === "pass" && "border-border-strong bg-accent-dim text-accent",
        status === "ready" && "border-border bg-bg-subtle text-warn",
        status === "blocked" && "border-border bg-bg-subtle text-danger",
      )}
      aria-label={status}
    >
      <Icon className="size-3" strokeWidth={1.8} />
    </span>
  );
}