 NEW GEN AI Chip Lab

A research-to-hardware portfolio by sandynation.

Selected work moving from an AI hardware problem to architecture, RTL, and synthesis — with every claim checked against simulation or synthesis output, not stated as fact.

Research Tracks
01 · Chip A Virtual Core

A bit-accurate digital correction core for a virtual in-memory computing tile, addressing sneak-path interference in passive crossbar arrays. Includes the LIF datapath model, reference-column correction logic, golden vectors, RTL, and SKY130 visualization artifacts.

Scope: digital correction core
Status: simulation and synthesis complete; not fabricated
Open work: post-silicon measurement, full mixed-signal Chip B, and validation beyond 32×32 analog arrays
02 · EDA and RTL Verification

RTL evaluation, Verilog/SystemVerilog test material, synthesis experiments, and OpenROAD-oriented flow work supporting the Chip A correction core and related digital design exercises.

03 · AI Accelerator Research

Quantization, model-efficiency, benchmarking, and hardware-aware research scripts, exploring how algorithmic choices map onto hardware cost.

04 · Design Tooling

Architecture-exploration and reproducible-analysis scripts drawn from broader EDA tool and silicon-design-suite work — not a packaged runtime, and not intended as one.

Honest Status
Post-silicon measurement: The chip has not been fabricated or probed. Results are simulation and synthesis until measured silicon confirms the sneak-path correction.
Chip B: The RRAM crossbar, column ADC, and voltage-isolation circuits are not tape-out ready. Chip A is the digital correction core; Chip B requires further analog design work.
Large-array validation: Available analog-array simulation covers 16×16 and 32×32. The reported 30% β reduction from 16×16 to 32×32 is encouraging, but 128×128 or larger tiles remain unverified.
Reproduction

Each track contains its own source notes and entry points. Generated builds, credentials, virtual environments, caches, packaged binaries, and private datasets are intentionally excluded.

Direction

Algorithm insight → Correction architecture → RTL implementation → Synthesis & verification

FPGA prototyping and ASIC tape-out are future direction, not current status.
