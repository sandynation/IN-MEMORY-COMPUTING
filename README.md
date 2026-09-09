# NEW GEN AI Chip Lab

A research-to-hardware portfolio by [sandynation](https://github.com/sandynation).

This repository collects selected work across the path from AI algorithm to architecture, RTL, verification, FPGA-oriented tooling, and ASIC exploration.

## Research Tracks

### 01 · Chip A Virtual Core

A bit-accurate digital correction core for a virtual in-memory computing tile. Includes the LIF datapath model, reference-column correction logic, golden vectors, RTL, and SKY130 visualization artifacts.

- Scope: digital correction core
- Status: simulation and synthesis complete; not fabricated
- Open work: post-silicon measurement, full mixed-signal Chip B, and validation beyond 32×32 analog arrays

### 02 · EDA and RTL Verification

Selected RTL evaluation, Verilog/SystemVerilog test material, synthesis experiments, and OpenROAD-oriented flow work.

### 03 · AI Accelerator Research

Selected quantization, model-efficiency, benchmarking, and hardware-aware research scripts from the CodexTest research workspace.

### 04 · Design Tooling

A compact selection from the EDA tool and silicon-design-suite work, focused on architecture exploration and reproducible analysis rather than packaged runtimes.

## Honest Status

- **Post-silicon measurement:** The chip has not been fabricated or probed. Results are simulation and synthesis until measured silicon confirms the sneak-path correction.
- **Chip B:** The RRAM crossbar, column ADC, and voltage-isolation circuits are not tape-out ready. Chip A is the digital correction core; Chip B requires further analog design work.
- **Large-array validation:** Available analog-array simulation covers 16×16 and 32×32. The reported 30% β reduction from 16×16 to 32×32 is encouraging, but 128×128 or larger tiles remain unverified.

## Reproduction

Each track contains its own source notes and entry points. Generated builds, credentials, virtual environments, caches, packaged binaries, and private datasets are intentionally excluded.

## Direction

`Frontier AI Algorithm → Architecture → RTL → Verification → FPGA Prototype → Benchmark → ASIC → Post-von-Neumann AI Chip`
