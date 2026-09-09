# Chip A Virtual Core

Public evidence package for a post-von-Neumann in-memory computing tile.

## Included

- Standard LIF datapath model and golden vectors
- Public RTL interface contract and testbench
- SKY130-oriented layout visualization
- Verification and DRC-flow notes

## Status

Chip A is a digital correction core validated through simulation and synthesis. It has not been fabricated or measured. The mechanism-bearing correction RTL is intentionally withheld; this public package exposes the interface, testbench, golden vectors, and presentation artifacts. The mixed-signal RRAM crossbar, column ADC, voltage-isolation circuits, and 128x128+ array validation remain open work.

## Public/private boundary

The public materials demonstrate implementation and verification ability without publishing the activity-weighted correction mechanism. The full correction implementation can be shared privately for an appropriate technical review.
