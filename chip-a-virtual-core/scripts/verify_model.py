#!/usr/bin/env python3
"""Public LIF golden-vector checks for Chip A."""

ADC_MAX = 63


def to_s16(value: int) -> int:
    unsigned = value & 0xFFFF
    return unsigned - 0x10000 if unsigned & 0x8000 else unsigned


def asr16(value: int, shift: int) -> int:
    return to_s16(value) >> max(0, min(7, shift))


def step(v_mem: int, i_col: int, v_th: int, leak_sh: int):
    column = max(0, min(ADC_MAX, i_col))
    v_next = to_s16(asr16(v_mem, leak_sh) + column)
    fire = v_next >= v_th
    return (0 if fire else v_next, 1 if fire else 0, v_next, fire)


def main() -> int:
    failures = 0

    def check(name, condition):
        nonlocal failures
        print(("PASS" if condition else "FAIL"), name)
        if not condition:
            failures += 1

    v_mem = 0
    spikes, values = [], []
    for _ in range(5):
        v_mem, spike, *_ = step(v_mem, 10, 40, 0)
        spikes.append(spike)
        values.append(v_mem)
    check("integrate then spike at Vth=40", spikes == [0, 0, 0, 1, 0] and values == [10, 20, 30, 0, 10])

    v_mem = 0
    for _ in range(3):
        v_mem, *_ = step(v_mem, 16, 1000, 1)
    check("lambda=1/2 barrel-shift leak", v_mem == 28)

    print("failures", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
