#!/usr/bin/env python3
"""Bit-accurate Chip A golden model — mirrors src/lib/virtual-core.ts / RTL."""

ADC_MAX = 63


def to_s16(x: int) -> int:
    u = x & 0xFFFF
    return u - 0x10000 if u & 0x8000 else u


def asr16(v: int, sh: int) -> int:
    s = max(0, min(7, sh))
    return to_s16(v) >> s


def sat6(c: int) -> int:
    if c < 0:
        return 0
    if c > ADC_MAX:
        return ADC_MAX
    return c & ADC_MAX


def step(v, c, i_col, i_ref, v_th, leak_sh, corr_en):
    i_col = max(0, min(ADC_MAX, i_col))
    i_ref = max(0, min(ADC_MAX, i_ref))
    c_next = to_s16(c + i_ref)
    dac = sat6(c_next) if corr_en else 0
    i_corr = i_col - dac
    leaked = asr16(v, leak_sh)
    v_next = to_s16(leaked + i_corr)
    fire = v_next >= v_th
    return (0 if fire else v_next, 0 if fire else c_next, 1 if fire else 0, dac, i_corr, v_next, fire)


def main() -> int:
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("PASS" if cond else "FAIL"), name)
        if not cond:
            fails += 1

    v = c = 0
    spikes, vs = [], []
    for _ in range(5):
        v, c, sp, *_ = step(v, c, 10, 0, 40, 0, True)
        spikes.append(sp)
        vs.append(v)
    check("integrate then spike at Vth=40", spikes == [0, 0, 0, 1, 0] and vs == [10, 20, 30, 0, 10])

    v, c, sp, *_ = step(0, 0, 30, 2, 40, 0, True)
    v, c, sp, *_ = step(v, c, 30, 2, 40, 0, True)
    check("shared reset clears C with Vmem", sp == 1 and v == 0 and c == 0)

    v, c, sp, dac, i_corr, *_ = step(0, 0, 20, 5, 1000, 0, True)
    check("same-cycle DAC uses C+Iref", dac == 5 and i_corr == 15 and v == 15 and c == 5)

    v = c = 0
    for _ in range(3):
        v, c, sp, dac, *_ = step(v, c, 20, 5, 1000, 0, True)
    check("correction subtracts growing DAC", v == 30 and c == 15 and dac == 15)

    v, c, sp, dac, *_ = step(0, 0, 20, 5, 40, 0, False)
    t1 = v == 20 and dac == 0 and sp == 0
    v, c, sp, dac, *_ = step(v, c, 20, 5, 40, 0, False)
    check("corr_en=0 ignores DAC", t1 and sp == 1 and v == 0)

    v = c = 0
    v, c, *_ = step(v, c, 16, 0, 1000, 1, True)
    v, c, *_ = step(v, c, 16, 0, 1000, 1, True)
    v, c, *_ = step(v, c, 16, 0, 1000, 1, True)
    check("lambda=1/2 barrel-shift leak", v == 28)

    v, c, sp, dac, i_corr, *_ = step(0, 60, 20, 10, 1000, 0, True)
    check("DAC saturates at 63", dac == 63 and i_corr == 20 - 63 and c == 70)

    print("fails", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
