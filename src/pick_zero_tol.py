"""STEP 0 -- report the small end of the |value| distribution.

Reports only. Chooses nothing, writes nothing, hardcodes nothing.
The author reads this and picks the cutoff.
"""

import pandas as pd

from checks_common import load

print("=" * 72)
print("STEP 0 -- ZERO THRESHOLD EVIDENCE (report only)")
print("=" * 72)
_, _, vars_, _ = load()

a = vars_["value"].abs().sort_values().reset_index(drop=True)
n_zero = int((a == 0.0).sum())
pos = a[a > 0.0]

print("\n" + "-" * 72)
print("COUNTS")
print("-" * 72)
print(f"  variables (OPTIMAL instances)        {len(a)}")
print(f"  |value| exactly 0.0                  {n_zero}")
print(f"  |value| strictly positive            {len(pos)}")

print("\n" + "-" * 72)
print("SORTED UNIQUE |value| BELOW 1.0")
print("-" * 72)
below = sorted(set(a[a < 1.0]))
print(f"  distinct values below 1.0: {len(below)}")
counts = a[a < 1.0].value_counts().to_dict()
for u in below:
    print(f"    {u!r:<26} count={counts[u]}")

print("\n" + "-" * 72)
print("GAP ABOVE ZERO")
print("-" * 72)
if len(pos):
    smallest = pos.iloc[0]
    print(f"  smallest strictly positive |value|   {smallest!r}")
    print(f"  gap (0, smallest)                    width = {smallest!r}")
    print(f"  any cutoff strictly inside (0.0, {smallest!r}) separates "
          f"the {n_zero} zeros from every positive value")
else:
    print("  no strictly positive values")

print("\n" + "-" * 72)
print("LARGEST EMPTY GAP BETWEEN CONSECUTIVE DISTINCT |value|, BELOW 1.0")
print("-" * 72)
uniq = sorted(set(a[a < 1.0])) + ([float(a[a >= 1.0].min())] if (a >= 1.0).any() else [])
gaps = [(uniq[i + 1] - uniq[i], uniq[i], uniq[i + 1]) for i in range(len(uniq) - 1)]
gaps.sort(reverse=True)
for w, lo, hi in gaps[:10]:
    print(f"    width={w!r:<26} between {lo!r} and {hi!r}")

print("\n" + "-" * 72)
print("HOW MANY VARIABLES WOULD FALL ON EACH SIDE OF A CANDIDATE CUTOFF")
print("-" * 72)
print(f"  {'cutoff':<14}{'<= cutoff (zero)':>20}{'> cutoff (positive)':>22}")
for c in [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    print(f"  {c:<14g}{int((a <= c).sum()):>20}{int((a > c).sum()):>22}")

print("\n  NO THRESHOLD SELECTED. Waiting for the number.")
