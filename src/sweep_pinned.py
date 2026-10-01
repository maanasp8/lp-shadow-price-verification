"""Sweep the pinned populations: AT_UPPER, and AT_LOWER sitting at a nonzero floor.

Clean models only, no mutants. Reads checkA_results.csv and tabulates the
OBSERVED relation between sum and obj_coeff -- equality vs strict inequality --
independently of which relation Check A required.

Observed relation uses the same frozen band as the fire test:
    EQ  if |sum - obj_coeff| <= REL_TOL * max(1, |obj_coeff|)
    GT  if sum >  obj_coeff + band
    LT  if sum <  obj_coeff - band
"""

import pandas as pd

from checks_common import REL_TOL, ZERO_TOL

df = pd.read_csv("data/checkA_results.csv", dtype={"problem_id": str})
objs = pd.read_csv("data/objectives.csv", dtype={"problem_id": str})[["problem_id", "sense"]]
df = df.merge(objs, on="problem_id", how="left")

band = REL_TOL * df["obj_coeff"].abs().clip(lower=1.0)
df["observed"] = "EQ"
df.loc[df["sum"] > df["obj_coeff"] + band, "observed"] = "GT"
df.loc[df["sum"] < df["obj_coeff"] - band, "observed"] = "LT"
df["at_zero"] = df["value"].abs() <= ZERO_TOL

# populations
df["population"] = "other"
df.loc[df.status == "AT_UPPER", "population"] = "AT_UPPER"
df.loc[(df.status == "AT_LOWER") & (~df.at_zero), "population"] = "AT_LOWER nonzero floor"
df.loc[(df.status == "AT_LOWER") & (df.at_zero), "population"] = "AT_LOWER at zero"
df.loc[df.status == "FREE", "population"] = "FREE"

print("=" * 78)
print("PINNED-POPULATION SWEEP  (clean models only)")
print("=" * 78)
print(f"  ZERO_TOL = {ZERO_TOL:g}   REL_TOL = {REL_TOL:g}")
print(f"  variable rows: {len(df)}")

print("\n" + "-" * 78)
print("POPULATION SIZES")
print("-" * 78)
for pop, n in df.population.value_counts().items():
    print(f"  {pop:<26} {n}")

TARGETS = ["AT_UPPER", "AT_LOWER nonzero floor"]

print("\n" + "-" * 78)
print("OBSERVED RELATION (sum vs obj_coeff), THE TWO REQUESTED POPULATIONS")
print("-" * 78)
print(f"  {'population':<26}{'n':>6}{'EQ':>8}{'GT':>8}{'LT':>8}{'branch=positive':>18}")
for pop in TARGETS:
    s = df[df.population == pop]
    o = s.observed.value_counts()
    print(f"  {pop:<26}{len(s):>6}{o.get('EQ', 0):>8}{o.get('GT', 0):>8}"
          f"{o.get('LT', 0):>8}{int((s.branch == 'positive').sum()):>18}")

print("\n" + "-" * 78)
print("SAME, SPLIT BY SENSE")
print("-" * 78)
print(f"  {'population':<26}{'sense':>6}{'n':>6}{'EQ':>8}{'GT':>8}{'LT':>8}")
for pop in TARGETS:
    for sense in ("max", "min"):
        s = df[(df.population == pop) & (df.sense == sense)]
        if not len(s):
            continue
        o = s.observed.value_counts()
        print(f"  {pop:<26}{sense:>6}{len(s):>6}{o.get('EQ', 0):>8}"
              f"{o.get('GT', 0):>8}{o.get('LT', 0):>8}")

print("\n" + "-" * 78)
print("FULL TABLE -- every population, for contrast")
print("-" * 78)
print(f"  {'population':<26}{'n':>6}{'EQ':>8}{'GT':>8}{'LT':>8}{'fired':>8}")
for pop in ["FREE", "AT_UPPER", "AT_LOWER nonzero floor", "AT_LOWER at zero"]:
    s = df[df.population == pop]
    o = s.observed.value_counts()
    print(f"  {pop:<26}{len(s):>6}{o.get('EQ', 0):>8}{o.get('GT', 0):>8}"
          f"{o.get('LT', 0):>8}{int(s.fired.sum()):>8}")

print("\n" + "-" * 78)
print("BRANCH ASSIGNMENT CHECK  (which Check A branch each population took)")
print("-" * 78)
print(df.groupby(["population", "branch"]).size().to_string())

print("\n" + "-" * 78)
print("STRICT-INEQUALITY ROWS IN THE TWO TARGET POPULATIONS")
print("-" * 78)
strict = df[(df.population.isin(TARGETS)) & (df.observed != "EQ")]
print(f"  count: {len(strict)}")
for r in strict.head(30).to_dict("records"):
    print(f"    {r['problem_id']:>13} {r['variable']:<22} {r['population']:<24} "
          f"{r['sense']:>3} obs={r['observed']} value={r['value']!r} "
          f"sum={r['sum']!r} obj_coeff={r['obj_coeff']!r}")
