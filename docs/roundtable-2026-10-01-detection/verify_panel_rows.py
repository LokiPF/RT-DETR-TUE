"""Independent re-implementation of the panel's two leading rows, scored only on positions 1059-1969.

Spec (panelist A's final proposal): at s1-s3, level m = mean |a| per channel, shape pi = log(t + 1e-6) - log(m + 1e-6)
with t the mean of the top 1% of positions. Neighbours: the 50 bank images nearest in standardised stage-4 means.
AC: per stage mean_c |pi - mean_N pi| / sd_bank(pi), z-scored on the 500 z-statistics images, summed over s1-s3.
M1: max(signed flatter arm, level arm), each arm's sum re-z-scored on the z-statistics images, where the flatter arm is
mean_c -(pi - mean_N pi) / sd and the level arm is Section 4 (mean_c |m - mean_N m| / sd).
Positions 1970 and later are never loaded.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/convtu-pilot")
from differential_uncertainty.baselines import protocol  # noqa: E402
from differential_uncertainty.baselines.activation_cdf import stage_zstats, zscored_sum  # noqa: E402
from differential_uncertainty.convtu import conditioned, confirmation  # noqa: E402
from differential_uncertainty.convtu.channels import fit_own_average  # noqa: E402

RUN = Path("/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines")
FIRST, LAST = 1059, 1970
names = [p.stem for p in protocol.evaluation_images(Path("/home/yuchen/YuchenZ/Datasets/coco/val2017"), seed=44)]
chosen = names[FIRST:LAST]
keys = [f"{s}_s{l}" for s in ("means", "top") for l in range(1, 5)]
test = {k: [] for k in keys}
for name in chosen:
    with np.load(RUN / "test_convtu_means" / f"{name}.npz") as item:
        for k in keys:
            test[k].append(item[k])
test = {k: np.stack(v).astype(np.float64) for k, v in test.items()}
with np.load(RUN / "convtu" / "channels_bank.npz") as b, np.load(RUN / "convtu" / "channels_zstats.npz") as z:
    bank = {k: b[k].astype(np.float64) for k in keys}
    zst = {k: z[k].astype(np.float64) for k in keys}
EPS = 1e-6
SCORED = ("s1", "s2", "s3")


def shape(values, layer):
    return np.log(values[f"top_{layer}"] + EPS) - np.log(values[f"means_{layer}"] + EPS)


centre, spread = fit_own_average(bank["means_s4"])
bank_keys = (bank["means_s4"] - centre) / spread


def arms(values):
    """Per-stage shape (two-sided) and flatter (signed) deviations for rows of `values` (rows, C)."""
    q = (values["means_s4"] - centre) / spread
    nb = conditioned.nearest_rows(q, bank_keys, 50)
    two, signed = [], []
    for layer in SCORED:
        pi_bank = shape(bank, layer)
        sd = fit_own_average(pi_bank)[1]
        dev = (shape(values, layer) - pi_bank[nb].mean(axis=1)) / sd
        two.append(np.abs(dev).mean(axis=1))
        signed.append((-dev).mean(axis=1))
    return np.stack(two, 1), np.stack(signed, 1)


flat = {k: v.reshape(-1, v.shape[-1]) for k, v in test.items()}
two_t, signed_t = arms(flat)
two_z, signed_z = arms(zst)
ac = zscored_sum(two_t, *stage_zstats(two_z)).reshape(len(chosen), 96)
flatter_t, flatter_z = zscored_sum(signed_t, *stage_zstats(signed_z)), zscored_sum(signed_z, *stage_zstats(signed_z))
level_t = conditioned.conditioned_scores({k: flat[k] for k in flat if k.startswith("means")},
                                         {k: bank[k] for k in bank if k.startswith("means")},
                                         {k: zst[k] for k in zst if k.startswith("means")})[0]
level_z = conditioned.conditioned_scores({k: zst[k] for k in zst if k.startswith("means")},
                                         {k: bank[k] for k in bank if k.startswith("means")},
                                         {k: zst[k] for k in zst if k.startswith("means")})[0]


def rez(t, z):
    return (t - z.mean()) / z.std()


m1 = np.maximum(rez(flatter_t, flatter_z), rez(level_t, level_z)).reshape(len(chosen), 96)
section4 = level_t.reshape(len(chosen), 96)
cdf = np.load(Path(__file__).parent / "data" / "coco5000.npz")["cdf_z"][FIRST:LAST]
rows = np.arange(len(chosen))
sev1 = [c for c, (f, s) in enumerate(protocol.CONDITIONS) if s == 1 and f in protocol.COMMON_FAMILIES]
from differential_uncertainty.baselines import metrics  # noqa: E402
print(f"positions {FIRST}-{LAST - 1} ({len(chosen)} images): AUROC common / extra / severity-1 common")
for label, score in (("AC (two-sided peak share)", ac), ("M1 (max of flatter and level arms)", m1),
                     ("Section 4 (level)", section4), ("Activation CDFs", cdf)):
    common, extra = confirmation.group_aurocs(score, rows)
    s1 = metrics.condition_aurocs(score[:, 0], score[:, sev1].T).mean()
    print(f"  {label:36s} {common:.3f} / {extra:.3f} / {s1:.3f}")
