"""
test.py — end-to-end demo of SMPL-Anthropometry using the current API
(as documented in README.md).

What this script does:
  1. Builds a body measurer (SMPL or SMPLX).
  2. Measures a body defined by shape `betas`  -> `from_body_model`.
  3. Measures a body defined by `vertices`      -> `from_verts`   (the
     "3D model / scan" workflow).
  4. Prints raw + standard-labeled measurements as tables.
  5. Produces a COMPARISON REPORT (mean absolute error) between two bodies.
  6. Demonstrates height normalization (useful when scale is unknown, e.g.
     a body regressed from an image).

NOTE on images:
  This repository measures an *existing* SMPL/SMPLX body. It does NOT turn a
  2D photo into a body by itself. To measure a person from an image, first run
  an image->SMPL regressor (e.g. SPIN, in the parent folder) to get `betas`
  (and gender) or the fitted `vertices`, then feed those into this script via
  `from_body_model(...)` or `from_verts(...)`.

Run:
  python test.py                 # measure + comparison report (text tables)
  python test.py --
            # also open the 3D measurement viz in browser
"""

import argparse

import numpy as np
import pandas as pd
import torch

from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
from evaluate import evaluate_mae


# ----------------------------------------------------------------------------
# Config — the models you have live in data/smplx/SMPLX_{GENDER}.pkl
# ----------------------------------------------------------------------------
MODEL_TYPE = "smplx"          # "smpl" or "smplx"
GENDER = "NEUTRAL"            # "MALE", "FEMALE" or "NEUTRAL"


def measure_from_betas(model_type, gender, betas):
    """Measure a body defined by shape parameters (betas)."""
    measurer = MeasureBody(model_type)
    measurer.from_body_model(gender=gender, shape=betas)
    measurer.measure(measurer.all_possible_measurements)
    measurer.label_measurements(STANDARD_LABELS)
    return measurer


def measurements_table(measurer):
    """Return a pandas DataFrame of label / name / value(cm)."""
    rows = []
    for label, name in measurer.labels2names.items():
        rows.append({
            "Label": label,
            "Measurement": name,
            "Value (cm)": round(measurer.measurements[name], 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="SMPL-Anthropometry demo / test.")
    parser.add_argument("--visualize", action="store_true",
                        help="Open the interactive 3D measurement visualization in the browser.")
    args = parser.parse_args()

    torch.manual_seed(0)  # reproducible random shapes

    # ------------------------------------------------------------------
    # 1) Measure body A from betas (here: the mean / zero shape)
    # ------------------------------------------------------------------
    betas_A = torch.zeros((1, 10), dtype=torch.float32)
    measurer_A = measure_from_betas(MODEL_TYPE, GENDER, betas_A)

    print(f"\n=== Body A: {MODEL_TYPE.upper()} {GENDER} mean shape ===")
    print(measurements_table(measurer_A).to_string(index=False))

    # ------------------------------------------------------------------
    # 2) Measure body B from betas (a random shape) — this is the "other"
    #    body we will compare against (e.g. a scan vs. a template, or a
    #    prediction vs. ground truth).
    # ------------------------------------------------------------------
    betas_B = torch.empty((1, 10)).normal_(mean=0.0, std=1.0)
    measurer_B = measure_from_betas(MODEL_TYPE, GENDER, betas_B)

    print(f"\n=== Body B: {MODEL_TYPE.upper()} {GENDER} random shape ===")
    print(measurements_table(measurer_B).to_string(index=False))

    # ------------------------------------------------------------------
    # 3) The "3D model / scan" workflow: measure directly from vertices.
    #    Here we reuse body A's own vertices as stand-in scan vertices and
    #    confirm we recover the same measurements. In your project, replace
    #    `verts` with the (N,3) vertices of your fitted SMPL/SMPLX mesh.
    # ------------------------------------------------------------------
    verts = torch.tensor(measurer_A.verts, dtype=torch.float32)
    measurer_V = MeasureBody(MODEL_TYPE)
    measurer_V.from_verts(verts=verts)
    measurer_V.measure(measurer_V.all_possible_measurements)
    verts_vs_betas = evaluate_mae(measurer_A.measurements, measurer_V.measurements)
    print("\n=== Sanity check: from_verts vs from_body_model (should be ~0 cm) ===")
    print(f"max abs diff: {max(verts_vs_betas.values()):.4f} cm")

    # ------------------------------------------------------------------
    # 4) COMPARISON REPORT: mean absolute error between body A and body B
    # ------------------------------------------------------------------
    MAE = evaluate_mae(measurer_A.measurements, measurer_B.measurements)
    report = pd.DataFrame({
        "Measurement": list(MAE.keys()),
        "Body A (cm)": [round(measurer_A.measurements[m], 2) for m in MAE],
        "Body B (cm)": [round(measurer_B.measurements[m], 2) for m in MAE],
        "MAE (cm)": [round(v, 2) for v in MAE.values()],
    }).sort_values("MAE (cm)", ascending=False)

    print("\n=== Comparison report: Body A vs Body B ===")
    print(report.to_string(index=False))
    print(f"\nOverall mean absolute error: {np.mean(list(MAE.values())):.2f} cm")

    # ------------------------------------------------------------------
    # 5) Height normalization (use when the body scale is unknown, e.g.
    #    regressed from an image). Scales all measurements so height == 175.
    # ------------------------------------------------------------------
    new_height = 175.0
    measurer_B.height_normalize_measurements(new_height)
    print(f"\n=== Body B normalized to height = {new_height} cm ===")
    norm = measurer_B.height_normalized_measurements
    print(pd.DataFrame({
        "Measurement": list(norm.keys()),
        "Normalized (cm)": [round(v, 2) for v in norm.values()],
    }).to_string(index=False))

    # ------------------------------------------------------------------
    # 6) Optional interactive 3D visualization (opens in browser)
    # ------------------------------------------------------------------
    if args.visualize:
        print("\nOpening 3D visualization for Body A in your browser...")
        measurer_A.visualize()


if __name__ == "__main__":
    main()
