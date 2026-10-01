"""measuring.py — measure a body directly from SMPL/SMPLX shape parameters
(betas); no mesh, no fitting. Shared by /measure/shape and by fit_worker.py
(which measures the betas a fit converged to)."""
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS


def from_betas(model_type: str, gender: str, betas, height_cm: float | None = None):
    """Run the full measurement pass on a body defined by `betas`.

    Returns (measurements, labeled, height_normalized); the last is None
    unless height_cm was given.
    """
    measurer = MeasureBody(model_type)
    measurer.from_body_model(gender=gender, shape=betas)
    measurer.measure(measurer.all_possible_measurements)
    measurer.label_measurements(STANDARD_LABELS)

    height_normalized = None
    if height_cm is not None:
        measurer.height_normalize_measurements(height_cm)
        height_normalized = dict(measurer.height_normalized_measurements)
    return (dict(measurer.measurements), dict(measurer.labeled_measurements),
            height_normalized)
