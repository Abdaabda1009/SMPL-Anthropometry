
def evaluate_mae(gt_measurements,estim_measurements):
    '''
    Compare two sets of measurements - given as dicts - by finding
    the mean absolute error (MAE) of each measurement.
    :param gt_measurement: dict of {measurement:value} pairs
    :param estim_measurements: dict of {measurement:value} pairs

    Returns
    :param errors: dict of {measurement:value} pairs of measurements
                    that are both in gt_measurement and estim_measurements
                    where value corresponds to the mean absoulte error (MAE)
                    in cm
    '''

    MAE = {}

    for m_name, m_value in gt_measurements.items():
        if m_name in estim_measurements.keys():
            error = abs(m_value - estim_measurements[m_name])
            MAE[m_name] = error

    if MAE == {}:
        print("Measurement dicts do not have any matching measurements!")
        print("Returning empty dict!")

    return MAE