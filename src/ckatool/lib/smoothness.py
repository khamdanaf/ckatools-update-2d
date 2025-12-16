"""
smoothness.py contains a list of functions for estimating movement smoothness.
"""

import numpy as np


def sampling_frequency_from_timestamp(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.size < 2:
        return float("nan")

    diffs = np.diff(timestamps)
    # buang dt yang tidak valid / nol / negatif
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")

    mean_dt = np.mean(diffs)
    if (not np.isfinite(mean_dt)) or mean_dt <= 0:
        return float("nan")

    return 1.0 / mean_dt



def sparc(speed_profile, fs, padlevel=4, fc=10.0, amp_th=0.05):
    speed_profile = np.asarray(speed_profile, dtype=float)

    # Guard: data terlalu pendek / fs tidak valid
    if speed_profile.size < 2 or (not np.isfinite(fs)) or fs <= 0:
        empty = (np.array([]), np.array([]))
        return float("nan"), empty, empty

    # Jika semua nol/flat, SPARC tidak bermakna
    if np.nanmax(np.abs(speed_profile)) <= 0:
        empty = (np.array([]), np.array([]))
        return float("nan"), empty, empty

    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(speed_profile))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)

    # Magnitude spectrum
    Mf = np.abs(np.fft.fft(speed_profile, nfft))
    den = np.nanmax(Mf) if Mf.size else 0.0
    if (not np.isfinite(den)) or den <= 0:
        return float("nan"), (f, Mf), (np.array([]), np.array([]))
    Mf = Mf / den

    # Low-pass selection
    fc_mask = f <= fc
    f_sel = f[fc_mask]
    Mf_sel = Mf[fc_mask]
    if f_sel.size < 2:
        return float("nan"), (f, Mf), (f_sel, Mf_sel)

    # Threshold selection
    inx = np.nonzero(Mf_sel >= amp_th)[0]
    if inx.size == 0:
        return float("nan"), (f, Mf), (f_sel, Mf_sel)

    f_sel = f_sel[inx[0] : inx[-1] + 1]
    Mf_sel = Mf_sel[inx[0] : inx[-1] + 1]
    if f_sel.size < 2:
        return float("nan"), (f, Mf), (f_sel, Mf_sel)

    denom = (f_sel[-1] - f_sel[0])
    if denom <= 0 or (not np.isfinite(denom)):
        return float("nan"), (f, Mf), (f_sel, Mf_sel)

    new_sal = -np.sum(
        np.sqrt((np.diff(f_sel) / denom) ** 2 + (np.diff(Mf_sel)) ** 2)
    )
    return new_sal, (f, Mf), (f_sel, Mf_sel)


def dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the
    dimensionless jerk metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.

    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.

    Notes
    -----


    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'

    """
    # first enforce data into an numpy array.
    movement = np.array(movement)

    # calculate the scale factor and jerk.
    movement_peak = max(abs(movement))
    dt = 1.0 / fs
    movement_dur = len(movement) * dt
    jerk = np.diff(movement, 2) / pow(dt, 2)
    scale = pow(movement_dur, 3) / pow(movement_peak, 2)

    # estimate dj
    return -scale * sum(pow(jerk, 2)) * dt


def log_dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the
    log dimensionless jerk metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.

    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.

    Notes
    -----


    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> ldl = log_dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'

    """
    return -np.log(abs(dimensionless_jerk(movement, fs)))


def dimensionless_jerk2(movement, fs, data_type="speed"):
    """
    Calculates the smoothness metric for the given movement data using the
    dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}

    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.

    Notes
    -----


    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'

    """
    # first ensure the movement type is valid.
    if data_type in ("speed", "accl", "jerk"):
        # first enforce data into an numpy array.
        movement = np.array(movement)

        # calculate the scale factor and jerk.
        movement_peak = max(abs(movement))
        dt = 1.0 / fs
        movement_dur = len(movement) * dt
        # get scaling factor:
        _p = {"speed": 3, "accl": 1, "jerk": -1}
        p = _p[data_type]
        scale = pow(movement_dur, p) / pow(movement_peak, 2)

        # estimate jerk
        if data_type == "speed":
            jerk = np.diff(movement, 2) / pow(dt, 2)
        elif data_type == "accl":
            jerk = np.diff(movement, 1) / pow(dt, 1)
        else:
            jerk = movement

        # estimate dj
        return -scale * sum(pow(jerk, 2)) * dt
    else:
        raise ValueError("\n".join(("The argument data_type must be either", "'speed', 'accl' or 'jerk'.")))


def log_dimensionless_jerk2(movement, fs, data_type="speed"):
    """
    Calculates the smoothness metric for the given movement data using the
    log dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}

    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.

    Notes
    -----


    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> ldl = log_dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'

    """
    return -np.log(abs(dimensionless_jerk2(movement, fs, data_type)))
