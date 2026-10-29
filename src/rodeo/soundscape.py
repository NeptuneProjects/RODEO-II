"""Functions for computing soundscape metrics from acoustic signals.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.stats import kurtosis

from rodeo.detect import Detection

logger = logging.getLogger(__name__)

REF_PRESSURE = 1.0  # (uPa)


@dataclass
class Metrics:
    """Dataclass for storing soundscape metrics.

    Attributes:
        index: Index of the detection in the database.
        uid: Unique identifier for the detection (not typically used).
        station: Station name.
        channel: Channel number.
        time_init: Start time of the detection.
        time_end: End time of the detection.
        unit: Unit of amplitude for the time series.
        SPL_rms: Root mean square sound pressure level in dB re 1 uPa.
        SPL_pk: Zero-to-peak sound pressure level in dB re 1 uPa.
        SPL_pkpk: Peak-to-peak sound pressure level in dB re 1 uPa.
        SPL_rms90: RMS sound pressure level over the 90% energy interval
            (T90) in dB re 1 uPa.
        SEL: Single-strike sound exposure level in dB re 1 uPa^2 s.
        t_rise: Rise time (s).
        t_decay: Decay time (s).
        kurtosis: Pearson kurtosis (beta_2; 3 for Gaussian noise).
        SEL_cum: Cumulative sound exposure level in dB re 1 uPa^2 s.
        metadata: Optional metadata for the detection.
    """

    index: int
    uid: str
    station: str
    channel: int
    time_init: np.datetime64
    time_end: np.datetime64
    unit: str
    SPL_rms: float
    SPL_pk: float
    SPL_pkpk: float
    SPL_rms90: float
    SEL: float
    t_rise: float
    t_decay: float
    kurtosis: float
    SEL_cum: float = 0.0
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.metadata:
            [setattr(self, k, v) for k, v in self.metadata.items()]


def build_detection(det: h5py.Group) -> Detection:
    """Build a Detection object from an HDF5 group.

    Args:
        det: HDF5 group containing detection data.

    Returns:
        Detection: Detection object.
    """
    if "metadata" in det:
        mdgrp = det["metadata"]
        metadata = {k: mdgrp.attrs[k] for k in mdgrp.attrs}
    else:
        metadata = None
    return Detection(
        uid=det.attrs["uid"],
        station=det.attrs["station"],
        waveform=np.array(det["waveform"]),
        time_vector=np.array(det["time_vector"], dtype=np.int64).astype(
            "datetime64[us]"
        ),
        channel=det.attrs["channel"],
        time_init=det.attrs["time_init"],
        time_end=det.attrs["time_end"],
        sampling_rate=det.attrs["sampling_rate"],
        unit=det.attrs["unit"],
        metadata=metadata,
    )


def compute_metrics(dbpath: Path) -> list[Metrics]:
    """Compute soundscape metrics from an HDF5 database.

    Args:
        dbpath: Path to the HDF5 database.

    Returns:
        list[Metrics]: List of Metrics objects.
    """
    with h5py.File(dbpath, "r") as fr:
        all_metrics = []
        for chan in fr:
            running_sel_cum = -np.inf
            for idx, k in enumerate(fr[chan].keys()):
                det = fr[chan][k]
                detection = build_detection(det)
                metrics = compute_pulse_metrics(idx, detection, running_sel_cum)
                running_sel_cum = metrics.SEL_cum
                all_metrics.append(metrics)

    return all_metrics


def compute_event_metrics(
    event: dict,
    database_path: Path,
) -> None:
    event_name = event["name"]
    event_start = event["start"][0:10]
    event_dir = f"{event_name}_{event_start}"

    logger.info(f"[{event_name}] Computing metrics for event.")

    savepath = database_path / event_dir
    savepath.mkdir(parents=True, exist_ok=True)

    for sensor in event["sensors"]:
        sensor_name = sensor["name"]
        logger.info(f"[{event_name}][{sensor_name}] Computing metrics for sensor.")

        dbpath = savepath / f"detections_{sensor_name}.hdf5"

        all_metrics = compute_metrics(dbpath)
        savename = savepath / f"metrics_{sensor_name}.csv"
        save_metrics_to_csv(all_metrics, savename)

        logger.info(f"[{event_name}][{sensor_name}] Metrics saved to {savename}.")
        logger.info(f"[{event_name}][{sensor_name}] Completed sensor.")
    logger.info(f"[{event_name}] Completed event.")


def compute_pulse_metrics(
    index: int, detection: Detection, running_sel_cum: float
) -> Metrics:
    """Compute soundscape metrics for a single pulse.

    Args:
        index: Index of the detection in the database.
        detection: Detection object.
        running_sel_cum: Running cumulative SEL (dB re 1 uPa^2 s); -inf for
            the first pulse.

    Returns:
        Metrics: Soundscape metrics for the pulse.
    """
    x = detection.waveform.squeeze()
    t = detection.time_vector

    rise_time, decay_time = rise_decay_times(x, t)
    sound_exp_level = sel(x, detection.sampling_rate)
    running_sel_cum = sel_cum(np.array([running_sel_cum, sound_exp_level]))
    return Metrics(
        index=index,
        uid=detection.uid,
        station=detection.station,
        channel=detection.channel,
        time_init=detection.time_init,
        time_end=detection.time_end,
        unit=detection.unit,
        SPL_rms=spl_rms(x),
        SPL_pk=spl_pk(x),
        SPL_pkpk=spl_pkpk(x),
        SPL_rms90=spl_rms90(x),
        SEL=sound_exp_level,
        t_rise=rise_time,
        t_decay=decay_time,
        kurtosis=kurtosis(x, fisher=False),
        SEL_cum=running_sel_cum,
        metadata=detection.metadata,
    )


def energy90(x: np.ndarray) -> float:
    """Compute the mean-square pressure over the 90% energy interval (T90),
    between 5% and 95% of the cumulative energy.

    Args:
        x: Time series data.

    Returns:
        Mean-square pressure over T90 (uPa^2).
    """
    energy = x**2
    start_index, end_index = find_t5t95(energy)
    return 1 / (end_index - start_index) * np.sum(energy[start_index:end_index])


def find_t5t95(energy: np.ndarray) -> tuple[int, int]:
    """Find the indices of the 5th and 95th percentiles of the cumulative energy.

    Args:
        energy: Energy time series.

    Returns:
        Starting index of the energy range.
        Ending index of the energy range.
    """
    cum_energy = np.cumsum(energy)
    norm_cum_energy = cum_energy / cum_energy[-1]

    lower_bound = 0.05
    upper_bound = 0.95

    start_index = np.searchsorted(norm_cum_energy, lower_bound)
    end_index = np.searchsorted(norm_cum_energy, upper_bound)
    return start_index, end_index


def p_ms(x: np.ndarray, axis: int = -1) -> float:
    """Compute the mean square sound pressure.

    Args:
        x: Time series data.
        axis: Axis along which to compute the mean square sound pressure.

    Returns:
        Mean square sound pressure.
    """
    return np.mean(x**2, axis=axis)


def p_pk(x: np.ndarray, axis: int = -1) -> float:
    """Compute the zero-to-peak (Pk) sound pressure.

    Args:
        x: Time series data.
        axis: Axis along which to compute the zero-to-peak sound pressure.

    Returns:
        Zero-to-peak sound pressure.
    """
    return np.abs(x).max(axis=axis)


def p_pkpk(x: np.ndarray, axis: int = -1) -> float:
    """Compute the peak-to-peak sound pressure.

    Args:
        x: Time series data.
        axis: Axis along which to compute the peak-to-peak sound pressure.

    Returns:
        Peak-to-peak sound pressure.
    """
    return x.max(axis=axis) - x.min(axis=axis)


def p_rms(x: np.ndarray, axis: int = -1) -> float:
    """Returns root mean square (RMS) sound pressure in uPa."""
    return np.sqrt(p_ms(x, axis=axis))


def rise_decay_times(x: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Compute the rise and decay times of the time series.

    Args:
        x: Time series data.
        t: Time vector.

    Returns:
        Rise time (s).
        Decay time (s).
    """
    energy = x**2
    start_index, end_index = find_t5t95(energy)
    peak_index = np.argmax(energy)
    rise_time = t[peak_index] - t[start_index]
    decay_time = t[end_index] - t[peak_index]
    return rise_time.astype(float) / 1e6, decay_time.astype(float) / 1e6


def save_metrics_to_csv(all_metrics: list[Metrics], destination: Path) -> None:
    """Save soundscape metrics to a CSV file.

    Args:
        all_metrics: List of Metrics objects.
        destination: Path to the CSV file.
    """
    rows = [
        {k: v for k, v in vars(m).items() if k != "metadata"} for m in all_metrics
    ]
    pl.DataFrame(rows).write_csv(destination)


def sel(x: np.ndarray, fs: float) -> float:
    """Compute single-strike sound exposure level (SEL), the time integral of
    squared pressure over the time series.

    Args:
        x: Time series data (uPa).
        fs: Sampling rate (Hz).

    Returns:
        Single-strike sound exposure level (dB re 1 uPa^2 s).
    """
    return 10 * np.log10(np.sum(x**2) / fs / (REF_PRESSURE**2))


def sel_cum(levels: np.ndarray) -> float:
    """Compute cumulative sound exposure level (SEL_cum).

    Args:
        levels: Array of sound exposure levels (dB).

    Returns:
        Cumulative sound exposure level (dB).
    """
    return 10 * np.log10(np.sum(10 ** (levels / 10)))


def spl(x: np.ndarray) -> np.ndarray:
    """Compute the sound pressure level (SPL) of the time series.

    Args:
        x: Time series data.

    Returns:
        Sound pressure level (dB).
    """
    return 20 * np.log10(np.abs(x) / REF_PRESSURE)


def spl_pk(x: np.ndarray, axis: int = -1) -> float:
    """Compute the zero-to-peak sound pressure level (SPL).

    Args:
        x: Time series data.
        axis: Axis along which to compute the zero-to-peak sound pressure level.

    Returns:
        Zero-to-peak sound pressure level (dB).
    """
    return spl(p_pk(x, axis=axis))


def spl_pkpk(x: np.ndarray, axis: int = -1) -> float:
    """Compute the peak-to-peak sound pressure (SPL).

    Args:
        x: Time series data.
        axis: Axis along which to compute the peak-to-peak sound pressure.

    Returns:
        Peak-to-peak sound pressure level (dB).
    """
    return spl(p_pkpk(x, axis=axis))


def spl_rms90(x: np.ndarray) -> float:
    """Compute the RMS sound pressure level over the 90% energy interval (T90).

    Args:
        x: Time series data.

    Returns:
        RMS sound pressure level over T90 (dB re 1 uPa).
    """
    return 10 * np.log10(energy90(x) / (REF_PRESSURE**2))


def spl_rms(x: np.ndarray, axis: int = -1) -> float:
    """Compute the root mean square (RMS) sound pressure level (SPL) of the
    time series.

    Args:
        x: Time series data.
        axis: Axis along which to compute the RMS sound pressure level.

    Returns:
        Root mean square sound pressure level (dB).
    """
    return spl(p_rms(x, axis=axis))
