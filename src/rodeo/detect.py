"""Detection module for acoustic data processing.

This module provides functions to detect acoustic pulses in time series data. The
`process_stream` function processes a `DataStream` object and saves the detected
pulses to an HDF5 file. The `Detection` data class is used to store information
about each detection.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy import signal
from tqdm import tqdm
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import TIME_PRECISION

from rodeo.utils import compute_range

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Data class for a single detection.

    Attributes:
        uid: Unique identifier (typically unused).
        station: Station name.
        waveform: Time series data.
        time_vector: Time vector.
        channel: Channel number.
        time_init: Start time of the detection time series.
        time_end: End time of the detection time series.
        sampling_rate: Sampling rate of the time series.
        unit: Amplitude unit of the time series.
        metadata: Metadata dictionary.
    """

    uid: str
    station: str
    waveform: np.ndarray
    time_vector: np.ndarray
    channel: int
    time_init: np.datetime64
    time_end: np.datetime64
    sampling_rate: float
    unit: str
    metadata: dict | None = None


def apply_linear_taper(time_series: np.ndarray, taper_length: int) -> np.ndarray:
    """
    Apply a linear taper to the beginning and end of a time series.

    Args:
      time_series: the original time series data
      taper_length: length of the taper in data points

    Returns:
      Tapered time series data
    """
    series_length = len(time_series)
    taper = np.ones(series_length)

    # Create the linear taper
    taper[:taper_length] = np.linspace(0, 1, taper_length)
    taper[-taper_length:] = np.linspace(1, 0, taper_length)

    # Apply the taper to the time series
    return time_series * taper


def characteristic_function(y: np.ndarray, stwin: int, ltwin: int) -> np.ndarray:
    """Compute the characteristic function of a time series.

    Args:
        y: Time series data.
        stwin: Short-time window (samples)
        ltwin: Long-time window (samples)

    Returns:
        Characteristic function of the time series.
    """
    y_rms = np.sqrt(signal.convolve(y**2, np.ones(stwin) / stwin, mode="same"))
    y_long = signal.convolve(y_rms, np.ones(ltwin) / ltwin, mode="same")
    return y_rms / y_long


def detect_pulses(
    ds: DataStream,
    stwin: float = 0.025,
    ltwin: float = 5.0,
    threshold: float = 2.25,
    distance: float = 0.75,
    taper: float = 5.0,
) -> np.ndarray:
    """Detect pulses in a time series.

    Args:
        ds: DataStream object.
        stwin: Short-time window (seconds).
        ltwin: Long-time window (seconds).
        threshold: Threshold for peak detection.
        distance: Minimum distance between peaks (seconds).
        taper: Taper length (seconds).

    Returns:
        Array of pulse times.
    """
    fs = ds.stats.sampling_rate
    time = ds.time_vector

    chf = characteristic_function(
        apply_linear_taper(ds.data.squeeze(), int(fs * taper)),
        int(stwin * fs),
        int(ltwin * fs),
    )
    peaks, _ = signal.find_peaks(chf, height=threshold, distance=distance * fs)
    return time[peaks]


def event_detections(
    inv_config: dict,
    event: dict,
    df_equip: pl.DataFrame,
    df_turb: pl.DataFrame,
    pulse_filter: str | None = None,
    pulse_filt_freq: float | None = None,
) -> None:

    def _compute_range(
        df_equip: pl.DataFrame, df_turb: pl.DataFrame, sensor_name: str, event_site: str
    ) -> float:
        # Compute the range between the sensor and the event site
        site = df_turb.filter(pl.col("Turbine") == event_site).row(0, named=True)
        sensor = df_equip.filter(pl.col("Name") == sensor_name).row(0, named=True)
        return compute_range(
            sensor["Longitude [dd]"], sensor["Latitude [dd]"], site["lon"], site["lat"]
        )

    event_name = event["name"]
    event_site = event["site"]
    event_start = event["start"][0:10]

    logger.info(f"Analyzing event {event_name}.")

    path = Path.cwd() / "data" / "events" / f"{event_name}_{event_start}"
    path.mkdir(parents=True, exist_ok=True)

    for sensor in event["sensors"]:
        sensor_name = sensor["name"]
        sensor_dec_factor = sensor.get("dec_factor", None)
        sensor_filter = sensor.get("filter", None)
        sensor_freq = sensor.get("freq", None)
        sensor_channels = sensor["channels"]

        logger.info(f"-Analayzing sensor {sensor_name}.")

        ds = read_inventory(
            file_path=Path(inv_config[sensor_name]["data"]["destination"]),
            time_start=np.datetime64(event["start"], TIME_PRECISION),
            time_end=np.datetime64(event["end"], TIME_PRECISION),
            channels=[i["channel"] for i in sensor_channels],
        )
        logger.info("-Data loaded from disk.")

        r = _compute_range(df_equip, df_turb, sensor_name, event_site)

        # Run analyses for individual pulses
        dbpath = path / f"detections_{sensor_name}.hdf5"
        process_stream(
            ds,
            pulse_start=[d["pulse_start"] for d in sensor["channels"]],
            pulse_end=[d["pulse_end"] for d in sensor["channels"]],
            stwin=[d["stwin"] for d in sensor["channels"]],
            ltwin=[d["ltwin"] for d in sensor["channels"]],
            threshold=[d["threshold"] for d in sensor["channels"]],
            distance=[d["distance"] for d in sensor["channels"]],
            taper=sensor["taper"],
            pulse_filter_type=pulse_filter,
            pulse_filter_freq=pulse_filt_freq,
            detector_filter_type=sensor_filter,
            detector_filter_freq=sensor_freq,
            detector_decimate_factor=sensor_dec_factor,
            station=sensor_name,
            dbpath=dbpath,
            metadata={
                "event_name": event_name,
                "event_site": event_site,
                "range": r,
                "filter": pulse_filter,
                "filter_freq": pulse_filt_freq,
            },
        )

        logger.info(f"-Completed sensor {sensor_name}.")
    logger.info(f"Completed {event_name}.")


def parse_detection(grp: h5py.Group) -> DataStreamStats:
    """Return `DataStreamStats` object from HDF5 group attributes.

    Args:
        grp: HDF5 group object.

    Returns:
        `DataStreamStats` object.
    """
    attrs = grp.attrs
    return DataStreamStats(
        channels=attrs.get("channel"),
        time_init=np.datetime64(attrs.get("time_init")[:-1], TIME_PRECISION),
        time_end=np.datetime64(attrs.get("time_end")[:-1], TIME_PRECISION),
        sampling_rate=attrs.get("sampling_rate"),
        units=attrs.get("unit"),
    )


def process_stream(
    ds: DataStream,
    pulse_start: int | list[int] = 500,
    pulse_end: int | list[int] = 500,
    stwin: float | list[float] = 0.025,
    ltwin: float | list[float] = 5.0,
    threshold: float | list[float] = 2.25,
    distance: float | list[float] = 0.75,
    taper: float | list[float] = 5.0,
    pulse_filter_type: str | None = None,
    pulse_filter_freq: float | None = None,
    detector_filter_type: str | None = None,
    detector_filter_freq: float | None = None,
    detector_decimate_factor: int | None = None,
    station: str = "unknown",
    dbpath: Path | None = None,
    metadata: dict | None = None,
) -> None:
    """Process a `DataStream` object and save detected pulses to an HDF5 file.

    Args:
        ds: DataStream object.
        pulse_start: Time before pulse to include in detection (ms).
        pulse_end: Time after pulse to include in detection (ms).
        stwin: Short-time window for characteristic function (s).
        ltwin: Long-time window for characteristic function (s).
        threshold: Threshold for peak detection.
        distance: Minimum distance between peaks (s).
        taper: Taper length (s).
        pulse_filter_type: Filter type for pulse data.
        pulse_filter_freq: Filter frequency for pulse data (Hz).
        detector_filter_type: Filter type for detector data.
        detector_filter_freq: Filter frequency for detector data (Hz).
        detector_decimate_factor: Decimation factor for detector data.
        station: Station name.
        dbpath: Path to the HDF5 file; `./detections.h5` if None.
        metadata: Metadata dictionary
    """
    if dbpath is None:
        dbpath = Path.cwd() / "detections.h5"

    # Ensure that all parameters are lists
    if not isinstance(pulse_start, list):
        pulse_start = [pulse_start] * ds.num_channels
    if not isinstance(pulse_end, list):
        pulse_end = [pulse_end] * ds.num_channels
    if not isinstance(stwin, list):
        stwin = [stwin] * ds.num_channels
    if not isinstance(ltwin, list):
        ltwin = [ltwin] * ds.num_channels
    if not isinstance(threshold, list):
        threshold = [threshold] * ds.num_channels
    if not isinstance(distance, list):
        distance = [distance] * ds.num_channels
    if not isinstance(taper, list):
        taper = [taper] * ds.num_channels

    for chan in range(ds.num_channels):
        logger.info(f"--Processing channel {chan}.")
        dsc = ds[chan]
        # Copy so that detector processing never alters the pulse data.
        dsc_det = dsc.copy()
        if detector_decimate_factor is not None:
            dsc_det.decimate(detector_decimate_factor)
            logger.info(f"--Detector data decimated by {detector_decimate_factor}.")
        if detector_filter_type is not None:
            dsc_det.filter(filt_type=detector_filter_type, freq=detector_filter_freq)
            logger.info(
                f"--Detector data filtered with {detector_filter_type} "
                f"at {detector_filter_freq} Hz."
            )

        pulse_times = detect_pulses(
            dsc_det,
            stwin=stwin[chan],
            ltwin=ltwin[chan],
            threshold=threshold[chan],
            distance=distance[chan],
            taper=taper[chan],
        )

        if pulse_filter_type is not None:
            dsc.filter(filt_type=pulse_filter_type, freq=pulse_filter_freq)
            logger.info(
                f"--Pulse filtered with {pulse_filter_type} at {pulse_filter_freq} Hz."
            )

        detections = []
        for pulse in tqdm(pulse_times):
            t_start = pulse - np.timedelta64(pulse_start[chan], "ms")
            t_end = pulse + np.timedelta64(pulse_end[chan], "ms")
            excerpt = dsc.slice(t_start, t_end)

            detections.append(
                Detection(
                    uid=uuid.uuid4().hex[:8],
                    station=station,
                    waveform=excerpt.data,
                    time_vector=excerpt.time_vector,
                    channel=excerpt.stats.channels,
                    time_init=excerpt.stats.time_init,
                    time_end=excerpt.stats.time_end,
                    sampling_rate=excerpt.stats.sampling_rate,
                    unit=excerpt.stats.units,
                    metadata=metadata,
                )
            )
        save_detections_to_hdf5(chan, detections, dbpath)
        logger.info(f"--Detections saved to {dbpath}.")
        logger.info(f"--Completed channel {chan}.")


def station_from_catalogue_name(fname: Path) -> str:
    """Return station name from catalogue file name.

    Args:
        fname: Path to the catalogue file.

    Returns:
        Station name.
    """
    return fname.stem.split("FileInfo")[0].rstrip("_")


def save_detections_to_hdf5(
    channel: int | str, detections: list[Detection], file_path: str
) -> None:
    """Save detections to an HDF5 file, appending to any already saved for
    the channel.

    Args:
        channel: Channel number.
        detections: List of `Detection` objects.
        file_path: Path to the HDF5 file.
    """
    with h5py.File(file_path, "a") as f:
        main_grp = f.require_group(f"chan={channel}")
        # Number after any detections already saved for this channel.
        for i, detection in enumerate(detections, start=len(main_grp)):
            grp = main_grp.create_group(f"det={i:04d}")

            # Save string attributes
            grp.attrs["uid"] = detection.uid
            grp.attrs["station"] = detection.station
            grp.attrs["unit"] = detection.unit

            # Save numpy arrays
            grp.create_dataset("waveform", data=detection.waveform)
            grp.create_dataset(
                "time_vector", data=detection.time_vector.astype("int64")
            )

            # Save scalar attributes
            grp.attrs["channel"] = detection.channel
            grp.attrs["time_init"] = np.datetime_as_string(
                detection.time_init, timezone="UTC", unit=TIME_PRECISION
            ).encode("utf-8")
            grp.attrs["time_end"] = np.datetime_as_string(
                detection.time_end, timezone="UTC", unit=TIME_PRECISION
            ).encode("utf-8")
            grp.attrs["sampling_rate"] = detection.sampling_rate
            if detection.metadata:
                grp.create_group("metadata")
                for key, value in detection.metadata.items():
                    grp["metadata"].attrs[key] = value
