"""Helper functions for plotting data.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""

import re
from collections.abc import Sequence
from pathlib import Path

import bathyreq
import cmasher as cmr
import cmocean as cmo  # noqa: F401
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.basemap import Basemap
from scipy import signal
from scipy.interpolate import interp1d
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_CONVERSION_FACTOR

from rodeo import detect, utils


def drop_nans(arr: np.ndarray) -> np.ndarray:
    rows_clean = arr[~np.isnan(arr).all(axis=1)]
    cols_clean = rows_clean[:, ~np.isnan(rows_clean).any(axis=0)]
    return cols_clean


def interpolate_nan_rows(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Linearly interpolates over NaN values along each row.

    Parameters:
    - arr (np.ndarray): 2D array with NaNs to interpolate.
    - x (np.ndarray): 1D array of x-coordinates.

    Returns:
    - np.ndarray: Interpolated array.
    """
    interpolated = np.copy(arr)
    for i in range(arr.shape[0]):
        valid = ~np.isnan(arr[i])
        if np.sum(valid) > 1:  # Only interpolate if there are at least 2 valid points
            f = interp1d(
                x[valid],
                arr[i][valid],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            interpolated[i] = f(x)
    return interpolated


def regularize_grid(
    windows: list[np.ndarray],
    tvec: np.ndarray,
    window: float,
    nbins: int = 100,
    drop_final_data_row=True,
) -> tuple[np.ndarray, np.ndarray]:
    pulse_indices = []
    time_points = []
    correlations = []

    for i, w in enumerate(windows):
        if len(w) == 0:
            continue
        times = w["dt"]
        corr = w["data"]

        pulse_indices.append(np.full_like(times, i))
        time_points.append(times)
        correlations.append(corr)

    # Convert to arrays:
    pulse_indices = np.concatenate(pulse_indices).astype(int)
    time_points = np.concatenate(time_points).astype(int)
    correlations = np.concatenate(correlations)

    # Create grid:
    time_bins = np.linspace(-window, window, num=nbins)
    pulse_bins = np.append(tvec, tvec[-1] + 1 / 1e3)
    time_grid, pulse_grid = np.meshgrid(time_bins, pulse_bins)

    if drop_final_data_row:
        corr_grid = np.full((len(windows), len(time_bins) - 1), np.nan)
    else:
        corr_grid = np.full((len(windows), len(time_bins)), np.nan)

    for i, (t, c) in enumerate(zip(time_points, correlations)):
        time_idx = np.argmin(np.abs(time_bins - t))
        pulse_idx = pulse_indices[i]
        corr_grid[pulse_idx, time_idx] = c

    if drop_final_data_row:
        corr_grid = interpolate_nan_rows(corr_grid, time_bins[:-1])
    else:
        corr_grid = interpolate_nan_rows(corr_grid, time_bins)

    return time_grid, pulse_grid, corr_grid


def stack_data(data: np.ndarray, dt: np.ndarray, window: float) -> list[np.ndarray]:
    windows = []
    for i in range(data.shape[0]):
        mask = (dt[i] >= -window) & (dt[i] <= window)
        data_window = data[i][mask]
        dt_window = dt[i][mask]

        structured = np.array(
            list(zip(dt_window, data_window)), dtype=[("dt", float), ("data", float)]
        )
        windows.append(structured)

    return windows


def plot_avg_power_timehist(
    mean_hist: np.ndarray,
    bins: np.ndarray,
    events: list[dict],
    title: str | None = None,
    power_range: tuple[float] = (100, 160),
    channels=None,
    figsize: tuple[float] = (8, 10),
) -> plt.Figure:

    LINE_STYLES = {
        "Ambient": "-",
        "Soft Start": "--",
        "Full Power": ":",
    }

    fig, axs = plt.subplots(nrows=len(channels), ncols=1, figsize=figsize)

    for i in range(len(channels)):
        for j, event in enumerate(events):
            ax = axs[i]
            base_name = re.sub(r" \d+$", "", event["name"])
            ls = LINE_STYLES.get(base_name, "-")
            ax.plot(bins[0:-1], mean_hist[j, i, :], linestyle=ls, label=event["name"])
            ax.set_xlim(power_range)
            ax.text(
                0.0,
                1.02,
                f"Channel {channels[i]}",
                fontsize=10,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
            if i < len(channels) - 1:
                ax.set_xticklabels([])

            if i == 0:
                ax.set_title(title)
                ax.legend()

    ax.set_xlabel("Power [dB re 1 uPa$^2$]")
    ax.set_ylabel("PDF")

    return fig


def plot_bathy(
    data: np.ndarray,
    lonvec: np.ndarray,
    latvec: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:

    if ax is None:
        ax = plt.gca()

    cmap = cmr.get_sub_cmap("cmo.deep", 0.0, 1.0)

    im = ax.imshow(
        np.abs(data),
        extent=(lonvec[0], lonvec[-1], latvec[0], latvec[-1]),
        cmap=cmap,
    )
    CS = ax.contour(
        lonvec, latvec, np.abs(np.flipud(data)), colors="k", levels=10, linewidths=0.5
    )
    ax.clabel(CS, inline=1, fmt="%1.0f", fontsize=plt.rcParams["font.size"] - 2)
    return im, ax


def plot_ctd_profiles(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 3),
    gridspec_kw: dict | None = None,
    dpi: int = 150,
    **plot_kwargs,
) -> plt.Figure:
    if gridspec_kw is None:
        gridspec_kw = {"wspace": 0.2}
    fig, axs = plt.subplots(figsize=figsize, ncols=3, gridspec_kw=gridspec_kw, dpi=dpi)
    ylim = (0, 35)

    ax0 = axs[0]
    ax0.plot(df["Temperature"], df["Depth"], **plot_kwargs["Temperature"])
    ax0.set_ylim(ylim)
    ax0.invert_yaxis()
    ax0.grid()
    ax0.set_xlabel("Temp. (C)")
    ax0.set_ylabel("Depth (m)")

    ax1 = axs[1]
    ax1.plot(df["Salinity"], df["Depth"], **plot_kwargs["Salinity"])
    ax1.set_ylim(ylim)
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_yticklabels([])
    ax1.set_xlabel("Sal. (PSU)")

    ax2 = axs[2]
    ax2.plot(df["Speed of sound"], df["Depth"], **plot_kwargs["Speed of sound"])
    ax2.set_ylim(ylim)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_yticklabels([])
    ax2.set_xlabel("Sound speed (m/s)")
    return fig


def plot_detection_diagnostic(
    ds: DataStream,
    stwin: float | list[float] = 0.025,
    ltwin: float | list[float] = 5.0,
    threshold: float | list[float] = 2.25,
    distance: float | list[float] = 0.75,
    taper: float = 5.0,
    sensor: str = "UnkSta",
    channel: int | None = None,
    figsize: tuple[float] = (24, 6),
    title: str | None = None,
) -> list[plt.Figure]:
    fs = ds.stats.sampling_rate
    time = ds.time_vector

    if channel is not None:
        channels = [channel]
        num_channels = 1
    else:
        channels = ds.stats.channels
        num_channels = ds.num_channels

    if not isinstance(stwin, list):
        stwin = [stwin] * num_channels
    if not isinstance(ltwin, list):
        ltwin = [ltwin] * num_channels
    if not isinstance(threshold, list):
        threshold = [threshold] * num_channels
    if not isinstance(distance, list):
        distance = [distance] * num_channels

    fig_list = []
    for i, chan in enumerate(channels):
        print(f"Processing channel {chan}...")

        data = detect.apply_linear_taper(ds.data[chan], int(fs * taper))
        chf = detect.characteristic_function(
            data, stwin=int(fs * stwin[i]), ltwin=int(fs * ltwin[i])
        )
        peaks, _ = signal.find_peaks(
            chf, height=threshold[i], distance=distance[i] * fs
        )

        if title is None:
            title = (
                f"{sensor} | Channel {chan} | STWIN: {stwin[i]} | LTWIN: {ltwin[i]} | "
                f"THRESHOLD: {threshold[i]} | DISTANCE: {distance[i]} | N_PEAKS: {len(peaks)}"
            )

        fig, axs = plt.subplots(
            nrows=2, ncols=1, figsize=figsize, gridspec_kw={"hspace": 0.25}, sharex=True
        )
        fig.suptitle(title)

        ax = axs[0]
        ax.plot(time, data)
        # ax.set_xticklabels([])
        ax.set_ylabel(f"Pressure [{ds.stats.units}]")

        # ax = axs[1]
        # ax.plot(time, detect.rms(data, int(stwin[i] * fs)))
        # ax.set_xticklabels([])
        # ax.set_ylabel("RMS")

        ax = axs[1]
        ax.plot(time, chf)
        ax.plot(time[peaks], chf[peaks], "rx")
        ax.axhline(threshold[i], color="r", linestyle="--")
        # ax.xaxis.set_tick_params(labelbottom=True, )
        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
        ax.set_xlabel("Time")
        ax.set_ylabel("Characteristic\nFunction")

        fig_list.append(fig)

        print(f"Channel {chan} complete.")

    return fig_list


def plot_detector_output(
    ds: DataStream,
    stwin: float | list[float] = 0.025,
    ltwin: float | list[float] = 5.0,
    threshold: float | list[float] = 2.25,
    distance: float | list[float] = 0.75,
    taper: float = 5.0,
    sensor: str = "UnkSta",
    channel: int | None = None,
    zoom_start: int = 0,
    zoom_end: int = 1000,
    figsize: tuple[float] = (8, 3),
    title: str | None = None,
) -> list[plt.Figure]:
    fs = ds.stats.sampling_rate
    time = ds.time_vector

    data = detect.apply_linear_taper(ds.data[channel], int(fs * taper))
    chf = detect.characteristic_function(
        data, stwin=int(fs * stwin), ltwin=int(fs * ltwin)
    )
    peaks, _ = signal.find_peaks(chf, height=threshold, distance=distance * fs)
    xlim = (time.min(), time.max())

    # Convert from uPa to Pa:
    data /= 1e6

    # Compute zoomed section:
    data_zoom = data[zoom_start:zoom_end]
    time_zoom = time[zoom_start:zoom_end]
    chf_zoom = chf[zoom_start:zoom_end]
    peaks_zoom = peaks[(peaks > zoom_start) & (peaks < zoom_end)] - zoom_start

    if title is None:
        title = (
            f"{sensor} | Channel {channel} | STWIN: {stwin} | LTWIN: {ltwin} | "
            f"THRESHOLD: {threshold} | DISTANCE: {distance} | N_PEAKS: {len(peaks)}"
        )

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        gridspec_kw={"hspace": 0.08, "wspace": 0.14, "width_ratios": [3, 2]},
    )
    fig.suptitle(title)

    # Time series, full
    ax = axs[0, 0]
    ax.plot(time, data, c="k")
    ax.axvspan(
        time[zoom_start], time[zoom_end], facecolor="gray", alpha=0.5, edgecolor="none"
    )
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xticklabels([])
    ax.set_ylabel(f"Pressure ({ds.stats.units[1:]})")

    # Time series, zoomed
    ax = axs[0, 1]
    ax.plot(time_zoom, data_zoom, c="k")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.set_xticklabels([])

    # Characteristic function, full
    ax = axs[1, 0]
    ax.plot(time, chf, c="k")
    ax.plot(time[peaks], chf[peaks], "rx")
    ax.axhline(threshold, color="r", linestyle="--")
    ax.axvspan(
        time[zoom_start], time[zoom_end], facecolor="gray", alpha=0.5, edgecolor="none"
    )
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(0, chf.max() * 1.1)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Characteristic\nFunction")

    # Characteristic function, zoomed
    ax = axs[1, 1]
    ax.plot(time_zoom, chf_zoom, c="k")
    ax.plot(time_zoom[peaks_zoom], chf_zoom[peaks_zoom], "rx")
    ax.axhline(threshold, color="r", linestyle="--")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.set_ylim(0, chf_zoom.max() * 1.1)

    return fig


def plot_event_metrics(df: pl.DataFrame, savepath: Path) -> None:
    METRICS = {
        "SPL_rms": "$L_{p,\\text{rms}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
        "SPL_pk": "$L_{p,\\text{pk}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
        "SPL_pkpk": "$L_{p,\\text{pk-pk}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
        "SEL": "$L_{E,p}$\n(dB re 1 $\\mathrm{\\mu Pa^2 s}$)",
        "SEL_cum": "cSEL\n(dB re 1 $\\mathrm{\\mu Pa^2 s}$)",
        "t_rise": "$t_\\mathrm{r}$ (s)",
        "t_decay": "$t_\\mathrm{d}$ (s)",
        "kurtosis": "$\\beta_2$",
    }
    events = sorted(df["event_name"].unique().to_list())
    stations = sorted(df["station"].unique().to_list(), key=lambda x: x.split("_")[-1])

    for event in events:
        print(f"Plotting event: {event}")

        total_strikes = None

        fig, axs = plt.subplots(
            nrows=len(METRICS),
            ncols=len(stations),
            figsize=(8, 10),
            gridspec_kw={"hspace": 0.1, "wspace": 0.05},
        )

        event_start = (
            df.filter(pl.col("event_name") == event)["time_init"].unique().min()
        )
        event_end = df.filter(pl.col("event_name") == event)["time_end"].unique().max()

        for i, (metric_key, metric_name) in enumerate(METRICS.items()):
            min_val = np.inf
            max_val = 0.0

            for j, station in enumerate(stations):
                ax = axs[i, j]
                station_name = station.replace("_", "/")

                df_sta = df.filter(
                    (pl.col("event_name") == event) & (pl.col("station") == station)
                )
                if df_sta.is_empty():
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                else:
                    distance = df_sta["range"].unique().item()
                    data = df_sta.select(["channel", "index", metric_key])

                    for l, ch in enumerate(sorted(data["channel"].unique().to_list())):
                        df_ch = df_sta.filter(pl.col("channel") == ch)
                        time = df_ch["time_init"].to_numpy()
                        data = df_ch[metric_key].to_numpy()
                        if not total_strikes:
                            total_strikes = len(time)

                        max_val = max(max_val, max(data))
                        min_val = min(min_val, min(data))

                        ax.scatter(
                            time,
                            data,
                            alpha=0.5,
                            s=1.0,
                            color="k",
                            zorder=10 + l,
                            label=f"CH {ch}",
                        )

                ax.grid()

                # Control column titles
                if i == 0:
                    if df_sta.is_empty():
                        ax.set_title(station_name)
                    else:
                        ax.set_title(f"{station_name}, R={distance / 1000:.1f} km")

                # Control row titles & ytick labels
                if j == 0:
                    ax.set_ylabel(metric_name)
                else:
                    ax.set_yticklabels([])

                # Control x-axis ticks and labels
                if i == len(METRICS) - 1:
                    ax.set_xlabel("Time")
                else:
                    ax.set_xticklabels([])
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                locator = mdates.AutoDateLocator(
                    minticks=2, maxticks=5
                )  # Min 4, max 8 ticks
                ax.xaxis.set_major_locator(locator)

            # Controlx x-axis limits:
            xlim_margin = 0.02 * (time.max() - time.min())
            xlim = (time.min() - xlim_margin, time.max() + xlim_margin)

            for j in range(len(stations)):
                axs[i, j].set_xlim(xlim)

            # Control y-axis limits
            for j in range(len(stations)):
                ylim_margin = 0.1 * (max_val - min_val)
                axs[i, j].set_ylim(min_val - ylim_margin, max_val + ylim_margin)

        fig.suptitle(
            f"Event {event} (all channels) | Total strikes: {total_strikes:,}\n{event_start.strftime('%d-%b-%Y %H:%M')} - "
            f"{event_end.strftime('%d-%b-%Y %H:%M %Z')}",
            y=0.935,
        )
        fig.savefig(
            savepath / f"{event}" / "event_metrics.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()


def plot_inst_power_timehist(
    hist_data: np.ndarray,
    title: str | None = None,
    power_range: tuple[float] = (100, 160),
    channels=None,
    fig_size: tuple[float] = (8, 10),
) -> plt.Figure:

    num_channels = hist_data.shape[0]
    num_intervals = hist_data.shape[1]
    vmax = hist_data.max()

    fig, axs = plt.subplots(
        hist_data.shape[0], 1, figsize=fig_size, gridspec_kw={"hspace": 0.3}
    )
    fig.suptitle(title)

    enumerator = enumerate(channels) if channels else enumerate(range(num_channels))

    for i, channel in enumerator:
        ax = axs[i]
        im = ax.imshow(
            hist_data[i].T,
            aspect="auto",
            origin="lower",
            cmap="jet",
            extent=[0, num_intervals, power_range[0], power_range[1]],
            interpolation="none",
            vmin=0,
            vmax=vmax,
        )

        ax.set_title(f"Channel {channel}", fontsize=10, ha="left", x=0)
        if channel == channels[-1]:
            ax.set_xlabel("Time Interval")
            ax.set_ylabel("Power [dB re 1 uPa$^2$]")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.set_yticklabels([])
            ax.set_ylabel("")

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Frequency")

    return fig


def plot_max_corr_mat(
    stations: list[str], data: np.ndarray, dt: np.ndarray, title: str | None = None
) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(8, 10), gridspec_kw={"hspace": 0.15, "wspace": 0.15}
    )

    def drop_nans(arr: np.ndarray) -> np.ndarray:
        rows_clean = arr[~np.isnan(arr).all(axis=1)]
        cols_clean = rows_clean[:, ~np.isnan(rows_clean).any(axis=0)]
        return cols_clean

    STATIONS = ["SHRU921/Tetra", "SHRU905/VLA1", "SHRU920/VLA2"]
    j_d = 0
    t_max = 0.0
    for j, station in enumerate(STATIONS):
        for i in range(4):
            ax = axs[i, j]

            if i == 0:
                ax.set_title(station)
            if j == 0:
                ax.set_ylabel(f"Channel {i + 1}\nTime ($10^3$ s)")
            if i == 3:
                ax.set_xlabel("Time ($10^3$ s)")

            if j == 0 and STATIONS[0] != stations[0]:
                j_d = j - 1
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                corr = drop_nans(data[j_d, i])
                dt_cl = drop_nans(dt[j_d, i])

                t = np.cumsum(np.insert(np.diff(dt_cl[0]), 0, 0.0))
                t_max = max(t_max, t[-1])
                X, Y = np.meshgrid(t, t)

                im = ax.pcolormesh(
                    X, Y, corr, cmap="magma", shading="auto", vmin=0.0, vmax=1.0
                )

        j_d += 1

    for ax in axs.flatten():
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, t_max)

    cax = fig.add_axes([0.92, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Maximum normalized cross-correlation")

    fig.suptitle(title, y=0.925)
    return fig


def plot_max_corr_stack(
    stations: list[str],
    data: np.ndarray,
    dt: np.ndarray,
    title: str | None = None,
    window: float = 30.0,
    nbins: int = 100,
) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(8, 10), gridspec_kw={"hspace": 0.15, "wspace": 0.15}
    )

    STATIONS = ["SHRU921/Tetra", "SHRU905/VLA1", "SHRU920/VLA2"]
    j_d = 0
    t_max = 0.0
    for j, station in enumerate(STATIONS):
        for i in range(4):
            ax = axs[i, j]

            if i == 0:
                ax.set_title(station)
            if j == 0:
                ax.set_ylabel(f"Channel {i + 1}\nTime ($10^3$ s)")
            if i == 3:
                ax.set_xlabel("Reduced time (s)")

            if j == 0 and STATIONS[0] != stations[0]:
                j_d = j - 1
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                corr = drop_nans(data[j_d, i])
                dt_cl = drop_nans(dt[j_d, i])
                t = np.cumsum(np.insert(np.diff(dt_cl[0]), 0, 0.0)) / 1e3
                t_max = max(t_max, t[-1])

                windows = stack_data(corr, dt_cl, window=window)
                time_grid, pulse_grid, corr_grid = regularize_grid(
                    windows, t, window=window, nbins=nbins
                )

                mesh = ax.pcolormesh(
                    time_grid,
                    pulse_grid,
                    corr_grid,
                    cmap="magma",
                    shading="auto",
                    vmin=0.0,
                    vmax=1.0,
                )
                ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

        j_d += 1

    for ax in axs.flatten():
        ax.set_xlim(-window, window)
        ax.set_ylim(t_max, 0)

    cax = fig.add_axes([0.92, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
    cbar.set_label("Maximum normalized cross-correlation")

    fig.suptitle(title, y=0.925)
    return fig


def plot_met_data(data: pl.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(
        2, 1, figsize=(6, 3), sharex=True, gridspec_kw={"hspace": 0.1}
    )

    ax = axs[0]
    ax.plot(data["datetime"], data["WVHT"], label="Wave Height (m)")
    ax.grid(True)
    ax.set_ylabel("Wave Height (m)")

    ax = axs[1]
    ax.plot(data["datetime"], data["WSPD"], label="Wind Speed (m/s)")
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_xlabel("Date")

    return fig


def plot_mode_function(
    environment: dict, phis: list[np.ndarray], freqs: list[float], zgrid: np.ndarray
) -> plt.Figure:
    fig, axs = plt.subplots(ncols=3, gridspec_kw={"wspace": 0.05}, figsize=(6, 4))
    for i, (ax, phi, freq) in enumerate(zip(axs, phis, freqs)):
        ax.plot(np.real(phi), zgrid, c="tab:blue", lw=1, label="$\\Phi_m$")
        ax.plot(np.real(phi).sum(axis=1), zgrid, c="k", lw=2, label="$\\Sigma\\Phi_m$")
        ax.axhline(environment["layerdata"][0]["z"].max(), color="k", linestyle="--")
        ax.axvline(0.0, color="k")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.1, zgrid.max() + 0.1)
        ax.invert_yaxis()
        if i == 0:
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Depth (m)")
        if i != 0:
            ax.set_yticklabels([])

        num_modes = phi.shape[1]
        modes_display = f"{num_modes} modes" if num_modes > 1 else "1 mode"
        ax.set_title(f"$f$ = {freq:.0f} Hz\n{modes_display}")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=fig.transFigure,
    )
    return fig


def plot_pressure_fields(
    pressures: list[np.ndarray],
    z: np.ndarray,
    seabed: float,
    labels: list[str] | None = None,
) -> plt.Figure:
    max_pressure = 1.1 * np.abs(np.concatenate(pressures)).max()
    fig, ax = plt.subplots(figsize=(2, 3))
    for i, pressure in enumerate(pressures):
        label = labels[i] if labels else f"Field {i}"
        ax.plot(np.abs(pressure), z, label=label)
    ax.axhline(seabed, color="k", linestyle="--")
    ax.invert_yaxis()
    ax.set_xlim(0, max_pressure)
    ax.set_ylim(z.max(), 0)
    ax.set_xlabel("Pressure amplitude")
    ax.set_ylabel("Depth (m)")
    ax.legend()
    return fig


def plot_psd_pdf(
    freq: np.ndarray,
    data: np.ndarray,
    freq_amb: np.ndarray,
    data_amb: np.ndarray,
    event_name: str,
    distance: np.ndarray,
) -> plt.Figure:
    # Data shape: (3 stations, 4 channels, # detections, freq bins)
    # Ambient noise shape: (3 stations, 4 channels, freq bins)

    channels = [0, 1, 2, 3]
    stations = ["SHRU921/Tetra", "SHRU905/VLA1", "SHRU920/VLA2"]
    # Reorder stations, `data`, and `data_amb` by distance, low to high:
    order = np.argsort(distance)
    stations = [stations[k] for k in order]
    data = data[order]
    data_amb = data_amb[order]
    distance = distance[order]

    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(8, 9), gridspec_kw={"hspace": 0.05, "wspace": 0.05}
    )

    for i, channel in enumerate(channels):
        for j, station in enumerate(stations):
            ax = axs[i][j]

            view_amb = data_amb[j, i]
            view = data[j, i]

            if np.isnan(view).all():
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                # Convert to dB
                view_amb = 10 * np.log10(view_amb)
                view = 10 * np.log10(view)
                mean = np.mean(view, axis=0)
                std = np.std(view, axis=0)
                ucb = mean + 2 * std
                lcb = mean - 2 * std

                ax.plot(freq, mean, c="tab:red", lw=2, label="Mean", zorder=20)
                ax.fill_between(
                    freq, ucb, lcb, color="tab:red", alpha=0.25, label="Mean ± 2σ"
                )
                ax.plot(
                    freq_amb,
                    view_amb,
                    c="tab:blue",
                    lw=1,
                    label="Ambient noise",
                    zorder=10,
                )

            ax.set_xscale("log")
            ax.set_xticks(
                [10, 100, 1000, 10000], ["$10^1$", "$10^2$", "$10^3$", "$10^4$"]
            )
            for ticklabel in ax.get_xticklabels():
                if ticklabel.get_text() == "$10^1$":
                    ticklabel.set_horizontalalignment("left")
                if ticklabel.get_text() == "$10^4$":
                    ticklabel.set_horizontalalignment("right")
            ax.set_xlim(10, 10000)
            ax.set_ylim(50, 150)
            ax.grid(True, which="both", ls="--", alpha=0.5)

            # Control column titles
            if i == 0:
                if distance[j] is None:
                    title = f"{station}"
                else:
                    title = f"{station} (R = {distance[j] / 1e3:.1f} km)"
                ax.set_title(title)

            # Control row titles
            if j == 0:
                ax.set_ylabel(
                    f"Channel {channel}\n PSD (dB re 1 $\\mathrm{{\\mu Pa^2/Hz}}$)"
                )
            else:
                ax.set_yticklabels([])

            # Control x-axis ticks and labels
            if i == len(channels) - 1:
                ax.set_xlabel("Frequency (Hz)")
            else:
                ax.set_xticklabels([])

    lines = [
        plt.Line2D([0], [0], color="tab:red", lw=2),
        plt.Rectangle((0, 0), 1, 1, color="tab:red", alpha=0.25),
        plt.Line2D([0], [0], color="tab:blue", lw=1),
    ]
    labels = ["Strikes (mean)", "Strikes (CI)", "Ambient"]
    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.04),
        bbox_transform=fig.transFigure,
    )

    fig.suptitle(
        f"Event {event_name} | Total strikes: {view.shape[0]:,}",
        y=0.925,
    )
    return fig


def plot_pulse_coherence(
    time_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    n_traces: int | None = None,
    title: str | None = None,
    channel: int | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    n_traces = corr_matrix.shape[0] if n_traces is None else n_traces
    [ax.plot(time_matrix[k], corr_matrix[k], "k-", alpha=0.5) for k in range(n_traces)]
    ax.set_xlim(0, 33)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("Time elapsed from pulse [s]")
    ax.set_ylabel("Normalized Correlation")
    ax.grid(True)
    ax.set_title(f"{n_traces} pulses from {title} Channel {channel}")
    return fig


def plot_pulse_dispersion(
    streams: list[DataStream],
    stations: list[str],
    window_func: callable,
    channel: int = 0,
    tperseg: int = 0.05,
    pcoverlap: float = 0.90,
    nfft_factor: float = 2,
    vmin: float = 80.0,
    vmax: float = 130.0,
    plot_freqs_between: Sequence[float] | None = (0.0, 500.0),
    filt: str = "highpass",
    filt_freq: float = 10.0,
) -> plt.Figure:
    fig = plt.figure(figsize=(3, 5))
    outer_gs = gridspec.GridSpec(3, 1, hspace=0.1)

    for i, ds in enumerate(streams):
        if filt is not None:
            ds.filter(filt, freq=filt_freq)
        data = ds.data[channel]
        if i == 0:
            fs = ds.stats.sampling_rate
            t_series = (ds.time_vector - ds.time_vector[0]).astype(
                float
            ) / TIME_CONVERSION_FACTOR
            xlim = (t_series.min(), t_series.max())

            nperseg, win, hop = utils.spectrogram_args(
                window_func, tperseg, fs, pcoverlap
            )
            nfft = int(nperseg * nfft_factor)

        STFT = signal.ShortTimeFFT(win, hop, fs, mfft=nfft, scale_to="psd")
        S = STFT.stft(data)
        SdB = 20 * np.log10(np.abs(S))
        f = STFT.f
        t_spec = STFT.t(len(data))

        # Plot only specified frequencies:
        if plot_freqs_between is not None:
            mask = (f >= plot_freqs_between[0]) & (f <= plot_freqs_between[1])
            f = f[mask]
            SdB = SdB[mask, :]

        inner_gs = outer_gs[i].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.0)

        # Time series
        ax = fig.add_subplot(inner_gs[0])
        ax.plot(t_series, data / 1e6, zorder=20, c="k", linewidth=0.5)
        ax.set_xlim(xlim)
        ax.grid(True)
        ax.set_xticklabels([])

        if i == 2:
            ax.set_ylabel("Pressure\n(Pa)")

        # Spectrogram
        ax = fig.add_subplot(inner_gs[1])
        ax.pcolormesh(
            t_spec, f, SdB, shading="auto", vmin=vmin, vmax=vmax, cmap="cmo.thermal"
        )
        ax.text(
            0.01,
            0.975,
            stations[i],
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
        ax.set_xlim(xlim)

        if i == 2:
            ax.set_xlabel("Reduced time (s)")
            ax.set_ylabel("Frequency\n(Hz)")
        else:
            ax.set_xticklabels([])

    # Add a colorbar to the bottom of the figure:
    cax = fig.add_axes([0.25, -0.01, 0.5, 0.03])
    cbar = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
    cbar.set_label("Power spectral density (dB re 1 $\\mu$Pa$^2$/Hz)")

    return fig


def plot_pulse_stack(
    sig_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    fs: float,
    title: str | None = None,
    channel: int | None = None,
):
    fig, axs = plt.subplots(
        nrows=20, ncols=1, figsize=(10, 10), gridspec_kw={"hspace": 0.0}
    )

    tvec = np.arange(0, sig_matrix.shape[1]) / fs
    for i, ax in enumerate(axs):
        sig = sig_matrix[i]
        ax.plot(tvec, sig / max(abs(sig)), "k")

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_xlim(0, max(tvec))
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f"{corr_matrix[0, i]:.2f}", rotation=0, ha="right", va="center")
        if i == len(axs) - 1:
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title(f"{title} Channel {channel}")

    return fig


def plot_signal(
    time: np.ndarray,
    signal_t: np.ndarray,
    freq: np.ndarray,
    signal_f: np.ndarray,
    suptitle: str | None = None,
):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    ax = axs[0]
    ax.plot(time, signal_t.T)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Time Domain")

    ax = axs[1]
    ax.plot(freq, np.abs(signal_f).T)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Domain")

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_spectrogram(f, t, Sxx, ax=None, vmin=None, vmax=None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    # return ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap="inferno", vmin=vmin, vmax=vmax)
    extent = (t[0], t[-1], f[0], f[-1])
    return ax.imshow(
        10 * np.log10(Sxx),
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )


def plot_spectrograms(
    ds: DataStream,
    title: str | None = None,
    tperseg: int = 1,
    overlap: float = 0.75,
    nfft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    fig_size: tuple[float, float] = (8.0, 6.0),
) -> plt.Figure:
    fs = ds.stats.sampling_rate

    nperseg = int(tperseg * fs)
    overlap_size = int(overlap * nperseg)
    if nfft is None:
        nfft = nperseg

    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=fig_size, gridspec_kw={"hspace": 0.3}
    )
    fig.suptitle(title)

    for i, channel in enumerate(ds.stats.channels):
        f, t, Sxx = signal.spectrogram(
            ds.data[i],
            fs=fs,
            nperseg=nperseg,
            noverlap=overlap_size,
            nfft=nfft,
        )

        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        if vmin is None:
            vmin = Sxx.min()
        if vmax is None:
            vmax = Sxx.max()

        ax = axs[i]
        im = plot_spectrogram(f, t, Sxx, ax=ax, vmin=vmin, vmax=vmax)

        ax.set_title(f"Channel {channel}", fontsize=10, ha="left", x=0)
        if channel == ds.stats.channels[-1]:
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD [$\\mathrm{{{ds.stats.units}}} / \\mathrm{{Hz}}^2$]")

    return fig


def plot_study_area(
    bounds: list[list[float]],
    turbines_df: pd.DataFrame,
    geo_equipment_df: pd.DataFrame,
    vla_equipment_df: pd.DataFrame,
) -> plt.Figure:
    req = bathyreq.BathyRequest()
    data, lonvec, latvec = req.get_area(
        longitude=bounds[0], latitude=bounds[1], size=[400, 400]
    )

    fig = plt.figure(figsize=(3.37, 2), dpi=300)
    ax = plt.gca()

    m = Basemap(
        projection="tmerc",
        llcrnrlat=bounds[1][0],
        urcrnrlat=bounds[1][1],
        llcrnrlon=bounds[0][0],
        urcrnrlon=bounds[0][1],
        resolution="i",
        lon_0=np.mean(lonvec),
        lat_0=np.mean(latvec),
    )
    m.drawmeridians(
        np.arange(bounds[0][0], bounds[0][1], 0.1), labels=[0, 0, 1, 0], ax=ax
    )
    m.drawparallels(
        np.arange(bounds[1][0], bounds[1][1], 0.05), labels=[1, 0, 0, 0], ax=ax
    )
    xlim = m(np.array(bounds[0]), np.ones_like(bounds[0]) * np.mean(bounds[1]))[0]
    ylim = m(np.ones_like(bounds[0]) * np.mean(bounds[0]), np.array(bounds[1]))[1]

    _, ax = plot_bathy(
        data,
        lonvec=m(lonvec, np.ones_like(lonvec) * np.mean(latvec))[0],
        latvec=m(np.ones_like(latvec) * np.mean(lonvec), latvec)[1],
        ax=ax,
    )
    [
        ax.scatter(
            *m(turbines_df.iloc[i]["lon"], turbines_df.iloc[i]["lat"]),
            marker="h",
            c="yellow",
            edgecolors="k",
            # linewidth=1,
            # s=150,
            zorder=50,
            label=turbines_df.iloc[i]["Turbine"],
        )
        for i in range(len(turbines_df))
    ]
    ax.scatter(
        *m(geo_equipment_df["Longitude [dd]"], geo_equipment_df["Latitude [dd]"]),
        marker="^",
        c="tab:green",
        edgecolors="k",
        # linewidth=1,
        # s=150,
        zorder=50,
        label="Geosled",
    )
    ax.scatter(
        *m(
            vla_equipment_df.iloc[0]["Longitude [dd]"],
            vla_equipment_df.iloc[0]["Latitude [dd]"],
        ),
        marker="v",
        c="tab:orange",
        edgecolors="k",
        # linewidth=1,
        # s=150,
        zorder=50,
        label="VLA 1",
    )
    ax.scatter(
        *m(
            vla_equipment_df.iloc[1]["Longitude [dd]"],
            vla_equipment_df.iloc[1]["Latitude [dd]"],
        ),
        marker="v",
        c="tab:red",
        edgecolors="k",
        # linewidth=1,
        # s=150,
        zorder=50,
        label="VLA 2",
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # leg = ax.legend(facecolor="white", edgecolor="black", loc="upper right")
    # Make the legend so it only shows one symbol for all the turbines
    handles, labels = ax.get_legend_handles_labels()
    labels = ["Turbines", "Geosled", "VLA1", "VLA2"]
    # handles = handles[:3]
    leg = ax.legend(
        handles[-4:], labels, facecolor="white", edgecolor="black", loc="upper right"
    )
    leg.get_frame().set_alpha(None)

    scalebar = AnchoredSizeBar(
        ax.transData,
        5000,
        "5 km",
        "lower right",
        pad=0.1,
        color="k",
        frameon=True,
        size_vertical=200,
    )
    ax.add_artist(scalebar)

    return fig


def plot_summed_pulses(sig_matrix: np.ndarray, fs: float):
    fig = plt.figure(figsize=(15, 5))
    ax = plt.gca()

    summed = sig_matrix.mean(axis=0)
    tvec = np.arange(0, sig_matrix.shape[1]) / fs
    [ax.plot(tvec, sig_matrix[i], "k", alpha=0.5) for i in range(10)]
    ax.plot(tvec, summed, "r", linewidth=2, zorder=50)
    ax.set_xlim(0, max(tvec))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True)
    return fig, summed


def plot_time_series(
    ds: DataStream,
    ymin: float | None = None,
    ymax: float | None = None,
    fig_size: tuple[float] = (12, 6),
    title: str | None = None,
):
    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=fig_size, gridspec_kw={"hspace": 0.25}
    )
    fig.suptitle(title)

    for i, channel in enumerate(ds.stats.channels):
        ax = axs[i]
        ax.plot(ds.time_vector, ds.data[i])
        ax.set_xlim(ds.time_vector[0], ds.time_vector[-1])

        ax.set_ylabel(f"Channel {channel}")
        if channel == ds.stats.channels[-1]:
            ax.set_xlabel("Time")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    if ymin is None:
        ymin = ds.data.min()
    if ymax is None:
        ymax = ds.data.max()
    [ax.set_ylim(ymin, ymax) for ax in axs]

    return fig
