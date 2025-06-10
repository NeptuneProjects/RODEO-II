# -*- coding: utf-8 -*-
"""
Constructors for environment models.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""
import concurrent.futures
import copy

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.fft as fft
from tqdm import tqdm
from tritonoa.modeling.at.kraken import run_kraken
from tritonoa.signal import waveforms


def build_base_environment(ctd_df: pd.DataFrame) -> dict:
    base_environment = construct_environment()
    layerdata = base_environment["layerdata"]
    z = ctd_df["Depth"].values
    c_p = ctd_df["Speed of sound"].values
    z = np.append(z, layerdata[0]["z"].max())
    c_p = np.append(c_p, c_p[-1])
    z = np.insert(z, 0, 0.0)
    # c_p = np.insert(c_p, 0, layerdata[0]["c_p"])
    c_p = np.insert(c_p, 0, c_p[0])

    new_layerdata = [
        {"z": z, "c_p": c_p},
        layerdata[1],
    ]
    base_environment.update({"layerdata": new_layerdata})
    return base_environment


def compute_greens_function(
    environment: dict,
    freq_bounds: tuple[float, float],
    fs: float,
    num_points: int,
    range_m: float,
    freq: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:

    def _enforce_symmetry(
        fft_data: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        """Make FFT data conjugate symmetric to ensure real IFFT output"""
        n = len(fft_data)
        symmetric_fft = fft_data.copy()

        # DC component should be real
        symmetric_fft[0] = np.real(symmetric_fft[0])

        # Nyquist frequency component should be real (if n is even)
        if n % 2 == 0:
            symmetric_fft[n // 2] = np.real(symmetric_fft[n // 2])

        # Make all other frequencies conjugate symmetric
        for k in range(1, n // 2):
            symmetric_fft[n - k] = np.conj(symmetric_fft[k])

        return symmetric_fft

    def _greens_function(freq: float) -> npt.NDArray[np.complex128]:
        model = run_kraken(env | {"freq": freq})
        return model.modes.p.squeeze()

    def _get_freqs_from_band(
        num_points: int, freq_band: tuple[float, float], fs: float
    ):
        f = fft.fftfreq(num_points, d=1 / fs)
        inds = np.argwhere((f >= freq_band[0]) & (f <= freq_band[1])).squeeze()
        return f[inds], inds

    def _pad_freqs(
        freq: npt.NDArray[np.float64],
        computed_freq_inds: npt.NDArray[np.float64],
        arr: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64]:
        """Pad frequencies to match the number of points in the FFT."""
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]

        padded_arr = np.zeros((len(freq), arr.shape[1]), dtype=complex)
        # Fill the padded array with the computed frequencies
        padded_arr[computed_freq_inds, :] = arr
        return padded_arr.T.squeeze()


    env = copy.deepcopy(environment)

    # Define frequencies
    computed_freqs, freq_inds = _get_freqs_from_band(num_points, freq_bounds, fs=fs)

    # Define the environment
    env.update({"rec_r": range_m / 1e3})

    # Compute Green's Function in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chan_resp_f = list(
            tqdm(
                executor.map(_greens_function, computed_freqs),
                total=len(computed_freqs),
            )
        )
    chan_resp_f_partial = np.array(chan_resp_f)

    chan_resp_freq = _pad_freqs(freq, freq_inds, chan_resp_f_partial)

    # Make chan_resp_freq conjugate symmetric:
    chan_resp_freq = _enforce_symmetry(chan_resp_freq)

    # Compute the IFFT of the channel response
    chan_resp_time = fft.ifft(chan_resp_freq * np.conj(chan_resp_freq), n=num_points)

    return chan_resp_time, chan_resp_freq


def construct_environment() -> dict:
    """Construct a simple environment model.

    Returns:
        Dictionary containing the environment model.
    """
    z = np.array([0, 35])
    return {
        # 1. General
        "title": "SouthFork",
        "model": "KRAKENC",
        # 2. Top medium (halfspace)
        "top_opt": "CVW    ",
        # 3. Layered media
        "layerdata": [
            # Water:
            {"z": z, "c_p": 1500.0, "rho": 1.0},
            # Medium Sand:
            {"z": [z.max(), 70], "c_p": 1640.0, "c_s": 950.0, "rho": 1.8},
        ],
        # 4. Bottom medium - limestone
        "bot_opt": "A",
        "bot_c_p": 3500.0,
        "bot_c_s": 2000.0,
        "bot_rho": 2.4,
        # "bot_a_p": 2.0,
        # 5. Speed constraints
        "clow": 0.0,
        "chigh": 2000.0,
        # 6. Receiver parameters
        "rec_z": 20.0,
        "rec_r": 2.0,
        # 7. Source parameters
        "src_z": 5.0,
    }


def generate_point_source_signals(
    num_signals: int,
    signal_duration: float,
    fs: float,
    pulse_duration: float,
    fc: float = 100.0,
    snr_db: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = int(signal_duration * fs)
    if isinstance(pulse_duration, float):
        pulse_duration = [pulse_duration] * num_signals
    if isinstance(fc, float):
        fc = [fc] * num_signals
    if isinstance(snr_db, float):
        snr_db = [snr_db] * num_signals

    source_signals_time = np.zeros((num_signals, num_samples))
    for i in range(num_signals):
        t, source_signals_time[i] = point_source_signal(
            signal_duration=signal_duration,
            fs=fs,
            pulse_duration=pulse_duration[i],
            fc=fc[i],
            snr_db=snr_db[i],
        )
    source_signals_freq = fft.fft(source_signals_time, axis=-1)
    freqvec = fft.fftfreq(num_samples, 1 / fs)

    return t, source_signals_time, freqvec, source_signals_freq


def generate_source_signal(
    type: str = "point", **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if type == "point":
        return generate_point_source_signals(**kwargs)
    else:
        raise ValueError(f"Invalid source type: {type}")


def point_source_signal(
    signal_duration: float,
    fs: float,
    pulse_duration: float,
    fc: float = 100.0,
    snr_db: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    pulse = waveforms.generate_gaussian_pulse(pulse_duration, fc, fs, sigma=0.005)

    t = np.arange(0, signal_duration, 1 / fs)
    num_samples = t.size
    x = np.concatenate([pulse, np.zeros(num_samples - pulse.size)])

    signal_power = np.mean(pulse**2)
    noise_rms = np.sqrt(signal_power / (10 ** (snr_db / 10)))
    noise = np.random.normal(0, 1, num_samples) * noise_rms
    return t, x + noise
