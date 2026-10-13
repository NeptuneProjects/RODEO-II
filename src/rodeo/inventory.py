import logging
import tomllib
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tritonoa.data.inventory import Inventory
from tritonoa.data.signal import SignalParams
from tritonoa.data.time import ClockParameters

logger = logging.getLogger(__name__)

load_dotenv()


def build_inventory(config_path: Path) -> None:
    """Build and save an inventory for each sensor in a TOML configuration.

    Each sensor has a `data` table (`directory`, `glob_pattern`, `destination`,
    optional `file_format`) and optional `clock` and `hydrophone` tables.

    TritonOA converts raw data to pressure on read as
    `counts * adc_vref / gain / sensitivity`, with `gain` and `sensitivity` in
    dB. For SHRU recorders, `adc_vref` must therefore be the volts per count,
    2.5 / 2**23, and `sensitivity` is in dB re 1 V/uPa (negative). Without a
    `hydrophone` table the inventory is unconditioned (raw values).
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for key in config:
        logger.info(f"Building inventory for {key}.")

        data_cfg = config[key]["data"]
        clock_cfg = config[key].get("clock")
        hyd_cfg = config[key].get("hydrophone")

        if clock_cfg is None:
            clock_params = ClockParameters()
        else:
            clock_params = ClockParameters(
                time_check_0=np.datetime64(clock_cfg["time_check_0"]),
                time_check_1=np.datetime64(clock_cfg["time_check_1"]),
                offset_0=clock_cfg["offset_0"],
                offset_1=clock_cfg["offset_1"],
            )

        if hyd_cfg is None:
            signal_params = SignalParams()
        else:
            if any(s > 0 for s in hyd_cfg["sensitivity"]):
                raise ValueError(
                    f"{key}: hydrophone sensitivity must be in dB re 1 V/uPa "
                    f"(negative), got {hyd_cfg['sensitivity']}."
                )
            signal_params = SignalParams(
                adc_vref=hyd_cfg.get("adc_vref", [1.0]),
                gain=hyd_cfg["fixed_gain"],
                sensitivity=hyd_cfg["sensitivity"],
            )

        inv = Inventory()
        inv.build(
            dataset_path=data_cfg["directory"],
            glob_pattern=data_cfg["glob_pattern"],
            clock_params=clock_params,
            conditioner=signal_params,
            file_format=data_cfg.get("file_format"),
        )
        savepath = Path(data_cfg["destination"])
        inv.save(savepath)

        logger.info(f"Inventory saved to {savepath.resolve()}")
