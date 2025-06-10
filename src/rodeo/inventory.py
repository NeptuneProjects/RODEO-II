#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import tomllib

from dotenv import load_dotenv
import numpy as np
from tritonoa.data.inventory import Inventory
from tritonoa.data.signal import SignalParams
from tritonoa.data.time import ClockParameters

load_dotenv()


def build_inventory(config_path: Path) -> Inventory:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for key in config:
        logging.info(f"Building inventory for {key}.")
        
        data_cfg = config[key]["data"]
        clock_cfg = config[key]["clock"]
        hyd_cfg = config[key]["hydrophone"]

        time_check_0 = np.datetime64(clock_cfg["time_check_0"])
        time_check_1 = np.datetime64(clock_cfg["time_check_1"])

        clock_params = ClockParameters(
            time_check_0=time_check_0,
            time_check_1=time_check_1,
            offset_0=clock_cfg["offset_0"],
            offset_1=clock_cfg["offset_1"],
        )
        signal_params = SignalParams(
            gain=hyd_cfg["fixed_gain"],
            sensitivity=hyd_cfg["sensitivity"],
        )

        inv = Inventory()
        inv.build(
            dataset_path=data_cfg["directory"],
            glob_pattern=data_cfg["glob_pattern"],
            clock_params=clock_params,
            conditioner=signal_params,
        )
        savepath = Path(data_cfg["destination"])
        inv.save(savepath)

        logging.info(f"Inventory saved to {savepath.resolve()}")
