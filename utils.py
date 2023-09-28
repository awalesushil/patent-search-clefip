"""Utility functions for the evaluator"""

import os

import random
import requests
import json

from typing import List, Union
from datetime import datetime

import yaml

import dask.dataframe as dd
from wasabi import msg

def get_experiment_name():
    """Create a random experiment name of form <adjective>-<noun>"""
    urls = [
        "https://www.randomlists.com/data/adjectives.json",
        "https://www.randomlists.com/data/cars.json"
    ]

    adjectives = requests.get(urls[0], timeout=10).json()["data"]
    cars = requests.get(urls[1], timeout=10).json()["data"]

    adjective = random.choice(adjectives).lower()
    car = "-".join(random.choice(cars).split(" ")).lower()
    return f"{adjective}-{car}"

def load_config(filename: str) -> dict:
    """Get the configuration"""
    with open(f"config/{filename}", "r", encoding="utf-8") as cfg:
        return yaml.safe_load(cfg)

def start_experiment(config: dict) -> dict:
    """Start the experiment"""
    config["experiment"] = get_experiment_name()
    os.makedirs(f"results/{config['experiment']}", exist_ok=True)
    config["datetime"] = datetime.now().strftime("%Y%m%d%H%M%S")
    with open(f"results/{config['experiment']}/config.yml", "w", encoding="utf-8") as file:
        yaml.dump(config, file)
    msg.info(f"Running experiment: {config['experiment']}")
    return config["experiment"]

def save_json(data: dict, path: str) -> None:
    """Save the json file"""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)

def load_json(path: str) -> dict:
    """Load the json file"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def get_metadata(dataframe: dd.DataFrame, key: str) -> dict:
    """Prepapre metadata for EPO dataset"""
    metadata = {}
    for row in dataframe.itertuples():
        metadata[row.pub] = {key: getattr(row, key)}
    return metadata

def get_dataframe(paths: Union[str, List[str]],
                  silent: bool = False,
                  header: bool = True) -> dd.DataFrame:
    """Get the dataframe"""

    if isinstance(paths, str):
        if not silent:
            msg.info(f"Reading dataframe from {paths}")
        return dd.read_csv(paths, dtype=str) if header else dd.read_csv(paths, dtype=str, header=None)

    dataframe = None
    for path in paths:
        if not silent:
            msg.info(f"Reading dataframe from {path}")
        if dataframe is None:
            if header:
                dataframe = dd.read_csv(path, dtype=str)
            else:
                dataframe = dd.read_csv(path, dtype=str, header=None)
        else:
            if header:
                dd.concat([dataframe, dd.read_csv(path, dtype=str)])
            else:
                dd.concat([dataframe, dd.read_csv(path, dtype=str, header=None)])
    return dataframe

