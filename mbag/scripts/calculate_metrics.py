import json
import logging
import os
import pathlib
import pickle
import zipfile
from typing import Dict, List

import tqdm
from sacred import Experiment

from mbag.evaluation.episode import MbagEpisode
from mbag.evaluation.metrics import (
    MbagEpisodeMetrics,
    calculate_mean_metrics,
    calculate_metrics,
)

ex = Experiment()


@ex.config
def sacred_config():
    # Path to directory containing episodes.
    evaluate_dir = ""  # noqa: F841
    # Path to output file containing metrics. Can only be provided if there is one episodes file in evaluate_dir and recursive=False.
    out_fname = ""  # noqa: F841
    # Flag to recursively search for episodes in subdirectories.
    recursive: bool = False  # noqa: F841
    # Flag to overwrite existing files.
    overwrite: bool = False  # noqa: F841


def _make_filename(directory: str, filename: str, overwrite: bool) -> str:
    """Create a filename for the metrics in the directory.

    If overwrite is False and the file already exists, a unique filename is created by
    appending a counter to the filename. Otherwise, the filename is returned as is.

    Args:
        directory: The directory to save the file in.
        filename: The filename to save the metrics to.
        overwrite: Flag to overwrite existing files.

    Returns:
        The full path to the file.
    """
    # Create the full path for the file
    full_path = os.path.join(directory, filename)
    if overwrite:
        return full_path

    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)
    # Set the initial unique number
    counter = 1

    # Check if the file already exists and find a unique name
    while os.path.exists(full_path):
        # Create a new filename with the counter appended
        new_filename = f"{name}_{counter}{extension}"
        full_path = os.path.join(directory, new_filename)
        counter += 1

    return full_path


@ex.automain
def main(  # noqa: C901
    evaluate_dir: str,
    out_fname: str,
    recursive: bool,
    overwrite: bool,
    _log: logging.Logger,
) -> List[Dict]:
    if not os.path.exists(evaluate_dir):
        raise FileNotFoundError(f"Directory {evaluate_dir} does not exist")
    if out_fname and recursive:
        raise ValueError("recursive and out_fname are mutually exclusive")
    if out_fname and os.path.exists(out_fname) and not overwrite:
        raise FileExistsError(
            f"{out_fname} already exists. Use overwrite=True to overwrite"
        )

    patterns = ["episodes.zip", "episodes.pickle", "episode_info.pickle"]
    if recursive:
        patterns = [f"**/{pattern}" for pattern in patterns]
    files = sum(
        (list(pathlib.Path(evaluate_dir).glob(pattern)) for pattern in patterns), []
    )
    if not files:
        raise FileNotFoundError(f"No episodes files found in {evaluate_dir}")
    if out_fname and len(files) > 1:
        raise ValueError(
            f"out_fname should not be specified because multiple episodes files were found: {files}"
        )

    results_list = []
    for file in tqdm.tqdm(files, desc="Files"):
        episodes: List[MbagEpisode]
        _log.info(f"Loading episodes from {file}...")
        if file.suffix == ".zip":
            with zipfile.ZipFile(file, "r") as episodes_zip:
                with episodes_zip.open("episodes.pickle") as episodes_file:
                    episodes = pickle.load(episodes_file)
        else:
            with open(file, "rb") as episodes_file:
                episodes = pickle.load(episodes_file)

        episode_metrics: List[MbagEpisodeMetrics] = []
        for episode in tqdm.tqdm(episodes, desc="Episodes", leave=False):
            episode_metrics.append(calculate_metrics(episode))

        results = {
            "mean_metrics": calculate_mean_metrics(episode_metrics),
            "episode_metrics": episode_metrics,
        }
        results_list.append(results)

        if out_fname:
            curr_out_fname = out_fname
        else:
            out_dir = os.path.dirname(file)
            curr_out_fname = _make_filename(
                out_dir, "metrics.json", overwrite=overwrite
            )
        _log.info(f"Saving metrics to {curr_out_fname}")
        with open(curr_out_fname, "w") as out_file:
            json.dump(results, out_file)

    return results_list
