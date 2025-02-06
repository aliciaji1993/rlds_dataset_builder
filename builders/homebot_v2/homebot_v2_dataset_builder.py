import argparse
import glob
import numpy as np
import os
import tqdm
import yaml

from typing import Iterator, Tuple, Any

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from data_utils.convert_dataset import parse_trajectory
from builders.rlds_dataset_builder import RLDSDatasetBuilder

CONFIG_FILE_PATH = "./process_data/config/dataset_config.yaml"


class Homebot_v2(RLDSDatasetBuilder):
    """DatasetBuilder for example dataset."""

    # VERSION = tfds.core.Version("2.0.0")
    # RELEASE_NOTES = {
    #     "1.0.0": "Initial release.",
    #     "2.0.0": "Actions predicting 8 future positions",
    # }

    def __init__(self, *args, **kwargs):
        super().__init__(dataset_name="homebot_v2", **kwargs)
