import argparse
import cv2
import numpy as np
import os
import pickle
import tqdm
import yaml

from typing import Iterator, Tuple, Any

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from process_data.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

CONFIG_FILE_PATH = "/home/yufeng/rlds_dataset_builder/config/nomad.yaml"
ENCODER_PATH = (
    "/home/yufeng/.cache/tfhub_modules/google/universal-sentence-encoder-large/5"
)


class SacsonDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        # workaround:
        # 1) manual download "https://hub.tensorflow.google.cn/google/universal-sentence-encoder-large/5"
        # 2) unzip and move to the tfhub cache folder
        # 3) load model directly from the cache folder by passing the path to hub.load
        self._embed = hub.load(ENCODER_PATH)

        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)

        self.dataset_name = "sacson"
        data_config = config["datasets"][self.dataset_name]
        self.data_folder = data_config["data_folder"]
        self.train_split_folder = data_config["train"]
        self.test_split_folder = data_config["test"]

        self.end_slack = data_config["end_slack"] if "end_slack" in data_config else 0
        self.waypoint_spacing = (
            data_config["waypoint_spacing"] if "waypoint_spacing" in data_config else 1
        )

        self.image_size = config["image_size"]

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(96, 96, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation."
                                        "Cropped to size 96x96x3, same as in nomad.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(2,),
                                        dtype=np.float32,
                                        doc="Robot current location state, in (x, y) coordinates",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(2,),
                                dtype=np.float32,
                                doc="Robot movement action, in (x, y) coordinates",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "traj_folder": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(split="train"),
            "val": self._generate_examples(split="test"),
        }

    def _generate_examples(self, split) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_trajectory(traj_name):
            traj_folder = os.path.join(self.data_folder, traj_name)
            with open(os.path.join(traj_folder, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            traj_len = len(traj_data["position"]) - self.end_slack

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(0, traj_len):
                # compute Kona language embedding
                language_instruction = "Where can the robot go?"
                language_embedding = self._embed([language_instruction])[0].numpy()

                # load image
                image_path = traj_folder + f"/{i}.jpg"
                # with open(image_path, "rb") as f:
                image = img_path_to_data(image_path, self.image_size)

                # compute relative (x, y) corrdinates of next pos w.r.t current pos
                position = traj_data["position"][i].astype(np.float32)
                action = to_local_coords(
                    traj_data["position"][i + 1],
                    traj_data["position"][i],
                    traj_data["yaw"][i],
                )

                episode.append(
                    {
                        "observation": {
                            "image": image,
                            "state": position,
                        },
                        "action": action,
                        "discount": 1.0,
                        "reward": float(i == (traj_len - 1)),
                        "is_first": i == 0,
                        "is_last": i == (traj_len - 1),
                        "is_terminal": i == (traj_len - 1),
                        "language_instruction": language_instruction,
                        "language_embedding": language_embedding,
                    }
                )

                if False:
                    info = f"state: {position}, action: {action}"
                    cv2.imshow(info, image)

            # create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {"traj_folder": traj_folder},
            }

            # if you want to skip an example for whatever reason, simply return None
            return traj_folder, sample

        data_split_folder = (
            self.train_split_folder if split == "train" else self.test_split_folder
        )

        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            traj_names = file_lines.split("\n")
        if "" in traj_names:
            traj_names.remove("")

        # for smallish datasets, use single-thread parsing
        for traj_name in tqdm.tqdm(traj_names, disable=True, dynamic_ncols=True):
            yield _parse_trajectory(traj_name)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(traj_names) | beam.Map(_parse_trajectory)
