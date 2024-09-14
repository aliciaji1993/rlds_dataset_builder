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

from trajectory_parser import parse_trajectory

CONFIG_FILE_PATH = "/home/yufeng/rlds_dataset_builder/config/nomad.yaml"
ENCODER_PATH = (
    "/home/yufeng/.cache/tfhub_modules/google/universal-sentence-encoder-large/5"
)


class Sacson(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "Actions predicting 8 future positions",
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

        data_config = config["datasets"]["sacson"]
        self.data_folder = data_config["data_folder"]
        self.train_folder = data_config["train"]
        self.test_folder = data_config["test"]

        self.end_slack = data_config["end_slack"] if "end_slack" in data_config else 0
        self.waypoint_spacing = (
            data_config["waypoint_spacing"] if "waypoint_spacing" in data_config else 1
        )
        self.image_size = config["image_size"]
        self.len_traj_pred = 8

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
                                        shape=(3,),
                                        dtype=np.float32,
                                        doc="Robot current absolute state (x, y, yaw)",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(8, 2),
                                dtype=np.float32,
                                doc="Robot movement action, in current (x, y) coordinates",
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

        # compute language embedding
        language_instruction = "continue the trajectory"
        language_embedding = self._embed([language_instruction])[0].numpy()

        def _parse_trajectory(traj_folder):
            steps = parse_trajectory(
                traj_folder=traj_folder,
                image_size=self.image_size,
                len_traj_pred=self.len_traj_pred,
                end_slack=self.end_slack,
            )

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(steps):
                episode.append(
                    {
                        "observation": {
                            "image": step["image"],
                            "state": step["state"].astype(np.float32),
                        },
                        "action": step["action"],
                        "discount": 1.0,
                        "reward": float(i == (len(steps) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(steps) - 1),
                        "is_terminal": i == (len(steps) - 1),
                        "language_instruction": language_instruction,
                        "language_embedding": language_embedding,
                    }
                )

            # create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {"traj_folder": traj_folder},
            }

            # if you want to skip an example for whatever reason, simply return None
            return traj_folder, sample

        split_folder = self.train_folder if split == "train" else self.test_folder

        traj_names_file = os.path.join(split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            traj_names = file_lines.split("\n")
        traj_names.remove("")

        # filter invalid trajectories
        # def validate_trajectory(traj_name):
        #     if not traj_name:
        #         return False
        #     # check if trajectory folder exists
        #     traj_folder = os.path.join(self.data_folder, traj_name)
        #     if not os.path.exists(traj_folder):
        #         print(f"Skipping non-existing trajectory {traj_name}...")
        #         return False
        #     # check if trajectory contains enough steps
        #     traj_len = len(glob.glob(os.path.join(traj_folder, "*.jpg")))
        #     if traj_len <= self.end_slack + self.len_traj_pred:
        #         print(f"Skipping short trajectory {traj_name} of length {traj_len}...")
        #         return False
        #     return True

        # traj_names = filter(validate_trajectory, traj_names)

        traj_folders = [os.path.join(self.data_folder, name) for name in traj_names]

        # for smallish datasets, use single-thread parsing
        for traj_folder in tqdm.tqdm(traj_folders, disable=True, dynamic_ncols=True):
            yield _parse_trajectory(traj_folder)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(traj_folders) | beam.Map(_parse_trajectory)
