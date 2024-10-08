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
from data_utils.gen_instruct.chat_wrapper import ChatGPT
from data_utils.gen_instruct.generate import (
    InstructType,
    ContextType,
    generate_instruction,
)
from data_utils.gen_instruct.template import INSTRUCT_TEMPLATES, INTRO_TEMPLATES


CONFIG_FILE_PATH = (
    "/home/yufeng/rlds_dataset_builder/data_utils/config/dataset_config.yaml"
)
ENCODER_PATH = (
    "/home/yufeng/.cache/tfhub_modules/google/universal-sentence-encoder-large/5"
)


class RLDSDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "Actions predicting 8 future positions",
        "3.0.0": "Actions predicting 8 future positions, plus language instruction",
    }

    def __init__(self, dataset_name: str, len_traj_pred: int = 8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # configurable params
        self.dataset_name = dataset_name
        self.len_traj_pred = len_traj_pred

        # embedding module
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        # workaround (no access to tfhub.dev):
        # 1) manual download "https://hub.tensorflow.google.cn/google/universal-sentence-encoder-large/5"
        # 2) unzip and move to the tfhub cache folder
        # 3) load model directly from the cache folder by passing the path to hub.load
        self._embed = hub.load(ENCODER_PATH)

        # language instruction generation
        self.context_type = ContextType.OBS_1_ACTIONS_STRING
        self.instruction_type = InstructType.FORMAT_ACTION
        self.instructions = INSTRUCT_TEMPLATES[self.instruction_type]
        self.chat = ChatGPT(system_prompt=INTRO_TEMPLATES[self.context_type])

        # dataset configs
        with open(CONFIG_FILE_PATH, "r") as f:
            config = yaml.safe_load(f)
        data_config = config["datasets"][dataset_name]
        self.data_folder = data_config["data_folder"]
        self.train_folder = data_config["train"]
        self.test_folder = data_config["test"]
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
                            "language_reasoning": tfds.features.Text(
                                doc="Language Reasoning."
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

        def _parse_trajectory(traj_folder):
            traj = parse_trajectory(
                traj_folder=traj_folder,
                image_size=self.image_size,
                len_traj_pred=self.len_traj_pred,
                end_slack=self.end_slack,
            )

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            traj_len = len(traj["images"])
            for i in range(traj_len):
                try:
                    response = generate_instruction(
                        chat=self.chat,
                        images=traj["images"][i],
                        actions=traj["actions"][i],
                        instructions=self.instructions,
                        context_type=self.context_type,
                    )
                    # compute lanuage embedding
                    language_reasoning = response["reasoning"]
                    language_instruction = response["instruction"]
                    language_embedding = self._embed([language_instruction])[0].numpy()
                except Exception as e:
                    language_reasoning = ""
                    language_instruction = "continue the trajectory"
                    language_embedding = self._embed([language_instruction])[0].numpy()

                # add step
                episode.append(
                    {
                        "observation": {
                            "image": traj["images"][i],
                            "state": traj["states"][i].astype(np.float32),
                        },
                        "action": traj["actions"][i],
                        "discount": 1.0,
                        "reward": float(i == (traj_len - 1)),
                        "is_first": i == 0,
                        "is_last": i == (traj_len - 1),
                        "is_terminal": i == (traj_len - 1),
                        "language_reasoning": language_reasoning,
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

        traj_folders = [os.path.join(self.data_folder, name) for name in traj_names]

        # for smallish datasets, use single-thread parsing
        # for traj_folder in tqdm.tqdm(traj_folders, dynamic_ncols=True):
        #     yield _parse_trajectory(traj_folder)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return beam.Create(traj_folders) | beam.Map(_parse_trajectory)
