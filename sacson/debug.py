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


def main():
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = "sacson"
    data_config = config["datasets"][dataset_name]
    data_folder = data_config["data_folder"]
    train_split_folder = data_config["train"]
    test_split_folder = data_config["test"]

    end_slack = data_config["end_slack"] if "end_slack" in data_config else 0
    waypoint_spacing = (
        data_config["waypoint_spacing"] if "waypoint_spacing" in data_config else 1
    )

    image_size = config["image_size"]

    data_split_folder = train_split_folder

    traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
    with open(traj_names_file, "r") as f:
        file_lines = f.read()
        traj_names = file_lines.split("\n")
    if "" in traj_names:
        traj_names.remove("")

    # debug iterations
    for traj_name in traj_names:
        print("Processsing trajectory data for: ", traj_name)
        traj_folder = os.path.join(data_folder, traj_name)
        with open(os.path.join(traj_folder, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        traj_len = len(traj_data["position"]) - end_slack

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        # episode = []
        for i in range(0, traj_len):
            # compute Kona language embedding
            # language_instruction = "Where can the robot go?"
            # language_embedding = self._embed([language_instruction])[0].numpy()

            # load image
            image_path = traj_folder + f"/{i}.jpg"
            # with open(image_path, "rb") as f:
            image = img_path_to_data(image_path, image_size)

            # compute relative (x, y) corrdinates of next pos w.r.t current pos
            position = traj_data["position"][i].astype(np.float32)
            action = to_local_coords(
                traj_data["position"][i + 1],
                traj_data["position"][i],
                traj_data["yaw"][i],
            )

            image = cv2.imread(image_path)
            info = f"state: {position}, action: {action}"
            cv2.imshow(info, image)
            cv2.waitKey(3000)
            print(info)

        break


if __name__ == "__main__":
    main()
