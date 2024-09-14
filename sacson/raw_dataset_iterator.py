import argparse
import cv2

import os
import yaml

from trajectory_parser import parse_trajectory

CONFIG_FILE_PATH = "/home/yufeng/rlds_dataset_builder/config/nomad.yaml"


def main():
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)

    # load data config
    data_config = config["datasets"]["sacson"]

    # load data config details
    data_folder = data_config["data_folder"]
    train_split_folder = data_config["train"]
    test_split_folder = data_config["test"]
    end_slack = data_config["end_slack"] if "end_slack" in data_config else 0
    image_size = config["image_size"]

    # list all trajectory names
    traj_names = []
    for split_folder in [train_split_folder, test_split_folder]:
        traj_names_file = os.path.join(split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            traj_names = file_lines.split("\n")
    if "" in traj_names:
        traj_names.remove("")

    # iterate through and parse all trajectory folders
    skipped_trajs = []
    for traj_name in traj_names:
        print("Processsing trajectory data for: ", traj_name)
        traj_folder = os.path.join(data_folder, traj_name)
        steps = parse_trajectory(
            traj_folder=traj_folder,
            image_size=image_size,
            len_traj_pred=8,
            end_slack=end_slack,
        )
        if not steps:
            skipped_trajs.append(traj_name)
    print("Skipped trajectories: ", skipped_trajs)
    print("Example output: ", steps)


if __name__ == "__main__":
    main()
