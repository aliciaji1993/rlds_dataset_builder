import os
import pickle
import numpy as np

from pathlib import Path

from .data_utils import img_path_to_data, to_local_coords


def parse_trajectory(traj_folder, image_size, len_traj_pred, end_slack):
    with open(os.path.join(traj_folder, "traj_data.pkl"), "rb") as f:
        traj_data = pickle.load(f)
    traj_len = len(traj_data["position"]) - end_slack - len_traj_pred

    images = []
    actions = []
    states = []
    for i in range(0, traj_len):
        # load image
        image_path = os.path.join(traj_folder, f"{i}.jpg")
        image = img_path_to_data(image_path, image_size)

        # load position and yaw
        start_index = i
        end_index = i + len_traj_pred + 1
        yaws = traj_data["yaw"][start_index:end_index].astype(np.float32)
        positions = traj_data["position"][start_index:end_index].astype(np.float32)

        # compute relative (x, y) coordinates of next n positions in current position
        waypoints = to_local_coords(positions, positions[0], yaws[0])
        action = waypoints[1:]

        # compute states as absolute (x, y, yaw)
        state = np.concatenate([positions, yaws.reshape(-1, 1)], -1)[0]

        # append image, position, yaw and action to data list
        images.append(image)
        actions.append(action)
        states.append(state)

    return dict(images=images, actions=actions, states=states)
