import tensorflow as tf
import numpy as np
import PIL
import io
import glob
import os

import matplotlib.pyplot as plt

OUTPUT_ROOT = "./output"

# load tensorflow dataset
dataset_folder = "/media/yufeng/tensorflow_datasets/sacson/2.0.0"
tfrecords = glob.glob(f"{dataset_folder}/*.tfrecord*")
raw_dataset = tf.data.TFRecordDataset(tfrecords[0])

num_empty = 0
# iterate each record (trajectory) through the entire dataset
for i, raw_record in enumerate(raw_dataset.as_numpy_iterator()):
    example = tf.train.Example()
    example.ParseFromString(raw_record)

    # get trajectory folder
    traj_folder = (
        example.features.feature.get("episode_metadata/traj_folder")
        .bytes_list.value[0]
        .decode("utf-8")
    )
    print("Processing data for record folder: ", traj_folder)

    # create output directory
    traj_name = traj_folder.split("/")[-1]
    traj_dir = os.path.join(OUTPUT_ROOT, traj_name)
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    # get images list
    images = example.features.feature.get("steps/observation/image").bytes_list.value
    print("length of trajectory is:", len(images))

    if len(images) == 0:
        num_empty += 1
        continue

    # get actions list
    actions = example.features.feature.get("steps/action").float_list.value
    actions = np.asarray(actions).reshape(-1, 8, 2)
    lim = np.max(np.abs(actions))

    # length of actions and images should be the same
    assert actions.shape[0] == len(images)

    # traverse through each step
    for j in range(len(images)):
        image_bytes_stream = io.BytesIO(images[j])
        image = PIL.Image.open(image_bytes_stream)
        # convert to numpy and inspect image size
        image_array = np.asarray(image)
        assert image_array.shape == (96, 96, 3)

        # save obs image to local path
        image.save(os.path.join(traj_dir, f"step_{j}.jpg"))

        # save action trajectory as image
        _, ax = plt.subplots()
        ax.plot(actions[j, :, 0], actions[j, :, 1], "bo")  # 'bo' means blue circles
        ax.plot(actions[j, :, 0], actions[j, :, 1], "b-")  # 'b-' means blue solid line
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.savefig(os.path.join(traj_dir, f"step_{j}_actions.jpg"), dpi=300)
        plt.close()


print(f"{num_empty} empty records empty out of total {i} records")
