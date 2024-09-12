import tensorflow as tf
import numpy as np
import PIL
import io
import glob
import os

OUTPUT_ROOT = "./output"

# load tensorflow dataset
dataset_folder = "/media/yufeng/tensorflow_datasets/sacson/1.0.0"
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
    output_dir = os.path.join(OUTPUT_ROOT, traj_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get images list
    images = example.features.feature.get("steps/observation/image").bytes_list.value
    print("length of trajectory is:", len(images))

    if len(images) == 0:
        num_empty += 1
        continue

    # get actions list
    actions = example.features.feature.get("steps/action").float_list.value
    actions = np.asarray(actions).reshape(-1, 2)

    # length of actions and images should be the same
    assert actions.shape[0] == len(images)

    # get first observation image
    for j in range(len(images)):
        image_bytes_stream = io.BytesIO(images[j])
        image = PIL.Image.open(image_bytes_stream)
        # convert to numpy and inspect image size
        image_array = np.asarray(image)
        assert image_array.shape == (96, 96, 3)
        # save to local path for later inspection
        image.save(os.path.join(output_dir, f"step_{j}_action_{actions[j]}.jpeg"))

print(f"{num_empty} empty records empty out of total {i} records")
