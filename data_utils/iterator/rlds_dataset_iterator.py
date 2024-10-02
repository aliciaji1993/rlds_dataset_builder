import draccus
import glob
import numpy as np
import io
import PIL
import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path

from data_utils.gen_instruct.generate import visualize_step

@dataclass
class IterateConfig:
    dataset_folder: str = Path("/media/yufeng/tensorflow_datasets/sacson/2.0.0")
    visualize_dataset: bool = False
    output_root: str = Path("./output")


@draccus.wrap()
def generate(cfg: IterateConfig) -> None:
    # load tensorflow dataset
    dataset_folder = Path(cfg.dataset_folder)
    tfrecords = glob.glob(f"{dataset_folder}/*.tfrecord*")
    raw_dataset = tf.data.TFRecordDataset(tfrecords)

    num_empty_traj = 0
    num_steps = 0
    num_default_instruct = 0

    # iterate each record (trajectory) through the entire dataset
    for i, raw_record in tqdm.tqdm(enumerate(raw_dataset.as_numpy_iterator())):
        example = tf.train.Example()
        example.ParseFromString(raw_record)

        # get trajectory folder
        traj_path = Path(
            example.features.feature.get("episode_metadata/traj_folder")
            .bytes_list.value[0]
            .decode("utf-8")
        )

        # create output directory
        traj_name = traj_path.name
        output_dir = Path(cfg.output_root) / traj_name
        if cfg.visualize_dataset and not output_dir.exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # get images list
        images = example.features.feature.get("steps/observation/image").bytes_list.value
        if len(images) == 0:
            num_empty_traj += 1
            continue
        num_steps += len(images) 

        # get actions list
        actions = example.features.feature.get("steps/action").float_list.value
        actions = np.asarray(actions).reshape(-1, 8, 2)
        lim = np.max(np.abs(actions))
        # length of actions and images should be the same
        assert actions.shape[0] == len(images)

        # get instructions and reasoning
        instructions = example.features.feature.get(
            "steps/language_instruction"
        ).bytes_list.value
        reasonings = example.features.feature.get(
            "steps/language_reasoning"
        ).bytes_list.value

        # traverse through each step
        for step in range(len(images)):
            image_bytes_stream = io.BytesIO(images[step])
            image = PIL.Image.open(image_bytes_stream)
            # convert to numpy and inspect image size
            image_array = np.asarray(image)
            assert image_array.shape == (96, 96, 3)
            
            if instructions[step].decode("utf-8") == "continue the trajectory":
                # print(f"default instruction found at {traj_path} step {step}")
                num_default_instruct += 1

            # visualize step
            if cfg.visualize_dataset:
                visualize_step(
                    image=image,
                    actions=actions[step],
                    instruction=instructions[step].decode("utf-8"),
                    reasoning=reasonings[step].decode("utf-8"),
                    save_path=output_dir / f"step_{step}.jpg",
                )

    print(f"Out of total {i+1} records, total steps {num_steps}: \n"
        f"{num_empty_traj} empty records and "
        f"{num_default_instruct} default instructions")

