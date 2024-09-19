import base64
import draccus
import itertools
import numpy as np
import random

from dataclasses import dataclass
from enum import IntEnum
from openai import OpenAI
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

from process_data.trajectory_parser import parse_trajectory

class EvalType(IntEnum):
    RANDOM_TRAJ = 1         # random step, random traj from specified data split (default to test)
    SINGLE_TRAJ = 3           # all steps in specified traj (use defaulf if not provided)
    SINGLE_SPLIT = 4          # all steps in all traj from specified data split (default to test)
    ENTIRE_DATASET = 5        # entire dataset

@dataclass
class EvalConfig:
    # eval settings
    eval_type: EvalType = EvalType.ENTIRE_DATASET
    eval_split: str = "test"

    # ground truth data directory, trajectory name & step for eval
    data_split_dir = Path("/data/jiyufeng/nomad_dataset/data_splits/sacson/")
    data_root_dir = Path("/data/jiyufeng/nomad_dataset/sacson")
    traj_name: str = "Dec-12-2022-bww8_00000034_1"
    step: int = 0
    window_size: int = 1

    # training settings
    run_root_dir = Path("/data/jiyufeng/openvla/lora/run")
    exp_id: str = "openvla-7b+sacson+b16+lr-0.0005+lora-r32+dropout-0.0"
    run_dir = run_root_dir / exp_id

    # output settings
    output_root_dir = Path("/data/jiyufeng/openvla/instruct")
    image_size = [96, 96]
    end_slack: int = 3
    len_traj_pred: int = 8


INTRODUCTION = "Assume you are giving instructions to direct a robot in an office or school building. Here are some examples: "

EXAMPLE_INSTRUCTIONS = [
    "Move forward and slightly to the right, then make a sharp right turn and continue straight.",
    "Turn slightly to the right and continue curving to the right.",
    "Move forward and curve to the left, then continue straight.",
    "Turn slightly to the right and align to the hallway, then continue straight ahead.",
    "Move a few steps forward, then curve to the right."
]

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def generate_text_image_content(instruction, base64_image):
    return [
        {
          "type": "text",
          "text": instruction
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
    ]

def send_message(openai, system_prompt, content, verbose=True):
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
    )
    response = completion.choices[0].message
    if verbose:
        print("Message text: ", " ".join([c["text"] for c in content[:2]]))
        print("Response: ", response.content)
    return response

@draccus.wrap()
def generate(cfg: EvalConfig) -> None:
    # format prompt
    system_prompt = [{"type": "text", "text": INTRODUCTION}]
    examples = [generate_text_image_content(EXAMPLE_INSTRUCTIONS[i], encode_image(f"{i}.jpg")) for i in range(5)]
    examples = list(itertools.chain(*examples))
    system_prompt.append(examples)

    print(system_prompt)

    # init openai
    openai = OpenAI()

    def eval_step(image: Image, actions: np.ndarray, save_path: Path):
        fig, axs = plt.subplots(1, 2)
        
        # show observation image
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[0].set_title(f"observation")

        def plot_actions(ax, actions, color="b"):
            lim = np.max(np.abs(actions))
            # switch x and y axis, as x represents forward movement in real world
            ax.plot(-actions[:, 1], actions[:, 0], f"{color}o")  # 'o' means circles
            ax.plot(-actions[:, 1], actions[:, 0], f"{color}-")  # '-' means solid line
            # mark start spot as green and end spot as red 
            ax.plot(-actions[0, 1], actions[0, 0], f"go")  # 'go' means green circles
            ax.plot(-actions[-1, 1], actions[-1, 0], f"ro")  # 'go' means red circles
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_box_aspect(aspect=1)

        # plot actions
        plot_actions(axs[1], actions)
        axs[1].set_title("trajectory")

        # set layout and draw
        fig.tight_layout()
        fig.canvas.draw()

        w,h = fig.canvas.get_width_height()
        # canvas = Image.frombytes('RGB', (w,h), fig.canvas.tostring_rgb())

        # generate instruction from gpt-4o
        prompt = "How will you instruct a robot to execute this trajectory?"
        canvas = base64.b64encode(fig.canvas.tostring_rgb()).decode('utf-8')
        instruction = send_message(openai, system_prompt, generate_text_image_content(prompt, canvas))

        # save and close
        fig.suptitle(instruction, fontsize=12)
        fig.savefig(save_path, dpi=300)
        plt.close()

    def eval_traj(traj_path: Path):
        steps = parse_trajectory(traj_path, cfg.image_size, cfg.len_traj_pred, cfg.end_slack)
        Path(cfg.output_root_dir / traj_path.name).mkdir(parents=True, exist_ok=True)
        for i, step in enumerate(steps):
            out_path = Path(cfg.output_root_dir / traj_path.name / f"step_{i}.jpg")
            eval_step(step["image"], step["action"], out_path)


    # eval
    if cfg.eval_type == EvalType.RANDOM_TRAJ:
        traj_path = random.choice(list(cfg.data_root_dir.iterdir()))
        eval_traj(traj_path)
    elif cfg.eval_type == EvalType.SINGLE_TRAJ:
        eval_traj(cfg.data_root_dir / cfg.traj_name)
    elif cfg.eval_type == EvalType.SINGLE_SPLIT:
        with open(Path(cfg.data_split_dir / cfg.eval_split / "traj_names.txt"), "rb") as f:
            traj_names = f.read().decode("utf-8").splitlines()
        traj_paths = [Path(cfg.data_root_dir / str(traj_name)) for traj_name in traj_names]
        for traj_path in traj_paths:
            eval_traj(traj_path)
    elif cfg.eval_type == EvalType.ENTIRE_DATASET:
        traj_paths = list(cfg.data_root_dir.iterdir())
        for traj_path in traj_paths:
            eval_traj(traj_path)
    else:
        raise KeyError("Not supported evaluation type: ", cfg.eval_type)


if __name__ == "__main__":
    generate()