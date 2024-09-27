import base64
import draccus
import itertools
import io
import numpy as np
import random
import shutil

from dataclasses import dataclass
from enum import IntEnum
from openai import OpenAI
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

from process_data.trajectory_parser import parse_trajectory


class EvalType(IntEnum):
    SINGLE_TRAJ = 0  # all steps in specified traj (use defaulf if not provided)
    SINGLE_SPLIT = 1  # all steps from specified data split (default to test)
    ENTIRE_DATASET = 2  # entire dataset


class InstructType(IntEnum):
    FREE_FORM_INSTRUCTIONS = 0
    MAIN_DIRECTIONS_4 = 1
    MAIN_DIRECTIONS_8 = 2
    FORMATTED_ACTIONS = 3


@dataclass
class EvalConfig:
    # eval settings
    instruction_type: InstructType = InstructType.MAIN_DIRECTIONS_4
    eval_type: EvalType = EvalType.ENTIRE_DATASET
    eval_split: str = "test"

    # ground truth data directory, trajectory name & step for eval
    data_split_dir = Path("/media/yufeng/nomad_dataset/data_splits/sacson/")
    data_root_dir = Path("/media/yufeng/nomad_dataset/sacson")
    traj_name: str = "Dec-12-2022-bww8_00000034_1"
    step: int = 0
    window_size: int = 1

    # output settings
    output_root_dir = Path("/media/yufeng/openvla/instruct")
    image_size = [96, 96]
    end_slack: int = 3
    len_traj_pred: int = 8
    sample_rate: float = 1.0


INTRODUCTION = (
    "Imagine you are providing natural language instructions to guide a robot in an indoor environment. "
    "On the left side, you will see an image showing the robot’s current view, and on the right side, "
    "an image of a 2D map representing the movement you want the robot to perform, starting at the "
    "green circle in the center and ending at the red circle on the edge.\n"
    "The positive x-axis indicates movement to the right, and the negative x-axis indicates movement to "
    "the left. The positive y-axis represents forward movement, while the negative y-axis indicates "
    "backward movement.\n"
    "IMPORTANT:\n"
    "1.	Start from the correct point—the green circle at the center of the map. \n"
    "2.	Ensure your instruction is concise and unambiguous, with only one object or goal that matches "
    "your description. Do not halluciate nor make up objects in your sight.\n"
    "3.	If multiple valid directions are visible, clarify your instruction to distinguish the intended "
    "direction from others. \n"
)

FREE_FORM_INSTRUCTIONS = [
    "Describe the trajectory using natural language. Example instructions:",
    "Example 1. make a sharp right turn to turn away from the wall.",
    "Example 2. Turn slightly to the right, continue curving to the right into the hallway.",
    "Example 3. Move straight ahead towards the door in the right corner.",
    "Example 4. Turn slightly to the right to align with the hallway, then continue straight ahead.",
    "Example 5. Move a few steps forward, then curve to the right.",
]

MAIN_DIRECTIONS_4 = [
    "Pick the instruction that best describes the direction you want the robot to take:",
    "take a left turn",
    "take a right turn",
    "move forward",
    "move backward",
]

MAIN_DIRECTIONS_8 = [
    "Pick the instruction that best describes the direction you want the robot to take:",
    "turn left",
    "turn right",
    "move forward",
    "move backward",
    "move forward-left",
    "move forward-right",
    "move backward-left",
    "move backward-right",
]

FORMATTED_ACTIONS = [
    "Pick the instruction that best describes the direction you want the robot to take, replacing the brackets:",
    "move forward",
    "move towards {describe a goal in sight}",
    "go along {wall or corridor}",
    "go around {obstacle or object to avoid}",
    "go through {door or door frame}",
    "turn {left or right}",
    "turn around and backwards",
]

INSTRUCTION_TEMPLATE = [
    FREE_FORM_INSTRUCTIONS,
    MAIN_DIRECTIONS_4,
    MAIN_DIRECTIONS_8,
    FORMATTED_ACTIONS,
]

DOWN_SAMPLE_KEYWORDS = [
    "move forward",
]


def send_message(openai, system_prompt, prompt, base64_image, verbose=True):
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        # response_format={"type": "json_object"},
    )
    response = completion.choices[0].message
    if verbose:
        print("Response: ", response.content)
    return response.content.strip(".").lower()


def generate_instruction(
    openai: OpenAI,
    image: Image,
    actions: np.ndarray,
    system_prompt: str,
    instruction_prompt: str,
    save_path: Path,
    keyword_down_sample_rate: float = 0.05,
):
    fig, axs = plt.subplots(1, 2)

    # show observation image
    axs[0].imshow(image)
    axs[0].axis("off")
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

    # draw fig on canvas and save to io buffer
    fig.tight_layout()
    fig.canvas.draw()
    # fig.savefig("test.jpg", dpi=300)
    fig_bytes = io.BytesIO()
    fig.savefig(fig_bytes, format="jpg", dpi=300)

    # convert to openai image format
    fig_bytes.seek(0)
    encoded_fig = base64.b64encode(fig_bytes.read()).decode("utf-8")

    # generate instruction from gpt-4o
    # prompt = "List all valid directions the robot can go without collision."
    instruction = send_message(openai, system_prompt, instruction_prompt, encoded_fig)
    # instruction = send_message(openai, system_prompt, generate_text_image_content(prompt, encode_image("test.jpg")))
    axs[1].set_title(instruction)
    fig.suptitle(str(save_path.stem), fontsize=12)

    # save and close
    if not (
        any([keyword in instruction for keyword in DOWN_SAMPLE_KEYWORDS])
        and random.random() > keyword_down_sample_rate
    ):
        fig.savefig(save_path, dpi=300)
    plt.close()


@draccus.wrap()
def generate(cfg: EvalConfig) -> None:
    # format prompt
    system_prompt = INTRODUCTION
    instruction_prompt = "\n".join(INSTRUCTION_TEMPLATE[cfg.instruction_type])
    print("================ System prompt ================")
    print(system_prompt)
    print("============= Instruction prompt ==============")
    print(instruction_prompt)
    print("============= End of Prompt Format ============")

    # init openai
    openai = OpenAI()

    # eval
    traj_paths = []
    if cfg.eval_type == EvalType.SINGLE_TRAJ:
        traj_paths.append(cfg.data_root_dir / cfg.traj_name)
    elif cfg.eval_type == EvalType.SINGLE_SPLIT:
        with open(Path(cfg.data_split_dir / cfg.eval_split / "traj_names.txt")) as f:
            traj_names = f.read().decode("utf-8").splitlines()
        for traj_name in traj_names:
            traj_paths.append(cfg.data_root_dir / traj_name)
    elif cfg.eval_type == EvalType.ENTIRE_DATASET:
        traj_paths = list(cfg.data_root_dir.iterdir())
    else:
        raise KeyError("Not supported evaluation type: ", cfg.eval_type)

    shutil.rmtree(cfg.output_root_dir)
    for traj_path in traj_paths:
        steps = parse_trajectory(
            traj_folder=traj_path,
            image_size=cfg.image_size,
            len_traj_pred=cfg.len_traj_pred,
            end_slack=cfg.end_slack,
        )
        Path(cfg.output_root_dir / traj_path.name).mkdir(parents=True, exist_ok=True)
        for i, step in enumerate(steps):
            if random.random() > cfg.sample_rate:
                continue
            out_path = (
                cfg.output_root_dir / traj_path.name / f"{traj_path.name}_step_{i}.jpg"
            )
            generate_instruction(
                openai,
                step["image"],
                step["action"],
                system_prompt,
                instruction_prompt,
                out_path,
            )


if __name__ == "__main__":
    generate()
