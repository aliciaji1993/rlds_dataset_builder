import draccus
import random
import shutil

from dataclasses import dataclass
from enum import IntEnum
from openai import OpenAI
from pathlib import Path

from .instruct_gen.generate import InstructType, generate_instruction
from .instruct_gen.template import INTRODUCTION, INSTRUCTION_TEMPLATE
from process_data.data_convert import parse_trajectory


class EvalType(IntEnum):
    SINGLE_TRAJ = 0  # all steps in specified traj (use defaulf if not provided)
    SINGLE_SPLIT = 1  # all steps from specified data split (default to test)
    ENTIRE_DATASET = 2  # entire dataset


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
