import draccus
import prettyprinter as pp
import random
import shutil

from dataclasses import dataclass
from enum import IntEnum
from openai import OpenAI
from pathlib import Path
from PIL import Image

from tqdm import tqdm

from gen_instruct.generate import InstructType, ContextType, generate_instruction
from gen_instruct.template import INSTRUCT_TEMPLATES, INTRO_TEMPLATES
from gen_instruct.chat_wrapper import *
from convert_dataset import parse_trajectory


class EvalType(IntEnum):
    SINGLE_TRAJ = 0  # all steps in specified traj (use defaulf if not provided)
    SINGLE_SPLIT = 1  # all steps from specified data split (default to test)
    ENTIRE_DATASET = 2  # entire dataset


@dataclass
class EvalConfig:
    # generation model settings
    chat_type: str = "vlm"  # or "gpt"
    model_name: str = "prism-dinosiglip+7b"  # or "gpt-4o"
    hf_token: str = Path("/home/yufeng/.hf_token_llama").read_text().strip()
    instruction_type: InstructType = InstructType.FORMAT_ACTION
    context_type: ContextType = ContextType.OBS_1_ACTIONS_STRING

    # dataset settings
    data_type: EvalType = EvalType.ENTIRE_DATASET
    data_split: str = "test"
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
    sample_rate: float = 0.001


@draccus.wrap()
def generate(cfg: EvalConfig) -> None:
    print("============== Generation Config ==============")
    pp.pprint(cfg)
    # format prompt
    system_prompt = INTRO_TEMPLATES[cfg.context_type]
    instruction_prompt = "\n".join(INSTRUCT_TEMPLATES[cfg.instruction_type])
    print("================ System prompt ================")
    print(system_prompt)
    print("============= Instruction prompt ==============")
    print(instruction_prompt)
    print("============= End of Prompt Format ============")

    # init chat
    if cfg.chat_type == "gpt":
        chat = ChatGPT(system_prompt=system_prompt)
    elif cfg.chat_type == "vlm":
        chat = ChatVLM(
            model_name=cfg.model_name,
            hf_token=cfg.hf_token,
            system_prompt=system_prompt,
        )

    # load traj paths
    traj_paths = []
    if cfg.data_type == EvalType.SINGLE_TRAJ:
        traj_paths.append(cfg.data_root_dir / cfg.traj_name)
    elif cfg.data_type == EvalType.SINGLE_SPLIT:
        with open(Path(cfg.data_split_dir / cfg.data_split / "traj_names.txt")) as f:
            traj_names = f.read().splitlines()
        for traj_name in traj_names:
            traj_paths.append(cfg.data_root_dir / traj_name)
    elif cfg.data_type == EvalType.ENTIRE_DATASET:
        traj_paths = list(cfg.data_root_dir.iterdir())
    else:
        raise KeyError("Not supported evaluation type: ", cfg.eval_type)

    # clear output root
    if cfg.output_root_dir.exists():
        shutil.rmtree(cfg.output_root_dir)
    Path(cfg.output_root_dir).mkdir(parents=True, exist_ok=True)

    # generate instructions
    random.shuffle(traj_paths)
    for traj_path in tqdm(traj_paths):
        steps = parse_trajectory(
            traj_folder=traj_path,
            image_size=cfg.image_size,
            len_traj_pred=cfg.len_traj_pred,
            end_slack=cfg.end_slack,
        )
        # Path(cfg.output_root_dir / traj_path.name).mkdir(parents=True, exist_ok=True)
        for i in range(len(steps["images"])):
            if random.random() > cfg.sample_rate:
                continue
            save_path = Path(cfg.output_root_dir / f"{traj_path.name}_step_{i}.jpg")
            images = (
                steps["images"][i : i + 8]
                if cfg.context_type is ContextType.OBS_8_ACTIONS_STRING
                else steps["images"][i]
            )
            try:
                generate_instruction(
                    chat=chat,
                    images=images,
                    actions=steps["actions"][i],
                    instruction_prompt=instruction_prompt,
                    context_type=cfg.context_type,
                    save_path=save_path,
                )
            except Exception as e:
                print("Unexpected error:", e)


if __name__ == "__main__":
    generate()
