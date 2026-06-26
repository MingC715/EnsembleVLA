import os
import pathlib
import sys

import dill
import hydra
import torch
from omegaconf import OmegaConf
from termcolor import cprint

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
sys.path.append(str(pathlib.Path(__file__).parent))

from diffusion_policy_3d.env_runner.robot_runner import RobotRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DP3:
    def __init__(self, cfg, usr_args) -> None:
        self.policy, self.env_runner = self.get_policy_and_runner(cfg, usr_args)

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self, observation=None):
        return self.env_runner.get_action(self.policy, observation)

    def get_policy_and_runner(self, cfg, usr_args):
        n_obs_steps = cfg["n_obs_steps"]
        n_action_steps = cfg["n_action_steps"]
        env_runner = RobotRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

        ckpt_file = usr_args.get("ckpt_path")
        if not ckpt_file:
            task_name = usr_args["task_name"]
            expert_data_num = usr_args["expert_data_num"]
            seed = usr_args["seed"]
            checkpoint_num = usr_args["checkpoint_num"]
            candidates = [
                f"./policy/DP3/checkpoints/{task_name}-{task_name}-{expert_data_num}_{seed}/{checkpoint_num}.ckpt",
                f"./policy/DP3/checkpoints/{task_name}-{expert_data_num}_{seed}/{checkpoint_num}.ckpt",
                f"./policy/DP3/checkpoints/{task_name}-{task_name}-{expert_data_num}_w_rgb_{seed}/{checkpoint_num}.ckpt",
                f"./policy/DP3/checkpoints/{task_name}-{expert_data_num}_w_rgb_{seed}/{checkpoint_num}.ckpt",
                f"./policy/DP3/checkpoints/{task_name}-beat_block_hammer-{expert_data_num}_{seed}/{checkpoint_num}.ckpt",
            ]
            ckpt_file = next((path for path in candidates if os.path.exists(path)), None)
            if ckpt_file is None:
                raise FileNotFoundError("DP3 checkpoint not found. Tried:\n" + "\n".join(candidates))

        ckpt_file = pathlib.Path(ckpt_file)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"DP3 checkpoint not found: {ckpt_file}")

        cprint(f"Loading DP3 checkpoint: {ckpt_file}", "magenta")
        payload = torch.load(ckpt_file.open("rb"), pickle_module=dill, map_location="cpu")
        ckpt_cfg = payload["cfg"]
        policy = hydra.utils.instantiate(ckpt_cfg.policy)
        state_key = "ema_model" if ckpt_cfg.training.use_ema and "ema_model" in payload["state_dicts"] else "model"
        policy.load_state_dict(payload["state_dicts"][state_key])
        policy.eval()
        policy.cuda()
        return policy, env_runner

    def prepare_data(self, observation=None):
        return self.env_runner.prepare_data(self.policy, observation)

    def prepare_data_no_append(self):
        return self.env_runner.prepare_data_no_append(self.policy)
