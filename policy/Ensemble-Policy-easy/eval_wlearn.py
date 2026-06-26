"""
Evaluation Script for Ensemble Policy - Weight Learning Version

 eval.py 
-  _wlearn  checkpoint 
-  ensemble_config_wlearn 

Usage:
    cd /path/to/EnsembleVLA-ICML2026
    python policy/Ensemble-Policy-easy/eval_wlearn.py \
        --config-name=ensemble_config_wlearn \
        task.name=beat_block_hammer \
        policy1.type=DP \
        policy2.type=DP3 \
        eval.checkpoint_num=best
"""
import sys
import os

# Expert demo seed validation needs the original Curobo planner. Some sweep
# launchers set FORCE_MPLIB=1 to avoid Curobo JIT during policy-only eval;
# if that leaks into expert-check runs, valid seeds are rejected almost
# universally. Keep FORCE_MPLIB only for explicit skip-expert-check evals.
def _cli_bool(name):
    prefix = name + "="
    for raw_arg in sys.argv[1:]:
        arg = raw_arg[1:] if raw_arg.startswith("+") else raw_arg
        if arg.startswith(prefix):
            return arg.split("=", 1)[1].lower() in {"1", "true", "yes", "on"}
    return None

_skip_expert_check_cli = _cli_bool("eval.skip_expert_check")
_force_mplib_for_expert_cli = _cli_bool("eval.force_mplib_for_expert_check")
if os.environ.get("FORCE_MPLIB") and _skip_expert_check_cli is not True and _force_mplib_for_expert_cli is not True:
    os.environ.pop("FORCE_MPLIB", None)
    print("[eval] Unset FORCE_MPLIB for expert-check evaluation; using Curobo planner for seed validation.", flush=True)

_CONDA_BIN = os.path.dirname(sys.executable)
_CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_HOME", _CUDA_HOME)
_PATH_PREFIX = [os.path.join(_CUDA_HOME, "bin"), _CONDA_BIN]
os.environ["PATH"] = ":".join([p for p in _PATH_PREFIX if os.path.isdir(p)] + [os.environ.get("PATH", "")])
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
if not _ext_dir:
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
import pathlib


os.environ['CUROBO_TORCH_COMPILE'] = '0'

print(f"Python PID: {os.getpid()}")

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

os.environ['SAPIEN_OFFSCREEN_ONLY'] = '1'
import logging
logging.getLogger('curobo').setLevel(logging.WARNING)
import torch
import numpy as np
import yaml
import traceback
import hydra
from omegaconf import OmegaConf
import importlib


ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.chdir(ROOT_DIR)

OmegaConf.register_new_resolver("eval", eval, replace=True)


from eval import (
    EnsemblePolicy,
    class_decorator,
    get_camera_config,
    get_embodiment_config,
    test_ensemble_policy,
)


@hydra.main(
    version_base=None,
    config_path='config',
    config_name='ensemble_config_wlearn',
)
def main(cfg: OmegaConf):
    print("\n" + "="*60)
    print("  Ensemble Policy Evaluation - Weight Learning Version")
    print("="*60 + "\n")
    
    task_name = cfg.task.name
    policy1_type = cfg.policy1.type
    policy2_type = cfg.policy2.type
    seed = cfg.training.seed
    checkpoint_num = cfg.eval.get('checkpoint_num', 'best')
    

    base_task_name = task_name.replace('_randomized', '')
    

    with open(f'./task_config/{task_name}.yml', 'r') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    

    args['task_name'] = base_task_name
    args['head_camera_type'] = cfg.head_camera_type
    args['expert_seed'] = seed
    args['expert_data_num'] = cfg.expert_data_num
    args['eval_mode'] = True
    

    head_cfg = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_cfg['fovy']
    args['head_camera_w'] = head_cfg['w']
    args['head_camera_h'] = head_cfg['h']
    

    with open('./task_config/_embodiment_config.yml', 'r') as f:
        embodiment_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    embodiment_type = args.get('embodiment', ['aloha-agilex'])
    
    if len(embodiment_type) == 1:

        args["left_robot_file"] = embodiment_config[embodiment_type[0]]["file_path"]
        args["right_robot_file"] = embodiment_config[embodiment_type[0]]["file_path"]
        args["embodiment_dis"] = 0.0
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:

        args["left_robot_file"] = embodiment_config[embodiment_type[0]]["file_path"]
        args["right_robot_file"] = embodiment_config[embodiment_type[1]]["file_path"]
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")
    
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    print(f"Configuration:")
    print(f"  Task: {task_name}")
    print(f"  Base Task: {base_task_name}")
    print(f"  Policies: {policy1_type} + {policy2_type}")
    print(f"  Seed: {seed}")
    print(f"  Checkpoint: {checkpoint_num}")
    print(f"  Weight Learning: ENABLED")
    print()
    

    task = class_decorator(base_task_name)
    


    original_task_name = task_name
    cfg.task.name = base_task_name
    

    if cfg.get('output_dir'):
        cfg.output_dir = cfg.output_dir.replace('_randomized', '')
        print(f"  Adjusted output_dir: {cfg.output_dir}")
    

    policy = EnsemblePolicy(cfg, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    

    cfg.task.name = original_task_name
    

    st_seed = 100000 * (1 + seed)
    test_num = cfg.eval.get('test_num', 20)
    
    skip_expert_check = cfg.eval.get('skip_expert_check', False)
    max_expert_seed_attempts = cfg.eval.get('max_expert_seed_attempts', None)
    force_default_instruction = cfg.eval.get('force_default_instruction', False)
    seed_list_file = cfg.eval.get('seed_list_file', None)
    if seed_list_file:
        seed_path = pathlib.Path(str(seed_list_file))
        if not seed_path.is_absolute():
            seed_path = pathlib.Path(ROOT_DIR) / seed_path
        raw_seed_text = seed_path.read_text()
        args['eval_seed_list'] = [
            int(tok) for tok in raw_seed_text.replace(',', ' ').split()
        ]
        print(f"Using fixed eval seed list: {seed_path} ({len(args['eval_seed_list'])} seeds)")
    final_seed, suc_num = test_ensemble_policy(
        task_name, task, args, policy, st_seed, test_num=test_num,
        expert_check=not skip_expert_check,
        max_expert_seed_attempts=max_expert_seed_attempts,
        force_default_instruction=force_default_instruction,
    )
    
    success_rate = suc_num / test_num
    print(f"\nFinal Success Rate: {success_rate:.1%} ({suc_num}/{test_num})")


if __name__ == "__main__":
    main()
