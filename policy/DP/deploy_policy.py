import numpy as np
from .dp_model import DP
import yaml

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):

    if 'ckpt_path' in usr_args and usr_args['ckpt_path']:
        ckpt_file = usr_args['ckpt_path']
        print(f"\033[96m[DP Model] Using custom checkpoint path: {ckpt_file}\033[0m")
    else:

        import os
        task_name = usr_args['task_name']
        expert_data_num = usr_args['expert_data_num']
        seed = usr_args['seed']
        checkpoint_num = usr_args['checkpoint_num']
        
        print(f"\033[96m[DP Model] Looking for checkpoint_num={checkpoint_num}\033[0m")
        
        possible_paths = [
            f"./policy/DP/checkpoints/{task_name}-{task_name}-{expert_data_num}-{seed}/{checkpoint_num}.ckpt",
            f"./policy/DP/checkpoints/{task_name}-{expert_data_num}-{seed}/{checkpoint_num}.ckpt",
            f"./policy/DP/checkpoints/{task_name}-beat_block_hammer-{expert_data_num}-{seed}/{checkpoint_num}.ckpt",
        ]
        
        ckpt_file = None
        for path in possible_paths:
            if os.path.exists(path):
                ckpt_file = path
                break
        
        if ckpt_file is None:
            raise FileNotFoundError(f"DP checkpoint not found. Tried paths:\n" + "\n".join(possible_paths))
        
        print(f"\033[96m[DP Model] Found checkpoint: {ckpt_file}\033[0m")
    
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
