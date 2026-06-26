"""
Evaluation Script for Ensemble Policy

Usage:
    cd /path/to/EnsembleVLA-ICML2026
    python policy/Ensemble-Policy-easy/eval.py \
        --config-name=ensemble_config \
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
import gc


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

# Import instruction generation for episode-specific instructions
sys.path.append("./description/utils")
from generate_episode_instructions import generate_episode_descriptions


ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.chdir(ROOT_DIR)

OmegaConf.register_new_resolver("eval", eval, replace=True)

from policy_loader import load_policy, freeze_model
from energy_head import ConservativeEnergyHead
from composition import PolicyComposer


class EnsemblePolicy:
    """
    Ensemble Policy for Evaluation
    
     Energy Head  refinement
    """
    
    def __init__(self, cfg, device='cuda:0'):
        self.device = torch.device(device)
        self.cfg = cfg
        
        task_name = cfg.task.name
        policy1_type = cfg.policy1.type
        policy2_type = cfg.policy2.type
        checkpoint_num = cfg.eval.get('checkpoint_num', 'best')
        

        policy1_kwargs = {
            'expert_data_num': cfg.expert_data_num,
            'seed': cfg.training.seed,
        }
        if cfg.policy1.get('checkpoint_num'):
            policy1_kwargs['checkpoint_num'] = cfg.policy1.checkpoint_num
        
        self.policy1_dict = load_policy(
            policy1_type, task_name,
            **policy1_kwargs
        )
        freeze_model(self.policy1_dict)
        
        policy2_kwargs = {
            'expert_data_num': cfg.expert_data_num,
            'seed': cfg.training.seed,
        }
        if cfg.policy2.get('checkpoint_num'):
            policy2_kwargs['checkpoint_num'] = cfg.policy2.checkpoint_num
        
        self.policy2_dict = load_policy(
            policy2_type, task_name,
            **policy2_kwargs
        )
        freeze_model(self.policy2_dict)
        

        self._set_language_instruction(task_name)
        

        ckpt_dir = cfg.get('output_dir') or f"./policy/Ensemble-Policy-easy/checkpoints/{task_name}_{policy1_type}_{policy2_type}"
        ckpt_path = os.path.join(ckpt_dir, f"{checkpoint_num}.pt")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        print(f"Loading checkpoint from {ckpt_path}...")
        payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        

        force_w1 = cfg.eval.get('force_w1', None)
        force_w2 = cfg.eval.get('force_w2', None)
        
        if force_w1 is not None and force_w2 is not None:

            self.w1 = float(force_w1)
            self.w2 = float(force_w2)
            print(f" Using FORCED weights: w1={self.w1:.4f}, w2={self.w2:.4f}")
            print(f"  (Checkpoint weights ignored: w1={payload.get('w1', 'N/A')}, w2={payload.get('w2', 'N/A')})")
        else:

            self.w1 = payload.get('w1', cfg.dp_w)
            self.w2 = payload.get('w2', cfg.dp3_w)
            print(f" Loaded weights: w1={self.w1:.4f}, w2={self.w2:.4f}")
        

        ebm_cfg = cfg.get('ebm', {})
        if 'energy_head' in payload:
            energy_head_state = payload['energy_head']
        elif 'energy_head_state_dict' in payload:
            energy_head_state = payload['energy_head_state_dict']
        else:
            raise KeyError(f"Checkpoint missing energy head weights. Available keys: {list(payload.keys())}")

        use_base_context = ebm_cfg.get('use_base_context', False)
        action_dim = int(cfg.action_dim)
        ckpt_in_dim = None
        for key in (
            'residual_head.delta_net.0.weight_orig',
            'residual_head.delta_net.0.weight',
            'residual_head.gate_net.0.weight',
        ):
            tensor = energy_head_state.get(key)
            if tensor is not None and getattr(tensor, 'ndim', 0) == 2:
                ckpt_in_dim = int(tensor.shape[1])
                break

        if ckpt_in_dim == action_dim * 3 and not use_base_context:
            use_base_context = True
            print(f" Auto-enabled base-context energy head for checkpoint input dim {ckpt_in_dim}.")
        elif ckpt_in_dim == action_dim and use_base_context:
            use_base_context = False
            print(f" Auto-disabled base-context energy head for checkpoint input dim {ckpt_in_dim}.")
        elif ckpt_in_dim not in (None, action_dim, action_dim * 3):
            print(
                f" Warning: checkpoint energy head input dim {ckpt_in_dim} does not match "
                f"action_dim={action_dim} or 3*action_dim={action_dim * 3}."
            )

        self.energy_head = ConservativeEnergyHead(
            action_dim=action_dim,
            hidden_dim=ebm_cfg.get('hidden_dim', 64),
            num_layers=ebm_cfg.get('num_layers', 2),
            max_delta=ebm_cfg.get('max_delta', 0.001),
            use_spectral_norm=ebm_cfg.get('use_spectral_norm', True),
            init_w1=self.w1,
            init_w2=self.w2,
            use_base_context=use_base_context,
        ).to(self.device)
        self.energy_head.load_state_dict(energy_head_state)
        self.energy_head.eval()
        

        self.composer = PolicyComposer(device=str(self.device))
        

        self.composition_mode = cfg.get('composition_mode', 'native_x0_tail')
        
        print(f"\n Ensemble Policy ready")
        print(f"  Policies: {policy1_type} + {policy2_type}")
        print(f"  Weights: w1={self.w1:.4f}, w2={self.w2:.4f}")
        print(f"  Composition mode: {self.composition_mode}")
        print(f"  Max delta: {self.energy_head.max_delta}\n")
    
    def reset_obs(self, instruction: str = None):
        """



        """
        model1 = self.policy1_dict['model']
        model2 = self.policy2_dict['model']
        
        if hasattr(model1, 'reset_obs'):
            model1.reset_obs()
        elif hasattr(model1, 'runner') and hasattr(model1.runner, 'reset_obs'):
            model1.runner.reset_obs()
        elif hasattr(model1, 'env_runner') and hasattr(model1.env_runner, 'reset_obs'):
            model1.env_runner.reset_obs()
        
        if hasattr(model2, 'reset_obs'):
            model2.reset_obs()
        elif hasattr(model2, 'runner') and hasattr(model2.runner, 'reset_obs'):
            model2.runner.reset_obs()
        elif hasattr(model2, 'env_runner') and hasattr(model2.env_runner, 'reset_obs'):
            model2.env_runner.reset_obs()
        

        self._first_iteration = True
        

        self._pi05_update_count = 0
        self._pi05_prepare_count = 0
        self._debug_action_count = 0
        


        self._set_language_instruction(self.cfg.task.name, instruction)
    
    def _set_language_instruction(self, task_name: str, instruction: str = None):
        """




        """

        if instruction is not None:
            final_instruction = instruction
            # print(f"[INFO] Using episode-specific instruction: {instruction}")
        else:

            task_instructions = {
                'click_alarmclock': 'press the alarm clock button',
                'beat_block_hammer': 'hit the block with the hammer',
                'open_laptop': 'open the laptop',
                'move_playingcard_away': 'move the playing card away',
                'dump_bin_bigbin': 'dump the bin into the big bin',
                'handover_block': 'hand over the block',
                'lift_pot': 'lift the pot',
            }
            final_instruction = task_instructions.get(task_name, task_name.replace('_', ' '))
            # print(f"[INFO] Using default instruction: {final_instruction}")
        

        if self.policy1_dict['name'] in ('pi05', 'pi0'):
            model1 = self.policy1_dict['model']
            if hasattr(model1, 'set_language'):
                model1.set_language(final_instruction)
        
        if self.policy2_dict['name'] in ('pi05', 'pi0'):
            model2 = self.policy2_dict['model']
            if hasattr(model2, 'set_language'):
                model2.set_language(final_instruction)

        self._dp3_first_query_policy1 = True
        self._dp3_first_query_policy2 = True
    
    @torch.no_grad()
    def get_action(self, obs: dict) -> np.ndarray:
        """
        
        
        Args:
            obs: 
        
        Returns:
            actions: [n_action_steps, action_dim]
        """

        composed_action = self._get_composed_action(obs)
        
        # Energy head refinement. Some released DP+pi0.5 checkpoints were
        # trained with base-context features, so pass the latest native base
        # actions stashed by FlowDiffusionComposer when available.
        composed_tensor = torch.from_numpy(composed_action).float().to(self.device)
        if composed_tensor.dim() == 2:
            composed_tensor = composed_tensor.unsqueeze(0)  # [1, T, action_dim]

        base1 = base2 = None
        if getattr(self.energy_head, 'use_base_context', False):
            flow_composer = getattr(self.composer, 'flow_diffusion_composer', None)
            base_dp = getattr(flow_composer, 'last_base_dp', None) if flow_composer is not None else None
            base_pi05 = getattr(flow_composer, 'last_base_pi05', None) if flow_composer is not None else None

            def _align_base(x):
                if x is None:
                    return None
                x = x.to(self.device).float()
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                if x.dim() == 3 and x.shape[1] != composed_tensor.shape[1]:
                    target_t = composed_tensor.shape[1]
                    if x.shape[1] >= target_t:
                        x = x[:, :target_t, :]
                    else:
                        pad = x[:, -1:, :].expand(x.shape[0], target_t - x.shape[1], x.shape[2])
                        x = torch.cat([x, pad], dim=1)
                return x[..., :composed_tensor.shape[-1]]

            base1 = _align_base(base_dp)
            base2 = _align_base(base_pi05)

        refined_tensor = self.energy_head.refine_action(composed_tensor, base1, base2)
        refined = refined_tensor.squeeze(0).cpu().numpy()

        return refined
    
    @torch.no_grad()
    def _get_composed_action(self, obs: dict) -> np.ndarray:
        """








        """
        model1 = self.policy1_dict['model']
        model2 = self.policy2_dict['model']
        policy1_name = self.policy1_dict['name']
        policy2_name = self.policy2_dict['name']
        

        obs1 = self._encode_obs_for_policy(obs, policy1_name)
        obs2 = self._encode_obs_for_policy(obs, policy2_name)
        




        if policy1_name == 'DP':
            self._update_obs(model1, obs1, policy1_name)

        
        if policy2_name == 'DP':
            self._update_obs(model2, obs2, policy2_name)

        

        self._first_iteration = False
        

        try:

            if policy1_name == 'DP':
                infer_data1 = model1.prepare_data()
            elif policy1_name == 'DP3':

                if hasattr(self, '_dp3_first_query_policy1') and not self._dp3_first_query_policy1:
                    infer_data1 = model1.prepare_data_no_append()
                else:
                    infer_data1 = model1.prepare_data(obs1)
                    self._dp3_first_query_policy1 = False
            elif policy1_name == 'OpenVLA':

                infer_data1 = self._get_openvla_infer_data(self.policy1_dict, obs1)
            elif policy1_name in ('pi05', 'pi0'):

                infer_data1 = self._get_pi05_infer_data(model1, obs1)
            else:
                infer_data1 = model1.prepare_data(obs1)
            if not isinstance(infer_data1, dict):
                raise ValueError(f"Policy1 ({policy1_name}) prepare_data returned {type(infer_data1)}, expected dict")
            infer_data1['policy_type'] = self.policy1_dict['policy_type']
            
            if policy2_name == 'DP':
                infer_data2 = model2.prepare_data()
            elif policy2_name == 'DP3':

                if hasattr(self, '_dp3_first_query_policy2') and not self._dp3_first_query_policy2:
                    infer_data2 = model2.prepare_data_no_append()
                else:
                    infer_data2 = model2.prepare_data(obs2)
                    self._dp3_first_query_policy2 = False
            elif policy2_name == 'OpenVLA':

                infer_data2 = self._get_openvla_infer_data(self.policy2_dict, obs2)
            elif policy2_name in ('pi05', 'pi0'):

                infer_data2 = self._get_pi05_infer_data(model2, obs2)
            else:
                infer_data2 = model2.prepare_data(obs2)
            if not isinstance(infer_data2, dict):
                raise ValueError(f"Policy2 ({policy2_name}) prepare_data returned {type(infer_data2)}, expected dict")
            infer_data2['policy_type'] = self.policy2_dict['policy_type']
            

            composed = self.composer.compose(
                infer_data1, infer_data2,
                self.w1, self.w2,
                composition_mode=self.composition_mode,
            )
            

            if isinstance(composed, torch.Tensor):
                composed_np = composed.cpu().numpy()
                if composed_np.ndim == 3:  # [B, T, D]
                    composed_np = composed_np.squeeze(0)  # [T, D]
            else:
                composed_np = composed
            

            # if not hasattr(self, '_debug_action_count'):
            #     self._debug_action_count = 0
            # 

            #     print(f"\n{'='*60}")
            #     print(f"[DEBUG] Composed Action #{self._debug_action_count}")
            #     print(f"{'='*60}")
            #     print(f"Policy1: {policy1_name} (type={self.policy1_dict['policy_type']}), w1={self.w1:.4f}")
            #     print(f"Policy2: {policy2_name} (type={self.policy2_dict['policy_type']}), w2={self.w2:.4f}")
            #     print(f"Action shape: {composed_np.shape}")
            #     print(f"Action stats:")
            #     print(f"  mean: {composed_np.mean():.6f}")
            #     print(f"  std:  {composed_np.std():.6f}")
            #     print(f"  min:  {composed_np.min():.6f}")
            #     print(f"  max:  {composed_np.max():.6f}")
            #     

            #     if composed_np.ndim == 2:
            #         print(f"\nAction per timestep (first 3 dims):")

            #             action_t = composed_np[t]
            #             print(f"  t={t}: [{action_t[0]:.4f}, {action_t[1]:.4f}, {action_t[2]:.4f}, ...]")
            #     

            #     if np.isnan(composed_np).any():

            #     if np.isinf(composed_np).any():

            #     if composed_np.std() < 1e-6:

            #     if np.abs(composed_np).max() > 10:

            #     
            #     print(f"{'='*60}\n")
            #     self._debug_action_count += 1
            
            return composed_np
                
        except Exception as e:
            import traceback
            print(f"\n{'='*60}")
            print(f" Distribution-level composition FAILED!")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"Traceback:")
            print(traceback.format_exc())
            print(f"{'='*60}")
            print(f"\nPolicy 1: {policy1_name}, Policy 2: {policy2_name}")
            print(f"infer_data1 keys: {infer_data1.keys() if 'infer_data1' in locals() else 'Not created'}")
            print(f"infer_data2 keys: {infer_data2.keys() if 'infer_data2' in locals() else 'Not created'}")
            print(f"{'='*60}\n")
            raise RuntimeError(f"Distribution-level composition failed: {e}") from e
    
    def update_obs_after_action(self, obs: dict):
        """



        """
        model1 = self.policy1_dict['model']
        model2 = self.policy2_dict['model']
        policy1_name = self.policy1_dict['name']
        policy2_name = self.policy2_dict['name']
        

        obs1 = self._encode_obs_for_policy(obs, policy1_name)
        obs2 = self._encode_obs_for_policy(obs, policy2_name)
        

        self._update_obs(model1, obs1, policy1_name)
        self._update_obs(model2, obs2, policy2_name)
    
    def _encode_obs_for_policy(self, obs: dict, policy_name: str):
        """"""
        if policy_name == 'DP':
            return self._encode_dp_obs(obs)
        elif policy_name == 'DP3':
            return self._encode_dp3_obs(obs)
        elif policy_name in ('pi05', 'pi0'):

            return self._encode_pi05_obs(obs)
        elif policy_name == 'OpenVLA':
            return self._encode_openvla_obs(obs)
        else:
            return obs
    
    def _update_obs(self, model, obs, policy_name: str):
        """






        """
        if policy_name == 'DP':
            if hasattr(model, 'update_obs'):
                model.update_obs(obs)
            elif hasattr(model, 'runner'):
                model.runner.update_obs(obs)
        elif policy_name == 'DP3':
            if hasattr(model, 'update_obs'):
                model.update_obs(obs)
            elif hasattr(model, 'env_runner'):
                model.env_runner.update_obs(obs)
        elif policy_name in ('pi05', 'pi0'):



            if hasattr(model, 'update_observation_window'):

                if isinstance(obs, tuple) and len(obs) == 2:
                    img_arr, state = obs

                    # if not hasattr(self, '_pi05_update_count'):
                    #     self._pi05_update_count = 0
                    # self._pi05_update_count += 1

                    #     print(f"[DEBUG] pi05 update_observation_window called #{self._pi05_update_count}")
                    #     print(f"  img_arr shape: {img_arr.shape}, state shape: {state.shape if hasattr(state, 'shape') else len(state)}")
                    #     print(f"  state[:4]: {state[:4]}")
                    model.update_observation_window(img_arr, state)
                else:
                    pass  # print(f"[WARNING] pi05 obs is not (img_arr, state) tuple: type={type(obs)}")
            elif hasattr(model, 'update_obs'):
                model.update_obs(obs)
            else:
                pass  # print(f"[WARNING] pi05 model has no update_observation_window or update_obs method")

    
    def _get_single_action(self, model, policy_name: str) -> np.ndarray:
        """"""
        if hasattr(model, 'get_action'):
            action = model.get_action()
        else:
            raise ValueError(f"Model {policy_name} does not have get_action method")
        
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        return action
    
    def _encode_dp_obs(self, obs: dict) -> dict:
        """"""
        head_cam = np.moveaxis(obs["observation"]["head_camera"]["rgb"], -1, 0) / 255
        left_cam = np.moveaxis(obs["observation"]["left_camera"]["rgb"], -1, 0) / 255
        right_cam = np.moveaxis(obs["observation"]["right_camera"]["rgb"], -1, 0) / 255
        return {
            'head_cam': head_cam,
            'left_cam': left_cam,
            'right_cam': right_cam,
            'agent_pos': obs["joint_action"]["vector"],
        }
    
    def _encode_dp3_obs(self, obs: dict) -> dict:
        """"""
        return {
            'point_cloud': obs['pointcloud'],
            'agent_pos': obs["joint_action"]["vector"],
        }
    
    def _encode_pi05_obs(self, obs: dict) -> tuple:
        """




        """

        img_front = obs["observation"]["head_camera"]["rgb"]
        img_left = obs["observation"]["left_camera"]["rgb"]
        img_right = obs["observation"]["right_camera"]["rgb"]
        

        img_arr = np.array([img_front, img_right, img_left])
        

        state = obs["joint_action"]["vector"]
        
        return (img_arr, state)
    
    def _encode_openvla_obs(self, obs: dict) -> dict:
        """"""
        return {
            'head_cam': obs["observation"]["head_camera"]["rgb"],
            'left_cam': obs["observation"]["left_camera"]["rgb"],
            'right_cam': obs["observation"]["right_camera"]["rgb"],
            'agent_pos': obs["joint_action"]["vector"],
        }
    
    def _get_pi05_infer_data(self, model, obs) -> dict:
        """
         pi0.5  distribution-level composition
        
         batch_infer_data  composition.py 
        _get_flow_velocity_batch()  None
        
         workspace_wlearn.py  _get_infer_data  pi05 
        
        Args:
            model: pi0.5 
            obs:  (img_arr, state) 
        
        Returns:
            infer_data:  batch_infer_data 
        """

        img_arr, state = obs
        

        # if not hasattr(self, '_pi05_prepare_count'):
        #     self._pi05_prepare_count = 0
        # self._pi05_prepare_count += 1

        #     print(f"[DEBUG] pi05 _get_pi05_infer_data called #{self._pi05_prepare_count}")
        #     print(f"  img_arr shape: {img_arr.shape}, state[:4]: {state[:4]}")
        



        if model.instruction is None:
            # print(f"[WARNING] pi0.5 instruction not set, using default mapping")
            task_name = self.cfg.task.name
            task_instructions = {
                'click_alarmclock': 'press the alarm clock button',
                'beat_block_hammer': 'hit the block with the hammer',
                'open_laptop': 'open the laptop',
                'move_playingcard_away': 'move the playing card away',
                'dump_bin_bigbin': 'dump the bin into the big bin',
                'handover_block': 'hand over the block',
                'lift_pot': 'lift the pot',
            }
            instruction = task_instructions.get(task_name, task_name.replace('_', ' '))
            model.set_language(instruction)
        
        try:


            single_infer_data = model.prepare_data(obs)
            
            if not isinstance(single_infer_data, dict):
                raise ValueError(f"pi0.5 prepare_data returned {type(single_infer_data)}, expected dict")
            


            single_infer_data['pi0_model'] = model
            
            infer_data = single_infer_data.copy()
            infer_data['batch_infer_data'] = [single_infer_data]
            infer_data['batch_size'] = 1
            infer_data['policy_type'] = 'flow'
            infer_data['n_action_steps'] = getattr(model, 'pi0_step', 50)
            
            return infer_data
            
        except Exception as e:
            import traceback
            print(f"[ERROR] pi0.5 _get_pi05_infer_data failed: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to get pi0.5 infer_data: {e}") from e
    
    def _get_openvla_infer_data(self, policy_dict: dict, obs: dict) -> dict:
        """
         OpenVLA  distribution-level 
        
         workspace_wlearn.py  _get_openvla_infer_data 
        """
        from pathlib import Path
        

        openvla_path = Path(__file__).parent.parent / "openvla-oft"
        sys.path.insert(0, str(openvla_path))
        
        try:
            from experiments.robot.openvla_utils import (
                prepare_images_for_vla,
                normalize_proprio,
            )
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK
        except ImportError as e:
            raise RuntimeError(f"Could not import OpenVLA utils: {e}")
        

        vla = policy_dict.get('vla')
        processor = policy_dict.get('processor')
        action_head = policy_dict.get('action_head')
        proprio_projector = policy_dict.get('proprio_projector')
        noisy_action_projector = policy_dict.get('noisy_action_projector')
        cfg = policy_dict.get('cfg')
        
        if vla is None or action_head is None:
            raise RuntimeError("OpenVLA components not found in policy_dict")
        
        device = next(vla.parameters()).device
        

        head_cam = obs.get('head_cam')
        if head_cam is None:
            raise RuntimeError("head_cam not found in obs")
        

        if head_cam.max() <= 1.0:
            head_cam = (head_cam * 255).astype('uint8')
        else:
            head_cam = head_cam.astype('uint8')
        

        all_images = [head_cam, head_cam, head_cam]
        

        processed_images = prepare_images_for_vla(all_images, cfg)
        primary_image = processed_images[0]
        wrist_images = processed_images[1:]
        

        agent_pos = obs.get('agent_pos')
        if agent_pos is None:
            raise RuntimeError("agent_pos not found in obs")
        
        proprio = agent_pos
        

        try:
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio, proprio_norm_stats)
        except Exception:
            pass
        

        task_label = "execute the task"
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        

        inputs = processor(prompt, primary_image).to(device, dtype=torch.bfloat16)
        

        if wrist_images:
            all_wrist_inputs = [
                processor(prompt, img).to(device, dtype=torch.bfloat16) 
                for img in wrist_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat(
                [primary_pixel_values] + all_wrist_pixel_values, dim=1
            )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        

        actual_action_dim = action_head.action_dim
        actual_chunk_len = NUM_ACTIONS_CHUNK
        

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], device=device)), dim=1
            )
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)), dim=1
            )
        
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
        

        placeholder_action_token_ids = torch.ones(
            (1, actual_action_dim * actual_chunk_len), device=device, dtype=input_ids.dtype
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)
        

        STOP_INDEX = 32000
        stop_token_id = torch.ones((1, 1), device=device, dtype=input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)
        

        mask_extension = torch.ones(
            (1, input_ids.shape[-1] - attention_mask.shape[-1]),
            device=device, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)
        

        IGNORE_INDEX = -100
        ACTION_TOKEN_BEGIN_IDX = 32000
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels_extension = torch.full(
            (1, actual_action_dim * actual_chunk_len + 1),
            ACTION_TOKEN_BEGIN_IDX + 1,
            device=device, dtype=labels.dtype
        )
        labels[:, -(actual_action_dim * actual_chunk_len + 1):] = labels_extension
        labels[:, -1] = STOP_INDEX
        

        input_embeddings = vla.get_input_embeddings()(input_ids)
        

        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask
        

        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            1, -1, input_embeddings.shape[2]
        )
        

        use_film = cfg.use_film if hasattr(cfg, 'use_film') else True
        if use_film and hasattr(vla, 'vision_backbone'):
            patch_features = vla.vision_backbone(pixel_values, language_embeddings)
        else:
            patch_features = vla.vision_backbone(pixel_values)
        
        projected_patch_embeddings = vla.projector(patch_features)
        

        if proprio_projector is not None:
            proprio_tensor = torch.tensor(proprio, device=device, dtype=torch.bfloat16).unsqueeze(0)
            proprio_features = proprio_projector(proprio_tensor).unsqueeze(1)
            projected_patch_embeddings = torch.cat(
                (projected_patch_embeddings, proprio_features), dim=1
            )
        

        NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
        if proprio_projector is not None:
            NUM_PATCHES += 1
        NUM_PATCHES += 1
        

        class OpenVLANormalizer:
            def __init__(self, norm_stats, unnorm_key):
                action_stats = norm_stats[unnorm_key]['action']
                self.action_high = torch.tensor(action_stats['max'], dtype=torch.float32)
                self.action_low = torch.tensor(action_stats['min'], dtype=torch.float32)
            
            def normalize(self, actions):
                if isinstance(actions, torch.Tensor):
                    device = actions.device
                    high = self.action_high.to(device)
                    low = self.action_low.to(device)
                else:
                    high = self.action_high.numpy()
                    low = self.action_low.numpy()
                return 2 * (actions - low) / (high - low + 1e-8) - 1
            
            def unnormalize(self, normalized_actions):
                if isinstance(normalized_actions, torch.Tensor):
                    device = normalized_actions.device
                    high = self.action_high.to(device)
                    low = self.action_low.to(device)
                else:
                    high = self.action_high.numpy()
                    low = self.action_low.numpy()
                return 0.5 * (normalized_actions + 1) * (high - low + 1e-8) + low
        
        normalizer = OpenVLANormalizer(vla.norm_stats, cfg.unnorm_key)
        

        infer_data = {
            'policy_type': 'diffusion',
            'scheduler': action_head.noise_scheduler,
            'num_inference_steps': 10,
            'Da': actual_action_dim,
            'To': 1,
            'action_normalizer': normalizer,
            'n_action_steps': actual_chunk_len,
            'cond_data': torch.zeros((1, actual_chunk_len, actual_action_dim), device=device, dtype=torch.float32),
            'cond_mask': torch.zeros((1, actual_chunk_len, actual_action_dim), device=device, dtype=torch.bool),
            'is_openvla': True,
            'vla': vla,
            'action_head': action_head,
            'noisy_action_projector': noisy_action_projector,
            'projected_patch_embeddings': projected_patch_embeddings,
            'input_embeddings': input_embeddings,
            'all_actions_mask': all_actions_mask,
            'attention_mask': attention_mask,
            'NUM_PATCHES': NUM_PATCHES,
            'NUM_PROMPT_TOKENS': NUM_PROMPT_TOKENS,
            'action_dim': actual_action_dim,
            'chunk_len': actual_chunk_len,
        }
        
        return infer_data


def class_decorator(task_name):
    """"""
    envs_module = importlib.import_module(f'envs.{task_name}')
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_camera_config(camera_type):
    """"""
    with open('./task_config/_camera_config.yml', 'r') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]


def get_embodiment_config(robot_file):
    """"""
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def test_ensemble_policy(task_name, Demo_class, args, policy: EnsemblePolicy, st_seed, test_num=20, expert_check=True, max_expert_seed_attempts=None, force_default_instruction=False):
    """




    """
    print(f"\n{'='*60}")
    print(f"Testing ENSEMBLE POLICY")
    print(f"Task: {task_name}")
    print(f"Weights: w1={policy.w1:.4f}, w2={policy.w2:.4f}")
    print(f"{'='*60}\n")
    
    Demo_class.suc = 0
    Demo_class.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    print(f"Expert demo check: {'ENABLED' if expert_check else 'DISABLED'}")
    print(f"Force default instruction: {'ENABLED' if force_default_instruction else 'DISABLED'}")
    if expert_check:
        if max_expert_seed_attempts is None:
            max_expert_seed_attempts = int(args.get('max_expert_seed_attempts', 500))
        max_expert_seed_attempts = int(max_expert_seed_attempts)
        print(f"Expert seed attempt limit: {max_expert_seed_attempts}")
    expert_attempts = 0
    expert_failure_counts = {}
    instruction_type = args.get('instruction_type', 'unseen')
    fixed_seed_list = args.get('eval_seed_list')
    if fixed_seed_list is not None:
        fixed_seed_list = [int(seed) for seed in fixed_seed_list]
        test_num = len(fixed_seed_list)
        print(f"Fixed eval seed list: ENABLED ({test_num} seeds)")
    
    while succ_seed < test_num:
        if fixed_seed_list is not None:
            now_seed = fixed_seed_list[succ_seed]
        render_freq = args.get('render_freq', 0)
        args['render_freq'] = 0
        args['eval_video_log'] = False
        
        episode_info = None
        

        if expert_check:
            try:
                print(f"[{succ_seed+1}/{test_num}] Verifying expert demo (seed={now_seed})...", flush=True)
                Demo_class.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_result = Demo_class.play_once()

                if isinstance(episode_result, dict) and 'info' in episode_result:
                    episode_info = episode_result['info']
                Demo_class.close_env(clear_cache=True)
            except Exception as e:
                expert_attempts += 1
                reason = str(e) or type(e).__name__
                expert_failure_counts[reason] = expert_failure_counts.get(reason, 0) + 1
                print(f"[WARNING] Expert demo failed: {reason}")
                print(traceback.format_exc(limit=8), flush=True)
                try:
                    Demo_class.close_env(clear_cache=True)
                except:
                    pass
                if expert_attempts >= max_expert_seed_attempts:
                    print(f"[ERROR] Expert check reached max attempts ({max_expert_seed_attempts}) before collecting {test_num} valid seeds.", flush=True)
                    print(f"[ERROR] Expert failure summary: {expert_failure_counts}", flush=True)
                    break
                now_seed += 1
                args['render_freq'] = render_freq
                continue
            
            if not (Demo_class.plan_success and Demo_class.check_success()):
                expert_attempts += 1
                reason = 'plan_success_or_check_success_false'
                expert_failure_counts[reason] = expert_failure_counts.get(reason, 0) + 1
                print(f"[SKIP] Expert demo failed for seed={now_seed}, skipping...")
                if expert_attempts >= max_expert_seed_attempts:
                    print(f"[ERROR] Expert check reached max attempts ({max_expert_seed_attempts}) before collecting {test_num} valid seeds.", flush=True)
                    print(f"[ERROR] Expert failure summary: {expert_failure_counts}", flush=True)
                    break
                now_seed += 1
                args['render_freq'] = render_freq
                continue
            
            print(f"[OK] Expert demo passed for seed={now_seed}")
        

        try:
            # Setup environment
            print(f"[{succ_seed+1}/{test_num}] Setting up environment (seed={now_seed})...", flush=True)
            Demo_class.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            



            instruction = None
            if episode_info is not None and not force_default_instruction:
                try:
                    episode_info_list = [episode_info]
                    results = generate_episode_descriptions(task_name, episode_info_list, 1)
                    if results and len(results) > 0 and instruction_type in results[0]:
                        instruction = np.random.choice(results[0][instruction_type])
                        print(f"[INFO] Generated episode-specific instruction: {instruction}")

                        Demo_class.set_instruction(instruction=instruction)
                except Exception as e:
                    print(f"[WARNING] Failed to generate episode-specific instruction: {e}")
                    print(f"[WARNING] Falling back to default instruction")
            
            print(f"[{succ_seed+1}/{test_num}] Environment ready, starting evaluation...", flush=True)
            


            policy.reset_obs(instruction=instruction)
            
            # Run policy evaluation loop
            succ = False
            action_query_count = 0
            while Demo_class.take_action_cnt < Demo_class.step_lim:
                if action_query_count % 10 == 0:
                    print(f"  Query {action_query_count}, Steps {Demo_class.take_action_cnt}/{Demo_class.step_lim}...", flush=True)
                observation = Demo_class.get_obs()
                actions = policy.get_action(observation)
                action_query_count += 1
                
                # Execute actions
                if actions.ndim == 1:
                    actions = actions[np.newaxis, :]
                
                for i, action in enumerate(actions):
                    Demo_class.take_action(action)
                    


                    new_observation = Demo_class.get_obs()
                    policy.update_obs_after_action(new_observation)
                    
                    if Demo_class.eval_success:
                        succ = True
                        break
                
                if succ:
                    break
            
            if succ:
                Demo_class.suc += 1
                print(f"\033[92m Success!\033[0m")
            else:
                print(f"\033[91m Fail\033[0m")
            
            Demo_class.test_num += 1
            now_id += 1
            Demo_class.close_env(clear_cache=True)
            succ_seed += 1
            

            gc.collect()
            torch.cuda.empty_cache()
            
            success_rate = Demo_class.suc / Demo_class.test_num
            print(f"{task_name} | Success rate: {Demo_class.suc}/{Demo_class.test_num} => {success_rate*100:.1f}%, seed: {now_seed}\n")
            
        except Exception as e:
            print(f'\033[91mError: {e}\033[0m')
            print(traceback.format_exc())
            try:
                Demo_class.close_env(clear_cache=True)
            except:
                pass
            now_seed += 1
            continue
        
        args['render_freq'] = render_freq
        now_seed += 1
    
    success_rate = Demo_class.suc / Demo_class.test_num if Demo_class.test_num > 0 else 0
    print(f"\n{'='*60}")
    print(f"ENSEMBLE POLICY Results")
    print(f"Success rate: {success_rate:.1%} ({Demo_class.suc}/{Demo_class.test_num})")
    print(f"{'='*60}\n")
    
    return now_seed, Demo_class.suc


@hydra.main(
    version_base=None,
    config_path='config',
    config_name='ensemble_config',
)
def main(cfg: OmegaConf):
    print("\n" + "="*60)
    print("  Ensemble Policy Evaluation")
    print("="*60 + "\n")
    
    task_name = cfg.task.name
    policy1_type = cfg.policy1.type
    policy2_type = cfg.policy2.type
    seed = cfg.training.seed
    checkpoint_num = cfg.eval.get('checkpoint_num', 'best')
    

    with open(f'./task_config/{task_name}.yml', 'r') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
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
    print(f"  Policies: {policy1_type} + {policy2_type}")
    print(f"  Seed: {seed}")
    print(f"  Checkpoint: {checkpoint_num}")
    print()
    

    task = class_decorator(task_name)
    

    policy = EnsemblePolicy(cfg, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    

    st_seed = 100000 * (1 + seed)
    test_num = cfg.eval.get('test_num', 20)
    
    skip_expert_check = cfg.eval.get('skip_expert_check', False)
    max_expert_seed_attempts = cfg.eval.get('max_expert_seed_attempts', None)
    force_default_instruction = cfg.eval.get('force_default_instruction', False)
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
