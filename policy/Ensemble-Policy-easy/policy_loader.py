"""
Policy Loader - 


- DP (Diffusion Policy)
- DP3 (3D Diffusion Policy)
- pi0.5 (Flow Matching)
- OpenVLA-oft (Diffusion)
- RDT (Diffusion)
"""
import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


ROBOTWIN_ROOT = Path(__file__).parent.parent.parent
POLICY_ROOT = ROBOTWIN_ROOT / "policy"


def get_policy_type(policy_name: str) -> str:
    """"""
    policy_name = policy_name.upper()
    if policy_name in ['DP', 'DP3', 'RDT']:
        return 'diffusion'
    elif policy_name in ['OPENVLA', 'OPENVLA-OFT']:
        return 'diffusion'
    elif policy_name in ['PI05', 'PI0.5', 'PI0']:
        return 'flow'
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def load_dp_model(
    task_name: str,
    expert_data_num: int = 100,
    seed: int = 0,
    checkpoint_num: int = 300,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """
     DP (Diffusion Policy) 
    
    Returns:
        dict containing model, runner, and metadata
    """

    dp_path = POLICY_ROOT / "DP"
    sys.path.insert(0, str(dp_path))
    
    from dp_model import DP
    

    if checkpoint_path is None:

        possible_paths = [

            str(dp_path / f"checkpoints/{task_name}-{task_name}-{expert_data_num}-{seed}/{checkpoint_num}.ckpt"),

            str(dp_path / f"checkpoints/{task_name}-{expert_data_num}-{seed}/{checkpoint_num}.ckpt"),

            str(dp_path / f"checkpoints/{task_name}-beat_block_hammer-{expert_data_num}-{seed}/{checkpoint_num}.ckpt"),
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"DP checkpoint not found. Tried paths:\n" + "\n".join(possible_paths))
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DP checkpoint not found: {checkpoint_path}")
    

    action_dim = 14
    config_path = dp_path / f"diffusion_policy/config/robot_dp_{action_dim}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    n_obs_steps = config['n_obs_steps']
    n_action_steps = config['n_action_steps']
    

    model = DP(str(checkpoint_path), n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
    
    print(f" Loaded DP model from {checkpoint_path}")
    
    return {
        'model': model,
        'policy_type': 'diffusion',
        'name': 'DP',
        'n_obs_steps': n_obs_steps,
        'n_action_steps': n_action_steps,
    }


def load_dp3_model(
    task_name: str,
    expert_data_num: int = 100,
    seed: int = 0,
    checkpoint_num: int = 3000,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda:0',
    use_rgb: bool = False,
) -> Dict[str, Any]:
    """
     DP3 (3D Diffusion Policy) 
    """
    import torch
    import dill
    

    dp3_path = POLICY_ROOT / "DP3"
    dp3_core_path = dp3_path / "3D-Diffusion-Policy"
    sys.path.insert(0, str(dp3_path))
    sys.path.insert(0, str(dp3_core_path))
    sys.path.insert(0, str(dp3_core_path / "diffusion_policy_3d"))
    

    if checkpoint_path is None:

        possible_paths = [

            str(dp3_path / f"checkpoints/{task_name}-{task_name}-{expert_data_num}_{seed}/{checkpoint_num}.ckpt"),

            str(dp3_path / f"checkpoints/{task_name}-{expert_data_num}_{seed}/{checkpoint_num}.ckpt"),

            str(dp3_path / f"checkpoints/{task_name}-{task_name}-{expert_data_num}_w_rgb_{seed}/{checkpoint_num}.ckpt"),
            str(dp3_path / f"checkpoints/{task_name}-{expert_data_num}_w_rgb_{seed}/{checkpoint_num}.ckpt"),

            str(dp3_path / f"checkpoints/{task_name}-beat_block_hammer-{expert_data_num}_{seed}/{checkpoint_num}.ckpt"),
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"DP3 checkpoint not found. Tried paths:\n" + "\n".join(possible_paths))
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DP3 checkpoint not found: {checkpoint_path}")
    

    payload = torch.load(checkpoint_path, map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    

    import hydra
    from diffusion_policy_3d.env_runner.robot_runner import RobotRunner
    

    model = hydra.utils.instantiate(cfg.policy)
    

    print("[DP3 Loading] Loading state dict...")
    if 'model' in payload['state_dicts']:
        model.load_state_dict(payload['state_dicts']['model'])
        print("[DP3 Loading] Model state dict loaded")
    

    if cfg.training.use_ema and 'ema_model' in payload['state_dicts']:
        print("[DP3 Loading] Loading EMA state dict...")
        model.load_state_dict(payload['state_dicts']['ema_model'])
        print("[DP3 Loading] EMA state dict loaded")
    
    print("[DP3 Loading] Setting model to eval mode...")
    sys.stdout.flush()
    model.eval()
    
    print(f"[DP3 Loading] Moving model to device: {device}...")
    sys.stdout.flush()
    model.to(device)
    print(f"[DP3 Loading] Model moved to {device}")
    sys.stdout.flush()
    

    n_obs_steps = cfg.n_obs_steps if hasattr(cfg, 'n_obs_steps') else 3
    n_action_steps = cfg.n_action_steps if hasattr(cfg, 'n_action_steps') else 8
    env_runner = RobotRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
    
    print(f" Loaded DP3 model from {checkpoint_path}")
    sys.stdout.flush()
    

    class DP3Wrapper:
        def __init__(self, model, runner, cfg):
            self.policy = model
            self.env_runner = runner
            self.cfg = cfg
        
        def update_obs(self, observation):
            self.env_runner.update_obs(observation)
        
        def get_action(self, observation=None):
            return self.env_runner.get_action(self.policy, observation)
        
        def prepare_data(self, observation=None):
            return self.env_runner.prepare_data(self.policy, observation)
        
        def prepare_data_no_append(self):
            """Prepare inference data without appending observation."""
            return self.env_runner.prepare_data_no_append(self.policy)
    
    wrapper = DP3Wrapper(model, env_runner, cfg)
    
    return {
        'model': wrapper,
        'policy_type': 'diffusion',
        'name': 'DP3',
        'n_obs_steps': n_obs_steps,
        'n_action_steps': n_action_steps,
    }


def load_pi05_model(
    task_name: str,
    train_config_name: str = "pi0_base_aloha_robotwin_full_pytorch",
    model_name: str = None,
    checkpoint_id: int = 1000,
    pi0_step: int = 50,
    device: str = 'cuda:0',
    use_pytorch: bool = None,
) -> Dict[str, Any]:
    """
     pi0.5 (Flow Matching) 
    
    Args:
        task_name:  checkpoint
        train_config_name: 
        model_name:  None  task_name
        checkpoint_id: checkpoint 
        pi0_step: Flow Matching  50 DP  8
        device: 
        use_pytorch: None=, True=PyTorch, False=JAX
    
    
    -  {checkpoint_id}_pytorch/model.safetensors PyTorch50-100
    -  JAX
    """

    pi05_path = POLICY_ROOT / "pi05"
    pi05_src_path = pi05_path / "src"
    openpi_client_path = pi05_path / "packages" / "openpi-client" / "src"
    lerobot_path = Path(os.environ.get("LEROBOT_ROOT", str(ROBOTWIN_ROOT / "lerobot")))
    sys.path.insert(0, str(pi05_path))
    sys.path.insert(0, str(pi05_src_path))
    sys.path.insert(0, str(openpi_client_path))
    sys.path.insert(0, str(lerobot_path))
    

    if model_name is None:

        task_checkpoint_path = pi05_path / "checkpoints" / train_config_name / task_name
        if task_checkpoint_path.exists():
            model_name = task_name
            print(f"[pi0.5] Using task-specific checkpoint: {task_name}")
        else:

            model_name = "demo_clean"
            print(f"[pi0.5] Task checkpoint not found for {task_name}, using demo_clean")
    

    if use_pytorch is None:
        checkpoint_base = pi05_path / "checkpoints" / train_config_name / model_name / str(checkpoint_id)
        

        pytorch_paths = [

            checkpoint_base.parent / f"{checkpoint_id}_pytorch" / "model.safetensors",

            checkpoint_base / "model.safetensors",

            checkpoint_base / str(checkpoint_id) / "model.safetensors",
        ]
        
        pytorch_found = False
        for pytorch_path in pytorch_paths:
            if pytorch_path.exists():
                pytorch_found = True
                print(f"[pi0.5]  Found PyTorch checkpoint: {pytorch_path}")
                break
        
        if pytorch_found:
            use_pytorch = True
        else:
            use_pytorch = False
            print(f"[pi0.5] PyTorch checkpoint not found, using JAX version")
            print(f"[pi0.5] Tried paths:")
            for p in pytorch_paths:
                print(f"  - {p}")
            print(f"[pi0.5] To convert: bash policy/pi05/convert_single_checkpoint.sh {task_name} {checkpoint_id}")
    
    from pi_model import PI0
    model = PI0(train_config_name, model_name, checkpoint_id, pi0_step, use_pytorch=use_pytorch)
    

    actual_is_pytorch = model._is_pytorch if hasattr(model, '_is_pytorch') else use_pytorch
    

    # CRITICAL: Check DISABLE_TORCH_COMPILE environment variable
    # torch.compile causes state accumulation bugs in long evaluation sequences
    # This causes only the first episode to succeed, subsequent episodes fail
    if os.environ.get("DISABLE_TORCH_COMPILE", "0") == "1":
        print(f"[pi0.5] torch.compile disabled (DISABLE_TORCH_COMPILE=1)")
    elif actual_is_pytorch and hasattr(model, '_model') and model._model is not None:
        try:
            import torch
            if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
                print(f"[pi0.5] Applying torch.compile optimization...")


                model._model = torch.compile(
                    model._model, 
                    mode='reduce-overhead',
                    fullgraph=False,
                )
                print(f"[pi0.5]  torch.compile applied successfully")
            else:
                print(f"[pi0.5] torch.compile not available (PyTorch version: {torch.__version__})")
        except Exception as e:
            print(f"[pi0.5] Warning: torch.compile failed: {e}")
            print(f"[pi0.5] Continuing without compilation optimization")
    
    print(f" Loaded pi0.5 model (model_name={model_name}, pytorch={actual_is_pytorch})")
    
    return {
        'model': model,
        'policy_type': 'flow',
        'name': 'pi05',
        'n_obs_steps': 2,
        'n_action_steps': pi0_step,
        'is_pytorch': actual_is_pytorch,
        'device': device if actual_is_pytorch else None,
    }


def load_pi0_model(
    task_name: str,
    train_config_name: str = "pi0_base_aloha_robotwin_full_pytorch_pi0",
    model_name: str = None,
    checkpoint_id: int = 1000,
    pi0_step: int = 50,
    device: str = 'cuda:0',
    use_pytorch: bool = None,
) -> Dict[str, Any]:
    """
     pi0 (Flow Matching) 
    
    pi0  pi0.5 
    - pi0 checkpoint : policy/pi05/checkpoints/pi0_base_aloha_robotwin_full_pytorch_pi0/<task>/1000
    - pi0.5 checkpoint : policy/pi05/checkpoints/pi0_base_aloha_robotwin_full_pytorch/<task>/1000
    
    Args:
        task_name:  checkpoint
        train_config_name:  ( pi0 )
        model_name:  None  task_name
        checkpoint_id: checkpoint 
        pi0_step: 
        device: 
        use_pytorch: None=, True=PyTorch, False=JAX
    
    
    -  {checkpoint_id}_pytorch/model.safetensors PyTorch50-100
    -  JAX
    """

    pi05_path = POLICY_ROOT / "pi05"
    pi05_src_path = pi05_path / "src"
    openpi_client_path = pi05_path / "packages" / "openpi-client" / "src"
    lerobot_path = Path(os.environ.get("LEROBOT_ROOT", str(ROBOTWIN_ROOT / "lerobot")))
    sys.path.insert(0, str(pi05_path))
    sys.path.insert(0, str(pi05_src_path))
    sys.path.insert(0, str(openpi_client_path))
    sys.path.insert(0, str(lerobot_path))
    

    if model_name is None:

        task_checkpoint_path = pi05_path / "checkpoints" / train_config_name / task_name
        if task_checkpoint_path.exists():
            model_name = task_name
            print(f"[pi0] Using task-specific checkpoint: {task_name}")
        else:

            model_name = "demo_clean"
            print(f"[pi0] Task checkpoint not found for {task_name}, using demo_clean")
    

    if use_pytorch is None:
        checkpoint_base = pi05_path / "checkpoints" / train_config_name / model_name / str(checkpoint_id)
        

        pytorch_paths = [

            checkpoint_base.parent / f"{checkpoint_id}_pytorch" / "model.safetensors",

            checkpoint_base / "model.safetensors",

            checkpoint_base / str(checkpoint_id) / "model.safetensors",
        ]
        
        pytorch_found = False
        for pytorch_path in pytorch_paths:
            if pytorch_path.exists():
                pytorch_found = True
                print(f"[pi0]  Found PyTorch checkpoint: {pytorch_path}")
                break
        
        if pytorch_found:
            use_pytorch = True
        else:
            use_pytorch = False
            print(f"[pi0] PyTorch checkpoint not found, using JAX version")
            print(f"[pi0] Tried paths:")
            for p in pytorch_paths:
                print(f"  - {p}")
            print(f"[pi0] To convert: bash policy/pi05/convert_single_checkpoint.sh {task_name} {checkpoint_id}")
    
    from pi_model import PI0
    model = PI0(train_config_name, model_name, checkpoint_id, pi0_step, use_pytorch=use_pytorch)
    

    actual_is_pytorch = model._is_pytorch if hasattr(model, '_is_pytorch') else use_pytorch
    

    # CRITICAL: Check DISABLE_TORCH_COMPILE environment variable
    # torch.compile causes state accumulation bugs in long evaluation sequences
    # This causes only the first episode to succeed, subsequent episodes fail
    if os.environ.get("DISABLE_TORCH_COMPILE", "0") == "1":
        print(f"[pi0] torch.compile disabled (DISABLE_TORCH_COMPILE=1)")
    elif actual_is_pytorch and hasattr(model, '_model') and model._model is not None:
        try:
            import torch
            if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
                print(f"[pi0] Applying torch.compile optimization...")


                model._model = torch.compile(
                    model._model, 
                    mode='reduce-overhead',
                    fullgraph=False,
                )
                print(f"[pi0]  torch.compile applied successfully")
            else:
                print(f"[pi0] torch.compile not available (PyTorch version: {torch.__version__})")
        except Exception as e:
            print(f"[pi0] Warning: torch.compile failed: {e}")
            print(f"[pi0] Continuing without compilation optimization")
    
    print(f" Loaded pi0 model (model_name={model_name}, pytorch={actual_is_pytorch})")
    
    return {
        'model': model,
        'policy_type': 'flow',
        'name': 'pi0',
        'n_obs_steps': 2,
        'n_action_steps': pi0_step,
        'is_pytorch': actual_is_pytorch,
        'device': device if actual_is_pytorch else None,
    }


# ============================================================================

# ============================================================================

# ============================================================================
# def _load_pi05_pytorch(
#     pi05_path: Path,
#     train_config_name: str,
#     model_name: str,
#     checkpoint_dir: Path,
#     pi0_step: int,
#     device: str,
# ) -> Any:
#     """

#     
#     Args:






#     """
#     from openpi.training import config as _config
#     from openpi.policies import policy_config as _policy_config
#     

#     assets_path = checkpoint_dir / "assets"
#     if assets_path.exists():
#         entries = os.listdir(assets_path)
#         assets_id = entries[0] if entries else None
#     else:
#         assets_id = None
#     
#     config = _config.get_config(train_config_name)
#     policy = _policy_config.create_trained_policy(
#         config,
#         str(checkpoint_dir),
#         robotwin_repo_id=assets_id,
#         pytorch_device=device,
#     )
#     

#     class PI0PyTorchWrapper:
#         def __init__(self, policy, pi0_step, device):
#             self.policy = policy
#             self._model = policy._model
#             self._is_pytorch = policy._is_pytorch_model
#             self._device = device
#             self._input_transform = policy._input_transform
#             self._output_transform = policy._output_transform
#             self._sample_kwargs = policy._sample_kwargs
#             
#             self.pi0_step = pi0_step
#             self.observation_window = None
#             self.instruction = None
#             
#             # Get action dimensions from model config
#             if hasattr(self._model, 'config'):
#                 self.action_horizon = self._model.config.action_horizon
#                 self.action_dim = self._model.config.action_dim
#             else:
#                 self.action_horizon = getattr(self._model, 'action_horizon', 50)
#                 self.action_dim = getattr(self._model, 'action_dim', 14)
#             
#             self.num_inference_steps = self._sample_kwargs.get('num_steps', 10)
#             self.runner = PI0RunnerWrapper(self)
#         
#         def set_language(self, instruction):
#             self.instruction = instruction
#             print(f"successfully set instruction:{instruction}")
#         
#         def update_observation_window(self, img_arr, state):
#             import numpy as np
#             img_front, img_right, img_left, puppet_arm = (
#                 img_arr[0], img_arr[1], img_arr[2], state,
#             )
#             img_front = np.transpose(img_front, (2, 0, 1))
#             img_right = np.transpose(img_right, (2, 0, 1))
#             img_left = np.transpose(img_left, (2, 0, 1))
#             
#             self.observation_window = {
#                 "state": state,
#                 "images": {
#                     "cam_high": img_front,
#                     "cam_left_wrist": img_left,
#                     "cam_right_wrist": img_right,
#                 },
#                 "prompt": self.instruction,
#             }
#         
#         def update_obs(self, obs):
#             if isinstance(obs, tuple) and len(obs) == 2:
#                 img_arr, state = obs
#                 self.update_observation_window(img_arr, state)
#             else:
#                 raise ValueError(f"Expected obs to be (img_arr, state) tuple, got {type(obs)}")
#         
#         def get_action(self):
#             assert self.observation_window is not None, "update observation_window first!"
#             return self.policy.infer(self.observation_window)["actions"]
#         
#         def reset_obsrvationwindows(self):
#             self.instruction = None
#             self.observation_window = None
#             print("successfully unset obs and language intruction")
#         
#         def prepare_data(self, observation=None):
#             """
#             Prepare inference data for distribution-level composition (Flow Matching).
#             PyTorch version.
#             """
#             import numpy as np
#             import jax
#             from openpi.models import model as _model
#             from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
#             
#             if observation is not None:
#                 img_arr, state = observation
#                 self.update_observation_window(img_arr, state)
#             
#             assert self.observation_window is not None, "update observation_window first!"
#             assert self.instruction is not None, "set language instruction first using set_language()!"
#             
#             # Transform inputs
#             inputs = {k: v for k, v in self.observation_window.items()}
#             inputs = self._input_transform(inputs)
#             
#             # Convert to PyTorch tensors
#             inputs = jax.tree.map(
#                 lambda x: torch.from_numpy(np.array(x)).to(self._device)[None, ...], 
#                 inputs
#             )
#             
#             observation_obj = _model.Observation.from_dict(inputs)
#             
#             # Preprocess observation
#             images, img_masks, lang_tokens, lang_masks, state = self._model._preprocess_observation(
#                 observation_obj, train=False
#             )
#             
#             # Compute prefix embeddings and KV cache
#             prefix_embs, prefix_pad_masks, prefix_att_masks = self._model.embed_prefix(
#                 images, img_masks, lang_tokens, lang_masks
#             )
#             
#             prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
#             prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
#             
#             # Compute KV cache
#             prefix_att_2d_masks_4d = self._model._prepare_attention_masks_4d(prefix_att_2d_masks)
#             self._model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
#             
#             _, past_key_values = self._model.paligemma_with_expert.forward(
#                 attention_mask=prefix_att_2d_masks_4d,
#                 position_ids=prefix_position_ids,
#                 past_key_values=None,
#                 inputs_embeds=[prefix_embs, None],
#                 use_cache=True,
#             )
#             
#             device = self._device if self._device is not None else (
#                 torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#             )
#             
#             return {
#                 "model": self,  # Return wrapper instead of _model for denoise() compatibility
#                 "state": state,
#                 "prefix_pad_masks": prefix_pad_masks,
#                 "past_key_values": past_key_values,
#                 "num_inference_steps": self.num_inference_steps,
#                 "action_horizon": self.action_horizon,
#                 "action_dim": self.action_dim,
#                 "device": device,
#                 "is_pytorch": True,
#                 "dt": -1.0 / self.num_inference_steps,
#                 "policy_type": "PI05",
#                 "output_transform": self._output_transform,
#             }
#         
#         def denoise(self, x_t, timestep, state, past_key_values=None, prefix_pad_masks=None):
#             """
#             Adapter method for composition.py compatibility.
#             Maps to PI0Pytorch.denoise_step with correct parameter order.
#             
#             Args:
#                 x_t: noisy action trajectory [B, T, action_dim]
#                 timestep: current timestep [B]
#                 state: robot state
#                 past_key_values: cached KV from prefix
#                 prefix_pad_masks: attention masks for prefix
#             
#             Returns:
#                 velocity: predicted velocity [B, T, action_dim]
#             """
#             return self._model.denoise_step(
#                 state=state,
#                 prefix_pad_masks=prefix_pad_masks,
#                 past_key_values=past_key_values,
#                 x_t=x_t,
#                 timestep=timestep,
#             )
#     
#     class PI0RunnerWrapper:
#         def __init__(self, pi0_model):
#             self.pi0_model = pi0_model
#         
#         def reset_obs(self):
#             self.pi0_model.reset_obsrvationwindows()
#     
#     return PI0PyTorchWrapper(policy, pi0_step, device)
# ============================================================================


# ============================================================================

# ============================================================================
# def _convert_jax_to_pytorch(
#     pi05_path: Path,
#     train_config_name: str,
#     model_name: str,
#     checkpoint_id: int,
# ) -> None:
#     """

#     """
#     import subprocess
#     
#     checkpoint_dir = pi05_path / "checkpoints" / train_config_name / model_name / str(checkpoint_id)

#     
#     convert_script = pi05_path / "examples" / "convert_jax_model_to_pytorch.py"
#     
#     if not convert_script.exists():
#         raise FileNotFoundError(f"Conversion script not found: {convert_script}")
#     
#     print(f"[pi0.5] Converting JAX checkpoint to PyTorch...")
#     print(f"  Checkpoint: {checkpoint_dir}")
#     print(f"  Output: {output_path}")
#     

#     env = os.environ.copy()
#     pi05_src_path = str(pi05_path / "src")
#     openpi_client_path = str(pi05_path / "packages" / "openpi-client" / "src")
#     

#     existing_pythonpath = env.get('PYTHONPATH', '')
#     new_pythonpath = f"{pi05_src_path}:{openpi_client_path}"
#     if existing_pythonpath:
#         new_pythonpath = f"{new_pythonpath}:{existing_pythonpath}"
#     env['PYTHONPATH'] = new_pythonpath
#     

#     cmd = [
#         sys.executable,
#         str(convert_script),
#         "--checkpoint_dir", str(checkpoint_dir),
#         "--output_path", str(output_path),
#         "--config_name", train_config_name,
#         "--precision", "bfloat16",
#     ]
#     
#     result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(pi05_path), env=env)
#     
#     if result.returncode != 0:
#         raise RuntimeError(f"Conversion failed:\n{result.stderr}")
#     
#     print(f"[pi0.5] Conversion completed successfully")
# ============================================================================


def load_openvla_model(
    task_name: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_num: Optional[int] = None,
    use_diffusion: bool = True,
    unnorm_key: Optional[str] = None,
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """
     OpenVLA-oft  (Diffusion )
    
    Args:
        task_name: 
        checkpoint_path: checkpoint  None 
        checkpoint_num: checkpoint  ( 30000)
        use_diffusion:  diffusion 
        unnorm_key:  action unnormalization  key task_name 
        device: 
    
    Returns:
        dict containing model and all components needed for distribution-level composition
    """
    import glob
    

    openvla_path = POLICY_ROOT / "openvla-oft"
    sys.path.insert(0, str(openvla_path))
    
    from deploy_policy import get_model, InferenceConfig
    from experiments.robot.openvla_utils import get_noisy_action_projector
    
    if checkpoint_path is None:


        candidate_dirs = [
            str(openvla_path / f"checkpoints/{task_name}"),
            *glob.glob(str(openvla_path / f"checkpoints/*{task_name}*")),
            *glob.glob(str(openvla_path / f"checkpoints/aloha_{task_name}*")),
        ]
        
        checkpoint_path = None
        for base_dir in candidate_dirs:
            if not os.path.isdir(base_dir):
                continue
            

            subdirs = os.listdir(base_dir)
            

            if checkpoint_num is not None:

                matching_subdirs = [
                    d for d in subdirs 
                    if (f"--{checkpoint_num}_chkpt" in d) or 
                       (d == str(checkpoint_num)) or
                       (d.isdigit() and int(d) == checkpoint_num)
                ]
                if matching_subdirs:
                    ckpt_dir = os.path.join(base_dir, matching_subdirs[0])

                    if (os.path.exists(os.path.join(ckpt_dir, "dataset_statistics.json")) or
                        os.path.exists(os.path.join(ckpt_dir, "config.json"))):
                        checkpoint_path = ckpt_dir
                        print(f"[OpenVLA] Found checkpoint: {ckpt_dir}")
                        break
            

            if (os.path.exists(os.path.join(base_dir, "dataset_statistics.json")) or
                os.path.exists(os.path.join(base_dir, "config.json"))):
                checkpoint_path = base_dir
                print(f"[OpenVLA] Found checkpoint: {base_dir}")
                break
            

            if checkpoint_num is None:

                chkpt_subdirs = [d for d in subdirs if "_chkpt" in d]
                if chkpt_subdirs:

                    def extract_num(name):
                        try:

                            parts = name.split("--")
                            if len(parts) >= 2:
                                num_part = parts[-1].replace("_chkpt", "")
                                return int(num_part)
                        except:
                            pass
                        return 0
                    
                    latest = max(chkpt_subdirs, key=extract_num)
                    ckpt_dir = os.path.join(base_dir, latest)
                    if (os.path.exists(os.path.join(ckpt_dir, "dataset_statistics.json")) or
                        os.path.exists(os.path.join(ckpt_dir, "config.json"))):
                        checkpoint_path = ckpt_dir
                        print(f"[OpenVLA] Found latest checkpoint: {ckpt_dir}")
                        break
                

                num_subdirs = [d for d in subdirs if d.isdigit()]
                if num_subdirs:
                    latest = max(num_subdirs, key=int)
                    ckpt_dir = os.path.join(base_dir, latest)
                    if (os.path.exists(os.path.join(ckpt_dir, "dataset_statistics.json")) or
                        os.path.exists(os.path.join(ckpt_dir, "config.json"))):
                        checkpoint_path = ckpt_dir
                        print(f"[OpenVLA] Found latest checkpoint: {ckpt_dir}")
                        break
        
        if checkpoint_path is None:
            tried_paths = candidate_dirs[:5]
            raise FileNotFoundError(
                f"OpenVLA checkpoint not found for task '{task_name}'. "
                f"checkpoint_num={checkpoint_num}\n"
                f"Tried paths:\n" + "\n".join(tried_paths) + "\n"
                f"Expected structure: checkpoints/aloha_{{task_name}}_*/openvla-7b+...--{{num}}_chkpt/"
            )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"OpenVLA checkpoint not found: {checkpoint_path}")
    


    if unnorm_key is None:
        dataset_stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
        if os.path.exists(dataset_stats_path):
            import json
            with open(dataset_stats_path, 'r') as f:
                dataset_stats = json.load(f)

            available_keys = list(dataset_stats.keys())
            if available_keys:

                matching_keys = [k for k in available_keys if task_name in k]
                unnorm_key = matching_keys[0] if matching_keys else available_keys[0]
                print(f"  Auto-detected unnorm_key: {unnorm_key}")
        if unnorm_key is None:
            unnorm_key = task_name
    
    usr_args = {
        'checkpoint_path': checkpoint_path,
        'use_l1_regression': not use_diffusion,
        'use_diffusion': use_diffusion,
        'use_film': True,
        'use_proprio': True,
        'unnorm_key': unnorm_key,
    }
    
    model = get_model(usr_args)
    

    noisy_action_projector = None
    if use_diffusion:

        class TempCfg:
            def __init__(self, ckpt_path):
                self.pretrained_checkpoint = ckpt_path
        
        temp_cfg = TempCfg(checkpoint_path)
        try:
            noisy_action_projector = get_noisy_action_projector(temp_cfg, model.vla.llm_dim)
            print(f"   Loaded noisy_action_projector for distribution-level composition")
        except Exception as e:
            print(f"   Could not load noisy_action_projector: {e}")
            print(f"    Distribution-level composition may not work properly")
    
    print(f" Loaded OpenVLA-oft model from {checkpoint_path}")
    print(f"  (diffusion={use_diffusion}, unnorm_key={unnorm_key})")
    
    return {
        'model': model,
        'policy_type': 'diffusion' if use_diffusion else 'regression',
        'name': 'OpenVLA',
        'n_obs_steps': 1,
        'n_action_steps': 8,
        'use_diffusion': use_diffusion,
        'checkpoint_path': checkpoint_path,
        'unnorm_key': unnorm_key,

        'vla': model.vla,
        'processor': model.processor,
        'action_head': model.action_head,
        'proprio_projector': model.proprio_projector,
        'noisy_action_projector': noisy_action_projector,
        'cfg': model.cfg,
    }


def load_rdt_model(
    task_name: str,
    ckpt_setting: str = "rdt",
    checkpoint_id: int = 10000,
    left_arm_dim: int = 6,
    right_arm_dim: int = 6,
    rdt_step: int = 8,
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """
     RDT (Robotics Diffusion Transformer) 
    """

    rdt_path = POLICY_ROOT / "RDT"
    sys.path.insert(0, str(rdt_path))
    
    from model import RDT
    
    checkpoint_path = str(rdt_path / f"checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model/mp_rank_00_model_states.pt")
    
    model = RDT(checkpoint_path, task_name, left_arm_dim, right_arm_dim, rdt_step)
    
    print(f" Loaded RDT model")
    
    return {
        'model': model,
        'policy_type': 'diffusion',
        'name': 'RDT',
        'n_obs_steps': 2,
        'n_action_steps': rdt_step,
    }


def load_policy(
    policy_name: str,
    task_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    
    
    Args:
        policy_name:  (DP, DP3, pi05, openvla, rdt)
        task_name: 
        **kwargs: 
    
    Returns:
        dict containing model and metadata
    """
    policy_name = policy_name.upper()
    

    dp_params = ['expert_data_num', 'seed', 'checkpoint_num', 'checkpoint_path', 'device']
    dp3_params = ['expert_data_num', 'seed', 'checkpoint_num', 'checkpoint_path', 'device', 'use_rgb']
    pi05_params = ['train_config_name', 'model_name', 'checkpoint_id', 'pi0_step', 'device', 'use_pytorch']
    openvla_params = ['checkpoint_path', 'checkpoint_num', 'use_diffusion', 'unnorm_key', 'device']
    rdt_params = ['ckpt_setting', 'checkpoint_id', 'left_arm_dim', 'right_arm_dim', 'rdt_step', 'device']
    
    def filter_kwargs(allowed_params):
        return {k: v for k, v in kwargs.items() if k in allowed_params}
    
    if policy_name == 'DP':
        return load_dp_model(task_name, **filter_kwargs(dp_params))
    elif policy_name == 'DP3':
        return load_dp3_model(task_name, **filter_kwargs(dp3_params))
    elif policy_name in ['PI05', 'PI0.5']:

        pi05_kwargs = filter_kwargs(pi05_params)
        if 'checkpoint_num' in kwargs and 'checkpoint_id' not in pi05_kwargs:
            pi05_kwargs['checkpoint_id'] = kwargs['checkpoint_num']
        return load_pi05_model(task_name, **pi05_kwargs)
    elif policy_name == 'PI0':

        pi0_kwargs = filter_kwargs(pi05_params)
        if 'checkpoint_num' in kwargs and 'checkpoint_id' not in pi0_kwargs:
            pi0_kwargs['checkpoint_id'] = kwargs['checkpoint_num']
        return load_pi0_model(task_name, **pi0_kwargs)
    elif policy_name in ['OPENVLA', 'OPENVLA-OFT']:
        return load_openvla_model(task_name, **filter_kwargs(openvla_params))
    elif policy_name == 'RDT':
        return load_rdt_model(task_name, **filter_kwargs(rdt_params))
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def freeze_model(model_dict: Dict[str, Any]):
    """"""
    model = model_dict['model']
    policy_type = model_dict.get('policy_type', '')
    policy_name = model_dict.get('name', '')
    is_pytorch = model_dict.get('is_pytorch', False)
    

    if policy_type == 'flow' or policy_name in ['pi05', 'PI05', 'pi0', 'PI0']:
        # Check if the model actually has PyTorch parameters
        # The model._is_pytorch attribute is the ground truth
        actual_is_pytorch = False
        if hasattr(model, '_is_pytorch'):
            actual_is_pytorch = model._is_pytorch
        elif hasattr(model, 'policy') and hasattr(model.policy, '_is_pytorch_model'):
            actual_is_pytorch = model.policy._is_pytorch_model
        
        if actual_is_pytorch:

            if hasattr(model, '_model') and hasattr(model._model, 'parameters'):
                for p in model._model.parameters():
                    p.requires_grad = False
                model._model.eval()
                print(f" Frozen {policy_name} PyTorch model parameters")
            elif hasattr(model, 'policy') and hasattr(model.policy, '_model') and hasattr(model.policy._model, 'parameters'):
                for p in model.policy._model.parameters():
                    p.requires_grad = False
                model.policy._model.eval()
                print(f" Frozen {policy_name} PyTorch model parameters")
            else:
                print(f" Could not freeze {policy_name} PyTorch model (no _model.parameters found)")
        else:

            print(f" {policy_name} is a JAX model, no need to freeze (JAX doesn't track gradients by default)")
        return
    

    if policy_name == 'OpenVLA':

        vla = model_dict.get('vla')
        if vla is not None and hasattr(vla, 'parameters'):
            for p in vla.parameters():
                p.requires_grad = False
            vla.eval()
        

        action_head = model_dict.get('action_head')
        if action_head is not None and hasattr(action_head, 'parameters'):
            for p in action_head.parameters():
                p.requires_grad = False
            action_head.eval()
        

        proprio_projector = model_dict.get('proprio_projector')
        if proprio_projector is not None and hasattr(proprio_projector, 'parameters'):
            for p in proprio_projector.parameters():
                p.requires_grad = False
            proprio_projector.eval()
        

        noisy_action_projector = model_dict.get('noisy_action_projector')
        if noisy_action_projector is not None and hasattr(noisy_action_projector, 'parameters'):
            for p in noisy_action_projector.parameters():
                p.requires_grad = False
            noisy_action_projector.eval()
        
        print(f" Frozen {policy_name} model parameters (VLA + action_head + projectors)")
        return
    

    if hasattr(model, 'policy') and hasattr(model.policy, 'parameters'):

        policy = model.policy
        for p in policy.parameters():
            p.requires_grad = False
        policy.eval()
    elif hasattr(model, 'runner') and hasattr(model.runner, 'policy'):

        policy = model.runner.policy if hasattr(model.runner, 'policy') else model
        if hasattr(policy, 'parameters'):
            for p in policy.parameters():
                p.requires_grad = False
            policy.eval()
    elif hasattr(model, 'parameters'):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    else:
        print(f" Could not freeze {policy_name} model (no parameters method found)")
        return
    
    print(f" Frozen {policy_name} model parameters")
