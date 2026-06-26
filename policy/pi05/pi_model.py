#!/usr/bin/env python3
# -- coding: UTF-8
"""
#!/usr/bin/python3

JAX Performance Optimization Notes:
===================================
1. JIT Compilation: Key JAX functions are wrapped with @jax.jit for performance
2. Compilation Cache: Set JAX_COMPILATION_CACHE_DIR env var to cache compiled functions
3. Warmup: Call warmup_jax() before rollout to pre-compile JAX functions
4. Static shapes: Use fixed input shapes to avoid recompilation
"""
import json
import sys
import jax
import jax.numpy as jnp
import os
import numpy as np
import torch
import functools

# Add openpi source directory to Python path
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
openpi_src_path = os.path.join(parent_directory, "src")
if openpi_src_path not in sys.path:
    sys.path.insert(0, openpi_src_path)

# Add openpi_client source directory to Python path
openpi_client_src_path = os.path.join(parent_directory, "packages", "openpi-client", "src")
if openpi_client_src_path not in sys.path:
    sys.path.insert(0, openpi_client_src_path)

# Add local lerobot directory to Python path (for lerobot.common.datasets).
release_root = os.path.abspath(os.path.join(parent_directory, "..", ".."))
lerobot_path = os.environ.get("LEROBOT_ROOT", os.path.join(release_root, "lerobot"))
if lerobot_path not in sys.path:
    sys.path.insert(0, lerobot_path)

from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# Import JAX helper functions for composition optimization
from openpi.models.pi0 import make_attn_mask, posemb_sincos
import einops

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


# ============================================================================
# JAX JIT-compiled helper functions for performance optimization
# ============================================================================

@functools.partial(jax.jit, static_argnums=(0,))
def _jit_embed_prefix(model, images_dict, image_masks_dict, tokenized_prompt, tokenized_prompt_mask):
    """JIT-compiled prefix embedding for JAX model."""
    # Create observation object
    obs = _model.Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        state=None,  # Not needed for prefix
    )
    return model.embed_prefix(obs)


@functools.partial(jax.jit, static_argnums=())
def _jit_make_attn_mask(input_mask, mask_ar):
    """JIT-compiled attention mask creation."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def _create_jit_denoise_step(model):
    """Create a JIT-compiled denoise step function for the given model."""
    @jax.jit
    def _denoise_step_jit(kv_cache, prefix_mask, x_t, timestep, state, 
                          images_dict, image_masks_dict, tokenized_prompt, tokenized_prompt_mask):
        """JIT-compiled single denoising step."""
        # Reconstruct observation
        obs = _model.Observation(
            images=images_dict,
            image_masks=image_masks_dict,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            state=state,
        )
        return model.denoise_step(obs, kv_cache, prefix_mask, x_t, timestep)
    return _denoise_step_jit


def _create_jit_sample_actions(model):
    """Create a JIT-compiled sample_actions function."""
    @functools.partial(jax.jit, static_argnums=(2,))
    def _sample_actions_jit(rng, obs_flat, num_steps):
        """JIT-compiled action sampling."""
        # Note: This requires flattened observation inputs
        return model.sample_actions(rng, obs_flat, num_steps=num_steps)
    return _sample_actions_jit


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, use_pytorch=None):
        """
        Args:
            train_config_name: OpenPI config name
            model_name: 模型名称
            checkpoint_id: checkpoint 编号
            pi0_step: 动作步数
            use_pytorch: None=自动检测, True=强制PyTorch, False=强制JAX
        """
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        
        # 支持 checkpoint 目录名与 config 名不同的情况
        # 例如：目录名 pi0_base_aloha_robotwin_full_pytorch，config 名 pi0_base_aloha_robotwin_full
        checkpoint_dir_name = train_config_name
        config_name = train_config_name
        
        # 如果目录名以 _pytorch 结尾，config 名去掉 _pytorch
        if train_config_name.endswith('_pytorch'):
            config_name = train_config_name[:-8]  # 去掉 '_pytorch'

        # Check for PyTorch checkpoint first
        base_path = f"policy/pi05/checkpoints/{checkpoint_dir_name}/{self.model_name}"
        pytorch_checkpoint_path = f"{base_path}/{self.checkpoint_id}_pytorch"
        jax_checkpoint_path = f"{base_path}/{self.checkpoint_id}"
        
        # 支持多种目录结构：
        # 1. .../checkpoint_id/model.safetensors (PyTorch 旧格式)
        # 2. .../checkpoint_id_pytorch/model.safetensors (PyTorch 带后缀)
        # 3. .../checkpoint_id/checkpoint_id/model.safetensors (嵌套格式，如 full_pytorch)
        # 4. .../checkpoint_id/assets/ (JAX 格式)
        
        # 检测 PyTorch checkpoint 的多种可能路径
        def find_pytorch_checkpoint(base):
            """查找 PyTorch checkpoint，返回 (checkpoint_path, model_file) 或 (None, None)"""
            # 路径1: base/model.safetensors
            if os.path.exists(f"{base}/model.safetensors"):
                return base, f"{base}/model.safetensors"
            # 路径2: base/checkpoint_id/model.safetensors (嵌套格式)
            nested = f"{base}/{self.checkpoint_id}"
            if os.path.exists(f"{nested}/model.safetensors"):
                return nested, f"{nested}/model.safetensors"
            return None, None
        
        # Determine which checkpoint to use
        if use_pytorch is None:
            # Auto-detect PyTorch vs JAX
            # 先检查 _pytorch 后缀路径
            ckpt_path, model_file = find_pytorch_checkpoint(pytorch_checkpoint_path)
            if ckpt_path is None:
                # 再检查无后缀路径（可能是 full_pytorch 目录下的嵌套结构）
                ckpt_path, model_file = find_pytorch_checkpoint(jax_checkpoint_path)
            
            if ckpt_path is not None:
                checkpoint_path = ckpt_path
                print(f"[PI0] ✓ Using PyTorch checkpoint: {model_file}")
            else:
                checkpoint_path = jax_checkpoint_path
                print(f"[PI0] Using JAX checkpoint")
        elif use_pytorch:
            # Force PyTorch - 检查多种路径
            ckpt_path, model_file = find_pytorch_checkpoint(pytorch_checkpoint_path)
            if ckpt_path is None:
                ckpt_path, model_file = find_pytorch_checkpoint(jax_checkpoint_path)
            
            if ckpt_path is not None:
                checkpoint_path = ckpt_path
                print(f"[PI0] ✓ Using PyTorch checkpoint (forced): {model_file}")
            else:
                raise FileNotFoundError(
                    f"PyTorch checkpoint not found. Tried:\n"
                    f"  - {pytorch_checkpoint_path}/model.safetensors\n"
                    f"  - {pytorch_checkpoint_path}/{self.checkpoint_id}/model.safetensors\n"
                    f"  - {jax_checkpoint_path}/model.safetensors\n"
                    f"  - {jax_checkpoint_path}/{self.checkpoint_id}/model.safetensors"
                )
        else:
            # Force JAX
            checkpoint_path = jax_checkpoint_path
            print(f"[PI0] Using JAX checkpoint (forced)")
        
        # Get assets_id - 支持两种目录结构
        specified_path = f"{checkpoint_path}/assets/"
        if not os.path.exists(specified_path):
            # 尝试新格式：checkpoint_id/checkpoint_id/assets/
            nested_path = f"{checkpoint_path}/{self.checkpoint_id}/assets/"
            if os.path.exists(nested_path):
                checkpoint_path = f"{checkpoint_path}/{self.checkpoint_id}"
                specified_path = nested_path
                print(f"[PI0] Using nested checkpoint structure: {checkpoint_path}")
        
        entries = os.listdir(specified_path)
        assets_id = entries[0]
        print(specified_path,entries,assets_id)
        
        config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_path,
            robotwin_repo_id=assets_id,
            pytorch_device="cuda:0",  # Use GPU for PyTorch
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None  # 初始化 instruction 属性
        self.pi0_step = pi0_step
        
        # Store model config for distribution-level composition
        self._model = self.policy._model
        self._is_pytorch = self.policy._is_pytorch_model
        self._device = self.policy._pytorch_device if self._is_pytorch else None
        self._input_transform = self.policy._input_transform
        self._output_transform = self.policy._output_transform
        self._sample_kwargs = self.policy._sample_kwargs
        
        # Get action dimensions from model config
        if hasattr(self._model, 'config'):
            self.action_horizon = self._model.config.action_horizon
            self.action_dim = self._model.config.action_dim
        else:
            self.action_horizon = getattr(self._model, 'action_horizon', 50)
            self.action_dim = getattr(self._model, 'action_dim', 14)
        
        # Default num_steps for flow matching
        self.num_inference_steps = self._sample_kwargs.get('num_steps', 10)
        
        # Create a simple runner wrapper for compatibility with composed policy evaluation
        self.runner = PI0Runner(self)
        
        # JAX JIT compilation cache
        self._jit_denoise_step = None
        self._jit_sample_actions = None
        self._jax_warmed_up = False
        
        # Fixed shape for JAX JIT compilation (避免重复编译)
        self._fixed_action_horizon = 50  # pi0.5 内部固定形状
        self._fixed_action_dim = 32      # pi0.5 内部固定形状
        self._jit_denoise_step_fixed = None  # 预编译的固定形状 denoise_step
        self._composition_warmed_up = False  # 是否已预热 composition 用的 JIT 函数
        
        # Auto-warmup JAX if not using PyTorch
        if not self._is_pytorch:
            print("[PI0] JAX model detected. Call warmup_jax() before rollout for best performance.")
    
    def warmup_jax(self, verbose=True):
        """
        Warmup JAX JIT compilation by running dummy inference.
        
        This pre-compiles JAX functions to avoid compilation overhead during rollout.
        Should be called once before evaluation starts.
        
        Args:
            verbose: Whether to print warmup progress
        """
        if self._is_pytorch:
            if verbose:
                print("[PI0] PyTorch model, no JAX warmup needed.")
            return
        
        if self._jax_warmed_up:
            if verbose:
                print("[PI0] JAX already warmed up.")
            return
        
        if verbose:
            print("[PI0] Starting JAX warmup (this may take 1-2 minutes)...")
        
        import time
        start_time = time.time()
        
        # Create dummy inputs with correct shapes
        # Note: action_dim for pi0.5 internal model is 32, but robot state is 14
        # We need to use the actual robot state dimension (14 for dual-arm)
        robot_state_dim = 14  # Dual-arm robot: 7 joints per arm
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_state = np.zeros(robot_state_dim, dtype=np.float32)
        dummy_img_arr = [dummy_img, dummy_img, dummy_img]
        
        # Set a dummy instruction if not set
        original_instruction = self.instruction
        if self.instruction is None:
            self.set_language("warmup task")
        
        # Run one full inference to trigger JIT compilation
        try:
            self.update_observation_window(dummy_img_arr, dummy_state)
            _ = self.get_action()
            
            # Also warmup prepare_data for distribution-level composition
            _ = self.prepare_data()
            
            # Warmup composition JIT function (方案2)
            self.warmup_composition_jax(verbose=False)
            
            self._jax_warmed_up = True
            elapsed = time.time() - start_time
            if verbose:
                print(f"[PI0] ✓ JAX warmup completed in {elapsed:.1f}s")
        except Exception as e:
            if verbose:
                print(f"[PI0] JAX warmup failed (non-critical): {e}")
                print(f"[PI0] Training will continue, first batch may be slow due to JIT compilation.")
            # Don't raise - warmup failure is non-critical
        finally:
            # Restore original instruction
            self.instruction = original_instruction
            self.observation_window = None

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
    
    def reset_obs(self):
        """Alias for reset_obsrvationwindows for compatibility with eval.py"""
        self.reset_obsrvationwindows()
    
    def denoise(self, x_t, timestep, state, past_key_values=None, prefix_pad_masks=None):
        """
        Adapter method for composition.py compatibility.
        Maps to the underlying model's denoise_step with correct parameter order.
        
        Args:
            x_t: noisy action trajectory [B, T, action_dim]
            timestep: current timestep [B]
            state: robot state
            past_key_values: cached KV from prefix (PyTorch) or kv_cache (JAX)
            prefix_pad_masks: attention masks for prefix
        
        Returns:
            velocity: predicted velocity [B, T, action_dim]
        """
        if self._is_pytorch:
            # PyTorch path: call denoise_step on _model
            return self._model.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
        else:
            # JAX path: call denoise_step on _model
            # Note: JAX uses different parameter names
            return self._model.denoise_step(
                state=state,
                prefix_mask=prefix_pad_masks,
                kv_cache=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
    
    def _get_norm_stats(self):
        """
        Extract norm_stats from _output_transform for velocity space conversion.
        
        The _output_transform is a CompositeTransform containing an Unnormalize transform
        which has the norm_stats used for action normalization.
        
        Returns:
            dict with 'actions' key containing NormStats, or None if not found
        """
        try:
            # _output_transform is a CompositeTransform with transforms list
            if hasattr(self._output_transform, 'transforms'):
                for transform in self._output_transform.transforms:
                    # Look for Unnormalize transform which has norm_stats
                    if hasattr(transform, 'norm_stats') and transform.norm_stats is not None:
                        return transform.norm_stats
            return None
        except Exception as e:
            print(f"[PI0] Warning: Failed to extract norm_stats: {e}")
            return None
    
    def update_obs(self, obs):
        """
        Compatibility method for composed policy evaluation.
        This is an alias for update_observation_window to maintain compatibility
        with DP/DP3 interface.
        
        Args:
            obs: Tuple of (img_arr, state) where:
                - img_arr: List of [front_img, right_img, left_img]
                - state: Robot state array
        """
        if isinstance(obs, tuple) and len(obs) == 2:
            img_arr, state = obs
            self.update_observation_window(img_arr, state)
        else:
            raise ValueError(f"Expected obs to be (img_arr, state) tuple, got {type(obs)}")

    def prepare_data(self, observation=None):
        """
        Prepare inference data for distribution-level composition (Flow Matching).
        
        This method prepares all the data needed for the flow matching denoising process,
        allowing external composition with other policies (DP, DP3).
        
        Args:
            observation: Optional (img_arr, state) tuple. If provided, will update observation_window.
                        If None, uses existing observation_window.
        
        Returns:
            dict containing:
                - model: The underlying PI0 model
                - observation: Preprocessed observation
                - state: Robot state
                - prefix_cache: KV cache from prefix (images + language)
                - prefix_pad_masks: Padding masks for prefix
                - num_inference_steps: Number of denoising steps
                - action_horizon: Action sequence length
                - action_dim: Action dimension
                - device: PyTorch device (if applicable)
                - is_pytorch: Whether using PyTorch model
                - dt: Time step size for flow matching
                - denoise_fn: Function to compute velocity field at each step
        """
        if observation is not None:
            # observation is (img_arr, state) tuple
            img_arr, state = observation
            self.update_observation_window(img_arr, state)
        
        assert self.observation_window is not None, "update observation_window first!"
        assert self.instruction is not None, "set language instruction first using set_language()!"
        
        # Transform inputs - make a copy to avoid modifying original
        inputs = {k: v for k, v in self.observation_window.items()}
        inputs = self._input_transform(inputs)
        
        if self._is_pytorch:
            # PyTorch path
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._device)[None, ...], 
                inputs
            )
            
            observation_obj = _model.Observation.from_dict(inputs)
            
            # Preprocess observation
            images, img_masks, lang_tokens, lang_masks, state = self._model._preprocess_observation(
                observation_obj, train=False
            )
            
            # Compute prefix embeddings and KV cache
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = self._model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks) \
                if hasattr(self._model, 'make_att_2d_masks') else None
            
            # Get prefix attention masks
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            
            # Compute KV cache
            prefix_att_2d_masks_4d = self._model._prepare_attention_masks_4d(prefix_att_2d_masks)
            self._model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            
            _, past_key_values = self._model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            
            # Ensure device is set (fallback to cuda if available)
            device = self._device if self._device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            
            # Extract norm_stats from _output_transform for velocity space conversion
            pi05_norm_stats = self._get_norm_stats()
            
            infer_data = {
                "model": self,  # Return PI0 wrapper (has denoise method) instead of _model
                "_model": self._model,  # Also store the underlying PI0Pytorch model for denoise_step_full
                "state": state,
                "prefix_embs": prefix_embs,  # For denoise_step_full (distribution-level composition)
                "prefix_pad_masks": prefix_pad_masks,
                "prefix_att_masks": prefix_att_masks,  # For denoise_step_full
                "past_key_values": past_key_values,
                "num_inference_steps": self.num_inference_steps,
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "device": device,
                "is_pytorch": True,
                "dt": -1.0 / self.num_inference_steps,
                "policy_type": "PI05",
                "output_transform": self._output_transform,
                "norm_stats": pi05_norm_stats,  # Add norm_stats for velocity space conversion
            }
        else:
            # JAX path - similar structure but using JAX operations
            # Note: JAX functions are JIT-compiled on first call, which can be slow.
            # Call warmup_jax() before rollout to pre-compile.
            import jax.numpy as jnp
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            
            observation_obj = _model.Observation.from_dict(inputs)
            observation_obj = _model.preprocess_observation(None, observation_obj, train=False)
            
            # Embed prefix and get KV cache
            # These operations are JIT-compiled by JAX automatically
            prefix_tokens, prefix_mask, prefix_ar_mask = self._model.embed_prefix(observation_obj)
            prefix_attn_mask = self._make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            
            # Use JIT-compiled LLM forward pass
            _, kv_cache = self._model.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )
            
            # Block until computation is complete (important for timing)
            # This ensures JIT compilation happens here, not during rollout
            jax.block_until_ready(kv_cache)
            
            # Extract norm_stats from _output_transform for velocity space conversion
            pi05_norm_stats = self._get_norm_stats()
            
            infer_data = {
                "model": self._model,
                "observation": observation_obj,
                "state": observation_obj.state,  # 添加 state 键
                "prefix_mask": prefix_mask,
                "prefix_pad_masks": prefix_mask,  # 添加别名以保持一致性
                "kv_cache": kv_cache,
                "past_key_values": kv_cache,  # 添加别名以保持一致性
                "num_inference_steps": self.num_inference_steps,
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "device": None,  # JAX doesn't use device in the same way
                "is_pytorch": False,
                "dt": -1.0 / self.num_inference_steps,
                "policy_type": "PI05",
                "output_transform": self._output_transform,
                "pi0_model": self,  # Add reference to PI0 wrapper for get_action()
                "norm_stats": pi05_norm_stats,  # Add norm_stats for velocity space conversion
            }
        
        return infer_data
    
    def prepare_data_batch(self, observations: list):
        """
        批量版本的 prepare_data。
        
        由于 JAX 模型的 kv_cache 结构复杂，批量处理后拆分会破坏其结构，
        所以这里采用逐个处理的方式，但保持接口一致。
        
        真正的优化在 composition.py 的 _prepare_jax_cache 中实现，
        它会复用 kv_cache 在去噪循环中。
        
        Args:
            observations: List of (img_arr, state) tuples
        
        Returns:
            dict containing batch inference data
        """
        if not observations:
            return None
        
        batch_size = len(observations)
        
        # 确保 instruction 已设置
        if self.instruction is None:
            self.set_language("execute the task")
        
        infer_data_list = []
        
        for img_arr, state in observations:
            try:
                single_data = self.prepare_data((img_arr, state))
                single_data['pi0_model'] = self
                infer_data_list.append(single_data)
            except Exception as e:
                print(f"Warning: pi0.5 prepare_data failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        if not infer_data_list:
            return None
        
        # 返回与原来格式兼容的结果
        result = infer_data_list[0].copy()
        result['batch_infer_data'] = infer_data_list
        result['batch_size'] = batch_size
        result['policy_type'] = 'flow'
        result['is_batched'] = True
        result['n_action_steps'] = 8
        
        return result
    
    def _make_attn_mask(self, input_mask, mask_ar):
        """Create attention mask for JAX model."""
        import jax.numpy as jnp
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        cumsum = jnp.cumsum(mask_ar, axis=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
        valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
        return jnp.logical_and(attn_mask, valid_mask)

    # ========================================================================
    # JAX Fixed-Shape Composition Methods (方案2: 预编译固定形状的 JAX 函数)
    # ========================================================================
    
    def _create_jit_denoise_step_fixed(self):
        """
        创建固定形状的 denoise_step 函数。
        
        注意：不能对包含 kv_cache 的函数使用 @jax.jit，因为 kv_cache 被 traced 后
        会导致 Gemma 模型的类型检查失败。
        
        优化策略：
        - 不使用 @jax.jit 装饰器
        - 依赖内部的 PaliGemma.llm 已经被 JIT 编译
        - 使用固定形状确保内部 JIT 缓存有效
        """
        if self._is_pytorch:
            return None
        
        model = self._model
        
        # 检查是否是 pi0.5 模型
        is_pi05 = getattr(model, 'pi05', False) or getattr(getattr(model, 'config', None), 'pi05', False)
        
        if is_pi05:
            # pi0.5: embed_suffix 不需要 observation.state
            # 注意：不使用 @jax.jit，避免 kv_cache 被 traced
            def _denoise_step_fixed_pi05(kv_cache, prefix_mask, x_t, timestep):
                """
                pi0.5 专用的固定形状 denoise_step。
                
                不需要 observation 参数，因为 pi0.5 的 state 在 prefix 中处理。
                内部的 PaliGemma.llm 已经被 JIT 编译，所以不需要外层 JIT。
                """
                import einops
                
                # embed_suffix for pi0.5 (不需要 obs.state)
                action_tokens = model.action_in_proj(x_t)
                time_emb = posemb_sincos(timestep, model.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
                
                # time MLP (for adaRMS)
                time_emb = model.time_mlp_in(time_emb)
                time_emb = jax.nn.swish(time_emb)
                time_emb = model.time_mlp_out(time_emb)
                time_emb = jax.nn.swish(time_emb)
                
                action_expert_tokens = action_tokens
                adarms_cond = time_emb
                
                # Create masks
                batch_size = x_t.shape[0]
                action_horizon = model.action_horizon
                
                suffix_mask = jnp.ones((batch_size, action_horizon), dtype=jnp.bool_)
                suffix_ar_mask = jnp.array([True] + ([False] * (action_horizon - 1)))
                
                # Create attention masks
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=action_horizon)
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                
                # Compute positions
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                
                # Forward pass through LLM with KV cache
                # 注意：PaliGemma.llm 内部已经 JIT 编译
                (prefix_out, suffix_out), _ = model.PaliGemma.llm(
                    [None, action_expert_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                
                # Project to action space
                v_t = model.action_out_proj(suffix_out[:, -action_horizon:])
                
                return v_t
            
            return _denoise_step_fixed_pi05
        else:
            # pi0: 需要 observation.state
            # 同样不使用 @jax.jit
            def _denoise_step_fixed(kv_cache, prefix_mask, x_t, timestep, observation):
                """
                固定形状的 denoise_step。
                """
                return model.denoise_step(
                    observation=observation,
                    kv_cache=kv_cache,
                    prefix_mask=prefix_mask,
                    x_t=x_t,
                    timestep=timestep,
                )
            
            return _denoise_step_fixed
    
    def warmup_composition_jax(self, verbose=True):
        """
        预热用于 composition 的 JAX 函数。
        
        由于不再使用外层 @jax.jit（避免 kv_cache 被 traced），
        这里主要是预热内部的 PaliGemma.llm JIT 缓存。
        """
        if self._is_pytorch:
            if verbose:
                print("[PI0] PyTorch model, no composition warmup needed.")
            return
        
        if self._composition_warmed_up:
            if verbose:
                print("[PI0] Composition JAX already warmed up.")
            return
        
        if verbose:
            print("[PI0] Starting composition JAX warmup...")
        
        import time
        start_time = time.time()
        
        try:
            # 创建 JIT 函数
            self._jit_denoise_step_fixed = self._create_jit_denoise_step_fixed()
            
            # 创建 dummy 数据进行预编译
            dummy_x_t = jnp.zeros((1, self._fixed_action_horizon, self._fixed_action_dim), dtype=jnp.float32)
            dummy_timestep = jnp.array([0.5], dtype=jnp.float32)
            
            # 需要先准备 observation 和 kv_cache
            # 使用 warmup 时的 observation
            robot_state_dim = 14
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_state = np.zeros(robot_state_dim, dtype=np.float32)
            dummy_img_arr = [dummy_img, dummy_img, dummy_img]
            
            original_instruction = self.instruction
            if self.instruction is None:
                self.set_language("warmup composition")
            
            self.update_observation_window(dummy_img_arr, dummy_state)
            
            # 准备 infer_data 获取 kv_cache 和 observation
            infer_data = self.prepare_data()
            
            if not infer_data.get('is_pytorch', True):
                kv_cache = infer_data['kv_cache']
                prefix_mask = infer_data['prefix_mask']
                observation = infer_data['observation']
                
                # 检查是否是 pi0.5 模型
                is_pi05 = getattr(self._model, 'pi05', False) or getattr(getattr(self._model, 'config', None), 'pi05', False)
                
                # 预编译：调用一次触发 JIT 编译
                if is_pi05:
                    # pi0.5: 不需要 observation
                    _ = self._jit_denoise_step_fixed(
                        kv_cache, prefix_mask, dummy_x_t, dummy_timestep
                    )
                else:
                    # pi0: 需要 observation
                    _ = self._jit_denoise_step_fixed(
                        kv_cache, prefix_mask, dummy_x_t, dummy_timestep, observation
                    )
                
                # 等待编译完成
                jax.block_until_ready(_)
                
                self._composition_warmed_up = True
                elapsed = time.time() - start_time
                if verbose:
                    print(f"[PI0] ✓ Composition JAX warmup completed in {elapsed:.1f}s")
            
        except Exception as e:
            if verbose:
                print(f"[PI0] Composition warmup failed (non-critical): {e}")
                import traceback
                traceback.print_exc()
        finally:
            self.instruction = original_instruction
            self.observation_window = None
    
    def denoise_step_for_composition(
        self,
        kv_cache,
        prefix_mask,
        x_t_fixed: "jnp.ndarray",
        timestep: "jnp.ndarray",
        observation=None,
    ) -> "jnp.ndarray":
        """
        专门用于 composition 的 denoise_step，使用固定形状确保 JIT 缓存有效。
        
        Args:
            kv_cache: KV cache from prefix forward pass
            prefix_mask: Mask for prefix tokens
            x_t_fixed: Noisy action with FIXED shape [1, 50, 32]
            timestep: Current timestep [1]
            observation: Preprocessed observation object (仅 pi0 需要，pi0.5 不需要)
        
        Returns:
            v_t: Velocity field [1, 50, 32]
        
        Note:
            调用者需要确保 x_t_fixed 的形状是 [1, 50, 32]，
            并在调用后自行截取需要的部分。
        """
        if self._is_pytorch:
            raise RuntimeError("denoise_step_for_composition is only for JAX models")
        
        # 确保 JIT 函数已创建
        if self._jit_denoise_step_fixed is None:
            self._jit_denoise_step_fixed = self._create_jit_denoise_step_fixed()
        
        # 验证输入形状
        expected_shape = (1, self._fixed_action_horizon, self._fixed_action_dim)
        if x_t_fixed.shape != expected_shape:
            raise ValueError(
                f"x_t_fixed must have shape {expected_shape}, got {x_t_fixed.shape}. "
                f"Use pad_to_fixed_shape() to prepare the input."
            )
        
        # 检查是否是 pi0.5 模型
        is_pi05 = getattr(self._model, 'pi05', False) or getattr(getattr(self._model, 'config', None), 'pi05', False)
        
        if is_pi05:
            # pi0.5: 不需要 observation
            return self._jit_denoise_step_fixed(kv_cache, prefix_mask, x_t_fixed, timestep)
        else:
            # pi0: 需要 observation
            if observation is None:
                raise ValueError("observation is required for pi0 models")
            return self._jit_denoise_step_fixed(kv_cache, prefix_mask, x_t_fixed, timestep, observation)
    
    def pad_to_fixed_shape(
        self,
        x: "np.ndarray | jnp.ndarray",
        input_time_dim: int,
        input_action_dim: int,
    ) -> "jnp.ndarray":
        """
        将输入 pad 到固定形状 [1, 50, 32]。
        
        使用最后一个时间步的值填充剩余时间步（而不是零），
        以减少 padding 对模型预测的影响。
        
        Args:
            x: Input tensor [1, input_time_dim, input_action_dim]
            input_time_dim: Original time dimension (e.g., 8 for DP)
            input_action_dim: Original action dimension (e.g., 14 for DP)
        
        Returns:
            x_fixed: Padded tensor [1, 50, 32]
        """
        import jax.numpy as jnp
        
        # 创建固定形状的零张量
        x_fixed = jnp.zeros(
            (1, self._fixed_action_horizon, self._fixed_action_dim),
            dtype=jnp.float32
        )
        
        # 确保输入是 JAX array
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x, dtype=jnp.float32)
        
        # 填充实际数据
        x_fixed = x_fixed.at[0, :input_time_dim, :input_action_dim].set(x[0, :input_time_dim, :input_action_dim])
        
        # 用最后一个时间步填充剩余时间步（减少 padding 影响）
        if input_time_dim < self._fixed_action_horizon:
            last_step = x[0, input_time_dim - 1:input_time_dim, :input_action_dim]
            for t in range(input_time_dim, self._fixed_action_horizon):
                x_fixed = x_fixed.at[0, t, :input_action_dim].set(last_step[0])
        
        return x_fixed
    
    def extract_from_fixed_shape(
        self,
        v_fixed: "jnp.ndarray",
        input_time_dim: int,
        input_action_dim: int,
    ) -> "jnp.ndarray":
        """
        从固定形状 [1, 50, 32] 中提取原始形状的数据。
        
        Args:
            v_fixed: Velocity with fixed shape [1, 50, 32]
            input_time_dim: Target time dimension (e.g., 8 for DP)
            input_action_dim: Target action dimension (e.g., 14 for DP)
        
        Returns:
            v: Extracted velocity [1, input_time_dim, input_action_dim]
        """
        return v_fixed[:, :input_time_dim, :input_action_dim]


class PI0Runner:
    """
    Simple runner wrapper for PI0 to provide compatibility with composed policy evaluation.
    This mimics the interface of DP/DP3 runners.
    """
    def __init__(self, pi0_model):
        self.pi0_model = pi0_model
    
    def reset_obs(self):
        """Reset observation window - delegates to PI0's reset method."""
        self.pi0_model.reset_obsrvationwindows()

