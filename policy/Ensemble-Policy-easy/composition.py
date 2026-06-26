"""Final policy composition used by the EnsembleVLA release.

Only the two released composition families are kept here:

1. DP + DP3: diffusion-level composition by combining denoising model outputs.
2. DP + pi0.5: native_x0_tail composition. pi0.5 is queried in its native
   action space, its clean action is mapped into the DP normalized x0 space,
   and the two clean predictions are composed inside the DP denoising loop.

Only the released composition paths are implemented.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


class DiffusionComposer:
    """Distribution-level composition for two diffusion policies."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)

    @torch.no_grad()
    def compose_generic_diffusion(
        self,
        infer_data1: Dict[str, Any],
        infer_data2: Dict[str, Any],
        w1: float,
        w2: float,
    ) -> torch.Tensor:
        policy1_type = infer_data1.get("policy_type", "diffusion")
        policy2_type = infer_data2.get("policy_type", "diffusion")
        if policy1_type != "diffusion" or policy2_type != "diffusion":
            raise ValueError(f"Expected diffusion + diffusion, got {policy1_type} + {policy2_type}")
        if infer_data1.get("is_openvla") or infer_data2.get("is_openvla"):
            raise ValueError("This diffusion composition path is not part of this release package")
        return self._compose_diffusion_diffusion(infer_data1, infer_data2, w1, w2)

    @torch.no_grad()
    def _compose_diffusion_diffusion(
        self,
        infer_data1: Dict[str, Any],
        infer_data2: Dict[str, Any],
        w1: float,
        w2: float,
    ) -> torch.Tensor:
        model1 = infer_data1["model"]
        model2 = infer_data2["model"]
        scheduler = infer_data2["scheduler"]
        cond_data = infer_data2["cond_data"]
        cond_mask = infer_data2["cond_mask"]
        local_cond1 = infer_data1.get("local_cond")
        global_cond1 = infer_data1.get("global_cond")
        local_cond2 = infer_data2.get("local_cond")
        global_cond2 = infer_data2.get("global_cond")
        num_inference_steps = infer_data2["num_inference_steps"]
        action_dim = infer_data2["Da"]
        obs_steps = infer_data2["To"]
        normalizer = infer_data2["action_normalizer"]
        n_action_steps = infer_data2["n_action_steps"]

        trajectory = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            if isinstance(cond_mask, torch.Tensor) and cond_mask.any():
                trajectory[cond_mask] = cond_data[cond_mask]
            output1 = model1(trajectory, t, local_cond=local_cond1, global_cond=global_cond1)
            output2 = model2(trajectory, t, local_cond=local_cond2, global_cond=global_cond2)
            model_output = float(w1) * output1 + float(w2) * output2
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        if isinstance(cond_mask, torch.Tensor) and cond_mask.any():
            trajectory[cond_mask] = cond_data[cond_mask]

        action_pred = normalizer.unnormalize(trajectory[..., :action_dim])
        start = obs_steps - 1
        return action_pred[:, start:start + n_action_steps]


class FlowDiffusionComposer:
    """Final DP + pi0.5 native_x0_tail composer."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.last_base_dp: Optional[torch.Tensor] = None
        self.last_base_pi05: Optional[torch.Tensor] = None

    def _extract_dp_norm_params(self, normalizer: Any) -> Optional[Dict[str, torch.Tensor]]:
        """Extract min/max tensors from DP's action normalizer."""
        try:
            if not hasattr(normalizer, "params_dict"):
                return None
            params_dict = normalizer.params_dict

            if "action" in params_dict:
                action_params = params_dict["action"]
                if hasattr(action_params, "params_dict"):
                    action_params = action_params.params_dict
                if "input_stats" in action_params:
                    stats = action_params["input_stats"]
                    if "min" in stats and "max" in stats:
                        return {"min": stats["min"], "max": stats["max"]}

            if "input_stats" in params_dict:
                stats = params_dict["input_stats"]
                if "min" in stats and "max" in stats:
                    return {"min": stats["min"], "max": stats["max"]}

            if "scale" in params_dict and "offset" in params_dict:
                scale = params_dict["scale"]
                offset = params_dict["offset"]
                value_range = 2.0 / (scale + 1e-8)
                min_val = -(offset + 1.0) * value_range / 2.0
                return {"min": min_val, "max": min_val + value_range}
        except Exception as exc:
            print(f"[warning] failed to extract DP normalizer params: {exc}")
        return None

    @torch.no_grad()
    def _get_pure_pi05_action(
        self,
        flow_infer_data: Dict[str, Any],
        diffusion_infer_data: Dict[str, Any],
    ) -> torch.Tensor:
        """Query pi0.5 in its native rollout space and return raw actions."""
        pi05_model = None
        batch_data = flow_infer_data.get("batch_infer_data") or []
        if batch_data:
            first = batch_data[0]
            pi05_model = first.get("model") or first.get("pi0_model")
        pi05_model = pi05_model or flow_infer_data.get("model") or flow_infer_data.get("pi0_model")
        if pi05_model is None:
            raise ValueError("Cannot find pi0.5 model in flow inference data")

        raw_action = pi05_model.get_action()
        if isinstance(raw_action, torch.Tensor):
            action_tensor = raw_action.detach().float()
        else:
            action_tensor = torch.from_numpy(np.asarray(raw_action)).float()
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)

        action_dim = diffusion_infer_data.get("Da", 14)
        pi0_step = getattr(pi05_model, "pi0_step", action_tensor.shape[1])
        return action_tensor[:, :pi0_step, :action_dim]

    def _raw_action_to_dp_x0_space(
        self,
        raw_action: torch.Tensor,
        dp_norm_params: Dict[str, torch.Tensor],
        target_shape: torch.Size,
        action_dim: int,
        obs_steps: int,
        n_action_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        debug: bool = False,
    ) -> torch.Tensor:
        if raw_action.dim() == 2:
            raw_action = raw_action.unsqueeze(0)

        batch, horizon, dims = target_shape
        raw_action = raw_action.to(device=device, dtype=dtype)
        if raw_action.shape[0] != batch:
            raw_action = raw_action.expand(batch, -1, -1) if raw_action.shape[0] == 1 else raw_action[:batch]

        dim = min(action_dim, dims, raw_action.shape[-1])
        if dim <= 0:
            raise ValueError(f"Invalid action dimension for target shape {tuple(target_shape)}")

        dp_min = dp_norm_params["min"][:dim].to(device=device, dtype=dtype).view(1, 1, -1)
        dp_max = dp_norm_params["max"][:dim].to(device=device, dtype=dtype).view(1, 1, -1)
        dp_range = torch.clamp(dp_max - dp_min, min=1e-8)

        raw_trimmed = raw_action[..., :dim]
        raw_aligned = raw_trimmed[:, :1, :].expand(batch, horizon, dim).clone()
        start = max(int(obs_steps) - 1, 0)
        end = min(start + int(n_action_steps), horizon)
        copy_len = min(max(end - start, 0), raw_trimmed.shape[1])
        if copy_len > 0:
            raw_aligned[:, start:start + copy_len, :] = raw_trimmed[:, :copy_len, :]
            if start + copy_len < horizon:
                tail = raw_trimmed[:, copy_len - 1:copy_len, :]
                raw_aligned[:, start + copy_len:, :] = tail.expand(batch, horizon - start - copy_len, dim)

        x0_dp_space = (raw_aligned - dp_min) / dp_range * 2.0 - 1.0
        x0_dp_space = torch.clamp(x0_dp_space, min=-3.0, max=3.0)
        full_x0 = torch.zeros(target_shape, device=device, dtype=dtype)
        full_x0[..., :dim] = x0_dp_space

        if debug:
            print(f"[native_x0_tail] pi0.5 raw shape={tuple(raw_action.shape)} target={tuple(target_shape)}")
            print(f"[native_x0_tail] aligned start={start} copy_len={copy_len} dim={dim}")
        return full_x0

    @torch.no_grad()
    def compose_flow_diffusion(
        self,
        flow_infer_data: Dict[str, Any],
        diffusion_infer_data: Dict[str, Any],
        w_flow: float,
        w_diffusion: float,
        debug: bool = False,
        composition_space: Optional[str] = "native_x0_tail",
        force_composer: bool = False,
    ) -> torch.Tensor:
        mode = composition_space or "native_x0_tail"
        if mode not in {"native_x0_tail", "raw_x0_tail", "x0_raw_tail"}:
            raise ValueError(f"Unsupported DP + pi0.5 composition mode in release: {mode}")
        return self._compose_native_x0_tail(flow_infer_data, diffusion_infer_data, w_flow, w_diffusion, debug)

    @torch.no_grad()
    def _compose_native_x0_tail(
        self,
        flow_infer_data: Dict[str, Any],
        diffusion_infer_data: Dict[str, Any],
        w_flow: float,
        w_diffusion: float,
        debug: bool = False,
    ) -> torch.Tensor:
        diff_model = diffusion_infer_data["model"]
        diff_scheduler = diffusion_infer_data["scheduler"]
        cond_data = diffusion_infer_data["cond_data"]
        cond_mask = diffusion_infer_data["cond_mask"]
        local_cond = diffusion_infer_data.get("local_cond")
        global_cond = diffusion_infer_data.get("global_cond")
        num_inference_steps = diffusion_infer_data["num_inference_steps"]
        action_dim = diffusion_infer_data["Da"]
        obs_steps = diffusion_infer_data["To"]
        normalizer = diffusion_infer_data["action_normalizer"]
        n_action_steps = diffusion_infer_data.get("n_action_steps", 8)

        dp_norm_params = self._extract_dp_norm_params(normalizer)
        if dp_norm_params is None:
            raise RuntimeError("native_x0_tail requires DP min/max normalizer parameters")

        trajectory = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)
        pi05_raw_action = self._get_pure_pi05_action(flow_infer_data, diffusion_infer_data)
        x0_flow_dp_space = self._raw_action_to_dp_x0_space(
            pi05_raw_action,
            dp_norm_params,
            trajectory.shape,
            action_dim,
            obs_steps,
            n_action_steps,
            trajectory.device,
            trajectory.dtype,
            debug=debug,
        )

        diff_scheduler.set_timesteps(num_inference_steps)
        prediction_type = getattr(diff_scheduler.config, "prediction_type", "epsilon")
        is_sample_prediction = prediction_type == "sample"
        eps = 1e-8

        if debug:
            print(f"[native_x0_tail] w_flow={float(w_flow):.4f} w_diffusion={float(w_diffusion):.4f}")
            print(f"[native_x0_tail] prediction_type={prediction_type} steps={num_inference_steps}")

        for step_idx, t_diff in enumerate(diff_scheduler.timesteps):
            if isinstance(cond_mask, torch.Tensor) and cond_mask.any():
                trajectory[cond_mask] = cond_data[cond_mask]

            if isinstance(t_diff, torch.Tensor):
                t_idx = int(t_diff.flatten()[0].item())
            else:
                t_idx = int(t_diff)
            alpha_t = diff_scheduler.alphas_cumprod[t_idx]
            if isinstance(alpha_t, torch.Tensor):
                alpha_t = alpha_t.to(device=trajectory.device, dtype=trajectory.dtype)
            else:
                alpha_t = torch.tensor(alpha_t, device=trajectory.device, dtype=trajectory.dtype)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            noise_pred_dp = diff_model(trajectory, t_diff, local_cond=local_cond, global_cond=global_cond)
            if is_sample_prediction:
                x0_dp = noise_pred_dp
                noise_pred_dp = (trajectory - sqrt_alpha_t * x0_dp) / (sqrt_one_minus_alpha_t + eps)
            else:
                x0_dp = (trajectory - sqrt_one_minus_alpha_t * noise_pred_dp) / (sqrt_alpha_t + eps)
                x0_dp = torch.clamp(x0_dp, min=-3.0, max=3.0)

            x0_combined = float(w_diffusion) * x0_dp + float(w_flow) * x0_flow_dp_space
            noise_combined = (trajectory - sqrt_alpha_t * x0_combined) / (sqrt_one_minus_alpha_t + eps)

            if debug and (step_idx < 3 or step_idx == num_inference_steps - 1):
                print(f"[native_x0_tail] step={step_idx} t={t_idx} x0_mean={x0_combined.mean().item():.4f}")
            trajectory = diff_scheduler.step(noise_combined, t_diff, trajectory).prev_sample

        if isinstance(cond_mask, torch.Tensor) and cond_mask.any():
            trajectory[cond_mask] = cond_data[cond_mask]

        action_pred = normalizer.unnormalize(trajectory[..., :action_dim])
        start = obs_steps - 1
        dp_window = action_pred[:, start:start + n_action_steps]
        final_action = dp_window

        if pi05_raw_action.shape[1] > final_action.shape[1]:
            flow_tail = pi05_raw_action.to(device=final_action.device, dtype=final_action.dtype)
            if flow_tail.shape[0] != final_action.shape[0]:
                flow_tail = flow_tail.expand(final_action.shape[0], -1, -1) if flow_tail.shape[0] == 1 else flow_tail[:final_action.shape[0]]
            flow_tail = flow_tail[:, final_action.shape[1]:, :action_dim]
            if flow_tail.numel() > 0:
                final_action = torch.cat([final_action, flow_tail], dim=1)

        self.last_base_dp = dp_window[..., :action_dim].detach()
        self.last_base_pi05 = pi05_raw_action[:, :final_action.shape[1], :action_dim].to(
            device=final_action.device, dtype=final_action.dtype
        ).detach()
        return final_action


class PolicyComposer:
    """Dispatch only the released DP+DP3 and DP+pi0.5 composition families."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.diffusion_composer = DiffusionComposer(device)
        self.flow_diffusion_composer = FlowDiffusionComposer(device)
        self._debug_count = 0
        self._debug_max = 0    # 禁用debug输出,加速评估

    def set_composition_params(self, **kwargs):
        return None

    @torch.no_grad()
    def compose(
        self,
        infer_data1: Dict[str, Any],
        infer_data2: Dict[str, Any],
        w1: float,
        w2: float,
        debug: Optional[bool] = None,
        composition_mode: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        if debug is None:
            debug = self._debug_count < self._debug_max
            if debug:
                self._debug_count += 1

        policy1_type = infer_data1.get("policy_type", "diffusion")
        policy2_type = infer_data2.get("policy_type", "diffusion")
        if debug:
            print(f"[PolicyComposer] {policy1_type} + {policy2_type}, w1={float(w1):.4f}, w2={float(w2):.4f}")
            print(f"[PolicyComposer] composition_mode={composition_mode or 'native_x0_tail'}")

        if policy1_type == "diffusion" and policy2_type == "diffusion":
            return self.diffusion_composer.compose_generic_diffusion(infer_data1, infer_data2, w1, w2)

        if policy1_type == "flow" and policy2_type == "diffusion":
            return self.flow_diffusion_composer.compose_flow_diffusion(
                infer_data1,
                infer_data2,
                w_flow=w1,
                w_diffusion=w2,
                debug=debug,
                composition_space=composition_mode or "native_x0_tail",
                force_composer=kwargs.get("force_composer", False),
            )

        if policy1_type == "diffusion" and policy2_type == "flow":
            return self.flow_diffusion_composer.compose_flow_diffusion(
                infer_data2,
                infer_data1,
                w_flow=w2,
                w_diffusion=w1,
                debug=debug,
                composition_space=composition_mode or "native_x0_tail",
                force_composer=kwargs.get("force_composer", False),
            )

        raise ValueError(
            f"Unsupported released composition: {policy1_type} + {policy2_type}. "
            "This package supports DP+DP3 and DP+pi0.5 only."
        )
