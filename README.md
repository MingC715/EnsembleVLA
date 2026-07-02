<div align="center">

# EnsembleVLA: Ensemble Learning for Vision-Language Action Models (ICML 2026)

<div>
Mingchen&nbsp;Song<sup>1 2</sup>,
<a target="_blank" href="https://homepage.hit.edu.cn/dengxiang">Xiang&nbsp;Deng</a><sup>1 3</sup>,
Jie&nbsp;Wei<sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?hl=en&user=Awsue7sAAAAJ">Dongmei&nbsp;Jiang</a><sup>2</sup>,
<a target="_blank" href="https://scholar.google.com/citations?hl=en&user=yywVMhUAAAAJ">Liqiang&nbsp;Nie</a><sup>1</sup>,
<a target="_blank" href="https://ieeexplore.ieee.org/author/37087008154">Weili&nbsp;Guan</a><sup>1 3</sup>
</div>
<sup>1</sup>Harbin Institute of Technology, Shenzhen&nbsp;&nbsp;
<sup>2</sup>PengCheng Laboratory&nbsp;&nbsp;
<sup>3</sup>Shenzhen Loop Area Institute
<br>
<br><br>
<a href="#"><img src="https://img.shields.io/badge/Paper-coming_soon-deepgreen" alt="Paper"></a>
<a href="https://github.com/MingC715/EnsembleVLA" target="_blank"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub"></a>
<a href="https://huggingface.co/mingchens/EnsembleVLA" target="_blank"><img src="https://img.shields.io/badge/Checkpoints-HuggingFace-yellow" alt="Hugging Face"></a>

</div>

## 📖 Introduction

Recent Vision-Language-Action (VLA) models have demonstrated strong capabilities
in robotic manipulation, yet how to effectively ensemble heterogeneous VLAs
remains largely underexplored. Unlike discriminative predictors, generative action
policies often represent high-dimensional and multimodal action distributions,
which makes conventional ensemble strategies such as voting or direct action
averaging insufficient for robust policy composition.

We propose **EnsembleVLA**, an energy-based framework for principled composition
of diverse VLA policies. EnsembleVLA formulates diffusion-based and flow-based
VLA models under a unified energy perspective, where additive energy aggregation
naturally induces policy composition at the distribution level. This formulation
allows multiple pretrained policies to remain frozen while being aggregated into
a stronger ensemble policy that can exploit their complementary action modes.

Building on this compositional view, EnsembleVLA introduces learnable composition
weights for dynamic policy balancing, together with a confidence-aware gating
mechanism that modulates bounded residual corrections. The resulting policy
maintains compatibility with the RoboTwin2 rollout interface: each base policy is
queried under its own observation and language context, and the composed action is
executed in the simulator for stable task completion.

<div align="center">
<img src="docs/images/ensemblevla_overview.png" width="100%" alt="EnsembleVLA overview">
</div>

## ⚙️ Environment Setup

This project should be installed on top of a working RoboTwin2 environment. Please
first follow the official RoboTwin2 documentation for installation, asset download,
configuration files, and policy evaluation:

- RoboTwin2 documentation: https://robotwin-platform.github.io/doc/index.html


A typical setup is:

```bash
git clone https://github.com/MingC715/EnsembleVLA.git
cd EnsembleVLA

conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

mkdir -p external
git clone https://github.com/RoboTwin-Platform/RoboTwin.git external/RoboTwin
cd external/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
python script/update_embodiment_config_path.py
cd ../..
```

After RoboTwin2 is installed, make the assets visible to this repository and check
that the task, camera, and embodiment configuration files point to your local
installation:

```bash
ln -s /path/to/RoboTwin/assets assets
```

Recommended runtime variables for headless GPU evaluation are:

```bash
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export SAPIEN_OFFSCREEN_ONLY=1
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}
```

The base policy backends should be available under `policy/`:

```text
policy/DP/       # Diffusion Policy backend
policy/DP3/      # 3D Diffusion Policy backend
policy/pi05/     # pi0.5 / openpi backend
```

Then install the EnsembleVLA composition/evaluation dependencies and the base
policy backends you plan to use:

```bash
# EnsembleVLA composition & evaluation dependencies
pip install -r policy/Ensemble-Policy-easy/requirements.txt

# Base policy backends (install the ones you evaluate):
pip install -e policy/DP                        # Diffusion Policy (DP)
pip install -e policy/DP3/3D-Diffusion-Policy   # 3D Diffusion Policy (DP3)
pip install -e policy/pi05                       # pi0.5 / openpi
```

For backend-specific setup, also follow the RoboTwin2 policy pages and the
corresponding upstream repositories.

## 📦 Checkpoints

We release both the lightweight EnsembleVLA heads and the required base policy
checkpoints on Hugging Face:
[mingchens/EnsembleVLA](https://huggingface.co/mingchens/EnsembleVLA).
After downloading the release assets, place or symlink them under
`best_checkpoint/` using the layout below.

Download with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/mingchens/EnsembleVLA hf_assets
rsync -a hf_assets/best_checkpoint/ best_checkpoint/
```

Expected best-checkpoint layout:

```text
best_checkpoint/
+-- dp+dp3/<task>/
|   +-- ensemble_checkpoint/best.pt
|   +-- base_dp/<ckpt>.ckpt
|   +-- base_dp3/<ckpt>.ckpt
+-- dp+pi0.5/<task>/
    +-- ensemble_checkpoint/best.pt
    +-- base_dp/<ckpt>.ckpt
    +-- base_pi05_checkpoint_dir/
        +-- model.safetensors
        +-- metadata.pt
        +-- assets/<task>/norm_stats.json
```

Only inference checkpoints are required for evaluation. Raw optimizer states,
training logs, and rollout logs are not required.

### DP + DP3 Checkpoints

| Task | Ensemble policy checkpoint | Base DP checkpoint | Base DP3 checkpoint |
| --- | --- | --- | --- |
| `beat_block_hammer` | [`best_checkpoint/dp+dp3/beat_block_hammer/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/beat_block_hammer/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/beat_block_hammer/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/beat_block_hammer/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/beat_block_hammer/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/beat_block_hammer/base_dp3/base_dp3.ckpt) |
| `open_laptop` | [`best_checkpoint/dp+dp3/open_laptop/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/open_laptop/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/open_laptop/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/open_laptop/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/open_laptop/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/open_laptop/base_dp3/base_dp3.ckpt) |
| `click_alarmclock` | [`best_checkpoint/dp+dp3/click_alarmclock/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/click_alarmclock/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/click_alarmclock/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/click_alarmclock/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/click_alarmclock/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/click_alarmclock/base_dp3/base_dp3.ckpt) |
| `move_playingcard_away` | [`best_checkpoint/dp+dp3/move_playingcard_away/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/move_playingcard_away/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/move_playingcard_away/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/move_playingcard_away/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/move_playingcard_away/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/move_playingcard_away/base_dp3/base_dp3.ckpt) |
| `place_bread_skillet` | [`best_checkpoint/dp+dp3/place_bread_skillet/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/place_bread_skillet/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/place_bread_skillet/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/place_bread_skillet/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/place_bread_skillet/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/place_bread_skillet/base_dp3/base_dp3.ckpt) |
| `dump_bin_bigbin` | [`best_checkpoint/dp+dp3/dump_bin_bigbin/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/dump_bin_bigbin/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/dump_bin_bigbin/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/dump_bin_bigbin/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/dump_bin_bigbin/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/dump_bin_bigbin/base_dp3/base_dp3.ckpt) |
| `handover_block` | [`best_checkpoint/dp+dp3/handover_block/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/handover_block/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/handover_block/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/handover_block/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/handover_block/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/handover_block/base_dp3/base_dp3.ckpt) |
| `stack_bowls_three` | [`best_checkpoint/dp+dp3/stack_bowls_three/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/stack_bowls_three/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+dp3/stack_bowls_three/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/stack_bowls_three/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+dp3/stack_bowls_three/base_dp3/base_dp3.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+dp3/stack_bowls_three/base_dp3/base_dp3.ckpt) |

### DP + pi0.5 Checkpoints

| Task | Ensemble policy checkpoint | Base DP checkpoint | Base pi0.5 checkpoint |
| --- | --- | --- | --- |
| `beat_block_hammer` | [`best_checkpoint/dp+pi0.5/beat_block_hammer/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/beat_block_hammer/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/beat_block_hammer/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/beat_block_hammer/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/beat_block_hammer/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/beat_block_hammer/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |
| `open_laptop` | [`best_checkpoint/dp+pi0.5/open_laptop/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/open_laptop/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/open_laptop/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/open_laptop/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/open_laptop/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/open_laptop/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |
| `click_alarmclock` | [`best_checkpoint/dp+pi0.5/click_alarmclock/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/click_alarmclock/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/click_alarmclock/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/click_alarmclock/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/click_alarmclock/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/click_alarmclock/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |
| `move_playingcard_away` | [`best_checkpoint/dp+pi0.5/move_playingcard_away/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/move_playingcard_away/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/move_playingcard_away/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/move_playingcard_away/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/move_playingcard_away/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/move_playingcard_away/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |
| `place_bread_skillet` | [`best_checkpoint/dp+pi0.5/place_bread_skillet/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/place_bread_skillet/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/place_bread_skillet/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/place_bread_skillet/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/place_bread_skillet/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/place_bread_skillet/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |
| `dump_bin_bigbin` | [`best_checkpoint/dp+pi0.5/dump_bin_bigbin/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/dump_bin_bigbin/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/dump_bin_bigbin/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/dump_bin_bigbin/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/dump_bin_bigbin/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/dump_bin_bigbin/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |
| `handover_block` | [`best_checkpoint/dp+pi0.5/handover_block/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/handover_block/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/handover_block/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/handover_block/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/handover_block/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/handover_block/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |
| `stack_bowls_three` | [`best_checkpoint/dp+pi0.5/stack_bowls_three/ensemble_checkpoint/best.pt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/stack_bowls_three/ensemble_checkpoint/best.pt) | [`best_checkpoint/dp+pi0.5/stack_bowls_three/base_dp/base_dp.ckpt`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/stack_bowls_three/base_dp/base_dp.ckpt) | [`best_checkpoint/dp+pi0.5/stack_bowls_three/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors`](https://huggingface.co/mingchens/EnsembleVLA/blob/main/dp+pi0.5/stack_bowls_three/base_pi05_checkpoint_dir/base_pi05_checkpoint_dir.safetensors) |

Each pi0.5 base also needs `metadata.pt` and the `assets/` folder from the same `base_pi05_checkpoint_dir/` (both on Hugging Face).

See `docs/checkpoints.md` for the full checkpoint manifest.

## 🧪 Evaluation

DP + DP3 example:

```bash
conda activate RoboTwin
bash policy/Ensemble-Policy-easy/eval_wlearn.sh \
  beat_block_hammer DP DP3 0 best L515 100 0 100 base base \
  best_checkpoint/dp+dp3/beat_block_hammer/ensemble_checkpoint
```

DP + pi0.5 example:

```bash
conda activate RoboTwin
bash policy/Ensemble-Policy-easy/eval_wlearn.sh \
  click_alarmclock DP pi05 0 best L515 100 0 100 100 1000 \
  best_checkpoint/dp+pi0.5/click_alarmclock/ensemble_checkpoint
```

Arguments:

```text
task policy1 policy2 gpu ensemble_ckpt camera data_num seed test_num policy1_ckpt policy2_ckpt output_dir [composition_mode] [policy2_use_pytorch]
```

For DP + pi0.5 evaluation, pass the released composition mode argument used by
`eval_wlearn.sh` and set `policy2_use_pytorch=true`.

## 🗂️ Project Structure

```text
EnsembleVLA-ICML2026/
+-- README.md
+-- docs/
|   +-- checkpoints.md
+-- description/
+-- envs/
+-- task_config/
+-- script/
|   +-- eval_policy.py
+-- policy/
|   +-- DP/
|   +-- DP3/
|   +-- pi05/
|   +-- Ensemble-Policy-easy/
|       +-- composition.py
|       +-- energy_head.py
|       +-- eval.py
|       +-- eval_wlearn.py
|       +-- eval_wlearn.sh
+-- best_checkpoint/
    +-- dp+dp3/
    +-- dp+pi0.5/
```

## 🙏 Acknowledgement

We thank the open-source projects [RoboTwin2](https://robotwin-platform.github.io/doc/index.html)
and [GPC](https://github.com/SageCao1125/GPC) for making their code and tools publicly available.

## 📚 Citation

```bibtex
@inproceedings{song2026ensemblevla,
  title={EnsembleVLA: Ensemble Learning for Vision-Language Action Models},
  author={Song, Mingchen and Deng, Xiang and Wei, Jie and Jiang, Dongmei and Nie, Liqiang and Guan, Weili},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
