# Checkpoint Manifest

This manifest records the released best checkpoints for both composition
families. Paths are relative to the repository root. Each task has exactly one
released ensemble checkpoint.

## DP + DP3

| Task | Ensemble checkpoint | DP base | DP3 base | Result |
| --- | --- | --- | --- | ---: |
| `beat_block_hammer` | `best_checkpoint/dp+dp3/beat_block_hammer/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_dp3/3000.ckpt` | 98/100 |
| `open_laptop` | `best_checkpoint/dp+dp3/open_laptop/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_dp3/3000.ckpt` | 93/100 |
| `click_alarmclock` | `best_checkpoint/dp+dp3/click_alarmclock/ensemble_checkpoint/best.pt` | `base_dp/100.ckpt` | `base_dp3/3000.ckpt` | 100/100 |
| `move_playingcard_away` | `best_checkpoint/dp+dp3/move_playingcard_away/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_dp3/3000.ckpt` | 89/100 |
| `place_bread_skillet` | `best_checkpoint/dp+dp3/place_bread_skillet/ensemble_checkpoint/best.pt` | `base_dp/100.ckpt` | `base_dp3/100.ckpt` | 57/100 |
| `dump_bin_bigbin` | `best_checkpoint/dp+dp3/dump_bin_bigbin/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_dp3/100.ckpt` | 89/100 |
| `handover_block` | `best_checkpoint/dp+dp3/handover_block/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_dp3/100.ckpt` | 70/100 |
| `stack_bowls_three` | `best_checkpoint/dp+dp3/stack_bowls_three/ensemble_checkpoint/best.pt` | `base_dp/100.ckpt` | `base_dp3/100.ckpt` | 76/100 |

## DP + pi0.5

Mode: `native_x0_tail + PyTorch pi0.5 + 100 eval + expert check enabled`.
`handover_block` uses pi0.5-2000; the other tasks use pi0.5-1000.

| Task | Ensemble checkpoint | DP base | pi0.5 base | Result |
| --- | --- | --- | --- | ---: |
| `beat_block_hammer` | `best_checkpoint/dp+pi0.5/beat_block_hammer/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_pi05_checkpoint_dir/1000` | 74/100 |
| `open_laptop` | `best_checkpoint/dp+pi0.5/open_laptop/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_pi05_checkpoint_dir/1000` | 93/100 |
| `click_alarmclock` | `best_checkpoint/dp+pi0.5/click_alarmclock/ensemble_checkpoint/best.pt` | `base_dp/100.ckpt` | `base_pi05_checkpoint_dir/1000` | 98/100 |
| `move_playingcard_away` | `best_checkpoint/dp+pi0.5/move_playingcard_away/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_pi05_checkpoint_dir/1000` | 89/100 |
| `place_bread_skillet` | `best_checkpoint/dp+pi0.5/place_bread_skillet/ensemble_checkpoint/best.pt` | `base_dp/100.ckpt` | `base_pi05_checkpoint_dir/1000` | 55/100 |
| `dump_bin_bigbin` | `best_checkpoint/dp+pi0.5/dump_bin_bigbin/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_pi05_checkpoint_dir/1000` | 78/100 |
| `handover_block` | `best_checkpoint/dp+pi0.5/handover_block/ensemble_checkpoint/best.pt` | `base_dp/300.ckpt` | `base_pi05_checkpoint_dir/2000` | 39/100 |
| `stack_bowls_three` | `best_checkpoint/dp+pi0.5/stack_bowls_three/ensemble_checkpoint/best.pt` | `base_dp/100.ckpt` | `base_pi05_checkpoint_dir/1000` | 53/100 |

## Release Notes

- `ensemble_checkpoint` files are lightweight and can be committed directly.
- `base_dp`, `base_dp3`, and `base_pi05_checkpoint_dir/model.safetensors`
  are large base policies. Keep them as local archives, Git LFS artifacts, or
  external downloads.
- Raw Hydra outputs and raw evaluation logs are not required
  for evaluation and are intentionally excluded from this release tree.


## Hugging Face Assets

The release checkpoint assets should be uploaded to Hugging Face and arranged to match the `best_checkpoint/` manifest paths. Users can download the assets and place or symlink them into this directory before running evaluation.
