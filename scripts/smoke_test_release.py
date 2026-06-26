from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
EXTERNAL_ARTIFACT_DIRS = {"base_dp", "base_dp3", "base_pi05_checkpoint_dir", "hydra", "eval_log"}
PRIVATE_PATTERNS = re.compile(
    r"/data[0-9]+/|/home/[^/]+|/Users/[^/]+",
    re.IGNORECASE,
)

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "script/eval_policy.py",
    "policy/DP/deploy_policy.py",
    "policy/DP/dp_model.py",
    "policy/DP/eval.sh",
    "policy/DP3/deploy_policy.py",
    "policy/DP3/eval.sh",
    "policy/DP3/3D-Diffusion-Policy/dp3_policy.py",
    "policy/Ensemble-Policy-easy/composition.py",
    "policy/Ensemble-Policy-easy/policy_loader.py",
    "policy/Ensemble-Policy-easy/energy_head.py",
    "policy/Ensemble-Policy-easy/eval.py",
    "policy/Ensemble-Policy-easy/eval_wlearn.py",
    "policy/Ensemble-Policy-easy/eval_wlearn.sh",
    "task_config/_camera_config.yml",
    "task_config/_embodiment_config.yml",
    "docs/checkpoints.md",
]

ENSEMBLE_CHECKPOINTS = [
    "best_checkpoint/dp+dp3/beat_block_hammer/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+dp3/open_laptop/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+dp3/click_alarmclock/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+dp3/move_playingcard_away/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+dp3/place_bread_skillet/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+dp3/dump_bin_bigbin/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+dp3/handover_block/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+dp3/stack_bowls_three/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/beat_block_hammer/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/open_laptop/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/click_alarmclock/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/move_playingcard_away/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/place_bread_skillet/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/dump_bin_bigbin/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/handover_block/ensemble_checkpoint/best.pt",
    "best_checkpoint/dp+pi0.5/stack_bowls_three/ensemble_checkpoint/best.pt",
]

FORBIDDEN_NAMES = {
    "train.py",
    "train.sh",
    "train_dp3.py",
    "train_rgb.sh",
    "process_data.py",
    "process_data.sh",
    "workspace.py",
    "workspace_wlearn.py",
    "dataset.py",
    "trainer.py",
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def release_files() -> list[Path]:
    return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def is_external_artifact(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    return any(part in EXTERNAL_ARTIFACT_DIRS for part in rel_parts)


def check_required() -> None:
    missing = [p for p in REQUIRED_FILES + ENSEMBLE_CHECKPOINTS if not (ROOT / p).is_file()]
    if missing:
        fail("Missing required files:\n" + "\n".join(missing))


def check_no_training_files() -> None:
    found = [str(p.relative_to(ROOT)) for p in release_files() if p.name in FORBIDDEN_NAMES]
    if found:
        fail("Training or data-processing files found:\n" + "\n".join(found))


def check_file_sizes() -> None:
    oversized = [
        f"{p.relative_to(ROOT)} ({p.lstat().st_size} bytes)"
        for p in release_files()
        if not p.is_symlink()
        and not is_external_artifact(p)
        and p.lstat().st_size > MAX_GITHUB_FILE_BYTES
    ]
    if oversized:
        fail("Files exceed GitHub's 100MB limit:\n" + "\n".join(oversized))


def check_private_markers() -> None:
    hits = []
    for p in release_files():
        if p == ROOT / "scripts/smoke_test_release.py" or is_external_artifact(p):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PRIVATE_PATTERNS.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{lineno}")
                break
    if hits:
        fail("Private path or account markers found:\n" + "\n".join(hits[:80]))


def check_shell_syntax() -> None:
    scripts = [
        "policy/DP/eval.sh",
        "policy/DP3/eval.sh",
        "policy/Ensemble-Policy-easy/eval_wlearn.sh",
        "scripts/download_dp_dp3_checkpoints.sh",
    ]
    for script in scripts:
        if (ROOT / script).exists():
            subprocess.check_call(["bash", "-n", str(ROOT / script)])


def main() -> None:
    check_required()
    check_no_training_files()
    check_file_sizes()
    check_private_markers()
    check_shell_syntax()
    print("[OK] Eval release smoke test passed.")


if __name__ == "__main__":
    main()
