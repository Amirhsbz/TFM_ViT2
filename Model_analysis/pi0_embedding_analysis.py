import argparse
import json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Inspect tactile embedding stored at the end of observation.state.")
parser.add_argument(
    "--dataset-root",
    type=Path,
    default=Path("outputs/turn_cleanser_water_bottle_gated_tactile_lerobot_tactile_emb_two_prompt"),
    help="LeRobot dataset root containing conversion_report.json, meta/, and data/.",
)
args = parser.parse_args()

root = args.dataset_root
report = json.loads((root / "conversion_report.json").read_text())
tactile_shape = report.get("tactile_shape")
print("tactile_shape from report:", tactile_shape)

if not tactile_shape:
    raise RuntimeError("conversion_report.json does not contain tactile_shape")

D = int(tactile_shape[-1])


def load_observation_state(dataset_root: Path) -> np.ndarray:
    """Load observation.state from a LeRobot v2 dataset."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Need LeRobot to read this dataset. Run this script in the OpenPI/LeRobot environment."
        ) from exc

    dataset = LeRobotDataset(repo_id=str(dataset_root), root=dataset_root)
    states = []
    for i in range(len(dataset)):
        state = dataset[i]["observation.state"]
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        states.append(np.asarray(state, dtype=np.float32).reshape(-1))
    return np.stack(states, axis=0)


states = load_observation_state(root)
print("observation.state shape:", states.shape)

if states.shape[1] < D:
    raise RuntimeError(f"state dim {states.shape[1]} is smaller than tactile dim {D}")

robot_state = states[:, :-D]
tactile_emb = states[:, -D:]

print("robot_state shape:", robot_state.shape)
print("tactile_emb shape:", tactile_emb.shape)

std = tactile_emb.std(axis=0)
mean = tactile_emb.mean(axis=0)
norm = np.linalg.norm(tactile_emb, axis=1)

centered = tactile_emb - tactile_emb.mean(axis=0, keepdims=True)
_, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
energy = singular_values**2
energy_ratio = energy / max(energy.sum(), 1e-12)
effective_rank_95 = int(np.searchsorted(np.cumsum(energy_ratio), 0.95) + 1)

delta = np.linalg.norm(np.diff(tactile_emb, axis=0), axis=1)

sample_count = min(len(tactile_emb), 2000)
sample = tactile_emb[:sample_count]
sample_norm = np.linalg.norm(sample, axis=1, keepdims=True)
sample_unit = sample / np.maximum(sample_norm, 1e-8)
cosine = sample_unit @ sample_unit.T
upper = cosine[np.triu_indices_from(cosine, k=1)]

print("\n=== tactile embedding stats ===")
print("D:", D)
print("mean abs:", float(np.mean(np.abs(mean))))
print("std mean:", float(std.mean()))
print("std min/max:", float(std.min()), float(std.max()))
print("near-zero std dims:", int((std < 1e-4).sum()), "/", D)
print("norm mean/std/min/max:", float(norm.mean()), float(norm.std()), float(norm.min()), float(norm.max()))
print("effective rank 95%:", effective_rank_95, "/", D)
print("top energy ratios:", np.round(energy_ratio[: min(8, D)], 4).tolist())
print("frame delta mean/median/max:", float(delta.mean()), float(np.median(delta)), float(delta.max()))
print("pairwise cosine mean/median/p95:", float(upper.mean()), float(np.median(upper)), float(np.quantile(upper, 0.95)))

print("\n=== quick interpretation ===")
if (std < 1e-4).sum() > 0:
    print("- Some tactile dimensions are almost constant; check for collapsed or unused dimensions.")
if effective_rank_95 <= max(2, D // 4):
    print("- Effective rank is low; this embedding may be over-compressed or too correlated.")
if np.quantile(upper, 0.95) > 0.98:
    print("- Many frames have very similar embeddings; tactile variation may be weak after compression.")
if delta.mean() < 1e-3:
    print("- Adjacent-frame changes are tiny; check whether gated tactile is mostly baseline frames.")
