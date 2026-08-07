import numpy as np

path = r"D:\Final_code\model_data\static_expert_2026-05-04_18-56-59\static_expert_dataset.npz"
data = np.load(path, allow_pickle=True)

states = data["states"]
actions = data["actions"]
dones = data["dones"]
episode_ids = data["episode_ids"]
step_ids = data["step_ids"]

v_max = states[:, 7]
dist = np.linalg.norm(states[:, :2], axis=1)

episodes = np.unique(episode_ids)

summary = []
for ep in episodes:
    idx = np.where(episode_ids == ep)[0]
    last = idx[-1]

    ep_vmax = states[idx[0], 7]
    ep_steps = len(idx)
    final_dist = dist[last]
    success = bool(dones[last]) or final_dist < 5.0

    dv_sat = np.mean(np.isclose(np.abs(actions[idx, 0]), 0.025, atol=1e-4))
    dtheta_sat = np.mean(np.isclose(np.abs(actions[idx, 1]), np.pi / 8, atol=1e-4))

    summary.append([ep_vmax, success, ep_steps, final_dist, dv_sat, dtheta_sat])

summary = np.asarray(summary, dtype=float)

print("Total episodes:", len(summary))
print("Success rate:", summary[:, 1].mean())
print("Mean steps:", summary[:, 2].mean())
print("Mean final distance:", summary[:, 3].mean())
print("Mean delta_v saturation:", summary[:, 4].mean())
print("Mean delta_theta saturation:", summary[:, 5].mean())

bins = np.linspace(5 * np.pi / 180, 16 * np.pi / 180, 6)
print("\nBy v_max bins:")
for low, high in zip(bins[:-1], bins[1:]):
    mask = (summary[:, 0] >= low) & (summary[:, 0] < high)
    if mask.sum() == 0:
        continue

    print(
        f"{low * 180 / np.pi:.1f}-{high * 180 / np.pi:.1f} deg: "
        f"N={mask.sum()}, "
        f"success={summary[mask, 1].mean():.3f}, "
        f"steps={summary[mask, 2].mean():.1f}, "
        f"final_dist={summary[mask, 3].mean():.2f}, "
        f"dv_sat={summary[mask, 4].mean():.3f}, "
        f"dtheta_sat={summary[mask, 5].mean():.3f}"
    )
