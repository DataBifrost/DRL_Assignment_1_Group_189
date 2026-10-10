"""Omang - Slides 1-2 (objective and motivation).

Toy illustration of the problem PPO addresses (Schulman et al., 2017, Sec. 1, 2.1, 3):
reusing ONE batch of data for many gradient steps. We compare
  - unclipped surrogate  L^CPI = r * A            (Eq. 6)
  - clipped surrogate    L^CLIP                    (Eq. 7)
on a tiny 3-action softmax policy with a fixed batch and fixed advantages.

This is an illustration of the paper's claim, NOT a reproduction of its experiments.
"""
import numpy as np
import matplotlib.pyplot as plt

EPS, K, LR, N = 0.2, 100, 0.5, 300
rng = np.random.default_rng(0)

old_logits = np.zeros(3)                       # old policy: uniform over 3 actions
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()
old_p = softmax(old_logits)

# One fixed batch collected with the OLD policy; advantage = true value + noise.
actions = rng.choice(3, size=N, p=old_p)
adv = np.array([1.0, 0.0, -1.0])[actions] + rng.normal(0, 0.3, size=N)
onehot = np.eye(3)[actions]


def run(clipped: bool):
    theta = old_logits.copy()
    kl_hist, ratio_hist = [], []
    for _ in range(K):                         # K epochs on the SAME batch
        p = softmax(theta)
        r = p[actions] / old_p[actions]
        raw = r * adv
        clip = np.clip(r, 1 - EPS, 1 + EPS) * adv
        active = (raw <= clip) if clipped else np.ones(N, bool)  # min() picks raw term
        grad = ((r * adv * active)[:, None] * (onehot - p)).mean(axis=0)
        theta += LR * grad                     # gradient ascent
        p = softmax(theta)
        kl_hist.append(float(np.sum(old_p * np.log(old_p / p))))
        ratio_hist.append(float(p[0] / old_p[0]))   # action 0 has the best advantage
    return kl_hist, ratio_hist


kl_u, r_u = run(clipped=False)
kl_c, r_c = run(clipped=True)
print(f"After {K} epochs on one batch (epsilon = {EPS}):")
print(f"  unclipped : KL(old||new) = {kl_u[-1]:.3f}, ratio of best action = {r_u[-1]:.2f}")
print(f"  clipped   : KL(old||new) = {kl_c[-1]:.3f}, ratio of best action = {r_c[-1]:.2f}")
assert kl_c[-1] < kl_u[-1], "clipping should keep the policy closer to the old one"
print("Check passed: clipping keeps the new policy closer to the old policy.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(kl_u, label="unclipped  r*A (Eq. 6)")
ax[0].plot(kl_c, label="clipped  L^CLIP (Eq. 7)")
ax[0].set_xlabel("epoch on the same batch"); ax[0].set_ylabel("KL(old || new)")
ax[0].set_title("Policy drift"); ax[0].legend()
ax[1].plot(r_u, label="unclipped"); ax[1].plot(r_c, label="clipped")
ax[1].axhline(1 + EPS, ls="--", lw=1, color="grey")
ax[1].set_xlabel("epoch on the same batch"); ax[1].set_ylabel("ratio r of best action")
ax[1].set_title("Probability ratio"); ax[1].legend()
plt.tight_layout(); plt.savefig("omang_motivation_demo.png", dpi=160)
print("Saved omang_motivation_demo.png")
