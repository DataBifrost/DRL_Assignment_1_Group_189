"""Avishek - Slides 8-9 (limitations, future scope, conclusion, references).

Supports the limitation 'epsilon is a hyperparameter' (Sec. 3, Sec. 6.1, Table 1):
plots L^CLIP for the three epsilon values compared in Table 1 and prints how the
clip window and the paper's score change. Not a reproduction of the experiments.
"""
import numpy as np
import matplotlib.pyplot as plt

TABLE1_CLIP = {0.1: 0.76, 0.2: 0.82, 0.3: 0.70}   # avg normalized score, Table 1
r = np.linspace(0.4, 1.6, 300)


def term(ratio, adv, eps):
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)


print("epsilon  clip window    max term (A=+1)  min term (A=-1 clipped side)  Table 1 score")
for eps, score in TABLE1_CLIP.items():
    cap = float(term(np.array(2.0), 1.0, eps))      # plateau for A>0 as r grows
    floor_ = float(term(np.array(0.0), -1.0, eps))  # plateau for A<0 as r shrinks
    assert np.isclose(cap, 1 + eps) and np.isclose(floor_, -(1 - eps))
    print(f"{eps:7.1f}  [{1-eps:.1f}, {1+eps:.1f}]   {cap:15.2f}  {floor_:28.2f}  {score:13.2f}")
best = max(TABLE1_CLIP, key=TABLE1_CLIP.get)
assert best == 0.2
print(f"\nBest of the three in Table 1: epsilon = {best} -> the setting is tuned, not universal.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for eps in TABLE1_CLIP:
    ax[0].plot(r, term(r, 1.0, eps), label=f"eps={eps}")
    ax[1].plot(r, term(r, -1.0, eps), label=f"eps={eps}")
ax[0].set_title("A > 0"); ax[1].set_title("A < 0")
for a in ax:
    a.set_xlabel("probability ratio r"); a.legend()
ax[0].set_ylabel("L^CLIP term")
plt.tight_layout(); plt.savefig("avishek_epsilon_sensitivity.png", dpi=160)
print("Saved avishek_epsilon_sensitivity.png")
