"""Disha - Slides 3-5: verification of PPO Equation (7), Schulman et al. (2017).

Checks the per-timestep clipped surrogate  min(r*A, clip(r, 1-eps, 1+eps)*A)
against hand-computed values and two properties stated in Sec. 3.
Not a reproduction of the paper's MuJoCo or Atari experiments.
"""
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.2


def clipped_surrogate(ratio, advantage, epsilon=EPSILON):
    clipped_ratio = float(np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon))
    raw_term = ratio * advantage
    clipped_term = clipped_ratio * advantage
    objective = min(raw_term, clipped_term)          # the min(...) in Eq. (7)
    return clipped_ratio, raw_term, clipped_term, objective


# (ratio, advantage, hand-computed objective)
CASES = [
    (0.6, 1.0, 0.6), (0.9, 1.0, 0.9), (1.0, 1.0, 1.0), (1.2, 1.0, 1.2), (1.4, 1.0, 1.2),
    (0.6, -1.0, -0.8), (0.8, -1.0, -0.8), (1.0, -1.0, -1.0), (1.4, -1.0, -1.4),
]


def run_checks():
    print(f"PPO Equation (7) check, epsilon = {EPSILON}")
    print("ratio   adv  clipped_r  raw_term  clipped_term  objective  expected")
    for ratio, adv, expected in CASES:
        cr, raw, clp, obj = clipped_surrogate(ratio, adv)
        print(f"{ratio:5.1f} {adv:5.1f} {cr:9.1f} {raw:9.2f} {clp:13.2f} {obj:10.2f} {expected:9.2f}")
        assert np.isclose(obj, expected), (ratio, adv, obj, expected)
        assert obj <= raw + 1e-12                    # lower bound on r*A (Sec. 3)
    for adv in (1.0, -1.0):                          # equals r*A at r = 1 (Sec. 3)
        assert np.isclose(clipped_surrogate(1.0, adv)[-1], adv)
    # No incentive to push r past the clip edge in the "good" direction:
    assert np.isclose(clipped_surrogate(1.4, 1.0)[-1], clipped_surrogate(1.2, 1.0)[-1])
    assert np.isclose(clipped_surrogate(0.6, -1.0)[-1], clipped_surrogate(0.8, -1.0)[-1])
    print("\nAll checks passed.")


def make_plot(path="ppo_eq7_plot.png"):
    r = np.linspace(0.4, 1.6, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (adv, label) in zip(axes, [(1.0, "A > 0"), (-1.0, "A < 0")]):
        ax.plot(r, [clipped_surrogate(x, adv)[-1] for x in r], label="L^CLIP term")
        ax.plot(r, r * adv, ls=":", color="grey", label="unclipped r*A")
        ax.axvline(1 - EPSILON, ls="--", lw=1); ax.axvline(1 + EPSILON, ls="--", lw=1)
        ax.set_xlabel("probability ratio r"); ax.set_title(f"PPO Eq. (7), {label}"); ax.legend()
    axes[0].set_ylabel("surrogate term")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    run_checks()
    make_plot()
