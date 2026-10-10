# DRL Assignment I – Group 189 – PPO proof of work

Paper: Schulman, Wolski, Dhariwal, Radford, Klimov (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347

Small, honest checks. None of these reproduces the paper's MuJoCo or Atari experiments.

| Contributor | Slides | Script | What it shows |
|---|---|---|---|
| Omang Agarwal | 1–2 | `omang_motivation_demo.py` | Toy policy: unclipped vs clipped objective over many epochs on one batch |
| Disha Agarwal | 3–5 | `ppo_eq7_verification.py` | Eq. (7) checked against hand-computed values; plot like Fig. 1 |
| Girish N | 6–7 | `girish_table_checks.py` | Setup arithmetic, Table 1 best setting, Table 2 row 2 recounted from Table 6 |
| Avishek Gulshan | 8–9 | `avishek_epsilon_sensitivity.py` | Clip window for ε = 0.1/0.2/0.3 next to Table 1 scores |

Run: `pip install numpy matplotlib` then `python <script>.py` (each saves a PNG next to the script).

## References
- Schulman et al. (2017), arXiv:1707.06347 — Sec. 2.1, 2.2, 3, 5, 6.1, 6.2, 6.4, 7; Tables 1, 2, 6; Figure 1.
- Cited in the paper (Sec. 6.2, footnote 3) as a baseline implementation only: github.com/berkeleydeeprlcourse/homework (hw4). Not our code.
# DRL_Assignment_1_Group_189
