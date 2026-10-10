"""Girish - Slides 6-7 (how results were obtained, results and interpretation).

Consistency checks using ONLY numbers printed in the paper (Schulman et al., 2017):
  1. Setup arithmetic (Sec. 6.1): 7 tasks x 3 seeds = 21 runs.
  2. Table 1: which setting scored highest.
  3. Table 2, row (2): recount the winners from Appendix Table 6 (final scores).
Table 2 row (1) cannot be recounted: Table 6 only reports final-100-episode scores.
This does not reproduce the experiments.
"""
import matplotlib.pyplot as plt

# ---- Table 1 (avg normalized score) ----
TABLE1 = {
    "No clip/penalty": -0.39, "Clip e=0.1": 0.76, "Clip e=0.2": 0.82, "Clip e=0.3": 0.70,
    "AdaptKL 0.003": 0.68, "AdaptKL 0.01": 0.74, "AdaptKL 0.03": 0.71,
    "FixedKL b=0.3": 0.62, "FixedKL b=1": 0.71, "FixedKL b=3": 0.72, "FixedKL b=10": 0.69,
}

# ---- Table 6 (mean final scores): game: (A2C, ACER, PPO) ----
T6 = {
 "Alien": (1141.7,1655.4,1850.3), "Amidar": (380.8,827.6,674.6), "Assault": (1562.9,4653.8,4971.9),
 "Asterix": (3176.3,6801.2,4532.5), "Asteroids": (1653.3,2389.3,2097.5),
 "Atlantis": (729265.3,1841376.0,2311815.0), "BankHeist": (1095.3,1177.5,1280.6),
 "BattleZone": (3080.0,8983.3,17366.7), "BeamRider": (3031.7,3863.3,1590.0),
 "Bowling": (30.1,33.3,40.1), "Boxing": (17.7,98.9,94.6), "Breakout": (303.0,456.4,274.8),
 "Centipede": (3496.5,8904.8,4386.4), "ChopperCommand": (1171.7,5287.7,3516.3),
 "CrazyClimber": (107770.0,132461.0,110202.0), "DemonAttack": (6639.1,38808.3,11378.4),
 "DoubleDunk": (-16.2,-13.2,-14.9), "Enduro": (0.0,0.0,758.3), "FishingDerby": (20.6,34.7,17.8),
 "Freeway": (0.0,0.0,32.5), "Frostbite": (261.8,285.6,314.2), "Gopher": (1500.9,37802.3,2932.9),
 "Gravitar": (194.0,225.3,737.2), "IceHockey": (-6.4,-5.9,-4.2), "Jamesbond": (52.3,261.8,560.7),
 "Kangaroo": (45.3,50.0,9928.7), "Krull": (8367.4,7268.4,7942.3),
 "KungFuMaster": (24900.3,27599.3,23310.3), "MontezumaRevenge": (0.0,0.3,42.0),
 "MsPacman": (1626.9,2718.5,2096.5), "NameThisGame": (5961.2,8488.0,6254.9),
 "Pitfall": (-55.0,-16.9,-32.9), "Pong": (19.7,20.7,20.7), "PrivateEye": (91.3,182.0,69.5),
 "Qbert": (10065.7,15316.6,14293.3), "Riverraid": (7653.5,9125.1,8393.6),
 "RoadRunner": (32810.0,35466.0,25076.0), "Robotank": (2.2,2.5,5.5), "Seaquest": (1714.3,1739.5,1204.5),
 "SpaceInvaders": (744.5,1213.9,942.5), "StarGunner": (26204.0,49817.7,32689.0),
 "Tennis": (-22.2,-17.6,-14.8), "TimePilot": (2898.0,4175.7,4342.0), "Tutankham": (206.8,280.8,254.4),
 "UpNDown": (17369.8,145051.4,95445.0), "Venture": (0.0,0.0,0.0),
 "VideoPinball": (19735.9,156225.6,37389.0), "WizardOfWor": (859.0,2308.3,4185.3),
 "Zaxxon": (16.3,29.0,5008.7),
}
PAPER_TABLE2_ROW2 = {"A2C": 1, "ACER": 28, "PPO": 19, "Tie": 1}

# 1. Setup arithmetic
tasks = ["HalfCheetah", "Hopper", "InvertedDoublePendulum", "InvertedPendulum",
         "Reacher", "Swimmer", "Walker2d"]
assert len(tasks) * 3 == 21
print(f"Setup: {len(tasks)} tasks x 3 seeds = {len(tasks) * 3} runs per setting  [OK]")

# 2. Table 1
best = max(TABLE1, key=TABLE1.get)
print(f"Table 1: best setting = {best} ({TABLE1[best]})")
assert best == "Clip e=0.2" and TABLE1[best] == 0.82

# 3. Table 2 row (2) recount from Table 6
algos = ["A2C", "ACER", "PPO"]
count = {"A2C": 0, "ACER": 0, "PPO": 0, "Tie": 0}
ties = []
for game, scores in T6.items():
    m = max(scores)
    winners = [a for a, s in zip(algos, scores) if s == m]
    if len(winners) > 1:
        count["Tie"] += 1; ties.append(game)
    else:
        count[winners[0]] += 1
assert len(T6) == 49
print(f"\nRecount from Table 6 ({len(T6)} games): {count}")
print(f"Paper's Table 2 row (2):               {PAPER_TABLE2_ROW2}")
print(f"Tied in the rounded Table 6: {ties}")
assert count["PPO"] == 19 and count["A2C"] == 1
assert count["ACER"] + count["Tie"] == PAPER_TABLE2_ROW2["ACER"] + PAPER_TABLE2_ROW2["Tie"]
print("PPO and A2C counts match exactly. The ACER/Tie split differs by one game "
      "because Table 6 is rounded (e.g. Pong shows 20.7 for both ACER and PPO).")

# Chart of Table 1
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(TABLE1)), list(TABLE1.values()),
       color=["tab:red" if v < 0 else "tab:green" if k == best else "tab:blue" for k, v in TABLE1.items()])
ax.set_xticks(range(len(TABLE1))); ax.set_xticklabels(TABLE1.keys(), rotation=40, ha="right")
ax.set_ylabel("avg normalized score"); ax.set_title("Table 1 (Schulman et al., 2017)")
plt.tight_layout(); plt.savefig("girish_table1_chart.png", dpi=160)
print("Saved girish_table1_chart.png")
