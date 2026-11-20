# Data

Five CSV files containing the data behind every quantitative figure in the paper.
All metrics are means over the 100 parallel agents per rollout, time-averaged
over the episode (success rate = % of agents that did not fall).

## Figure → file map

| Figure | File |
|--------|------|
| Main Fig 3 (optimal-gait checkerboard) | `gait_metrics.csv` |
| Main Fig 4b (gait selection on parkour) | `parkour_trajectory_fig4b.csv` |
| Main Fig 4c,d (gait usage % + metric bars) | `stage2_parkour_gait_usage.csv` |
| Supplementary Fig 2 (gait-tracking accuracy) | `gait_tracking_accuracy.csv` |
| Supplementary Figs 3–11 (per-gait metric line plots) | `gait_metrics.csv` |
| Supplementary Fig 12 (various Stage 2 policies gait usage) | `supp_fig12_gait_usage.csv` |


## `gait_metrics.csv` — one row per (gait, commanded velocity, terrain, difficulty)

11 gaits (0–10) + a no-coupling baseline (gait 11); 4 commanded velocities; 7 terrains; 5 difficulty levels.

| Column | Meaning |
|--------|---------|
| `gait`, `gait_name` | gait index 0–11 and name (11 = baseline, no coupling) |
| `commanded_velocity_mps` | commanded forward velocity (0.5 / 1.0 / 1.5 / 2.0) (m/s) |
| `terrain`, `terrain_name` | terrain index 1–7 and name |
| `terrain_difficulty` | 1 (easiest) – 5 (hardest) |
| `Success_Rate` | % of agents that did not fall (success rate) |
| `COT` | cost of transport |
| `Lin_Vel_Vx` | achieved mean forward velocity (m/s) — compare to the command |
| `Mean_Angular_Velocity` | mean body angular velocity (rad/s) |
| `Mean_Joint_Acceleration` | mean abs joint acceleration (rad/s²) |
| `Mean_CPG_Amp` | mean CPG oscillator amplitude (≈ step length) |
| `Mean_CPG_freq` | mean CPG oscillator frequency (Hz, ≈ step frequency) |
| `Mean_Body_Height` | mean base height (m) |
| `Mean_Ground_Clearance` | mean swing-foot ground clearance (m) |
| `Mean_Xoff` | mean fore–aft foot offset (m) |

## `parkour_trajectory_fig4b.csv` — Fig 4b one row per control step of one parkour run

The Stage 2 power-minimizing policy at 1.5 m/s traversing the 30 m parkour.
Plotted against `x_position` (not time).

| Column | Meaning |
|--------|---------|
| `x_position` | forward position on the course (m) |
| `gait_num` | gait selected by the Stage 2 policy at that step |
| `body_height`, `ground_clearance`, `foot_x_offset` | style parameters (m) |

## `stage2_parkour_gait_usage.csv` — Fig 4c,d (3 Stage 2 policies)

One row per (policy, commanded velocity). Stage 2 policies minimizing respectively: Power, Angular Velocity,
Joint Acceleration.

| Column | Meaning |
|--------|---------|
| `reward_objective` | which quantity the policy minimizes |
| `commanded_velocity_mps` | 0.5 / 1.0 / 1.5 / 2.0  (m/s)|
| `COT`, `Mean_Angular_Velocity`, `Mean_Joint_Acceleration` | metric-bar values (Fig 4d) |
| `usage_pct_<gait>` | % of the course spent in each gait (Fig 4c) |

## `gait_tracking_accuracy.csv` — Supplementary Fig 2 (gait-tracking accuracy)

One row per (gait, commanded velocity, terrain, difficulty): 11 gaits (0–10), 4 commanded
velocities, 7 terrains, 5 difficulty levels. Gait-tracking accuracy is the fraction 
of commanded stance time during which the corresponding foot is actually in terrain contact.

| Column | Meaning |
|--------|---------|
| `gait`, `gait_name` | gait index 0–10 and name |
| `commanded_velocity_mps` | commanded forward velocity (0.5 / 1.0 / 1.5 / 2.0) (m/s) |
| `terrain`, `terrain_name` | terrain index 1–7 and name |
| `terrain_difficulty` | 1 (easiest) – 5 (hardest) |
| `realization_pct` | gait-tracking accuracy (%); panel a = mean over difficulties, panel b = per difficulty |
| `realization_robot_std` | std across the 100 robots (±1σ band in panel b) |

## `supp_fig12_gait_usage.csv` — Supplementary Fig 12 (various Stage 2 policies gait usage)

One row per (reward objective, commanded velocity); `usage_pct_<gait>` = % of the
parkour spent in each gait. `(+)` objectives maximize rather than minimize the
quantity. Gait-usage percentages only.
