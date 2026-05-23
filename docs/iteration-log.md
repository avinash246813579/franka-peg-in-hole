# Iteration Log: Sim-Based Manipulation Pipeline

This is the engineering history of the work in this repo. Numbers cited here come from real evaluation outputs across multiple seeds and configurations. Where numbers vary across configurations, both are shown.

---

## Phase 6 — Stacking (predecessor work, not in this repo)

The pipeline in this repo originated from a longer cube-stacking effort. That work closed at **27.3% success on wide-spawn cube stacking** with a single wrist camera + learned perception net + scripted-expert controller. Multiple architectural improvements (v3, v4 of the perception stack) actually *regressed* performance from that baseline.

**Root cause identified (the lesson that shaped this repo):** training data covered *trajectories* (the arm in motion) but the perception net was deployed to predict from *static* viewpoints near the target — viewpoints the training distribution barely contained. Architecture changes had been compensating for a data-coverage gap.

**Pivot:** introduce an explicit **SCAN pre-phase** during training (the arm visits randomized static poses around the target so the perception net sees the deployment-viewpoint distribution); move the same pipeline from cube-stacking (a research toy) to industrial peg-in-hole (a task with paying customers and cleaner success metrics).

---

## Phase 7.1 — Peg-in-hole baseline

Brought the SCAN-augmented controller stack onto a fresh peg-in-hole task. Initial implementation used a kinematic peg (pose-driven, no physics) for controller-only validation. Result: 4/4 = 100% success at 1mm clearance, privileged-state controller.

**Then added real physics:** replaced the kinematic peg with a PhysX `FixedJoint`-attached dynamic body (gravity, collision, force feedback). Geometry tuned so the peg sits between the gripper fingers visually. Validated at 4/4 = 100% at 20mm clearance.

---

## Phase 7.1 clearance sweep — first compliance ceiling

Tightened clearance progressively at 5 seeds × 4 envs each:

| Clearance per side | Success |
|---|---|
| 20 mm | 4/4 (100%) |
| 10 mm | 4/4 (100%) |
| 5 mm  | 4/4 (100%) |
| 2 mm  | 4/4 (100%) |
| **1 mm**  | **2/4 (50%)** |

Drop at 1mm looked like a controller precision limit. Initial response: build force-feedback spiral-search compliance + lift-and-recover logic. Got us to ~65–70%. Felt like a hard ceiling.

---

## Phase 7.2 — Finger-pinch breakthrough

While debugging the apparent precision limit, instrumented the actuation interface and found:

- The peg was attached to the wrist via a rigid PhysX `FixedJoint`.
- The gripper fingers were *also* applying ~12 N of constant lateral pinch force on the peg's sides (from the closed-gripper command).
- Two physical attachments were fighting each other. The IK wasn't lacking precision — it was being perturbed by the system's own gripper command.

**Fix:** open the gripper (let the rigid joint hold the peg alone). Same rigid IK controller hit ~10 µm precision. Success at 1 mm jumped to 100%. Subsequent clearance sweep characterized the precision ceiling all the way down to **20 µm/side clearance** with 100% success.

**Key insight (portable rule):** before adding any compliance machinery, audit the actuation interface for self-perturbations the system is generating against itself.

---

## Phase 7.3 — Hole randomization

For perception-driven deployment, the hole pose can't be fixed in sim. Added a reset event that randomizes hole xy per env across the workspace (`x ∈ [0.40, 0.60]`, `y ∈ [-0.20, 0.05]`). Privileged-state controller hits **20/20 (100%) at 1 mm/side clearance** with random hole positions (5 seeds × 4 envs, current codebase). An earlier run at the same clearance with 8 seeds × 4 envs = 32/32 produced the same 100% result — see iteration notes; the current repo has been re-verified at 5 seeds and the 0 mm noise row of the README results table is the same configuration.

---

## Phase 7.3 — Noise-injection diagnostic (the methodology lesson)

Before training a perception net, tested the controller's tolerance to *synthetic* perception error. Injected per-episode Gaussian noise of varying std into the privileged hole pose passed to the expert, then measured success. Result:

| Perception noise σ | Success (5 seeds × 4 envs) |
|---|---|
| 0 mm | 100% |
| 1 mm | 85–90% |
| 2 mm | 55–70% |
| 5 mm | 35% |
| 10 mm | 15% |

**The compliance code (spiral search + lift-and-recover) had a hard ceiling at ~2–3mm noise.** No spiral tuning, no max-radius tweak, no lift-amount change moved it past that. The architectural envelope was a real limit, not a tuning problem.

**Methodology lesson:** this 30-minute test before training a perception net surfaces the controller envelope. If the envelope is narrower than typical perception net precision (2–5mm), you know upfront that training the net won't be enough.

---

## Phase 7.3b — Stepped chamfer (geometry beats algorithm)

To push noise tolerance past the compliance ceiling, tried geometric guidance: a chamfered (funnel) hole entrance.

First attempts (4 tilted cuboids at 45°) had two failed iterations — the rotated cuboids overlapped existing walls or caught the peg during descent. Both reverted to clean state.

Final attempt: **stepped chamfer** — 3 levels of horizontal cuboid rings at increasing widths (`inner_half + 5mm / 12mm / 20mm`) stacked above the existing walls. No rotation math, just box stacking. Results at 1 mm/side clearance (5 seeds × 4 envs per cell):

| Perception noise σ | No chamfer¹ | + Stepped chamfer² |
|---|---|---|
| 0 mm  | 100% | 100% |
| 1 mm  | 85%  | 90% |
| 2 mm  | 55–70% | **80%** |
| 5 mm  | 35%  | 50% |
| 10 mm | 15%  | 20% |

¹ Pre-chamfer numbers come from earlier iterations with the same controller but plain (un-chamfered) hole walls.
² Re-run on the current codebase, confirmed reproducible — see `results/eval_outputs.md`.

The chamfer adds 5–15 percentage points across all noise levels and noticeably improves the 1–2 mm regime (the most likely real perception net operating point). The geometric solution provides reliable lift exactly where algorithmic compliance was bottoming out.

**Why this works:** when geometry math is fighting you, simplify the geometry. Three flat steps beat one tilted face for both implementation effort and reliability. Industrial peg-in-hole has used chamfered entries since the 1970s; ML-first roboticists often reach for algorithmic fixes first.

---

## Summary of portable lessons

1. **Failure-mode discipline beats architectural novelty.** Most "controller precision" problems are actuation perturbation problems. Audit before adding compliance.
2. **Test the controller envelope before training the perception net.** A 30-minute noise-injection diagnostic surfaces the upper bound on success and saves multi-day misadventures.
3. **Geometry first, algorithm second.** When the task allows it (chamfered entry, tapered peg, mechanical alignment features), geometric guidance is more reliable and easier to implement than learned or hand-coded compliance.

---

## Current status (as of last update)

- All Phase 7.3b results above reproducible by running `scripts/smoke_test_peg_scripted.py` with appropriate `--seed` and `--hole_noise_std` values.
- Pipeline is sim-only. Perception net training and real-hardware transfer are pending (see Roadmap in main README).
