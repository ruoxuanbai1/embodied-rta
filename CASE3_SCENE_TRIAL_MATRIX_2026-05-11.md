# Case 3 Scene Logic and Trial Matrix

This note fixes two things:

1. what the current **complete executable scene set** really is
2. what **minimum sample count** is acceptable before training `R2/R3`

## 1. Scene logic

The benchmark should stay layered and interpretable:

1. `reference`
2. `single-axis scenes`
3. `mixed scenes`
4. `geometry scenes` once the backend supports them

This matters because we need:

- single-axis scenes for clean calibration
- mixed scenes for interaction validation
- geometry scenes only when a real clutter backend exists

## 2. Current full executable scene set

The long-term benchmark target is 10 scenes, but the **current real executable
set is 8 scenes**.

### Executable now

1. `nominal_clear_reference`
2. `perception_stressed_mild`
3. `perception_stressed_strong`
4. `dynamic_disturbance_sparse`
5. `dynamic_disturbance_dense`
6. `dynamic_disturbance_late`
7. `mixed_perception_dynamics_mild`
8. `mixed_perception_dynamics_strong`

These 8 scenes are the **current complete training-and-testing scene set**.

### Defined but not executable yet

9. `static_clutter_light`
10. `static_clutter_heavy`

These two are still blocked because the current real `gym-aloha` path does not
provide a true clutter-capable variant.

They must stay in the benchmark design, but they must **not** be silently
replaced by a toy backend.

## 3. Trial matrix inside one scene

One scene is not one rollout.

Each supported scene variant should be repeated through a fixed trial matrix:

- normal repeats: `5`
- seeds per fault: `5`
- fault timings:
  - `early`
  - `mid`
  - `late`
- intensity bands:
  - `mild`
  - `medium`
  - `severe`

For each `scene x fault`, this gives:

- `5 x 3 x 3 = 45` fault trials
- plus `5` normal control trials

This is the **minimum useful training-scale block**.

If we collect only 1-3 repeats, that can support:

- smoke tests
- quick diagnostics
- early method debugging

But it is **not enough** to trust `R2/R3` after the scene set expands.

The final repeat count should not be chosen by intuition alone.

It should be stopped by **metric-mean convergence**:

- keep adding seeds / repeats
- rerun online evaluation
- compare the recent mean values of:
  - `precision`
  - `recall`
  - `f1_score`
  - `false_alarm_rate`
- `mean_lead_time`
- stop only when those means stop moving meaningfully

Raw variance is useful as a dispersion diagnostic, but the stop rule should not
use variance alone.

The more decision-relevant quantity is the **confidence-interval half-width**,
especially the **relative half-width** with respect to the current metric mean.

Current default convergence rule:

- use the most recent `2` result windows as the uncertainty-estimation window
- require the current `95%` confidence-interval relative half-width
  to be `<= 0.10`
- do not even check convergence before at least `5` scene seeds are present

Window-to-window mean drift can still be reported as a diagnostic quantity, but
it is no longer part of the stopping criterion.

## 4. Why this sample size matters

If we only tune `R2/R3` on a few runs from a few scenes, we risk:

- overfitting scene-specific thresholds
- learning scene-specific reward-drop patterns
- unstable branch calibration when a new scene arrives

So the right protocol is:

1. first complete the full executable 8-scene set
2. then train on all 8 supported scenes together
3. then test online on all 8 supported scenes
4. and keep scene-wise results, not only one global average

## 5. Collection order

### Phase 1: perception axis

- `nominal_clear_reference`
- `perception_stressed_mild`
- `perception_stressed_strong`

Primary faults:

- `F1_lighting`
- `F2_occlusion`
- `F3_adversarial`

### Phase 2: dynamics + timing axis

- `dynamic_disturbance_sparse`
- `dynamic_disturbance_dense`
- `dynamic_disturbance_late`

Primary faults:

- `F4_payload`
- `F5_friction`
- `F6_dynamic`
- `F8_compound`

### Phase 3: mixed axis

- `mixed_perception_dynamics_mild`
- `mixed_perception_dynamics_strong`

Primary faults:

- `F3_adversarial`
- `F6_dynamic`
- `F8_compound`

### Phase 4: geometry axis

- `static_clutter_light`
- `static_clutter_heavy`

Only enter this phase after a real clutter-capable backend exists.

## 6. Reporting rule

For every method (`R1`, `R2`, `R3`, `Full`), results should be reported at
three levels:

1. all supported scenes combined
2. per scene
3. per scene and per fault

This keeps the benchmark honest when scene coverage expands.
