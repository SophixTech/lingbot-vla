# Package A2D Training Completion Protocol

## Purpose

This protocol separates checkpoint selection from deployment approval. A low
training loss or open-loop action error is not proof that a robot will grasp a
package, keep its label up, and place it on a conveyor correctly.

## Evidence From Robot Learning Practice

- RT-1 evaluates the ability to generalize on real robots as data, model size,
  and diversity change. It does not use training loss as the policy-quality
  claim. [Brohan et al., 2023](https://arxiv.org/abs/2212.06817)
- OpenVLA reports downstream robot task success rates across held-out tasks and
  embodiments, including its parameter-efficient fine-tuning results.
  [Kim et al., 2024](https://arxiv.org/abs/2406.09246)
- Octo evaluates fine-tuned policies on fixed task conditions and reports
  success rates averaged over 20 real-robot trials. [Octo Model Team,
  2024](https://arxiv.org/abs/2405.12213)
- Diffusion Policy shows that smooth training objectives need not imply stable
  policy behavior, evaluates checkpoints by rollout success, and reports both
  best-checkpoint and late-checkpoint averages. [Chi et al.,
  2023](https://arxiv.org/abs/2303.04137)
- ACT selects a candidate by validation loss, then reports real-world task
  success over 25 trials. [Zhao et al., 2023](https://arxiv.org/abs/2304.13705)

The statistical component follows the standard paired bootstrap principle:
because every checkpoint is evaluated on the same episodes, resampling
episode-level differences tests whether a measured improvement is likely to be
more than sample noise.

## Layer 1: Automatic Offline Checkpoint Selection

1. Save a checkpoint every 1,000 optimization steps. The original 8,000-step
   budget is a mandatory evaluation point. With explicit authorization, the
   run may continue in 1,000-step increments while the patience criterion has
   not been met; each increment must retain the same holdout cohort and safety
   monitoring.
2. Evaluate every checkpoint on the same 24 evenly spaced holdout episodes
   (2204-2447), never included in the 2,204 training episodes.
3. Evaluate ten 16-action chunks per episode and save episode-level MAE. The
   cohort and action horizon must not change between checkpoints.
4. A checkpoint is `improved` only when both conditions hold against the best
   previous checkpoint:
   - mean episode MAE decreases by at least 1%; and
   - the lower bound of a 10,000-resample paired-bootstrap 95% confidence
     interval for (previous MAE - current MAE) is greater than zero.
5. After step 4,000, automatically stop after three consecutive
   `not_improved` checkpoints. This is a patience rule, not a claim that the
   task is physically solved. If the rule is not met at a normal max-step
   completion, increase the limit by 1,000 steps and resume from the complete
   checkpoint; safety-guard stops and abnormal exits never auto-resume.
6. Select the checkpoint with the best fixed-cohort MAE as the deployment
   candidate. The previous six-episode, one-chunk metric remains historical
   information only and is not used for automatic early stopping.

The results are recorded in `holdout_metrics_v2.tsv`. Per-episode metrics are
kept with each checkpoint's holdout output so comparisons are auditable.

## Layer 2: Locked Offline Test

Before touching the robot, run the selected candidate once on a separate,
locked group of held-out episodes that was never used for checkpoint choice.
Report mean and per-episode action errors; do not use this result to choose a
different checkpoint. This estimates offline generalization, not task success.

## Layer 3: Real-Robot Deployment Approval

Only an authorized, supervised real-robot rollout can establish task success.
For the package task, a trial succeeds only if all are true:

- the right gripper grasps a soft package;
- the label is facing up after placement; and
- the package is placed on the conveyor without a safety stop or drop.

Run 50 pre-specified, randomized trials with varied package pose and box
position, recording `grasped`, `label_up`, `placed_on_conveyor`, the joint
success flag, and a failure reason. Report the Wilson 95% confidence interval
for joint success. For an 80% deployment target, require at least 46 successes
out of 50: its Wilson lower 95% bound is approximately 81%. Any safety fault is
reported separately and blocks unattended deployment regardless of success
rate.

## Decision Terms

- `training complete`: the process stopped by the above patience rule or the
  8,000-step cap, with final artifacts generated.
- `offline converged`: three fixed-cohort evaluations did not produce a
  statistically supported 1% MAE improvement.
- `deployment candidate`: the best checkpoint under Layer 1, which passed the
  locked offline test.
- `task validated`: an authorized real-robot evaluation met the joint success
  and safety criteria in Layer 3.
