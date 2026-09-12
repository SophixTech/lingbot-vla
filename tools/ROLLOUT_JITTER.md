# G1 Rollout Jitter Diagnostics

This stage records data only. It does not change model weights, action values,
chunk length, `use_length`, inference, control frequency, PD gains, filtering,
interpolation, temporal ensemble, SEAM, or RTC behavior. Diagnostics are opt-in
with `--diagnostic-log-dir`; omit that flag to retain the original path.

## Verified deployment chain

Serial native path:

```text
RobotDds.arm_joint_states/gripper_states + CosineCamera
  -> wait_for_observation/read_observation
  -> WebsocketClientPolicy.infer (blocking WebSocket)
  -> LingbotVLAServer.infer/FeatureTransform.apply/select_action/unapply
  -> server returns full [50, 16] action
  -> _validate_action_chunk
  -> execute_prefix selects frames_per_observation (default 30)
  -> RobotDds.move_arm/move_gripper
  -> RobotDds feedback on next read
```

The model action is an unnormalized position vector with 14 arm joints and 2
gripper values. The model chunk is 50 steps (`config.chunk_size` and
`n_action_steps`). `use_length` is applied in
`deploy/lingbot_vla_policy.py:314-317` before transport. The current G1 client
contract requires the response to still be `[50,16]`, so the deployed model
server must be launched with `--use_length 50`; a shorter value is rejected by
`_validate_action_chunk` and cannot provide a complete chunk for this logger.
The G1 client then selects its execution prefix at
`g1/运控test/g1_native_vla_deploy.py:execute_prefix`.
The serial inference and execution are synchronous.

Realtime path:

```text
InferenceWorker.capture -> client.infer (background thread)
  -> ActionTimeline installs the full [50,16] chunk
  -> run_control_loop samples at 30 Hz
  -> apply_frame -> RobotDds.move_arm/move_gripper
```

This path is asynchronous: one inference request is in flight while the
control loop executes the current timeline. The logger records request,
capture, inference, ready, and execution monotonic timestamps.

The current `RobotDds` interface used by this deployment exposes joint and
gripper position feedback only. `dq_actual`, EE pose, torque, Kp, and Kd are
therefore recorded as empty fields, never fabricated. A real low-level PD
diagnosis requires a separately validated feedback topic/API.

## One rollout

Source the GDK environment first. Dry-run is the default and publishes no
motion:

```bash
source /home/bjtc/Sophix/g1/g1_pc_gdk_env.sh
cd /home/bjtc/Sophix/g1/运控test
python g1_native_vla_deploy.py --loop \
  --diagnostic-log-dir /home/bjtc/Sophix/lingbot-vla/logs/rollout_$(date +%Y%m%d_%H%M%S)
```

For the already-approved native execution gate, retain the existing
`--execute --confirm RUN_NATIVE_PACKAGE_A2D_VLA` requirements and add the same
diagnostic flag. No new gate is bypassed by logging.

To record the asynchronous scheduler instead, use `--realtime-chunking
--loop --frames-per-observation 50` and the diagnostic flag.

## Offline analysis

```bash
source /home/bjtc/Sophix/g1/g1_pc_gdk_env.sh
python /home/bjtc/Sophix/lingbot-vla/tools/analyze_rollout_jitter.py \
  --log_dir /home/bjtc/Sophix/lingbot-vla/logs/rollout_YYYYmmdd_HHMMSS
```

The command writes `jitter_analysis.json` plus separate PNGs for command vs
actual, action/boundaries, velocity/acceleration/jerk, boundary delta,
latency, observation age, execution/inference timeline, and jerk-boundary
alignment. Classification is evidence-based; missing feedback is reported as
`insufficient_data` rather than inferred.
