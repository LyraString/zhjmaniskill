from types import SimpleNamespace

import numpy as np

from sim.maniskill.expert.primitives import DeltaPoseMotionPrimitives


class _BackendStub:
    render_mode = None

    def __init__(self, absolute_action):
        self.absolute_action = np.asarray(absolute_action, dtype=np.float32)
        self.prepared_chunks = []
        self.step_actions = []

    def prepare_action_chunk(self, action_chunk):
        self.prepared_chunks.append(np.asarray(action_chunk, dtype=np.float32).copy())
        return self.absolute_action[None, :].copy()

    def step(self, action):
        self.step_actions.append(action)
        return SimpleNamespace(
            observation=SimpleNamespace(data={"agent_pos": np.zeros(14, dtype=np.float32)}),
            info={},
        )


class _RecorderStub:
    def __init__(self):
        self.steps = []

    def append_step(self, action, next_obs):
        self.steps.append((np.asarray(action, dtype=np.float32).copy(), next_obs))


def test_delta_expert_executes_absolute_target_but_records_source_delta():
    delta_action = np.zeros(14, dtype=np.float32)
    delta_action[0] = 0.01
    delta_action[6] = 0.4
    delta_action[7] = -0.01
    delta_action[13] = 0.8
    absolute_action = np.asarray(
        [0.4, -0.2, 0.9, 0.1, -0.2, 0.3, 0.4,
         -0.4, 0.2, 0.9, -0.1, 0.2, -0.3, 0.8],
        dtype=np.float32,
    )
    backend = _BackendStub(absolute_action)
    recorder = _RecorderStub()

    expert = DeltaPoseMotionPrimitives.__new__(DeltaPoseMotionPrimitives)
    expert.backend = backend
    expert.base_env = SimpleNamespace(ACTION_DIM=14)
    expert.steps_taken = 0
    expert.step_limit = 10
    expert.last_status = "running"
    expert.last_plan_debug = {}
    expert.controller_failure_count = 0
    expert.delta_step_count = 0
    expert.success_detection_enabled = False
    expert.commanded_arm_qpos = {0: np.zeros(6), 1: np.zeros(6)}
    expert.commanded_gripper = {0: 0.0, 1: 0.0}
    expert.current_qpos = lambda arm_index: np.full(6, float(arm_index + 1))

    _result, success = expert.apply_delta_action(delta_action, recorder)

    assert success is False
    np.testing.assert_allclose(backend.prepared_chunks[0], delta_action[None, :])
    np.testing.assert_allclose(backend.step_actions[0].value, absolute_action)
    assert backend.step_actions[0].action_type == "bimanual_pd_ee_delta_pose"
    np.testing.assert_allclose(recorder.steps[0][0], delta_action)
    assert not np.allclose(recorder.steps[0][0], absolute_action)
    np.testing.assert_allclose(expert.commanded_arm_qpos[0], np.ones(6))
    np.testing.assert_allclose(expert.commanded_arm_qpos[1], np.full(6, 2.0))
    np.testing.assert_allclose(
        [expert.commanded_gripper[0], expert.commanded_gripper[1]],
        [0.4, 0.8],
    )
    assert expert.steps_taken == 1
    assert expert.delta_step_count == 1
