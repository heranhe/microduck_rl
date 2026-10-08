"""CPU-only configuration locks for the mjlab Jumper port."""

from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab_microduck.robot import jumper_model as J
from mjlab_microduck.tasks.jumper_env_cfg import (
    gait_phase,
    jumper_rl_cfg,
    make_jumper_env_cfg,
    tripod_match,
)


def test_asset_facts_match_the_local_lab_copy():
    local = (
        Path(__file__).resolve().parents[2]
        / "microduck-lab/microduck_local/src/microduck_local/robots"
    )
    if local.exists():
        assert (local / "jumper_assets.json").read_bytes() == Path(
            J.__file__
        ).with_name("jumper_assets.json").read_bytes()
        assert (local / "jumper_model.py").read_bytes() == Path(J.__file__).read_bytes()


def test_jumper_layout_has_20_ordered_actions_and_no_history():
    cfg = make_jumper_env_cfg()
    assert cfg.actions["joint_pos"].actuator_names == J.JOINT_NAMES
    assert cfg.actions["joint_pos"].preserve_order
    assert tuple(cfg.observations["actor"].terms) == (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    )
    assert all(
        term.history_length == 0 for term in cfg.observations["actor"].terms.values()
    )
    assert cfg.sim.mujoco.timestep * cfg.decimation == J.CTRL_DT
    assert cfg.scene.sensors[0].primary.pattern == J.FOOT_GEOMS
    assert jumper_rl_cfg().clip_actions == 4.0
    assert jumper_rl_cfg().actor.obs_normalization


def test_command_gate_and_six_foot_grouping():
    cmd = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0, 0], [0.1, 0, 0]])
    found = torch.tensor(
        [
            [[1], [1], [1], [1], [1], [1]],
            [[0], [1], [1], [0], [0], [1]],
            [[1], [0], [0], [1], [1], [0]],
        ],
        dtype=torch.bool,
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        episode_length_buf=torch.tensor([0, 0, 8]),
        command_manager=SimpleNamespace(get_command=lambda _: cmd),
        scene={
            "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=found))
        },
    )
    assert torch.equal(tripod_match(env), torch.ones(3))
    found[1, 0] = True
    assert tripod_match(env)[1] < 1
    assert torch.equal(gait_phase(env)[0], torch.zeros(2))
    assert torch.allclose(gait_phase(env)[1], torch.tensor([0.0, 1.0]))


def test_stand_commands_and_play_randomizers():
    cfg = make_jumper_env_cfg(stand=True)
    assert cfg.commands["twist"].rel_standing_envs == 1
    assert not cfg.commands["twist"].heading_command
    assert cfg.commands["twist"].ranges.heading is None
    cfg = make_jumper_env_cfg(play=True)
    assert not cfg.observations["actor"].enable_corruption
    assert (
        not {"push_robot", "base_com", "encoder_bias", "foot_friction"}
        & cfg.events.keys()
    )


def test_export_stamp_preserves_metadata_and_refuses_history(tmp_path):
    import json

    import onnx
    from onnx import TensorProto, helper

    def graph(width):
        return helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Constant",
                        [],
                        ["actions"],
                        value=helper.make_tensor(
                            "zero", TensorProto.FLOAT, [1, 20], [0.0] * 20
                        ),
                    )
                ],
                "policy",
                [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
                [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 20])],
            )
        )

    path = tmp_path / "policy.onnx"
    model = graph(71)
    helper.set_model_props(model, {"run_path": "retained"})
    onnx.save(model, path)
    J.stamp_policy(path)
    J.stamp_policy(path)
    props = {p.key: p.value for p in onnx.load(path).metadata_props}
    assert props["run_path"] == "retained"
    assert json.loads(props["contract"]) == J.policy_contract()
    assert props["contract_id"] == "jumper-lab-71-v1"
    onnx.save(graph(355), path)
    import pytest

    with pytest.raises(ValueError, match="71 observations"):
        J.stamp_policy(path)
