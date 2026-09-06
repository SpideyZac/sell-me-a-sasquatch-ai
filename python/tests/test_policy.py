"""Tests for the pointer policy and its numpy mirror.

These need the optional `[train]` extra (torch + sb3-contrib), so they skip
cleanly when it is not installed.
"""

# pylint: disable=C0413

import os

import numpy as np  # type: ignore
import pytest  # type: ignore

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import torch as th  # type: ignore  # noqa: E402
from sb3_contrib import MaskablePPO  # type: ignore  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # type: ignore  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore  # noqa: E402

from sell_me_a_sasquatch.policy import (
    MaskablePointerPolicy,
    NumpyPointerPolicy,
)  # noqa: E402
from sell_me_a_sasquatch.selfplay_env import (
    SasquatchSelfPlayEnv,
    random_masked_policy,
)  # noqa: E402

DECK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "deck.toml")
)


def make_model(players=(2, 4, 6), **kwargs):
    """Builds a small `MaskablePointerPolicy` model for fast tests."""

    def _factory():
        env = SasquatchSelfPlayEnv(
            players=players,
            deck_config_path=DECK_PATH,
            opponent_policy=random_masked_policy,
        )
        return ActionMasker(env, lambda e: e.action_masks())  # type: ignore

    vec = DummyVecEnv([_factory])
    model = MaskablePPO(
        MaskablePointerPolicy,
        vec,
        seed=0,
        device="cpu",
        n_steps=kwargs.pop("n_steps", 64),
        batch_size=kwargs.pop("batch_size", 64),
        policy_kwargs=dict(net_arch=[32, 32], action_embed_dim=16),
        **kwargs,
    )
    return model, vec


def test_pointer_policy_trains_end_to_end():
    """Exercises every path MaskablePPO takes through the custom head:
    rollout collection (`forward`), bootstrapping (`predict_values`) and the
    update (`evaluate_actions`)."""
    model, vec = make_model()
    model.learn(total_timesteps=256, progress_bar=False)
    vec.close()


def test_numpy_policy_matches_torch():
    """The numpy mirror exists purely for speed, so it has to be the same
    function, not merely a similar one."""
    model, vec = make_model()
    env = vec.envs[0].env  # type: ignore
    obs, _ = env.reset(seed=3)
    mirror = NumpyPointerPolicy.from_model(model)

    tensor_obs, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        latent_pi, _ = model.policy.mlp_extractor(
            model.policy.extract_features(tensor_obs)
        )
        # pylint: disable=protected-access
        torch_logits = model.policy._action_logits(latent_pi, tensor_obs["actions"])[  # type: ignore  # pylint: disable=line-too-long
            0
        ].numpy()

    np.testing.assert_allclose(mirror.logits(obs), torch_logits, rtol=1e-4, atol=1e-5)
    vec.close()


def test_numpy_policy_only_ever_returns_a_legal_action():
    """The numpy policy never picks an index outside the current legal count."""
    model, vec = make_model()
    env = vec.envs[0].env  # type: ignore
    obs, _ = env.reset(seed=5)
    mirror = NumpyPointerPolicy.from_model(model, rng=np.random.default_rng(0))
    for _ in range(50):
        legal = env.legal_count
        action = mirror(obs, env.action_masks(), legal)
        assert 0 <= action < legal
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            obs, _ = env.reset()
    vec.close()


def test_saved_model_round_trips(tmp_path):
    """`action_embed_dim` is a constructor argument of ours, not SB3's; if
    it is not carried in the saved constructor parameters, a reloaded model
    silently rebuilds the head at the wrong width."""
    model, vec = make_model()
    path = str(tmp_path / "model.zip")
    model.save(path)
    reloaded = MaskablePPO.load(path, device="cpu")
    assert reloaded.policy.action_embed_dim == model.policy.action_embed_dim

    env = vec.envs[0].env  # type: ignore
    obs, _ = env.reset(seed=9)
    mask = env.action_masks()
    before, _ = model.predict(obs, action_masks=mask, deterministic=True)
    after, _ = reloaded.predict(obs, action_masks=mask, deterministic=True)
    assert int(before) == int(after)
    vec.close()


def test_action_features_distinguish_candidates():
    """The pointer head is only useful if candidates actually look different
    to it; identical rows would make every logit identical too."""
    env = SasquatchSelfPlayEnv(
        players=4, deck_config_path=DECK_PATH, opponent_policy=random_masked_policy
    )
    obs, _ = env.reset(seed=13)
    rows = obs["actions"][: env.legal_count]
    assert env.legal_count > 1
    assert len(np.unique(rows, axis=0)) == len(
        rows
    ), "legal actions must have distinct feature rows"
