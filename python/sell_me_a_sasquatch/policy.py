"""A pointer-style maskable policy, and a fast numpy copy of it.

Why not a plain MLP head: the action space is ordinal, index `i` means
"the i-th entry of `legal_actions()` right now" (see `spaces.py`). A normal
policy head maps a state latent to a fixed logit per index, which silently
assumes index `i` means the same thing in every state, and here it does
not. Index 7 might be "reveal the Giant" one micro-step and "steal two
cards from the seat on my left" the next. A head like that has to reverse
engineer the engine's enumeration order from the state before it can mean
anything, which is close to hopeless.

So instead the environment hands the policy a description of every
candidate action (`obs["actions"]`, one feature row each: action type,
which card classes it commits, which seat it targets, how many face-down
cards it touches), and the policy scores each candidate from its own
description:

    logit(i) = <encode(action_i), query(state)> / sqrt(d)  +  bias(action_i)

This is the standard pointer/attention formulation. The score of an action
now depends on what the action does, so it transfers across states, across
table sizes, and across the shuffling of the action list, and adding a new
action type does not invalidate what the policy already knows.

`NumpyPointerPolicy` is the same computation in numpy, for the self-play
opponents. Opponent moves outnumber the learner's by (table size - 1) to
one, and a torch forward pass on a single observation is dominated by
framework overhead rather than arithmetic; the numpy copy sidesteps it.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np  # type: ignore
import torch as th  # type: ignore
from gymnasium import spaces  # type: ignore
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # type: ignore
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule  # type: ignore
from torch import nn

from sb3_contrib.common.maskable.distributions import MaskableDistribution  # type: ignore
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # type: ignore


class StateExtractor(BaseFeaturesExtractor):
    """Feeds only `obs["state"]` to the shared trunk.

    The action rows are not trunk input - they are consumed by the policy
    head, one per candidate - so flattening them in here would just make a
    (max_actions x feature) block of mostly-masked noise the value network
    has to learn to ignore.
    """

    def __init__(self, observation_space: spaces.Dict):
        """Builds an extractor sized to the `state` field's width."""
        super().__init__(
            observation_space, features_dim=int(observation_space["state"].shape[0])  # type: ignore
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        """Passes `obs["state"]` through unchanged."""
        return observations["state"]


class MaskablePointerPolicy(MaskableActorCriticPolicy):
    """`MaskableActorCriticPolicy` with a per-action scoring head.

    The shared trunk and value head are stock SB3; only the action head
    changes, so `MaskablePPO` needs no modification. The four methods that
    would otherwise route through `action_net` are overridden to route
    through `_action_logits` instead.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *args,
        action_embed_dim: int = 64,
        **kwargs,
    ):
        """Builds the policy, deferring the actual head to `_build`."""
        if (
            not isinstance(observation_space, spaces.Dict)
            or "actions" not in observation_space.spaces
        ):
            raise ValueError(
                "MaskablePointerPolicy needs a Dict observation space with 'state' and 'actions'"
            )
        self.action_feat_dim = int(observation_space["actions"].shape[1])  # type: ignore
        self.action_embed_dim = action_embed_dim
        kwargs.setdefault("features_extractor_class", StateExtractor)
        # the head reads obs["actions"] directly, so a second, unshared
        # copy of the trunk would have nothing extra to extract
        kwargs["share_features_extractor"] = True
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Builds the trunk, the pointer head, and the optimizer."""
        self._build_mlp_extractor()
        dim = self.action_embed_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_feat_dim, dim),
            self.activation_fn(),
            nn.Linear(dim, dim),
        )
        self.query_net = nn.Linear(self.mlp_extractor.latent_dim_pi, dim)
        # a state-independent preference per action, lets the policy learn
        # "passing is usually fine" without spending query capacity on it
        self.action_bias = nn.Linear(dim, 1)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # unused: the base class documents action_net as the action head,
        # and keeping the attribute (as a no-op) means anything reaching
        # for it fails loudly rather than silently reading stale weights
        self.action_net = nn.Identity()
        self._logit_scale = float(dim) ** -0.5

        if self.ortho_init:
            module_gains: dict[nn.Module, float] = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_encoder: np.sqrt(2),
                self.query_net: 0.01,
                self.action_bias: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs  # type: ignore
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Adds `action_embed_dim` to the params SB3 needs to rebuild this policy."""
        data = super()._get_constructor_parameters()
        data.update(action_embed_dim=self.action_embed_dim)
        return data

    # head

    def _action_logits(
        self, latent_pi: th.Tensor, action_feats: th.Tensor, action_masks=None
    ) -> th.Tensor:
        """Scores every candidate action against the state.

        Given a mask, only the live candidates are embedded at all. The
        action block is sized for the worst case (a 6-player thingamabob
        window, hundreds of options) while a typical micro-turn offers a
        few dozen, so embedding the padded tail would be an order of
        magnitude of wasted work on every forward and backward pass.
        Masked-out logits stay 0 here and are driven to -inf by
        `apply_masking` immediately afterwards."""
        query = self.query_net(latent_pi)  # (batch, dim)
        if action_masks is None:
            embedded = self.action_encoder(action_feats)  # (batch, actions, dim)
            scores = th.einsum("bad,bd->ba", embedded, query) * self._logit_scale
            return scores + self.action_bias(embedded).squeeze(-1)

        live = th.as_tensor(action_masks, dtype=th.bool, device=query.device).reshape(
            action_feats.shape[:2]
        )
        embedded = self.action_encoder(action_feats[live])  # (live, dim)
        queries = query.unsqueeze(1).expand(-1, action_feats.shape[1], -1)[live]
        scores = (embedded * queries).sum(-1) * self._logit_scale + self.action_bias(
            embedded
        ).squeeze(-1)
        return th.zeros(
            live.shape, dtype=scores.dtype, device=scores.device
        ).masked_scatter(live, scores)

    def _distribution(
        self, latent_pi: th.Tensor, obs: PyTorchObs, action_masks: np.ndarray | None
    ) -> MaskableDistribution:
        """Builds the masked action distribution for one state."""
        logits = self._action_logits(latent_pi, obs["actions"], action_masks)  # type: ignore
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    # overrides that would otherwise go through `action_net`

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        """Samples an action and returns it with its value estimate and log probability."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._distribution(latent_pi, obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob  # type: ignore

    def evaluate_actions(  # type: ignore
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        action_masks: np.ndarray | None = None,
    ):
        """Scores given actions under the current policy, for the PPO loss."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(latent_pi, obs, action_masks)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(
        self, obs: PyTorchObs, action_masks: np.ndarray | None = None
    ) -> MaskableDistribution:
        """The action distribution for one observation, without a value estimate."""
        features = super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.pi_features_extractor
        )
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._distribution(latent_pi, obs, action_masks)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """The value estimate for one observation, without an action distribution."""
        features = super(MaskableActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return self.value_net(self.mlp_extractor.forward_critic(features))


# numpy inference copy


def _linear_stack(module: nn.Module) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Flattens a `Linear`/activation chain into `(weight, bias, is_tanh)`.

    Raises on anything it does not recognize rather than silently producing
    a policy that disagrees with the torch one.
    """
    layers: list[tuple[np.ndarray, np.ndarray, bool]] = []
    children = [module] if isinstance(module, nn.Linear) else list(module.children())
    pending: tuple[np.ndarray, np.ndarray] | None = None
    for child in children:
        if isinstance(child, nn.Linear):
            if pending is not None:
                layers.append((*pending, False))
            pending = (
                child.weight.detach().cpu().numpy().T.copy(),
                child.bias.detach().cpu().numpy().copy(),
            )
        elif isinstance(child, nn.Tanh):
            assert pending is not None, "activation before any linear layer"
            layers.append((*pending, True))
            pending = None
        elif isinstance(child, (nn.Identity, nn.Flatten)):
            continue
        else:
            raise TypeError(f"NumpyPointerPolicy cannot mirror {type(child).__name__}")
    if pending is not None:
        layers.append((*pending, False))
    return layers


def _apply(
    x: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray, bool]]
) -> np.ndarray:
    """Runs `x` through a flattened linear/tanh stack from `_linear_stack`."""
    for weight, bias, tanh in layers:
        x = x @ weight + bias
        if tanh:
            x = np.tanh(x)
    return x


class NumpyPointerPolicy:
    """Frozen `MaskablePointerPolicy` weights, evaluated in numpy.

    Drop-in for the `opponent_policy` callback: same `(obs, mask) -> index`
    contract, same arithmetic as the torch policy (see
    `test_policy.py::test_numpy_policy_matches_torch`), without torch's
    per-call dispatch overhead on batch-of-one observations.
    """

    def __init__(
        self,
        trunk,
        query,
        encoder,
        bias,
        scale: float,
        rng: np.random.Generator | None = None,
    ):
        """Wraps already-flattened weight stacks; use `from_model` in practice."""
        self.trunk = trunk
        self.query = query
        self.encoder = encoder
        self.bias = bias
        self.scale = scale
        self.rng = rng or np.random.default_rng()

    @classmethod
    def from_model(
        cls, model, rng: np.random.Generator | None = None
    ) -> "NumpyPointerPolicy":
        """Builds a numpy copy from a trained SB3 model's weights."""
        policy = getattr(model, "policy", model)
        if not isinstance(policy, MaskablePointerPolicy):
            raise TypeError("NumpyPointerPolicy only mirrors MaskablePointerPolicy")
        with th.no_grad():
            return cls(
                trunk=_linear_stack(policy.mlp_extractor.policy_net),
                query=_linear_stack(policy.query_net),
                encoder=_linear_stack(policy.action_encoder),
                bias=_linear_stack(policy.action_bias),
                # pylint: disable=protected-access
                scale=policy._logit_scale,
                rng=rng,
            )

    def logits(self, obs: dict) -> np.ndarray:
        """Raw action logits for one observation, before masking."""
        latent = _apply(np.asarray(obs["state"], dtype=np.float32), self.trunk)
        query = _apply(latent, self.query)
        embedded = _apply(np.asarray(obs["actions"], dtype=np.float32), self.encoder)
        return embedded @ query * self.scale + _apply(embedded, self.bias)[..., 0]

    def __call__(self, obs: dict, mask: np.ndarray, deterministic: bool = False) -> int:
        """Picks a legal action index, sampling unless `deterministic`."""
        logits = self.logits(obs)
        legal = np.asarray(mask, dtype=bool)
        if not legal.any():
            return 0
        logits = np.where(legal, logits, -np.inf)
        if deterministic:
            return int(np.argmax(logits))
        probs = np.exp(logits - logits.max())
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return int(np.argmax(logits))
        return int(self.rng.choice(probs.size, p=probs / total))
