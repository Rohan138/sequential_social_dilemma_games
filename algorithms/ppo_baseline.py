from __future__ import absolute_import, division, print_function

from ray.rllib.agents.ppo.ppo import execution_plan, validate_config
from ray.rllib.agents.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    ValueNetworkMixin,
    compute_and_clip_gradients,
    kl_and_loss_stats,
    postprocess_ppo_gae,
    ppo_surrogate_loss,
    setup_config,
    setup_mixins,
    vf_preds_fetches,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy import build_tf_policy
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule

from algorithms.common_funcs_baseline import BaselineResetConfigMixin


def build_ppo_baseline_trainer(config):
    """
    Creates a PPO policy class, then creates a trainer with this policy.
    :param config: The configuration dictionary.
    :return: A new PPO trainer.
    """
    policy = build_tf_policy(
        name="PPOTFPolicy",
        get_default_config=lambda: config,
        loss_fn=ppo_surrogate_loss,
        stats_fn=kl_and_loss_stats,
        extra_action_out_fn=vf_preds_fetches,
        postprocess_fn=postprocess_ppo_gae,
        compute_gradients_fn=compute_and_clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin],
    )

    ppo_trainer = build_trainer(
        name="BaselinePPOTrainer",
        default_policy=policy,
        default_config=config,
        validate_config=validate_config,
        execution_plan=execution_plan,
        mixins=[BaselineResetConfigMixin],
    )
    return ppo_trainer
