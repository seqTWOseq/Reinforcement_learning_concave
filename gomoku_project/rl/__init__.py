"""Reinforcement-learning components for Gomoku."""

__all__ = [
    "AlphaZeroPolicyValueNet",
    "AlphaZeroTrainer",
    "PPOActorCritic",
    "PPOTrainer",
    "SelfPlayTrainer",
    "SimplePolicyNetwork",
]


def __getattr__(name: str):
    if name in {"AlphaZeroPolicyValueNet", "SimplePolicyNetwork"}:
        from gomoku_project.rl.network import AlphaZeroPolicyValueNet, SimplePolicyNetwork

        exports = {
            "AlphaZeroPolicyValueNet": AlphaZeroPolicyValueNet,
            "SimplePolicyNetwork": SimplePolicyNetwork,
        }
        return exports[name]

    if name == "PPOActorCritic":
        from gomoku_project.rl.ppo_network import PPOActorCritic

        return PPOActorCritic

    if name in {"AlphaZeroTrainer", "SelfPlayTrainer"}:
        from gomoku_project.rl.trainer import AlphaZeroTrainer, SelfPlayTrainer

        exports = {
            "AlphaZeroTrainer": AlphaZeroTrainer,
            "SelfPlayTrainer": SelfPlayTrainer,
        }
        return exports[name]

    if name == "PPOTrainer":
        from gomoku_project.rl.ppo_trainer import PPOTrainer

        return PPOTrainer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
