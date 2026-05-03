import optax

from src.config import TrainingConfig


def build_learning_rate_schedule(
    training_config: TrainingConfig,
    total_steps: int,
    warmup_steps: int,
) -> optax.Schedule:
    """Warmup + cosine-decay schedule built from input parameters
    and default values that are defined in configs.

    The learning rate schedule determines how the learning rate
    will change during a training run.
    """
    return optax.warmup_cosine_decay_schedule(
        init_value=training_config.lr_init_value,
        peak_value=training_config.lr_peak_value,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=training_config.lr_end_value,
    )
