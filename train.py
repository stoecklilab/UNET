import random

import torch

from unet.utils import load_config, copy_config, get_logger
from unet.trainer import create_trainer

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config, config_path = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = create_trainer(config)
    # Copy config file
    copy_config(config, config_path)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
 