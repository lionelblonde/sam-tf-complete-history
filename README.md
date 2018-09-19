# Sample Efficient Imitation Learning

Write-up in progress.

ArXiv link: [Sample-efficient Imitation Learning via Generative Adversarial Nets](
https://arxiv.org/abs/1809.02064)

## How to launch?

Example: locally launch a PPO expert training job in the environment 'BreakoutNoFrameskip-v4'
from the [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment).
* Docker:
    * CPU only:
`docker run -i -t --rm docker-rl-tf-cpu:latest bash -c "cd launchers/expert && ./atari_train_ppo_expert.sh 2 BreakoutNoFrameskip-v4"`
    * GPU support:
`docker run -i -t --rm docker-rl-tf-gpu:latest bash -c "cd launchers/expert && ./atari_train_ppo_expert.sh 2 BreakoutNoFrameskip-v4"`
* Non-Docker:

## TODOs

* Fix [UNREAL](https://arxiv.org/abs/1611.05397) prioritized replay buffer
(need efficient sorted-insert)

## Acknowledgments

Code initially inspired from [openai/baselines](https://github.com/openai/baselines)
