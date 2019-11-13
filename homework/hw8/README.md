## IE 534 Assignment: Reinforcement Learning

#### Getting Started
You can either:

* Fork your own copy of the repo, and work on it
* or, download a zip file containing everything: https://github.com/mikuhatsune/ie534_rl_hw/archive/master.zip
* or, directly clone the repo to local (not recommended):
```bash
git clone https://github.com/mikuhatsune/ie534_rl_hw.git
```

Please follow instructions in the Jupyter notebook [rl.ipynb](rl.ipynb).

An example of finished homework is in [example_solution/rl.ipynb](example_solution/rl.ipynb) and [example_solution/rl.pdf](example_solution/rl.pdf).

Example training logs [example_solution/log_breakout_dqn.txt](example_solution/log_breakout_dqn.txt), and [example_solution/log_breakout_a2c.txt](example_solution/log_breakout_a2c.txt).
Format:
```
iter: iteration
n_ep: number of episodes (games played)
ep_len: running averaged episode length
ep_rew: running averaged episode clipped reward
raw_ep_rew: running averaged raw episode reward (actual raw game score)
env_step: number of environment simulation steps
time, rem: time passed, estimated time remain

iter    500 |loss   0.00 |n_ep    28 |ep_len   31.3 |ep_rew  -0.22 |raw_ep_rew   1.76 |env_step   1000 |time 00:04 rem 281:49
```
