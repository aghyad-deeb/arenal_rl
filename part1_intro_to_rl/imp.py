# %%
import os
import random
import sys
from pathlib import Path
from typing import TypeAlias

import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

Arr: TypeAlias = np.ndarray

max_episode_steps = 1000
N_RUNS = 200

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part1_intro_to_rl"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import tests as tests
import utils as utils
from plotly_utils import imshow

# %%
ActType: TypeAlias = int
ObsType: TypeAlias = int

class MultiArmedBandit(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray

    def __init__(self, num_arms=10, stationary=True):
        super().__init__()
        self.num_arms = num_arms
        self.stationary = stationary
        self.observation_space= gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(num_arms)
        self.reset()

    def step(self, arm: ActType) -> tuple[ObsType, float, bool, dict]:
        assert self.action_space.contains(arm)
        if not self.stationary:
            q_drift = self.np.random.normal(
                    loc=0.0, scale=0.01, size=self.num_arms
            )
            self.arm_reward_means += q_drift
            self.best_arm = int(np.argmax(self.arm_reward_means))
        reward = self.np_random.normal(
                loc=self.arm_reward_means[arm], scale=1.0
        )
        obs = 0 
        terminated = False
        truncated = False
        info = dict(best_arm=self.best_arm)
        return (obs, reward, terminated, truncated, info)


    def reset(
        self, seed: int | None = None, options=None
    ) -> tuple[ObsType, dict]:
        super().reset(seed=seed)
        if self.stationary:
            self.arm_reward_means = self.np_random.normal(
                loc=0.0, scale=1.0, size=self.num_arms
            )
        else:
            self.arm_reward_means = np.zeros(shape=[self.num_arms])
        self.best_arm = int(np.argmax(self.arm_reward_means))

        obs = 0
        info = dict()
        return obs, info

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported"
        bandit_samples = list()
        for arm in range(self.action_space.n):
            bandit_samples += [
                np.random.normal(
                    loc=self.arm_reward_means[arm], scale=1.0, size=1000
                )
            ]
        plt.voilinplot(bandit_samples, showmeans=True)
        plt.xlabel("Bandit Arm")
        plt.ylabel("Reward Distribution")
        plt.show()



# %%
gym.envs.registration.register(
    id="ArmedBanditTestbed-v0",
    entry_point=MultiArmedBandit,
    max_episode_steps=max_episode_steps,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={"num_arms": 10, "stationary": True},
)

env = gym.make("ArmedBanditTestbed-v0")
print(f"{env=}")

# %%
class Agent:
    rng: np.random.Generator

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)
    
    def get_action(self) -> ActType:
        raise NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
    

def run_episode(env: gym.Env, agent: Agent, seed: int) -> tuple[Arr, Arr]:
    rewards, was_bests = ([], [])
    env.reset(seed=seed)
    agent.reset(seed=seed)

    done = False
    while done == False:
        arm = agent.get_action()
        obs, reward, terminated, truncated, info = env.step(arm)
        done = terminated or truncated
        agent.observe(arm, reward, info)
        rewards.append(reward)
        was_bests.append(1 if arm == info["best_arm"] else 0)
    
    rewards = np.array(rewards, dtype=float)
    was_bests = np.array(was_bests, dtype=int)
    return rewards, was_bests


def run_agent(
    env: gym.Env, agent: Agent, n_runs=200, base_seed=1
) -> tuple[Arr, Arr]:
    all_rewards = []
    all_was_bests = []
    base_rng = np.random.default_rng(base_seed)
    for n in tqdm(range(n_runs)):
        seed = base_rng.integers(low=0, high=10_000, size=1).item()
        rewards, was_bests = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_was_bests.append(was_bests)
    all_rewards = np.array(all_rewards)
    all_was_bests = np.array(all_was_bests)
    return all_rewards, all_was_bests


# %%
class RandomAgent(Agent):
    def get_action(self) -> ActType:
        return self.rng.integers(low=0, high=self.num_arms)
    
    def __repr__(self):
        return "RandomAgent"
    

num_arms = 10
stationary = True
env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
agent = RandomAgent(num_arms, 0)
all_rewards, all_corrects = run_agent(env, agent)

print(f"Expected correct freq: {1/10}, actual: {all_corrects.mean():.6f}")
assert np.isclose(all_corrects.mean(), 1 / 10, atol=0.05), "Random agent is not random enough!"

print(f"Expected average reward: 0.0, actual: {all_rewards.mean():.6f}")
assert np.isclose(
    all_rewards.mean(), 0, atol=0.05
), "Random agent should be getting mean arm reward, which is zero."

print("All tests passed!")


# %%
class RewardAveraging(Agent):
    reward_averages: Arr[float]
    action_counts: Arr[int]
    epsilon: float
    optimism: float

    def __init__(self, num_arms: int, seed: int, epsilon=0.01, optimism=0):
        self.epsilon = epsilon
        self.optimism = optimism
        super().__init__(num_arms, seed) 
    
    def get_action(self) -> ActType:
        if self.rng.binomial(n=1, p=self.epsilon):
            return self.rng.choice(list(range(self.num_arms)))
        else:
            return self.rng.choice(
                self.reward_averages.argmax(axis=-1, keepdims=True)
            )
    
    def observe(self, action: int, reward: float, info: dict) -> None:
        # Q_{n + 1} = Q_n + 1/n ( R_n - Q_n)
        self.action_counts[action] += 1
        self.reward_averages[action] += (
            (reward - self.reward_averages[action]) / self.action_counts[action]
        )
    
    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.reward_averages = np.zeros(shape=(num_arms), dtype=float)
        self.reward_averages += self.optimism
        self.action_counts = np.zeros(shape=(num_arms), dtype=int)
    
    def __repr__(self):
        return f"RewardAveraging({self.epsilon=}, {self.optimism=})"


num_arms = 10
stationary = True
names = []
all_rewards = []
env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)

for optimism in [0, 5]:
    agent = RewardAveraging(num_arms, 0, epsilon=0.01, optimism=optimism)
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    all_rewards.append(rewards)
    names.append(str(agent))
    print(agent)
    print(f" -> Frequency of correct arm: {num_correct.mean():.4f}")
    print(f" -> Average reward: {rewards.mean():.4f}")

utils.plot_rewards(all_rewards, names, moving_avg_window=15)


# %%
class CheaterAgent(Agent):
    best_arm: ActType

    def get_action(self) -> ActType:
        return self.best_arm

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.best_arm = info["best_arm"]
    
    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.best_arm = self.rng.choice(list(range(self.num_arms)))

    def __repr__(self):
        return "Cheater"


cheater = CheaterAgent(num_arms, 0)
reward_averaging = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=0)
random = RandomAgent(num_arms, 0)

names = []
all_rewards = []

for agent in [cheater, reward_averaging, random]:
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    names.append(str(agent))
    all_rewards.append(rewards)

utils.plot_rewards(all_rewards, names, moving_avg_window=15)

assert (all_rewards[0] < all_rewards[1]).mean() < 0.001, "Cheater should be better than reward averaging"
print("Tests passed!")


# %%
class UCBAgent(RewardAveraging):
    c: float
    
    def __init__(self, num_arms: int, seed: int, c: float):
        self.c = c
        super().__init__(num_arms, seed)

    def get_action(self) -> int:
        if (self.action_counts == 0).any():
            return self.rng.choice(np.argwhere((self.action_counts == 0))).item()
        t = self.action_counts.sum(axis=-1).item()
        UCB = self.reward_averages + self.c * np.sqrt(np.log(t) / self.action_counts)
        return self.rng.choice(UCB.argmax(axis=-1, keepdims=True))
    
    def __repr__(self):
        return f"UCB({self.c=})"


cheater = CheaterAgent(num_arms, 0)
reward_averaging = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=0)
reward_averaging_optimism = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=5)
ucb = UCBAgent(num_arms, 0, c=2.0)
random = RandomAgent(num_arms, 0)

names = []
all_rewards = []

for agent in [cheater, reward_averaging, reward_averaging_optimism, ucb, random]:
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    names.append(str(agent))
    all_rewards.append(rewards)

utils.plot_rewards(all_rewards, names, moving_avg_window=15)
# %%
