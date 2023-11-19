import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.ppo.policies import MlpPolicy
env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)