import gymnasium as gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, prices):
        super().__init__()
        self.prices = prices.astype(np.float32)
        self.n_steps = len(prices)
        self.reset()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.cash = 10000.0
        self.shares = 0
        self.value = [10000.0]
        return self._get_obs(), {}
    
    def _get_obs(self):
        price = self.prices[min(self.step_idx, len(self.prices)-1)]
        
        # Simple price momentum (5-step)
        momentum = 0
        if self.step_idx >= 5:
            momentum = (price / self.prices[self.step_idx-5] - 1) * 100
        
        portfolio = self.cash + self.shares * price
        return np.array([
            price / 1000,
            momentum,
            self.shares * price / 5000,
            self.cash / 10000,
            portfolio / 10000,
            self.step_idx / 10000
        ], dtype=np.float32)
    
    def step(self, action):
        price = self.prices[min(self.step_idx, len(self.prices)-1)]
        prev_value = self.value[-1]
        
        # Calculate momentum for reward
        momentum = 0
        if self.step_idx >= 5:
            momentum = (price / self.prices[self.step_idx-5] - 1) * 100
        
        # Simple trading logic
        if action == 1 and self.shares == 0 and self.cash > 100:  # Buy
            self.shares = self.cash * 0.5 / price
            self.cash -= self.shares * price
        
        elif action == 2 and self.shares > 0:  # Sell
            self.cash += self.shares * price
            self.shares = 0
        
        # Calculate value
        new_value = self.cash + self.shares * price
        self.value.append(new_value)
        
        # Simple reward: profit + momentum bonus
        reward = (new_value - prev_value) / 100  # Scale down
        
        # Bonus for buying in uptrend
        if action == 1 and momentum > 0.5:
            reward += 0.1
        
        # Bonus for selling in downtrend
        if action == 2 and momentum < -0.5:
            reward += 0.1
        
        self.step_idx += 1
        done = self.step_idx >= min(self.n_steps - 1, 5000)
        return self._get_obs(), reward, done, False, {}