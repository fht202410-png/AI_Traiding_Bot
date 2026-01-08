import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv

def main():
    # Load 100k rows
    df = pd.read_csv('ETHUSDT_1m_binance.csv', nrows=100000)
    prices = df.iloc[:, 4].values.astype(np.float32)
    prices = (prices / prices[0]) * 100
    
    print(f"Training on {len(prices)} prices")
    
    # Simple environment
    env = TradingEnv(prices[:10000])
    
    # Basic PPO
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        verbose=1,
        device='cpu'
    )
    
    # Train
    print("Training...")
    model.learn(total_timesteps=50000)
    model.save('trading_agent')
    print("Model saved!")
    
    # Quick test
    test_prices = prices[10000:15000]
    test_env = TradingEnv(test_prices)
    obs, _ = test_env.reset()
    actions = []
    
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = test_env.step(action)
        actions.append(action)
        if done:
            break
    
    profit = (test_env.value[-1] - 10000) / 10000 * 100
    print(f"Test profit: {profit:.2f}%")
    print(f"Buys: {actions.count(1)}, Sells: {actions.count(2)}")
    return profit > 0

if __name__ == "__main__":
    main()