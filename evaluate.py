import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv

def main():
    model = PPO.load('trading_agent', device='cpu')
    
    # Test on different data
    df = pd.read_csv('ETHUSDT_1m_binance.csv', skiprows=200000, nrows=50000)
    prices = df.iloc[:, 4].values.astype(np.float32)
    prices = (prices / prices[0]) * 100
    
    env = TradingEnv(prices[:5000])
    obs, _ = env.reset()
    
    # Track everything
    actions = []
    portfolio_values = []
    
    for i in range(5000):
        action, _ = model.predict(obs, deterministic=False)  # Try stochastic
        obs, reward, done, _, _ = env.step(action)
        actions.append(action)
        portfolio_values.append(env.value[-1])
        
        if done:
            break
    
    # Calculate
    final = env.value[-1]
    profit = (final - 10000) / 10000 * 100
    
    # Buy and hold comparison
    buy_hold = (prices[min(4999, len(prices)-1)] - 100) / 100 * 10000
    buy_hold_profit = (10000 + buy_hold - 10000) / 10000 * 100
    
    print(f"\n{'='*50}")
    print(f"AGENT: ${final:.2f} ({profit:.2f}%)")
    print(f"BUY & HOLD: ${10000 + buy_hold:.2f} ({buy_hold_profit:.2f}%)")
    print(f"Buys: {actions.count(1)}")
    print(f"Sells: {actions.count(2)}")
    print(f"Holds: {actions.count(0)}")
    print(f"Final cash: ${env.cash:.2f}")
    print(f"Final shares: {env.shares:.4f}")
    print(f"{'='*50}")
    
    # Success if beats buy & hold or >0%
    success = profit > 0 or profit > buy_hold_profit
    if success:
        print("✅ POSITIVE RETURNS!")
    else:
        print("❌ Needs improvement")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)