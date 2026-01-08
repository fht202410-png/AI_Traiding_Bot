from binance.client import Client
import pandas as pd
import os, csv, time

client = Client()
symbol = "ETHUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE
filename = "ETHUSDT_1m_binance.csv"

# Resume logic
if os.path.exists(filename):
    df = pd.read_csv(filename)
    last_time = pd.to_datetime(df["open_time"].iloc[-1])
    start_ts = int(last_time.timestamp() * 1000) + 60_000
    print("Resuming from:", last_time)
else:
    start_ts = int(pd.Timestamp("2017-01-01").timestamp() * 1000)

with open(filename, "a", newline="") as f:
    writer = csv.writer(f)

    if os.stat(filename).st_size == 0:
        writer.writerow(["open_time","open","high","low","close","volume"])

    while True:
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                limit=1000
            )

            if not klines:
                print("Finished. No more data.")
                break

            for k in klines:
                writer.writerow([
                    pd.to_datetime(k[0], unit="ms"),
                    float(k[1]), float(k[2]),
                    float(k[3]), float(k[4]),
                    float(k[5])
                ])

            start_ts = klines[-1][0] + 60_000

            print("Saved up to:", pd.to_datetime(start_ts, unit="ms"))
            time.sleep(0.3)  # polite to Binance

        except Exception as e:
            print("ERROR:", e)
            print("Sleeping 30 seconds...")
            time.sleep(30)
