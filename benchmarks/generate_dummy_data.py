import os
import numpy as np
import pandas as pd

def generate_dummy():
    path = "E:/Agent/baselines/JaxMARL-HFT/data/rawLOBSTER/AMZN/2017Jan_oneday/"
    os.makedirs(path, exist_ok=True)

    num_rows = 100000

    times = 34200.0 + np.arange(num_rows) * 0.1
    types = np.ones(num_rows, dtype=int)
    order_ids = 1000 + np.arange(num_rows, dtype=int)
    qtys = np.ones(num_rows, dtype=int) * 10
    prices = np.ones(num_rows, dtype=int) * 100000
    directions = np.where(np.arange(num_rows) % 2 == 0, 1, -1)

    df_msg = pd.DataFrame({
        'time': times,
        'type': types,
        'order_id': order_ids,
        'qty': qtys,
        'price': prices,
        'direction': directions
    })

    msg_file = os.path.join(path, "AMZN_2017-01-03_34200000_57600000_message_10.csv")
    df_msg.to_csv(msg_file, header=False, index=False)
    print(f"Wrote dummy message CSV to {msg_file}")

    row = []
    for i in range(10):
        row.extend([100100 + i * 100, 10, 99900 - i * 100, 10])

    row_str = ",".join(map(str, row)) + "\n"

    book_file = os.path.join(path, "AMZN_2017-01-03_34200000_57600000_orderbook_10.csv")
    with open(book_file, "w") as f:
        f.writelines([row_str] * num_rows)
    print(f"Wrote dummy orderbook CSV to {book_file}")

if __name__ == "__main__":
    generate_dummy()
