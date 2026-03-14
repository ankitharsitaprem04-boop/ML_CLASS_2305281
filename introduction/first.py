import pandas as pd
import numpy as np
n_rows = 10
n_cols = 5
min_val = 1
max_val = 100
data = np.random.randint(min_val, max_val+1, size=(n_rows, n_cols))
col_names = [f"subject{i+1}" for i in range(n_cols)]
df = pd.DataFrame(data, columns=col_names)
df.to_csv("random_data.csv", index=False)
print("Generated DataFrame")
print(df)
print("\nSaved to random_data.csv")