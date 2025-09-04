import pandas as pd

df = pd.read_csv("../data/processed/train.csv", usecols=["criterion_text"])
df_sampled = df.sample(n=1000, random_state=42)
df_sampled.to_csv("../data/processed/sample.txt", index=False, header=False)

print("Done")
