from hashlib import md5
import pandas as pd

infile = "data/WDC/Recipe/recipe.csv"
df = pd.read_csv(infile)
df["id"] = df["source"].apply(lambda x: md5(str(x).encode()).hexdigest())
df.to_csv(infile, index=False)