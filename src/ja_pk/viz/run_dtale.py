#%%
import dtale
import pandas as pd
from pathlib import Path

#path = Path("C:/")
#filepath = path / "Data" / "projects" / "kaggle" / "titanic" / "train.csv"
df = pd.read_csv("C:\Data\projects\kaggle\\titanic\\train.csv")
df.head()

#%%
df.describe()

#%%
d = dtale.show(df)
d
#d.main_url()
# %%
