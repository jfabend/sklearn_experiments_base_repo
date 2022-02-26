#%%
import dtale
import pandas as pd
from pathlib import Path

#%%
#path = Path("C:/")
#filepath = path / "Data" / "projects" / "kaggle" / "titanic" / "train.csv"
#df = pd.read_csv("C:\Data\path\data.csv",
#                    sep=',',
#                    encoding = 'utf8',
#                    lineterminator='\n'
#                )
#df.head()
df = pd.read_csv("D:\Sonstiges\sklearn_experiments_base_repo\\train_prepped.csv",
                    sep=',')
df.head()
#%%
df.describe()

#%%
df.columns

#%%
d = dtale.show(df)
d
#d.main_url()
# %%
