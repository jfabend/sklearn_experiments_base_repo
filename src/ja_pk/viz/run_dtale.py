#%%
import dtale
import pandas as pd
from pathlib import Path

#%%
#path = Path("C:/")
#filepath = path / "Data" / "projects" / "kaggle" / "titanic" / "train.csv"
#df = pd.read_csv("C:\Data\projects\\nomoko_ass\\immo_data.csv",
#                    sep=',',
#                    encoding = 'utf8',
#                    lineterminator='\n'
#                )
#df.head()
df = pd.read_csv("C:\Data\projects\sklearn_experiments_base_repo\\alldata_prepped.csv",
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
