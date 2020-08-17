# %%
import os
import pandas as pd

base_dir = os.getcwd()

# %%
db = pd.read_csv(base_dir + '/models/treemodel/output_2_1_2.csv')
# db['prob'] = db['prob'] * 5 / 3 if db['prob'] < 0.3 else 0.5 + (db['prob'] - 0.3) * 5 / 7
# for i in range(len(db)):
#     db['prob'].loc[i] = db['prob'].loc[i] * 5 / 2.3 if db['prob'].loc[i] < 0.23 else 0.5 + (db['prob'].loc[i] - 0.23) * 5 / 7.7

for i in range(len(db)):
    db['prob'].loc[i] = 0 if db['prob'].loc[i] < 0.23 else 1

# %%
db.to_csv(base_dir + '/models/treemodel/output_2_1_6.csv', index=False)