import numpy as np, pandas as pd
from pathlib import Path

# Adjust these lists as needed
train_subj = ['SA01','SA02','SA03','SA04','SA05','SA06','SA07','SA08','SA09','SA10','SA11','SA12','SA13','SA14','SA15']
val_subj   = ['SA16','SA17']
test_subj  = ['SA18','SA19','SA20']

def group_files(subj_list):
    files = []
    for f in Path("data/windows").glob("*.npz"):
        meta = np.load(f, allow_pickle=True)["meta"].item()
        if meta["subject"] in subj_list:
            files.append(f)
    return files
