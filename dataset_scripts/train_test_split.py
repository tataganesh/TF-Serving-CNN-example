import csv
import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
import config

np.random.seed(0)
TRAIN_FOLDER = config.TRAIN_FOLDER
TEST_FOLDER = config.TEST_FOLDER
RENAMED_COLUMNS = ['filename', 'width', 'height','class', 'xmin', 'ymin', 'xmax', 'ymax']
DATASET = 'dataset.csv'
df = pd.read_csv(DATASET, names=['imagePath', 'width', 'height','class', 'xmin', 'ymin', 'xmax', 'ymax'])
num_rows = df.shape[0]
NUM_TRAIN_IMAGES = int(0.8 * num_rows)
indices = np.random.permutation(num_rows)
training_idx, test_idx = indices[:NUM_TRAIN_IMAGES], indices[NUM_TRAIN_IMAGES:]
train_df, test_df = df.iloc[training_idx], df.iloc[test_idx]

def copy_files(dataframe, folder_path):
    for i, row in enumerate(tqdm(dataframe.iterrows())):
        source_path = row[1][0]
        image_name = "_".join(source_path.split("/")[-2:])
        dest_path = os.path.join(folder_path,image_name)
        shutil.copyfile(source_path, dest_path)
# copy_files(train_df, TRAIN_FOLDER)
# copy_files(test_df, TEST_FOLDER)

train_df["imagePath"] = train_df["imagePath"].apply(lambda x: "_".join(x.split("/")[-2:]))
test_df["imagePath"] = test_df["imagePath"].apply(lambda x: "_".join(x.split("/")[-2:]))
train_df.columns = RENAMED_COLUMNS
test_df.columns = RENAMED_COLUMNS

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)