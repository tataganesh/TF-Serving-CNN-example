import csv
import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm

TRAIN_FOLDER = 'images/train'
TEST_FOLDER = 'images/test'
DATASET = 'dataset.csv'
df = pd.read_csv(DATASET, names=["image_path", "height", "width", "x1", "y1", "x2", "y2"])
num_rows = df.shape[0]
NUM_TRAIN_IMAGES = int(0.8 * num_rows)
indices = np.random.permutation(num_rows)
training_idx, test_idx = indices[:NUM_TRAIN_IMAGES], indices[NUM_TRAIN_IMAGES:]
train_df, test_df = df.iloc[training_idx], df.iloc[test_idx]

def copy_files(dataframe, folder_path):
    for i, row in tqdm(enumerate(dataframe.iterrows())):
        source_path = row[1][0]
        image_name = "_".join(source_path.split("/")[-2:])
        dest_path = os.path.join(folder_path,image_name)
        shutil.copyfile(source_path, dest_path)
copy_files(train_df, TRAIN_FOLDER)
copy_files(test_df, TEST_FOLDER)

train_df["image_path"] = train_df["image_path"].apply(lambda x: os.path.join(TRAIN_FOLDER, "_".join(x.split("/")[-2:])))
test_df["image_path"] = test_df["image_path"].apply(lambda x: os.path.join(TEST_FOLDER, "_".join(x.split("/")[-2:])))
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)