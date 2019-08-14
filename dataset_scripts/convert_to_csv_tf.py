import pandas as pd
column_names = ['filename', 'width', 'height','class', 'xmin', 'ymin', 'xmax', 'ymax']

def convert_to_csv(file_path, dataset_type):
    df = pd.read_csv(file_path)   
    df['image_path'] = df['image_path'].apply(lambda x: x.split("/")[-1])
    df['class'] = "hand"
    df.columns = ['filename', 'height', 'width', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    df = df[column_names]
    df.to_csv(dataset_type + ".csv", index=False)

convert_to_csv("train.csv", "images/train")
convert_to_csv("test.csv", "images/test")
