import scipy.io as sio
import IPython
import cv2
import os
import numpy as np
import csv
import config
from tqdm import tqdm

# mat['video'][0] - Each of the 48 element list contains 7 sub elements as mentioned in the README
# Each sample has  - folder_name, partner folder, viewer_label, partner_label, setting, activity 
# Index 6 has shape (1, 100) - Containing framde_id and hand_segmentation 
# Frame number, _, _, segment1, segment2
CLASS_NAME = 'hand'

def get_bbox_from_mask(segmentation_mask):
    x1 = int(np.amin(segmentation_mask[:, 0]))
    x2 = int(np.amax(segmentation_mask[:, 0]))
    y1 = int(np.amin(segmentation_mask[:, 1]))
    y2 = int(np.amax(segmentation_mask[:, 1]))
    return x1, y1, x2, y2



def create_dataset(mat, csv_writer):
    for sample in tqdm(mat['video'][0]): 
        folder_name = sample[0][0]
        all_annotations = sample[6][0]
        base_path = os.path.join(config.IMAGES_FOLDER_PATH, folder_name)
        image_paths = sorted(os.listdir(base_path)) # Sorted to get in the same order as annotations
        for i, annotation in enumerate(all_annotations):
            image_path = os.path.join(base_path, image_paths[i])
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            frame_number = annotation[0][0][0]
            # Handle cases where only one of the hands is present 
            if annotation[3].shape[0] != 0:
                x1, y1, x2, y2 = get_bbox_from_mask(annotation[3])
                data_row = [image_path, height, width, x1, y1, x2, y2]
                csv_writer.writerow(data_row)
                # cv2.rectangle(image, hand1_bbox[:2], hand1_bbox[2:4], (0, 255, 0))
            if annotation[4].shape[0] != 0:
                x1, y1, x2, y2 = get_bbox_from_mask(annotation[4])
                data_row = [image_path, width, height, CLASS_NAME, x1, y1, x2, y2]
                csv_writer.writerow(data_row)
                # cv2.rectangle(image, hand2_bbox[:2], hand2_bbox[2:4], (0, 255, 0))
            # cv2.imshow("image", image)
            # cv2.waitKey(0)


mat = sio.loadmat(config.DATASET_METADATA_PATH)

csv_file = open('dataset.csv', 'wb')
csv_writer = csv.writer(csv_file)
create_dataset(mat, csv_writer)
csv_file.close()


