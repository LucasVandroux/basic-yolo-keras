from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.show_frame import show_frame
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import os.path as path
import json
import imageio
from xml.etree import cElementTree as ElementTree
from tqdm import tqdm, trange

def import_pred(file_path, class_mapping, im_shape):
    with open(file_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        list_row = [row for row in csvreader]
        row_count = len(list_row)

        pred_bb = np.zeros((row_count, 4))
        pred_cls = np.zeros(row_count)
        pred_conf = np.zeros(row_count)

        for idx, row in enumerate(list_row):
            pred_cls[idx] = int(class_mapping[row[0]][1])

            pred_x1 = float(row[1]) / im_shape[1]
            pred_y1 = float(row[2]) / im_shape[0]
            pred_x2 = float(row[3]) / im_shape[1]
            pred_y2 = float(row[4]) / im_shape[0]
            pred_bb[idx,:] = np.array([pred_x1, pred_y1, pred_x2, pred_y2])

            pred_conf[idx] = float(row[5])

    return pred_bb, pred_cls, pred_conf



class_mapping = {
      "Person_sitting": [4, 3],
      "Cyclist": [5, 5],
      "Pedestrian": [3, 3],
      "Van": [1, 0],
      "Truck": [2, 2],
      "Misc": [7, 7],
      "DontCare": [8, 8],
      "Car": [0, 0],
      "Tram": [6, 6]
    }
# TODO detecting a DontCare doesn't influence the mAP
# TODO convert as not all the classes are the same (Person_sitting and Pedestrian are the same)
# TODO save images
# TODO get the variables in arguments for the main function

if __name__ == '__main__':
    showFrame = False
    im_ext = '.png'
    pred_ext = '.csv'
    gt_ext = '.xml'

    path_im = '/Users/lucas/Desktop/yolo_test_mAP/images'
    path_pred = '/Users/lucas/Desktop/yolo_test_mAP/predictions'
    path_gt = '/Users/lucas/Desktop/yolo_test_mAP/labels'

    path_savefig = '/Users/lucas/Desktop/yolo_test_mAP/pr_curve.png'

    list_frames = []

    # Create class mapping for testing predictions
    n_class = int(len(set([item[1] for item in class_mapping.values()])))
    class_mapping_int = {str(item[0]): int(item[1]) for item in class_mapping.values()}

    # Create mAP object
    mAP = DetectionMAP(n_class)

    # Import list images
    set_im = set([filename[:-4] for filename in os.listdir(path_im) if filename.endswith(im_ext)])

    # Import list ground truth
    set_gt = set([filename[:-4] for filename in os.listdir(path_gt) if filename.endswith(gt_ext)])

    # Import list predictions
    set_pred = set([filename[:-4] for filename in os.listdir(path_pred) if filename.endswith(pred_ext)])

    list_files = list(set_im.intersection(set_gt).intersection(set_pred))
    list_files.sort()

    # print(list_files)

    for idx in trange(len(list_files)):
        filename = list_files[idx]
        # Create path to different files
        im_path = path.join(path_im, filename + im_ext)
        gt_path = path.join(path_gt, filename + gt_ext)
        pr_path = path.join(path_pred, filename + pred_ext)

        # Import gt from .xml file
        tree = ElementTree.parse(gt_path)
        root = tree.getroot()

        # Get image dimension from xml file
        im_size = root.find('size')
        im_height = int(im_size.find('height').text)
        im_width = int(im_size.find('width').text)
        # print(im_height, im_width)

        # Get gt from xml file
        num_obj = len(root.findall('object'))
        gt_cls = np.zeros(num_obj)
        gt_bb = np.zeros((num_obj, 4))

        for idx, bbox in enumerate(root.iter('object')):
            # Get object class
            gt_class = bbox.find('name').text
            if gt_class == 'car':
                gt_class = 'Car'
            if gt_class == 'person':
                gt_class = 'Pedestrian'

            gt_cls[idx] = int(class_mapping[gt_class][1])

            # Get bbox dimensions
            bbox_dim = bbox.find('bndbox')
            gt_x1 = float(bbox_dim.find('xmin').text) / im_width
            gt_y1 = float(bbox_dim.find('ymin').text) / im_height
            gt_x2 = float(bbox_dim.find('xmax').text) / im_width
            gt_y2 = float(bbox_dim.find('ymax').text) / im_height

            gt_bb[idx, :] = np.array([gt_x1, gt_y1, gt_x2, gt_y2])

        # Get pred from csv file
        with open(pr_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            list_row = [row for row in csvreader]
            row_count = len(list_row)

            pred_bb = np.zeros((row_count, 4))
            pred_cls = np.zeros(row_count)
            pred_conf = np.zeros(row_count)

            for idx, row in enumerate(list_row):
                pred_cls[idx] = int(class_mapping[row[0]][1])

                pred_x1 = float(row[1]) / im_width
                pred_y1 = float(row[2]) / im_height
                pred_x2 = float(row[3]) / im_width
                pred_y2 = float(row[4]) / im_height
                pred_bb[idx,:] = np.array([pred_x1, pred_y1, pred_x2, pred_y2])

                pred_conf[idx] = float(row[5])

        # Create frame
        frame = (pred_bb, pred_cls, pred_conf, gt_bb, gt_cls)

        # Show frame
        if showFrame:
            background_im = imageio.imread(im_path).astype('uint8')
            show_frame(*frame, background=background_im, show_confidence=True)

        # Evaluate mAP on the frame
        mAP.evaluate(*frame)

    # Plot the mAP and save the file
    mAP.plot()
    plt.savefig(path_savefig)
    plt.show()
