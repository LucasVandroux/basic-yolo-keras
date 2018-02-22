#! /usr/bin/env python

import argparse
import os
import os.path as path
import cv2
import numpy as np
from tqdm import tqdm, trange
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import csv

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format) or a folder')

argparser.add_argument(
    '-p',
    '--predictionPath',
    default=False,
    help='path to folder where to export the prediction and images.')

argparser.add_argument(
    '-s',
    '--saveImages',
    default=1,
    help='save or not the images with the predictions (0/1)')

def predict_single_image(yolo, img_path, labels):
    image = cv2.imread(img_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, labels)

    pred_list = []
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])
        label = labels[box.get_label()].encode("utf-8")
        score = box.get_score()
        pred_list.append([label, xmin, ymin, xmax, ymax, score])

    return image, pred_list

def _main_(args):

    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    save_images  = int(args.saveImages)

    if not args.predictionPath:
        pred_path = path.dirname(image_path)
    else:
        pred_path = args.predictionPath

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'),
                               50.0,
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()

    elif path.isdir(image_path):
        # list the pictures in the folder
        list_img = [filename for filename in os.listdir(image_path) if filename.endswith(('.png', '.jpg'))]

        # predict for each pictures
        for idx in trange(len(list_img)):
            img_name = list_img[idx]
            img_path = path.join(image_path, img_name)
            im, list_labels = predict_single_image(yolo, img_path, config['model']['labels'])

            if save_images:
                if args.predictionPath:
                    img_export_path = path.join(pred_path, img_name)
                else:
                    img_export_path = path.join(image_path, img_name[:-4] + '_detected' + img_name[-4:])

                print(img_export_path)
                cv2.imwrite(img_export_path, im)

            # save prediction in .csv file
            csv_path = path.join(pred_path, img_name[:-4] + '.csv')
            with open(csv_path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerows(list_labels)

    elif path.exists(image_path):
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        pred_list = []
        for box in boxes:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])
            label = config['model']['labels'][box.get_label()].encode("utf-8")
            score = box.get_score()
            pred_list.append([label, xmin, ymin, xmax, ymax, score])

        csv_path = path.join(pred_path, img_name[:-4] + '.csv')
        with open(pred_path, "w") as csv_file: # TODO change file name
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(pred_list)
            # pred_list.append([class_mapping[cls_num]] + b.tolist())

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
    else:
        print('ERROR: could not find ' + image_path)
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
