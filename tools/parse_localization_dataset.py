# -*- coding: utf-8 -*-
'''
Date : January 2017

Authors : Zhiming Luo from the University of Sherbrooke

Description : code used to parse the MIO-TCD localization dataset, localize
            each image and save results in the proper csv format.  Please see
            http://tcd.miovision.com/ for more details on the dataset

Execution : simply type the following command in a terminal:

   >> python parse_localization_dataset.py ./train/ your_results_train.csv
or
   >> python parse_localization_dataset.py ./test/ your_results_test.csv


NOTE: this code was developed and tested with Python 3.5.2 and Linux
      (Ubuntu 14.04)

Disclamer:

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
from collections import defaultdict
from os import listdir
from os.path import join, splitext, basename
from scipy.misc import imread
import numpy as np
import csv
from tqdm import tqdm
import sys
import _init_paths
from modeling.model_builder import Generalized_RCNN
from core.test import im_detect_all


classes = ['articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle',
           'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian',
           'pickup_truck', 'single_unit_truck', 'work_van']


def localize_obj_in_image(path_to_image,maskRCNN, timers):
    '''
    Localize foreground objects in the image of 'path_to_image'.

    You may replace this function with a call to your localization method
    '''

    img = imread(path_to_image)
    base = basename(path_to_image)
    name = splitext(base)[0]

    h, w = img.shape[0:2]
    bbox = []
    cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, img, timers=timers)

    import pdb
    pdb.set_trace()
    for x in cls_boxes:
        print(x)
        break
    # randomly generate up to 5 bounding boxes for each image
    # with size between [50, 100)
    '''
    for k in range(np.random.randint(5)):

        label = classes[np.random.randint(11)]
        score = np.random.rand(1)[0]

        bb_w = np.random.randint(50, 100, 1)[0]  # width
        bb_h = np.random.randint(50, 100, 1)[0]  # height

        bb_x1 = np.random.randint(0, w-100, 1)[0]
        bb_y1 = np.random.randint(0, h-100, 1)[0]

        bb_x2 = bb_x1 + bb_w - 1
        bb_y2 = bb_y1 + bb_h - 1

        bbox.append([name, label, score, bb_x1, bb_y1, bb_x2, bb_y2])
    '''
    return bbox

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes

def parse_dataset(path_to_dataset):
    '''
    Parse every image contained in 'path_to_dataset' (a path to the training
    or testing set), classify each image and save in a csv file the resulting
    assignment

    dataset_result: dict structure returned by the function.  It contains the
            label of each image
    '''
    llist = listdir(path_to_dataset)
    dataset_result = {}

    maskRCNN = Generalized_RCNN()
    maskRCNN.cuda()
    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU
    maskRCNN.eval()

    for name in tqdm(llist):
        dn = join(path_to_dataset, name)
        timers = defaultdict(Timer)
        dataset_result[dn] = localize_obj_in_image(dn,maskRCNN,timers)

    return dataset_result


def save_localization_result(dataset_result, output_csv_file_name):
    '''
    Save the dataset_result (a dict structure containing the class of every image)
    into a valid csv file.
    '''

    csvfile = open(output_csv_file_name, 'w')
    writer = csv.writer(csvfile, delimiter=',')

    for file_number in dataset_result.keys():
        for bbox in dataset_result[file_number]:
            writer.writerow(bbox)

    csvfile.close()

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("\nUsage : \n\t python parse_localization_dataset.py PATH OUTPUT_CSV_FILE_NAME\n")
        print("\t PATH : path to the training or the testing dataset")
        print("\t OUTPUT_CSV_FILE_NAME : name of the resulting csv file\n")
    else:
        print('\nProcessing: ', sys.argv[1], '\n')

        dataset_result = parse_dataset(sys.argv[1])
        save_localization_result(dataset_result, sys.argv[2])