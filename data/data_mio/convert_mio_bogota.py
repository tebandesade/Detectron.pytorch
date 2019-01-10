# Convert a detection model trained for COCO into a model that can be fine-tuned
# on bogota
#
# bogota_to_mio

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle

import argparse
import os
import sys
import numpy as np
import torch

import mio_to_bogota_id as cs
import sys

#sys.path.append('../../../tools/')
#import _init_paths
#from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg



#from modeling.model_builder import Generalized_RCNN

NUM_BGTA_CLS = 12
NUM_MIO_CLS = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a MIO pre-trained model wit COCO for use with Bogota')
    parser.add_argument(
        '--mio_model', dest='mio_model_file_name',
        help='Pretrained network weights file path',
        default=None, type=str)

    parser.add_argument(
        '--convert_func', dest='convert_func',
        help='Blob conversion function',
        default='bogota_to_mio', type=str)

    '''
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
        '''
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def convert_mio_blobs_to_bogota_blobs(model_dict):
    for k, v in model_dict['blobs'].items():
        if v.shape[0] == NUM_MIO_CLS or v.shape[0] == 4 * NUM_MIO_CLS:
            coco_blob = model_dict['blobs'][k]
            print(
                'Converting MIO blob {} with shape {}'.
                format(k, coco_blob.shape)
            )
            cs_blob = convert_mio_blob_to_bogota_blob(
                coco_blob, args.convert_func
            )
            print(' -> converted shape {}'.format(cs_blob.shape))
            model_dict['blobs'][k] = cs_blob


def convert_mio_blob_to_bogota_blob(coco_blob, convert_func):
    # coco blob (12, ...) or (12*4, ...)
    coco_shape = coco_blob.shape
    leading_factor = int(coco_shape[0] / NUM_MIO_CLS)
    tail_shape = list(coco_shape[1:])
    assert leading_factor == 1 or leading_factor == 4

    # Reshape in [num_classes, ...] form for easier manipulations
    coco_blob = coco_blob.reshape([NUM_MIO_CLS, -1] + tail_shape)
    # Default initialization uses Gaussian with mean and std to match the
    # existing parameters
    std = coco_blob.std()
    mean = coco_blob.mean()
    cs_shape = [NUM_BGTA_CLS] + list(coco_blob.shape[1:])
    cs_blob = (np.random.randn(*cs_shape) * std + mean).astype(np.float32)

    # Replace random parameters with COCO parameters if class mapping exists
    for i in range(NUM_BGTA_CLS):
        coco_cls_id = getattr(cs, convert_func)(i)
        if coco_cls_id >= 0:  # otherwise ignore (rand init)
            cs_blob[i] = coco_blob[coco_cls_id]

    cs_shape = [NUM_BGTA_CLS * leading_factor] + tail_shape
    return cs_blob.reshape(cs_shape)


def openPikle(file_):
    with open(file_, 'rb') as f:
        model_dict = pickle.load(f,encoding='latin1')
    return model_dict

def reset_bbox(k):
   # chunks = [k[x:x+4] for x in range(0, len(k), 4)]
    #changing motorized and no motirized to car
    k[20] , k[21], k[22], k[23] = k[12], k[13], k[14], k[15]
    k[24] , k[25], k[26], k[27] = k[12], k[13], k[14], k[15]
    #chunks[6] = chunks[4]
    #chunks[7] = chunks[4]

def reset_cls(k):
    k[6] = k[4]
    k[7] = k[4]

def iter_model(md):
    for k in md.values():
        if k.shape[0]==12:
            reset_cls(k)
        elif k.shape[0]==48:
            reset_bbox(k)  

def load_and_convert_mio_model(args):
    ##cfg_from_file(args.cfg_file)
    load_name = args.mio_model_file_name
    #maskRCNN = Generalized_RCNN()
    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    iter_model(model)

    save_name = load_name.split('/')[-1:][0]
    save_name = 'MiotoBgta_'+save_name
    torch.save({
        'step': checkpoint['step'],
        'train_size': checkpoint['train_size'],
        'batch_size': checkpoint['batch_size'],
        'model': model,
        'optimizer': checkpoint['optimizer']}, save_name)
    return True


if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.mio_model_file_name), \
        'Weights file does not exist'
    weights = load_and_convert_mio_model(args)
    print(weights)