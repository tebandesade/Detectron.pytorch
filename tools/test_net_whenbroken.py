import sys
import _init_paths
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
import pickle


def open_pkl(file_):
	with open(file_, 'rb') as f:
		data = pickle.load(f)
	return data


dataset_name = sys.argv[1]

detection    = open_pkl(sys.argv[2])

all_boxes    = detection['all_boxes']
all_segms    = detection['all_segms']
all_keyps    = detection['all_keyps']
output_dir   = sys.argv[3]

dataset = JsonDataset(dataset_name) 
results = task_evaluation.evaluate_all(dataset, all_boxes, all_segms, all_keyps, output_dir)