import time
import torch
import subprocess
import sys
import os
import numpy as np
import shutil
import sys
import argparse

from model import EAST
from detect import detect_dataset


def eval_model(model_name, test_img_path, validation_zip, submit_path, save_flag=True, quantize=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = EAST(False, is_quant=quantize).to(device)

	if quantize:
		model.prepare_for_quantization()   # must include quant/dequant stubs
		torch.quantization.convert(model, inplace=True)  # convert BEFORE loading
		model_name = './pths_quantized/east_quantized.pth'

	model.load_state_dict(torch.load(model_name, map_location=device))
	model.eval()
	
	base_dir = os.getcwd()
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput(f'zip -q {os.path.join(base_dir, "submit.zip")} *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir(base_dir)
	res = subprocess.getoutput(f'python ./evaluate/script.py –g={validation_zip} –s=./submit.zip')
	print(res)
	os.remove(os.path.join(base_dir, "submit.zip"))
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)

 
def parse_args():
	
	parser = argparse.ArgumentParser(description="Evaluate the accuracy of the model")
	
	parser.add_argument('--expected', help="A path to a zip file containing files which contain the true bounding boxes for text", required=True)
	parser.add_argument('--input', help="The path to a directory of images which will be evaluated", required=True)
	parser.add_argument('--out', help="The desired name of the output directory. Will contain the generated bounding boxes for text in the images in text format", required=True)
	parser.add_argument('--quantize', default=True, action='store_true')
	parser.add_argument('--no-quantize', dest='quantize', action='store_false')
	
	args = parser.parse_args()
	
	file_in = args.input
	validation_zip = args.expected
	destination = args.out
	quantize = args.quantize

	if not os.path.exists(file_in): 
		raise ValueError(f"Input path does not exists: {file_in}")
	if not os.path.exists(validation_zip): 
		raise ValueError(f"Validation zip does not exists: {validation_zip}")
	if os.path.exists(destination): 
		raise ValueError(f"Output destination already exists: {destination}")

	if not os.path.isdir(file_in): 
		raise ValueError("Input path must be a directory")
		
	return file_in, validation_zip, destination, quantize 

def main(): 
	"""
	Example usage: python3 eval.py --input ../ICDAR_2015/test_img/ --expected ./evaluate/gt.zip --out eval_results
	"""
	
	model_name = './pths/east_vgg16.pth'
	input_dir, validation_zip, output_dir, quantize = parse_args()

	eval_model(model_name, input_dir, validation_zip, output_dir, quantize=quantize)
	
 
if __name__ == '__main__': 
	main()
