import time
import torch
import subprocess
import sys
import os
import numpy as np
import shutil

from model import EAST
from detect import detect_dataset


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)

 
 def parse_args():
	if len(sys.argv) != 3: 
		raise ValueError(f"Usage: dir_input destination")
	args = sys.argv[1:]
	file_in, destination = args

	if not os.path.exists(file_in): 
		raise ValueError(f"Input path does not exists: {file_in}")
	if os.path.exists(destination): 
		raise ValueError(f"Output destination already exists: {destination}")
	
	if not os.path.isdir(file_in): 
		raise ValueError("Input path must be a directory")
		
	return file_in, destination 

def main(): 
	model_name = './pths/east_vgg16.pth'
	input_dir, output_dir = parse_args()
	eval_model(model_name, input_dir, output_dir)
	
 
if __name__ == '__main__': 
	main()
