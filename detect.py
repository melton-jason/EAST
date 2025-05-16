import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
import os
import sys
from dataset import get_rotate_mat
import numpy as np
import lanms
import time
import argparse


def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		model_start = time.time()
		score, geo = model(load_pil(img).to(device))
		model_time = time.time() - model_start
	box_start = time.time()
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	box_time = time.time() - box_start
	return adjust_ratio(boxes, ratio_w, ratio_h), model_time, box_time 


def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img


def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		boxes, model_time, box_time = detect(Image.open(img_file), model, device)
		seq = []
		if boxes is not None:
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)

def process_single(model, device, img_path, out_path):
	img = Image.open(img_path)
	print(f"Processing bounding boxes for: {img_path}")
	boxes, model_time, box_time = detect(img, model, device)
	print(f"Model evaluation time: {model_time} seconds")
	print(f"Bounding box evaluation time: {box_time} seconds")
	print(f"Total text bounding box evaluation time: {model_time + box_time} seconds")
	print("Plotting boxes to image copy")
	plotted_bxs = plot_boxes(img, boxes)
	print(f"Saving outputput to: {out_path}")
	plotted_bxs.save(out_path)
	return model_time, box_time


def procces_batch(model, device, dir_path, output_path):
	os.mkdir(output_path)
	model_times = []
	box_times = []
	result_times = []
	with os.scandir(dir_path) as entries: 
		for entry in entries: 
			if entry.is_file():
				image_out_path = os.path.join(output_path, f"{'.'.join(entry.name.split('.')[:-1])}.bmp")
				model_time, box_time = process_single(model, device, entry.path, image_out_path)
				model_times.append(model_time)
				box_times.append(box_time)
				result_times.append(model_time + box_time)
	
	print(f"Finished plotting bounding boxes for {dir_path}")
	print("Model Times")
	total_time, average_time, num_images, min_time, max_time = get_statistics(model_times)
	print(f"Number of images: {num_images}")
	print(f"Evaluation time: {total_time}")
	print(f"Average time: {average_time}")
	print(f"Minimum time: {min_time}")
	print(f"Maximum time: {max_time}\n")
	
	print("Bounding Box times")
	total_time, average_time, num_images, min_time, max_time = get_statistics(box_times)
	print(f"Number of images: {num_images}")
	print(f"Evaluation time: {total_time}")
	print(f"Average time: {average_time}")
	print(f"Minimum time: {min_time}")
	print(f"Maximum time: {max_time}\n")
	
	print("Overall Times")
	total_time, average_time, num_images, min_time, max_time = get_statistics(result_times)
	print(f"Number of images: {num_images}")
	print(f"Evaluation time: {total_time}")
	print(f"Average time: {average_time}")
	print(f"Minimum time: {min_time}")
	print(f"Maximum time: {max_time}")
	
	
def get_statistics(input_list): 
	minimum = 999999999
	maximum = 0
	length = 0
	running_sum = 0
	for number in input_list: 
		if number < minimum: 
			minimum = number
		elif number > maximum: 
			maximum = number
		
		running_sum += number
		length += 1
	average = running_sum / length
	return running_sum, average, length, minimum, maximum 
	
	

MODES = ('single', 'batch')
def parse_args():
	parser = argparse.ArgumentParser(description="Given an image or a directory of images, detect text in the image and generate images with colored bounding boxes around text")
	
	parser.add_argument('--mode',required=True , choices=MODES)
	parser.add_argument('--input', help="The input image or directory. Expected directory if --mode is batch and single file if --mode is single", required=True)
	parser.add_argument('--out', help="The name of the output directory or file. If --mode is single, the outputted image will be in bitmap format. Otherwise, the output will be a directory", required=True)
	parser.add_argument('--quantize', default=True, action='store_true')
	parser.add_argument('--no-quantize', dest='quantize', action='store_false')
	
	args = parser.parse_args()
	
	mode = args.mode
	file_in = args.input
	destination = args.out
	quantize = args.quantize
	
	if not os.path.exists(file_in): 
		raise ValueError(f"Input path does not exists: {file_in}")
	if os.path.exists(destination): 
		raise ValueError(f"Output destination already exists: {destination}")
	
	if mode == 'batch' and not os.path.isdir(file_in): 
		raise ValueError("Input path must be a directory for batch mode")
	
	if mode == 'single' and not os.path.isfile(file_in): 
		raise ValueError("Input path must be a file in single mode")
	
		
	return mode, file_in, destination, quantize

def init_model(quantize):
	model_path  = './pths/east_vgg16.pth' 
	device = torch.device("cpu")

	model = EAST(is_quant=quantize).to(device)

	if quantize:
		model.prepare_for_quantization()   # must include quant/dequant stubs
		torch.quantization.convert(model, inplace=True)  # convert BEFORE loading
		model_path = './pths_quantized/east_quantized.pth'

	model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
	model.eval()
	return model, device

def main(): 
	mode, file_in, destination, quantize = parse_args()
	model, device = init_model(quantize)
	if mode == 'single': 
		process_single(model, device, file_in, destination)
	elif mode == 'batch':
		procces_batch(model, device, file_in, destination)


if __name__ == '__main__':
	"""
	Example usage: python3 detect.py --mode batch --input ../ICDAR_2015/demo --out ./results
	"""
	main()


