import argparse
import os
import sys
from shutil import copyfile

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.models import DIFNet, DIFNet2
from models.pwcNet import PwcNet
from metrics import metrics
from frame2vid import frame2vid

from PIL import Image
import numpy as np
import math
import pdb
import time

#python run_seq2.py --cuda --n_iter 3 --skip 2


parser = argparse.ArgumentParser()
parser.add_argument('--model1', default='./trained_models/DIFNet2.pth') ####2
parser.add_argument('--in_file', default='./data/Stab_te_reg/07/')
parser.add_argument('--out_file', default='./output/OurStabReg2/07/')
parser.add_argument('--n_iter', type=int, default=1, help='number of stabilization interations')
parser.add_argument('--skip', type=int, default=1, help='number of frame skips for interpolation')
#parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
#parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
#parser.add_argument('--height', type=int, default=720, help='size of the data crop (squared assumed)')
#parser.add_argument('--width', type=int, default=1280, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

##########################################################

### Networks
DIFNet = DIFNet2()

# Place Network in cuda memory
if opt.cuda:
	DIFNet.cuda()

### DataParallel
DIFNet = nn.DataParallel(DIFNet)
DIFNet.load_state_dict(torch.load(opt.model1))
DIFNet.eval()

##########################################################

frameList = os.listdir(opt.in_file)
frameList.sort()

if os.path.exists(opt.out_file):
	copyfile(opt.in_file + frameList[0], opt.out_file + frameList[0])
	copyfile(opt.in_file + frameList[-1], opt.out_file + frameList[-1])
else:
	os.makedirs(opt.out_file)
	copyfile(opt.in_file + frameList[0], opt.out_file + frameList[0])
	copyfile(opt.in_file + frameList[-1], opt.out_file + frameList[-1])
#end

### Generate output sequence
for num_iter in range(opt.n_iter):
	idx = 1
	print('\nIter: ' + str(num_iter+1))
	for f in frameList[1:-1]:
		if f.endswith('.png'):
			if num_iter == 0:
				src = opt.in_file
			else:
				src = opt.out_file
			#end

			if idx < opt.skip or idx > (len(frameList)-1-opt.skip):
				skip = 1
			else:
				skip = opt.skip
			#end

			fr_g1 = torch.cuda.FloatTensor(np.array(Image.open(opt.out_file + f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
			#fr_g2 = torch.cuda.FloatTensor(np.array(Image.open(src + f)).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
			fr_g3 = torch.cuda.FloatTensor(np.array(Image.open(src + f[:-9] + '%05d.png' % (int(f[-9:-4])+skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)

			#fr_o1 = torch.cuda.FloatTensor(np.array(Image.open(opt.in_file + f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
			fr_o2 = torch.cuda.FloatTensor(np.array(Image.open(opt.in_file + f)).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
			#fr_o3 = torch.cuda.FloatTensor(np.array(Image.open(opt.in_file + f[:-9] + '%05d.png' % (int(f[-9:-4])+skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)

			with torch.no_grad():
				fhat, I_int = DIFNet(fr_g1, fr_g3, fr_o2, fr_g3, fr_g1, 0.5) # Notice 0.5

			# Save image
			#img = Image.fromarray(np.uint8(fhat.cpu().squeeze().permute(1,2,0)*255))
			#img.save(opt.out_file + f)

			sys.stdout.write('\rFrame: ' + str(idx) + '/' + str(len(frameList)-2))
			sys.stdout.flush()

			idx += 1
		#end
	#end
#end

### Make video
#print('\nMaking video...')
#frame2vid(src=opt.out_file, vidDir=opt.out_file[:-1] + '.avi')

### Assess with metrics
#print('\nComputing metrics...')
#metrics(in_src=opt.in_file, out_src=opt.out_file)
