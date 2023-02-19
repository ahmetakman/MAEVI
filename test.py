import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
import torchvision.transforms
from tqdm import tqdm

from event_voxel_non_sparse import EventVoxel

#from event_voxel import EventVoxel


import config
from common import myutils
import torchvision.utils as utils
import math
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model.MAEVI import UNet_3D_3D

from git import Repo
import datetime

local_repo = Repo(path="")
local_branch = local_repo.active_branch.name
marker = datetime.datetime.now()
save_loc = os.path.join("results",local_branch,str(marker.day)+"_"+str(marker.month))

if not os.path.exists(save_loc):
    os.makedirs(save_loc)


##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"]='0'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# for sparse
# test_set = EventVoxel(args.data_root, args.sparse_data_root, is_hsergb=False, is_training=False, is_validation=False, number_of_time_bins= args.voxel_grid_size)
# test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


#for non-sparse
test_set = EventVoxel(args.data_root, is_hsergb=False, is_training=False, is_validation=False, number_of_time_bins= args.voxel_grid_size)
test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



print("Building model: %s"%args.model)
model = UNet_3D_3D(n_inputs=args.nbr_frame, joinType=args.joinType)

model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))

myTransform = torchvision.transforms.ToPILImage()

def save_image(recovery, image_name):
    recovery_image = torch.split(recovery, 1, dim=0)
    batch_num = len(recovery_image)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    for ind in range(batch_num):
        sequence = image_name[ind].split("/")[-3]
        number = image_name[ind].split("/")[-1].split(".")[0]
        
        if not os.path.exists(save_loc+'/{}'.format(sequence)):
            os.makedirs(save_loc+'/{}'.format(sequence))
        utils.save_image(recovery_image[ind], save_loc+'/{}/{}.png'.format(sequence,number))

def to_psnr(rect, gt):
    mse = F.mse_loss(rect, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    psnr_list = [-10.0 * math.log10(mse) for mse in mse_list]
    return psnr_list

def test(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    with torch.no_grad():
        for i, (images, voxel, gt_image ,paths) in enumerate(tqdm(test_loader)):
            images = [img_.cuda() for img_ in images]
            gt = gt_image.cuda()

            torch.cuda.synchronize()
            start_time = time.time()
            out = model(images, voxel)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)


            save_image(out,paths)
            myutils.eval_metrics(out, gt, psnrs, ssims)

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , " , sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.load_from is not None

    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
