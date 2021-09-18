import glob
import os
import cv2
import json
import numpy as np

def read_exr(filename):
    return cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

#root_dir = '/home/trump/raid/data/32_FBX_scaled/*/'
#root_dir = '/home/trump/raid/data/falling_google_scenes/*/'
#root_dir = '/home/trump/raid/data/32_FBX/Ambulance/'
root_dir = '/home/trump/raid/code/nvisii_mvs/clean/output/*/'

all_depths = glob.glob(root_dir+'*.depth.exr')
depth_min = 10000
depth_max = 0
for i in range(len(all_depths)):
    depth = read_exr(all_depths[i])

    json_file = all_depths[i].replace("depth.exr", "json")
    with open(json_file, 'r') as f:
        meta = json.load(f)

    mask = cv2.imread(all_depths[i].replace('depth','seg'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mask[mask < -3.4028235e+37] = 0
    mask[mask > 3.4028235e+37] = 0

    # remove the ground
    # seg_ids = []
    # for i in range(len(meta['objects'])):
    #     seg_id = meta['objects'][i]['segmentation_id']
    #     seg_ids.append(seg_id)
    # if 0 in mask:
    #     seg_ids.append(0)
    # bg_ids = list(set(np.unique(mask).tolist()) - set(seg_ids))
    # assert len(bg_ids) == 1, "more than one floor segmentation ids / no floor found"
    # mask[mask == bg_ids[0]] = 0
    mask = (mask[:,:,0] > 0)

    #import pdb; pdb.set_trace()
    #mask[mask == mask.max()] = 0
    #mask[mask > 0] = 255
    #mask = (mask[:,:,0] > 0)
    depth_i = depth[mask]
 
    depth_i_min = depth_i.min()
    depth_i_max = depth_i.max()

    # print("depth max is {} for {}".format(depth_i_max, all_depths[i]))
    # if depth_i_max > depth_max:
    #     print("depth max updated. new max {} from {}".format(depth_i_max, all_depths[i]))

    depth_max = depth_i_max if depth_i_max > depth_max else depth_max
    depth_min = depth_i_min if depth_i_min < depth_min else depth_min

print("min:", depth_min)
print("max:", depth_max)
