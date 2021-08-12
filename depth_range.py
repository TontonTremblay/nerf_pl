import glob
import os
import cv2

def read_exr(filename):
    return cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

root_dir = './falling_google_1/'

all_depths = glob.glob(root_dir+'*.depth.exr')
depth_min = 10000
depth_max = 0
for i in range(len(all_depths)):
    depth = read_exr(all_depths[i])

    mask = cv2.imread(all_depths[i].replace('depth','seg'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mask[mask == mask.max()] = 0
    mask[mask > 0] = 255
    mask = (mask[:,:,0] > 0)
    depth_i = depth[mask]
 
    depth_i_min = depth_i.min()
    depth_i_max = depth_i.max()
    depth_max = depth_i_max if depth_i_max > depth_max else depth_max
    depth_min = depth_i_min if depth_i_min < depth_min else depth_min

print("min:", depth_min)
print("max:", depth_max)
