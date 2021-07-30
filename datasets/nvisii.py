import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

try:
    from .ray_utils import *
except :
    from ray_utils import *

import glob
import cv2

class NvisiiDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):

        # with open(os.path.join(self.root_dir,
        #                        f"transforms_{self.split}.json"), 'r') as f:
        #     self.meta = json.load(f)


        json_files = sorted(glob.glob(os.path.join(self.root_dir, f'*.json')))
        self.json_files = json_files
        if self.split == 'train':
            # json_files = json_files[0:2]
            json_files = json_files[0:100]
        elif self.split == 'val':            
            # json_files = json_files[100:101]
            json_files = json_files[100:105]
        elif self.split == 'test':
            json_files = json_files[105:]

        transforms = []
        for i_index, json_file in enumerate(json_files):   
            with open(json_file, 'r') as f:
                meta = json.load(f)   
                transforms.append(np.array(meta['camera_data']['cam2world']).T)
            # pass



        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*0.785398) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 0.3
        self.far = 2
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        # if self.split == 'train': # create buffer of all rays and rgb data
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        for i_trans,trans in enumerate(transforms):
            pose = trans[:3,:4]
            # print(trans)
            # print(trans[:3,:4])
            self.poses += [torch.FloatTensor(pose)]
            c2w = self.poses[-1]
            # print(c2w)

            image_path = json_files[i_trans].replace("json",'png')
            self.image_paths += [image_path]
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)


            # load the mask
            mask_path = json_files[i_trans].replace("json",'seg.exr')
            mask = cv2.imread(mask_path,  
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
            mask = cv2.resize(mask,self.img_wh,cv2.INTER_NEAREST)
            mask[mask == mask.max()] = 0
            mask[mask > 0] = 255
            mask = mask.astype(np.uint8)
            img = Image.fromarray(np.concatenate([np.array(img), mask[:, :, 0:1]], -1), 'RGBA')

            img = self.transform(img) # (4, h, w)
            # print(img.shape)
            # raise()
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            self.all_rgbs.append(img)
            
            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
            
            self.all_rays += [torch.cat([rays_o, rays_d, 
                                         self.near*torch.ones_like(rays_o[:, :1]),
                                         self.far*torch.ones_like(rays_o[:, :1])],
                                         1)] # (h*w, 8)

        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return len(self.poses)
            # return 8 # only validate 8 images (to support <=8 gpus)
        # return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            c2w = self.poses[idx]


            image_path = self.image_paths[idx]
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            # load the mask
            mask_path = image_path.replace("json",'seg.exr')
            mask = cv2.imread(mask_path,  
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
            mask = cv2.resize(mask,self.img_wh,cv2.INTER_NEAREST)
            mask[mask == mask.max()] = 0
            mask[mask > 0] = 255
            mask = mask.astype(np.uint8)
            img = Image.fromarray(np.concatenate([np.array(img), mask[:, :, 0:1]], -1), 'RGBA')
            img = self.transform(img) # (4, h, w)
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            valid_mask = (torch.tensor(mask)[:,:,0]>0).flatten()

            # print(valid_mask.min(),valid_mask.max(),valid_mask.shape)


            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            print(rays.min(),rays.max(),rays.shape)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample


if __name__ == '__main__':

    def visualize_ray(camera_position, ray_dirs):
        fig = plt.figure(figsize=(12,8))

        print(camera_position.unique())
        print(ray_dirs.shape)

        ax = fig.add_subplot(111, projection='3d')
        for i in range(50):
            i_sample = np.random.randint(0,camera_position.shape[0]-1)
            end_point = camera_position[i] + 0.5 * ray_dirs[i]
            ax.plot([camera_position[i][0], end_point[0]], 
                    [camera_position[i][1], end_point[1]], 
                    zs=[camera_position[i][2], end_point[2]])
        plt.show()

    train_ds = NvisiiDataset(
        root_dir='/home/titans/code/nerf_pytorch/data_tmp/falling_google_1/', 
        split='val', 
        img_wh=(400, 400)
    )

    train_ds[0]['rays']