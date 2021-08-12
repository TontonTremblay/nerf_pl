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

class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta['frames']:
                # print(frame['transform_matrix'])

                pose = np.array(frame['transform_matrix'])[:3, :4]
                # print(pose)
                # raise()
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                # print(self.all_rays)
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            # print(c2w)
            # raise()
            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            # print(valid_mask.min(),valid_mask.max(),valid_mask.shape)
            print(rays)
            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import random 
    def visualize_ray(camera_position, ray_dirs,rgbs):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        # print(np.unique(camera_position))
        for i in range(50):
            # ii_pick = random.randint(0,len(camera_position)-1)
            # i_sample = np.random.randint(0,camera_position.shape[0]-1)
            end_point = camera_position[i] + 2 * ray_dirs[i]
            # end_point = ray_dirs[i]
            ax.plot([camera_position[i][0], end_point[0]], 
                    [camera_position[i][1], end_point[1]], 
                    zs=[camera_position[i][2], end_point[2]])
            ax.scatter([camera_position[i][0]], 
                    [camera_position[i][1]], 
                    zs=[camera_position[i][2]],c=[rgbs[i].tolist()])

        plt.show()

    # train_ds = BlenderDataset(
    #     root_dir='../nerf_synthetic/lego/', 
    #     split='val', 
    #     img_wh=(800, 800)
    # )

    train_ds = BlenderDataset(
        root_dir='/home/jtremblay/Downloads/lego-20210727T000712Z-001/lego/', 
        split='train', 
        img_wh=(400, 400)
    )
    train_ds[0]
    # print(train_ds[0]['rays'])
    
    # for i in range(len(train_ds)):
    #     item = train_ds[i]
    #     c2w = item["c2w"]
    #     c2w = torch.cat((c2w, torch.FloatTensor([[0, 0, 0, 1]])), dim=0)
    #     #np.save("blender_c2ws/c2w{}.npy".format(i), c2w.numpy())
    
    cam_pos = []
    ray_end = []
    rgbs = []

    for ii in range(100):
        i = random.randint(0,len(train_ds)-1)
        
        data = train_ds[i]['rays']
        rgbs.append(train_ds[i]['rgbs'])
        cam_pos.append([data[0],data[1],data[2],])
        ray_end.append([data[3],data[4],data[5],])
    # print(train_ds[0]['rays']

    visualize_ray(np.array(cam_pos),np.array(ray_end),np.array(rgbs))
