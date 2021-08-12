import os
import torch
import numpy as np
import imageio
from tqdm import tqdm

from models.nerf import *
from datasets.ray_utils import *
from datasets.depth_utils import *
from utils import load_ckpt
from eval import batched_inference, get_opts

args = get_opts()
w, h = args.img_wh

# target extrinsics
with open('extrinsics.txt') as f:
    lines = f.readlines()

all_ext_mat = []
for line in lines:
    line = line.rstrip()
    ext_mat = []
    for elem in line.split(','):
        num = float(re.sub('[\[ \]]', '', elem))
        ext_mat.append(num)
    ext_mat = np.array(ext_mat).reshape(4, 4)
    all_ext_mat.append(ext_mat)
#all_ext_mat = np.stack(all_ext_mat)

focal = 0.5 * 800 / np.tan(0.5 * 0.785398)
focal *= w / 800

near = 0.03
far = 4.5

directions = get_ray_directions(h, w, focal).cuda()

# generate results
embedding_xyz = Embedding(3, 10)
embedding_dir = Embedding(3, 4)
nerf_coarse = NeRF()
nerf_fine = NeRF()
load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
nerf_coarse.cuda().eval()
nerf_fine.cuda().eval()

models = [nerf_coarse, nerf_fine]
embeddings = [embedding_xyz, embedding_dir]
dir_name = f'results/{args.dataset_name}/{args.scene_name}'
os.makedirs(dir_name, exist_ok=True)
imgs = []
for i in tqdm(range(len(all_ext_mat))):
    ext_mat = all_ext_mat[i]
    pose = torch.tensor(ext_mat[:3, :4]).float().cuda()
    rays_o, rays_d = get_rays(directions, pose)
    rays = torch.cat([rays_o, rays_d,
                      near * torch.ones_like(rays_o[:, :1]),
                      far * torch.ones_like(rays_o[:, :1])],
                      1)

    results = batched_inference(models, embeddings, rays,
                                args.N_samples, args.N_importance, args.use_disp,
                                args.chunk,
                                True)
    img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()

    if args.save_depth:
        depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
        depth_pred = np.nan_to_num(depth_pred)
        if args.depth_format == 'pfm':
            save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
        else:
            with open(f'depth_{i:03d}', 'wb') as f:
                f.write(depth_pred.tobytes())

    img_pred_ = (img_pred*255).astype(np.uint8)
    imgs += [img_pred_]
    imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
    
imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)
