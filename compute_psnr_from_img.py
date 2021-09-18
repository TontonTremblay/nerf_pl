import imageio
import metrics
import numpy as np
import torch
from PIL import Image

indices = [100, 101, 102, 103, 104]
target_size = (400, 400)
psnrs = []
for index in indices:
    #gt = torch.tensor(imageio.imread(f'{index:05d}.png')[..., :3]) / 255.
    gt = Image.open(f'{index:05d}.png').convert('RGB')
    gt = torch.tensor(np.array(gt.resize(target_size, Image.LANCZOS))) / 255.
    #pred = torch.tensor(imageio.imread(f'{index:05d}_pred.png')[..., :3]).cuda()
    pred = Image.open(f'{index:05d}_pred.png').convert('RGB')
    pred = np.array(pred.resize(target_size, Image.LANCZOS)) / 255.
    psnrs.append(metrics.psnr(gt, pred).item())
mean_psnr = np.mean(psnrs)
print(f'mean PSNR: {mean_psnr:.2f}')
