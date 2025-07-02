import os
import sys
import argparse
from src.model import (ResidualDiffusion, Trainer, Unet, UnetRes, set_seed)
from data.combined_dataset import CombinedDataset


def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='./MillionIRData/Train')
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument("--batch_size", type=int, default=80, help='batch size of dataloader')
    parser.add_argument('--load_size', type=int, default=268, help='scale images to this size') #572,268
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--meta", type=str, default='./MillionIRData_train_meta_info.txt', help='choose data for training based on meta info')
    parser.add_argument("--bsize", type=int, default=2)
    opt = parser.parse_args()
    return opt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
sys.stdout.flush()
set_seed(10)

save_and_sample_every = 1000
if len(sys.argv) > 1:
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 10

num_samples = 1
sum_scale = 0.01
image_size = 512
condition = True
opt = parsr_args()
train_batch_size = opt.batch_size
print(train_batch_size)

results_folder = "./ckpt_single_multi"

dataset = CombinedDataset(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='meta_info')
num_unet = 1
objective = 'pred_res'
test_res_or_noise = "res"
train_num_steps = 500000 # for single degradation training
# train_num_steps = 2000000 # for all training
sum_scale = 0.01
delta_end = 1.4e-3

model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_unet=num_unet,
    condition=condition,
    objective=objective,
    test_res_or_noise = test_res_or_noise
)

diffusion = ResidualDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,           # number of steps
    delta_end = delta_end,
    sampling_timesteps=sampling_timesteps,
    objective=objective,
    loss_type='l1',            # L1 or L2
    condition=condition,
    sum_scale=sum_scale,
    test_res_or_noise = test_res_or_noise,
)

trainer = Trainer(
    diffusion,
    dataset,
    opt,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=1e-4,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    results_folder = results_folder,
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    num_unet=num_unet,
)

# train
# trainer.load()
# trainer.load(500) ### load from 50k steps single degradation training
trainer.train()
