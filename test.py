import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
from data.combined_dataset import CombinedDataset
from src.model import (ResidualDiffusion,Trainer, Unet, UnetRes,set_seed)
def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='./MillionIRData/Test')
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size') #568
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', type=bool, default=True, help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--meta", type=str, default=None, help='choose data for training based on meta info')
    parser.add_argument("--bsize", type=int, default=2)
    opt = parser.parse_args()
    return opt

sys.stdout.flush()
set_seed(10)

save_and_sample_every = 1000

train_num_steps = 100000

condition = True

train_batch_size = 1
num_samples = 1
image_size = 1024


opt = parsr_args()

results_folder = 'premodel'

## For our testset
dataset = CombinedDataset(opt, image_size, augment_flip=False, equalizeHist=True, crop_patch=False, generation=False, task='meta_info')

## For your own data
# dataset = CombinedDataset(opt, image_size, augment_flip=False, equalizeHist=True, crop_patch=False, generation=False, task=None)

num_unet = 1
objective = 'pred_res'
test_res_or_noise = "res"
# sampling_timesteps = 3 ## systhesis
sampling_timesteps = 4 ## real-world
sum_scale = 0.01
ddim_sampling_eta = 0.
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
    ddim_sampling_eta=ddim_sampling_eta,
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
    train_lr=2e-4,
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

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(2000)
    trainer.set_results_folder('./results')
    # trainer.test(last=True, crop_phase='weight', crop_size=1024, crop_stride=512)
    trainer.test(last=True, crop_phase='im2overlap', crop_size=1024, crop_stride=512) ## for large image test
    # trainer.test(last=True) ## for no crop test