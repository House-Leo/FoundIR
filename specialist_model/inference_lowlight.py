import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.wavemamba_arch import WaveMamba
from basicsr.utils.download_util import load_file_from_url

import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')


import torch.nn.functional as F


def check_image_size(x, window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % (window_size)) % (
        window_size)
    mod_pad_w = (window_size - w % (window_size)) % (
        window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='./dataset/lowlight/input/',
                        help='Input image or folder')

    parser.add_argument('-w', '--weight', type=str,
                        default='./ckpt/lowlight.pth',
                        help='path for model weights')

    parser.add_argument('-o', '--output', type=str, default='./results/lowlight', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enhance_weight_path = args.weight

    EnhanceNet = WaveMamba(in_chn=3,
                           wf=32,
                           n_l_blocks=[1, 2, 4],
                           n_h_blocks=[1, 1, 2],
                           ffn_scale=2.0).to(device)

    EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=True)
    EnhanceNet.eval()
    print_network(EnhanceNet)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    num_img = 0
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)
        b, c, h, w = img_tensor.size()
        print('b, c, h, w = img_tensor.size()', img_tensor.size())
        img_tensor = check_image_size(img_tensor)

        with torch.no_grad():
            output = EnhanceNet.restoration_network(img_tensor)
        output = output

        output = output[:, :, :h, :w]
        output_img = tensor2img(output)

        num_img += 1
        print('num_img', num_img)

        save_path = os.path.join(args.output, img_name.split('.')[0] + ".png")
        imwrite(output_img, save_path)

        pbar.update(1)
    pbar.close()



if __name__ == '__main__':
    main()
