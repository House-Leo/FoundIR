import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.Restormer_arch import Restormer
from basicsr.utils.download_util import load_file_from_url
from collections import OrderedDict

import torch

_ = torch.manual_seed(123)

import torch.nn.functional as F


def check_image_size(x, window_size=512):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % (window_size)) % (
        window_size)
    mod_pad_w = (window_size - w % (window_size)) % (
        window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    #print(model)
    print("The number of parameters: {}".format(num_params))


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='./dataset/weather/',
                        help='Input image or folder')

    parser.add_argument('-w', '--weight', type=str,
                        default='./ckpt/weather.pth',
                        help='path for model weights')

    parser.add_argument('-o', '--output', type=str, default='./results/weather', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enhance_weight_path = args.weight
    print(enhance_weight_path)

    EnhanceNet = Restormer().to(device)

    state_dict = torch.load(enhance_weight_path)['params']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('restoration_network.', '')
        new_state_dict[new_key] = v

    EnhanceNet.load_state_dict(new_state_dict, strict=True)

    # EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=True)
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

        file_name = path.split('/')[-1]

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        patch_size = 512
        overlap = 0
        _, _, H, W = img_tensor.size()
        # print('Input size:', img_tensor.size())
        img_tensor = check_image_size(img_tensor)
        b, c, h, w = img_tensor.size()
        # print('Input size:', img_tensor.size())

        output = torch.zeros_like(img_tensor)

        step = patch_size - overlap

        patches_h = (h + step - 1) // step
        patches_w = (w + step - 1) // step
        with torch.no_grad():
            for i in range(patches_h):
                for j in range(patches_w):
                    h_start = i * step
                    w_start = j * step
                    h_end = min(h_start + patch_size, h)
                    w_end = min(w_start + patch_size, w)

                    img_patch = img_tensor[:, :, h_start:h_end, w_start:w_end]

                    pad_h = max(0, patch_size - (h_end - h_start))
                    pad_w = max(0, patch_size - (w_end - w_start))
                    if pad_h > 0 or pad_w > 0:
                        img_patch = F.pad(img_patch, (0, pad_w, 0, pad_h), mode='reflect')

                    output_patch = EnhanceNet(img_patch)

                    output[:, :, h_start:h_end, w_start:w_end] = output_patch[:, :, :h_end - h_start, :w_end - w_start]
        #####################################################################################################################################
        output = output[:, :, :H, :W]

        output_img = tensor2img(output)

        num_img += 1
        print('num_img', num_img)

        save_path = os.path.join(args.output, img_name.split('.')[0] + ".png")

        imwrite(output_img, save_path)

        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()

