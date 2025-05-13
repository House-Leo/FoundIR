# Image Quality Assessment Script
# Evaluates metrics like PSNR, SSIM, LPIPS, FID, DISTS, etc., for a set of images.

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import numpy as np
import torch
from PIL import Image
import pyiqa

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def get_timestamp():
    """Returns the current timestamp in a specific format."""
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """
    Sets up a logger with specified configurations.

    Args:
        logger_name (str): Name of the logger.
        root (str): Root directory for log files.
        phase (str): Phase name (e.g., 'test').
        level (int, optional): Logging level. Defaults to logging.INFO.
        screen (bool, optional): Whether to log to the screen. Defaults to False.
        tofile (bool, optional): Whether to log to a file. Defaults to False.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

def dict2str(opt, indent=1):
    """
    Converts a dictionary to a formatted string for logging.

    Args:
        opt (dict): The dictionary to convert.
        indent (int, optional): Indentation level. Defaults to 1.

    Returns:
        str: Formatted string representation of the dictionary.
    """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += ' ' * (indent * 2) + "]\n"
        else:
            msg += ' ' * (indent * 2) + f"{k}: {v}\n"
    return msg

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def main():
    parser = argparse.ArgumentParser(description="Image Quality Assessment Script")

    parser.add_argument(
        "--inp_imgs",
        type=str,
        default='./Visual_Results',
        help="Path(s) to the input (SR) images directories."
    )

    parser.add_argument(
        "--gt_imgs",
        type=str,
        default='./Visual_Results/GT/',
        help="Path(s) to the ground truth (GT) images directories."
    )

    parser.add_argument(
        "--log",
        type=str,
        default='./metrics',
        help="Directory path to save the log files."
    )

    parser.add_argument(
        "--log_name",
        type=str,
        default='METRICS',
        help="Base name for the log files."
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create log directory if it doesn't exist
    os.makedirs(args.log, exist_ok=True)

    # Initialize logger
    args.log_name = 'METRICS'
    setup_logger('base', args.log, f'test_{args.log_name}', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info("==========================\n")

    # Initialize IQA metrics excluding FID
    logger.info("Initializing IQA metrics...")
    iqa_metrics = {
        'PSNR': pyiqa.create_metric('psnr').to(device),
        'SSIM': pyiqa.create_metric('ssim').to(device),
        'LPIPS': pyiqa.create_metric('lpips', device=device),
        'NIMA': pyiqa.create_metric('nima', device=device),
        'CLIPIQA': pyiqa.create_metric('clipiqa+_vitL14_512', device=device),
        'NIQE': pyiqa.create_metric('niqe', device=device),
        'MUSIQ': pyiqa.create_metric('musiq', device=device),
        'MANIQA': pyiqa.create_metric('maniqa-pipal', device=device)
    }

    # Initialize FID separately
    fid_metric = pyiqa.create_metric('fid', device=device)
    logger.info("IQA metrics initialized.\n")

    categories = ['AirNet', 'PromptIR', 'DiffIR', 'DiffUIR', 'DA-CLIP', 'X-Restormer', 'Ours']

    logger.info("\n===== Starting Evaluation =====\n")

    gt_dir = args.gt_imgs

    for category in categories:
        if os.path.isdir(os.path.join(args.inp_imgs, category)):
            restored_dir = os.path.join(args.inp_imgs, category)
            img_list_gt = sorted(list(scandir(gt_dir, recursive=True, full_path=True)))
            img_list_restored = sorted(list(scandir(restored_dir, recursive=True, full_path=True)))

            metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}

            # Iterate over each image pair
            for img_idx, sr_path in enumerate(img_list_restored):
                gt_path = img_list_gt[img_idx]
                img_name, _ = os.path.splitext(os.path.basename(sr_path))

                # Read and preprocess images
                sr_img = Image.open(sr_path).convert("RGB")
                gt_img = Image.open(gt_path).convert("RGB")
                sr_img = np.array(sr_img)
                gt_img = np.array(gt_img)
                if sr_img is None or gt_img is None:
                    logger.warning(f"Image read failed for {img_name}. Skipping.")
                    continue

                sr_tensor = img2tensor(sr_img, bgr2rgb=False, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
                gt_tensor = img2tensor(gt_img, bgr2rgb=False, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

                # Compute metrics
                with torch.no_grad():
                    metrics = {}
                    for name, metric in iqa_metrics.items():
                        if name in ['NIMA', 'CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA', 'PAQ2PIQ']:
                            metrics[name] = metric(sr_tensor).item()
                        else:
                            metrics[name] = metric(sr_tensor, gt_tensor).item()

                # Accumulate metrics
                for name in metrics_accum:
                    metrics_accum[name] += metrics[name]

                # Log per-image metrics and runtime
                metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
                logger.info(f"{category}: {img_name} | {metrics_str}")

            # Compute average metrics
            num_images = len(img_list_restored)
            avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}
            logger.info(f"\n===== Average Metrics for [{category}]: {avg_metrics}\n")

            # Compute FID for the directory
            fid_value = fid_metric(gt_dir, restored_dir).item()

        logger.info(f"\n{category} | FID: {fid_value:.6f}\n")


    logger.info("===== Evaluation Completed =====")

if __name__ == "__main__":
    main()