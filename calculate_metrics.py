import argparse
import cv2
import glob
import os
import numpy as np
from log_util import setup_logging
from metrics import MSE, PSNR, SSIM, LPIPS, load_lpips_alex_model, MAE, load_lpips_vgg_model

def main(args):
  logger = setup_logging('metrics', 'metrics_' + args.model)
  logger.info(f'Starting {args.model} metrics calculation')

  filename_pattern = '[0-9][0-9][0-9][0-9].png'
  predicted = glob.glob(os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset', 'output', args.model, filename_pattern)))

  logger.info(f'Found {len(predicted)} predicted images')

  logger.info('Loading LPIPS model')
  # lpips_loss_fn = load_lpips_alex_model()
  lpips_loss_fn = load_lpips_vgg_model()
  # logger.info('LPIPS Alex model loaded successfully')
  logger.info('LPIPS VGG model loaded successfully')

  mae_list = []
  mse_list = []
  psnr_list = []
  ssim_list = []
  lpips_list = []
  for image in predicted:
    logger.info('-----------------------------------')
    original_image = os.path.abspath(os.path.join('..', 'dataset', 'processed', args.resolution, os.path.basename(image)))
    logger.info(f'Processing image: "{original_image}"')
    original = cv2.imread(original_image)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pred = cv2.imread(image)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    if original is None or pred is None:
      logger.error(f"Error reading images: {original_image} or {image}")
      continue
    if original.shape != pred.shape:
      logger.error(f"Shape mismatch: {original.shape} vs {pred.shape}")
      continue

    mae_value = MAE(original, pred)
    mse_value = MSE(original, pred)
    psnr_value = PSNR(original, pred)
    ssim_value = SSIM(original, pred)
    lpips_value = LPIPS(lpips_loss_fn, original, pred)
    mae_list.append(mae_value)
    mse_list.append(mse_value)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)
    lpips_list.append(lpips_value)
    logger.info(f'MAE: {mae_value:0.4f}')
    logger.info(f'MSE: {mse_value:0.4f}')
    logger.info(f'PSNR: {psnr_value:0.4f} db')
    logger.info(f'SSIM: {ssim_value:0.4f}')
    logger.info(f'LPIPS: {lpips_value:0.4f}')
  
  logger.info('-----------------------------------')
  logger.info('Metrics calculation completed')
  logger.info(f'-- {args.model} metrics --')
  logger.info(f'Number of images: {len(predicted)}')
  logger.info(f'MAE: {np.mean(mae_list)}')
  logger.info(f'PSNR: {np.mean(psnr_list)} db')
  logger.info(f'MSE: {np.mean(mse_list)}')
  logger.info(f'SSIM: {np.mean(ssim_list)}')
  logger.info(f'LPIPS: {np.mean(lpips_list)}')


if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument(
    '-m', '--model',
    choices=['hyper_u_net', 'AH_U_NET', 'H_U_NET', 'AH_U_NET_ALEX', 'AH_U_NET_VGG'],
  )
  argparser.add_argument(
    '-r', '--resolution',
    choices=['64', '128', '512'],
    default='512'
  )
  args = argparser.parse_args()
  main(args)