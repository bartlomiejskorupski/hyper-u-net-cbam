from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
import torch

def MSE(original, predicted):
  original = np.array(original).astype(np.float32)
  predicted = np.array(predicted).astype(np.float32)
  if original.shape != predicted.shape:
    raise ValueError("Original and predicted images must have the same shape.")
  return np.mean((original - predicted) ** 2)

def MAE(original, predicted):
  original = np.array(original).astype(np.float32)
  predicted = np.array(predicted).astype(np.float32)
  if original.shape != predicted.shape:
    raise ValueError("Original and predicted images must have the same shape.")
  return np.mean(np.abs(original - predicted))

def load_lpips_alex_model():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()
  for p in loss_fn_alex.parameters():
      p.requires_grad_(False)
  return loss_fn_alex

def load_lpips_vgg_model():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
  for p in loss_fn_vgg.parameters():
      p.requires_grad_(False)
  return loss_fn_vgg

def LPIPS(loss_fn, original, predicted):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  orig_norm = (original.astype('float32') / 255.0) * 2 - 1 # Normalize to [-1, 1]
  orig_tensor = torch.from_numpy(orig_norm) # shape: (h, w, 3)
  orig_tensor = orig_tensor.permute(2, 0, 1).unsqueeze(0) # shape: (1, 3, h, w)
  orig_tensor = orig_tensor.to(device)
  pred_norm = (predicted.astype('float32') / 255.0) * 2 - 1 # Normalize to [-1, 1]
  pred_tensor = torch.from_numpy(pred_norm) # shape: (h, w, 3)
  pred_tensor = pred_tensor.permute(2, 0, 1).unsqueeze(0) # shape: (1, 3, h, w)
  pred_tensor = pred_tensor.to(device)
  with torch.no_grad():
    dist = loss_fn(orig_tensor, pred_tensor) 
  return dist.mean().item()

def PSNR(original, predicted):
  mse = MSE(original, predicted)
  if mse == 0:
    return 100
  max_pixel = 255.0
  return 20 * np.log10(max_pixel / np.sqrt(mse))

def SSIM(original, predicted):
  original = np.array(original)
  predicted = np.array(predicted)
  if original.shape != predicted.shape:
    raise ValueError("Original and predicted images must have the same shape.")
  
  # ssim_total = 0.0
  # for i in range(3): 
  #   channel_ssim, _ = ssim(original[:, :, i], predicted[:, :, i], full=True)
  #   ssim_total += channel_ssim
  # ssim_total = ssim_total / 3.0
  
  return ssim(original, predicted, channel_axis=2)
