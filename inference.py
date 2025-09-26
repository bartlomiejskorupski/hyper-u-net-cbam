import argparse
import torch
import torch.nn as nn
import cv2
import kornia as K
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
import torch, torch.backends.cudnn as cudnn
from model import UNetCBAM
import time

def load_model(model_path, device="cuda", benchmark=False, use_cbam=True):
  model = UNetCBAM(use_cbam=use_cbam).to(device)
  model = model.to(memory_format=torch.channels_last)
  state = torch.load(model_path, map_location=device)
  model.load_state_dict(state, strict=True)
  if device == "cuda":
    cudnn.benchmark = benchmark
  model.eval()
  return model

def load_img_rgb(img_path, dim=None):
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  if dim is not None:
    img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def infer_img(model: nn.Module, rgb_img: np.ndarray, device="cuda"):
    CHANNELS_LAST = True
    # rgb_img - nparray H,W,3 in [0,255]
    img_tensor = torch.from_numpy(rgb_img).float().div_(255.0).permute(2,0,1).unsqueeze(0).to(device, non_blocking=True)
    if CHANNELS_LAST:
      img_tensor = img_tensor.to(memory_format=torch.channels_last)

    # Convert to LAB and normalize
    lab = K.color.rgb_to_lab(img_tensor) # FP32
    L = lab[:, :1].div_(100.0)  # [0,1], FP32

    # L = L.mul_(1.15).clamp_(0.0, 1.0)  # DELETE THIS

    # Without autocast to FP16/TF32, if not in google colab
    # with torch.amp.autocast(device, enabled=img_tensor.is_cuda):
    with torch.inference_mode():
      L_model = L.to(memory_format=torch.channels_last) if CHANNELS_LAST else L
      ab_pred = model(L_model) # FP32

    # Denormalize LAB back to real values
    lab_pred = torch.cat([L*100.0, ab_pred*128.0], dim=1) # FP32
    rgb_pred = K.color.lab_to_rgb(lab_pred).clamp_(0, 1)

    gray = L[0, 0].cpu().numpy()
    colorized = rgb_pred.squeeze_(0).cpu().permute(1,2,0).numpy()
    return gray, colorized, rgb_img

# DATA_PATH = "../dataset/test/test1.png"
# DATA_PATH = "../dataset/processed/val/1.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_infer_img(model, image_path):
    print(f'Loading image from {image_path}...')
    img_rgb = load_img_rgb(image_path)
    print('Running inference...')
    gray_img, colorized, original_img = infer_img(model, img_rgb, device=DEVICE)
    print('Image colorized!')

    joined = np.concatenate((original_img, (colorized*255).astype('uint8'), gray_img), axis=1)  
    plt.imshow(joined)
    plt.axis("off")
    plt.show()


def infer_multiple_images_and_save(model, device="cuda"):
  image_paths = glob('../dataset/processed/val/*.png')
  # output_dir = '../dataset/output/temp/'
  # output_dir = '../dataset/output/H_U_NET/'
  # output_dir = '../dataset/output/AH_U_NET_ALEX/'
  output_dir = '../dataset/output/AH_U_NET_VGG/'

  timings = []
  warmup_count = 5
  for img_path in image_paths:  
    img_base_name = os.path.basename(img_path)
    rgb_img = load_img_rgb(img_path, dim=None)
    start_time = time.time()
    _, colorized, _ = infer_img(model, rgb_img, device=device)
    end_time = time.time()
    print(f'Inference time: {end_time - start_time:.4f} seconds')
    if warmup_count > 0:
      warmup_count -= 1
    else:
      timings.append(end_time - start_time)
    print('Image colorized: ' + img_base_name)
    cv2.imwrite(output_dir + img_base_name, cv2.cvtColor((colorized * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
  avg_time = np.mean(timings) * 1000  # in milliseconds
  print(f"Average inference time (excluding warm-up): {avg_time} ms")

def infer_test_img(model, image_path, dim=512):
    print(f'Loading test image: {image_path}')
    img_rgb = load_img_rgb(image_path, dim)
    print('Running inference...')
    print(img_rgb.shape)

    gray_img, colorized, original_img = infer_img(model, img_rgb, device=DEVICE)
    print('Image colorized!')

    plt.subplot(1,2,1)
    plt.title("Colorized")
    plt.imshow(colorized)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Gray")
    plt.imshow(gray_img, cmap='gray')
    plt.axis("off")
    plt.show()

def infer_test_and_save(model, outdir):
    image_paths = glob('../dataset/test/*')
    os.makedirs(outdir, exist_ok=True)
    for img_path in image_paths:
      img_base_name = os.path.basename(img_path)
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      size = min(img.shape[0], img.shape[1])
      size = 1024 if size > 1024 else 512
      img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
      rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      gray_img, colorized, original_img = infer_img(model, rgb_img, device=DEVICE)
      colorized_rgb = (colorized*255).astype('uint8')
      # joined = np.concatenate((original_img, (colorized*255).astype('uint8')), axis=1)
      cv2.imwrite(os.path.join(outdir, img_base_name), cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR))
      print('Saved: ' + img_base_name)
    
def concat_test_and_save():
  paths = glob('../dataset/test/*')
  for impath in paths:
    basename = os.path.basename(impath)
    print(basename)
    testim = cv2.imread(impath)
    outim1 = cv2.imread(f"../dataset/test_out/hyper_u_net/{basename}")
    outim2 = cv2.imread(f"../dataset/test_out/AH_U_NET/{basename}")
    outim3 = cv2.imread(f"../dataset/test_out/AH_U_NET_VGG/{basename}")
    if outim1 is None:
      print(f"Missing: {basename} in hyper_u_net")
      continue
    max_height = max(outim1.shape[0], outim2.shape[0], outim3.shape[0])
    testim = cv2.resize(testim, (max_height, max_height), interpolation=cv2.INTER_AREA)
    outim1 = cv2.resize(outim1, (max_height, max_height), interpolation=cv2.INTER_AREA)
    v1 = np.concatenate((testim, outim1), axis=1)
    v2 = np.concatenate((outim2, outim3), axis=1)
    joined = np.concatenate((v1, v2), axis=0)
    cv2.imwrite(f"../dataset/test_out/combined/{basename}", joined)


# MODEL_PATH = "./models/AH_U_NET_20250901_091427_MAE.pth"
# MODEL_PATH = "./models/AH_U_NET_20250912_123424_MAE_ep44_0.0278.pth"
# MODEL_PATH = "./models/AH_U_NET_20250913_182413_LPIPS_ep43_0.0979.pth"
# MODEL_PATH = "./models/H_U_NET_20250914_175024_MAE_ep52_0.0292.pth"
# MODEL_PATH = "./models/AH_U_NET_20250915_173637_LPIPS_ep29_0.0823.pth"
MODEL_PATH = "./models/AH_U_NET_20250915_212231_LPIPS_ep57_0.0994.pth"

def main():
  argparser = argparse.ArgumentParser()
  # first argument
  argparser.add_argument(
    'model_path',
    type=str,
    help='Path to the model file'
  ) 
  argparser.add_argument(
    'input_image',
    type=str,
    help='Path to the input image'
  )
  argparser.add_argument(
    '-o', '--output_path',
    type=str,
    help='Path to save the colorized image to'
  )
  argparser.add_argument(
    '--no-cbam',
    action='store_true',
    help='Using Hyper U-Net model without CBAM'
  )

  args = argparser.parse_args()
  model_path = args.model_path
  input_image = args.input_image
  output_path = args.output_path
  use_cbam = not args.no_cbam


  print('Loading model...')
  # USE_CBAM = True
  model = load_model(model_path, DEVICE, benchmark=False, use_cbam=use_cbam)
  print('Model loaded!')

  print(f'Loading image from {input_image}...')
  img_rgb = load_img_rgb(input_image)
  print('Running inference...')
  gray_img, colorized, original_img = infer_img(model, img_rgb, device=DEVICE)
  print('Image colorized!')

  if output_path is not None:
    cv2.imwrite(output_path, cv2.cvtColor((colorized * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
    print(f'Colorized image saved to {output_path}')
  else:
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(original_img)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Colorized")
    plt.imshow((colorized * 255).astype('uint8'))
    plt.axis("off")
    plt.show()

  # load_and_infer_img(model, "../dataset/processed/val/2227.png")
  # infer_multiple_images_and_save(model, DEVICE)
  # DIM = 1248
  # infer_test_img(model, "../dataset/test/small 1_3086.jpg", DIM)
  # infer_test_img(model, "../dataset/test/small 2_0007.jpg", dim=1248)
  # infer_test_img(model, "../dataset/test/small 6_8046.jpg", dim=1248)
  # infer_test_img(model, "../dataset/test/small 16_4766.jpg", dim=1248)
  # infer_test_img(model, "../dataset/test/small 19_6925.jpg", dim=512)

  # infer_test_and_save(model, outdir="../dataset/test_out/AH_U_NET_VGG/")
  # infer_test_and_save(model, outdir="../dataset/test_out/AH_U_NET/")
  # infer_test_and_save(model, outdir="../dataset/test_out/AH_U_NET/")

  # infer_test_and_save(model, outdir="../dataset/test_out/lucky/")
  # concat_test_and_save()

  # load_and_infer_img(model, "../dataset/test_geoportal/7.png")


if __name__ == "__main__":
  main()