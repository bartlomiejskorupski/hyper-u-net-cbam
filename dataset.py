import argparse
import math
import glob
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2000000000
import cv2
import numpy as np
import shutil

def train_val_split(ratio=0.8):
  imgs = glob.glob('..\\dataset\\processed\\512\\*.png')
  print(f'Found {len(imgs)} images')

  imgs = np.random.permutation(imgs)

  train_imgs = imgs[:int(len(imgs) * ratio)]
  val_imgs = imgs[int(len(imgs) * ratio):]
  print(f'Train images: {len(train_imgs)}, Val images: {len(val_imgs)}')

  os.makedirs(os.path.join('..', 'dataset', 'processed', 'train'), exist_ok=True)
  os.makedirs(os.path.join('..', 'dataset', 'processed', 'val'), exist_ok=True)

  for img in train_imgs:
    newpath = os.path.join('..', 'dataset', 'processed', 'train', os.path.basename(img))
    shutil.copy(img, newpath)
    print('Copied to train:', newpath)

  for img in val_imgs:
    newpath = os.path.join('..', 'dataset', 'processed', 'val', os.path.basename(img))
    shutil.copy(img, newpath)
    print('Copied to val:', newpath)

def process_dataset(cmperpx = 25, kmperchunk = 0.5):
  imgs = glob.glob(f'..\\dataset\\raw\\{cmperpx}\\*.tif', recursive=True)
  # print(imgs)
  processed = glob.glob('..\\dataset\\processed\\512\\*.png')
  highest_number = max([0] + [ int(os.path.splitext(os.path.basename(f))[0]) for f in processed ])
  print('Highest number:', highest_number)
  counter = highest_number + 1
  kmperpx = 0.00001 * cmperpx  # km per pixel
  # kmperchunk = 0.5  # km per chunk
  for image in imgs:
    print('Processing image:', image)
    img = Image.open(image).convert('RGB')
    w, h = img.size
    w_chunks, h_chunks, chunk_size = calc_chunks(w, h, kmperpx, kmperchunk)
    for x in range(w_chunks):
      for y in range(h_chunks):
        left = x * chunk_size
        upper = y * chunk_size
        right = left + chunk_size
        lower = upper + chunk_size
        print(f'Cropping image: {left}, {upper}, {right}, {lower}')
        img_cropped = img.crop((left, upper, right, lower))

        resize_and_save(img_cropped, 512, counter)
        # resize_and_save(img_cropped, 128, counter)
        counter += 1

def resize_and_save(img, size, counter):
  img_resized = img.resize((size, size), Image.Resampling.BOX)
  newpath = f'..\\dataset\\processed\\{size}\\{str(counter).rjust(4, "0")}.png'
  if os.path.exists(newpath):
    print('File already exists, skipping:', newpath)
    return newpath
  img_resized.save(newpath, compress_level=6)
  print('Image saved to', newpath)
  return newpath


def calc_chunks(w, h, kmperpx, kmperchunk):
  w_km = w * kmperpx / kmperchunk
  h_km = h * kmperpx / kmperchunk
  print(f'w: {w}, h: {h}, w_km: {w_km}, h_km: {h_km}')
  w_rem = w_km % 1
  h_rem = h_km % 1
  w_chunks = math.floor(w_km)
  h_chunks = math.floor(h_km)
  print(f'w_chunks: {w_chunks}, h_chunks: {h_chunks}, w_rem: {w_rem}, h_rem: {h_rem}')
  if w_rem < h_rem:
    chunk_size = math.floor(w / w_chunks)
  else:
    chunk_size = math.floor(h / h_chunks)
  print(f'chunk_size: {chunk_size}')
  print(f'new_w: {w_chunks * chunk_size} new_h: {h_chunks * chunk_size}')
  return w_chunks, h_chunks, chunk_size

def print_info():
  imgs = glob.glob('../dataset/**/*.tif', recursive=True)
  print(len(imgs))

def main(args):
  if args.info:
    print_info()
  elif args.process:
    process_dataset(args.cmperpx, args.kmperchunk)
  elif args.split:
    train_val_split()

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument(
    '-i', '--info',
    action='store_true',
  )
  argparser.add_argument(
    '-p', '--process',
    action='store_true',
  )
  argparser.add_argument('--cmperpx', type=int, default=25, choices=[25, 50, 60, 30])
  argparser.add_argument('--kmperchunk', type=float, default=0.5, choices=[0.5, 0.66, 1.0, 1.5])
  argparser.add_argument(
    '-s', '--split',
    action='store_true',
  )
  args = argparser.parse_args()
  main(args)