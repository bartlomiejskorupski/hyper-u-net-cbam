# Hyper U-Net + CBAM

Implementation used in my master's thesis: "Study of algorithms for colouring aerial photographs".

Based on the Hyper U-Net model introduced in:

```
@article{farella2022colorizing,
  title={Colorizing the past: Deep learning for the automatic colorization of historical aerial images},
  author={Farella, Elisa Mariarosaria and Malek, Salim and Remondino, Fabio},
  journal={Journal of Imaging},
  volume={8},
  number={10},
  pages={269},
  year={2022},
  publisher={MDPI}
}
```

Original implementation: https://github.com/3DOM-FBK/Hyper_U_Net

My proposed implementation extends the Hyper U-Net model with Convolutional Block Attention Module (CBAM) inserted after each encoder block.

Pre-trained model variants:
- Hyper U-Net (no CBAM), L1 loss
- Hyper U-Net + CBAM, L1 loss
- Hyper U-Net + CBAM, Alex LPIPS loss
- Hyper U-Net + CBAM, VGG LPIPS loss

## Dataset

Dataset used for training (Bridge of Knowledge - Open Research Data):

```
Skorupski, B., & Odya, P. (2025). Aerial Image Patches Dataset (1–) [Dataset]. Gdańsk University of Technology. https://doi.org/10.34808/mn4m-w821
```

## Requirements

For local inference and utility scripts install directly:

```powershell
pip install -r requirements.txt
```

Google Colab is highly recommended for model training.

## Inference

```powershell
python inference.py <model_path> <input_image> -o <output_path>
```

Arguments:
- `model_path` (required): path to a `.pth` checkpoint.
- `input_image` (required): path to an RGB image.
- `-o / --output_path`: save result to file instead of displaying.
- `--no-cbam`: load the architecture without CBAM blocks (for non-CBAM checkpoints).

Example:

```powershell
python inference.py .\models\hyper_u_net_cbam_vgg_lpips.pth .\sample_images\0736.png -o .\out.png
```

## Training

The notebook `train_hyper_u_net_cbam.ipynb` contains the training workflow (data loading, loss configuration for L1 or LPIPS-based objectives, checkpoint saving, and TensorBoard logging). Google Colab is highly recommended for GPU resources.

## Training logs

TensorBoard event files are stored under `./pytorch_logs/<run-name>/`. These can be inspected with:

```powershell
tensorboard --logdir ./pytorch_logs
```

## File Overview

- `model.py` - model definition (Hyper U-Net + optional CBAM).
- `inference.py` - command-line inference utilities.
- `calculate_metrics.py` - batch metric calculation for generated images.
- `metrics.py` - MAE / MSE / PSNR / SSIM / LPIPS metric helpers.
- `dataset.py` - dataset preparation & splitting utilities.
- `log_util.py` - helper for timestamped file + console logging.
- `train_hyper_u_net_cbam.ipynb` - training notebook.
- `models/` - pre-trained checkpoints.
- `sample_images/` - example grayscale inputs for quick testing.

## License

MIT - see `LICENSE`.

## Citation

Primary referenced work:

```
@article{farella2022colorizing,
  title={Colorizing the past: Deep learning for the automatic colorization of historical aerial images},
  author={Farella, Elisa Mariarosaria and Malek, Salim and Remondino, Fabio},
  journal={Journal of Imaging},
  volume={8},
  number={10},
  pages={269},
  year={2022},
  publisher={MDPI}
}
```
