# Image Inpainting and Wide Image Generation

## Environment for Image Experiments 

The reference environment uses Python 3.12, PyTorch 2.6.0 with CUDA 11.8, Diffusers 0.33.1, and Transformers 4.53.3.




### Conda with uv (Recommended)

```bash
conda create -n ALM python=3.12 -y
conda activate ALM
python -m pip install uv
uv pip install -r requirements.txt
```



### Conda with pip

```bash
conda create -n ALM python=3.12 -y
conda activate ALM
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```





### Model Access

Image inpainting uses [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
Wide image generation uses [Stable Diffusion 2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).
Accept the relevant model terms and authenticate if requested:

```bash
huggingface-cli login
```

The model loaders use the local Hugging Face cache after the initial download.





## Image Inpainting

Run the included example:

```bash
conda activate ALM
bash scripts/run_inpainting.sh
```

The equivalent module command is:

```bash
python -m alm.inpainting \
  --input data/images/sample_cat.jpg \
  --mask data/masks/sample_mask.png \
  --prompt "a photo of a cat" \
  --output-dir outputs/inpainting
```

White mask pixels denote generated regions; 
black pixels denote observed source regions. 
Images and masks are resized to $512×512$ with the interpolation behavior used by the original experiment. 
`--input` may reference either one image or a directory of JPG and PNG files.

The reference configuration uses 50 DDIM steps, guidance scale 7.5, `w1=1.0`, and `w2=0.005`. The output directory contains:

```text
source/<name>.png
masked_source/<name>.png
generated/<name>.png
blended/<name>.png
mask.png
```





## Wide Image Generation

Run the included example:

```bash
conda activate ALM
bash scripts/run_wide_image.sh
```

The equivalent module command is:

```bash
python -m alm.wide_image \
  --prompt "A photo of a forest with a misty fog" \
  --output-dir outputs/wide_image
```

The reference configuration generates five overlapping $512×512$ patches with a 384-pixel stride, producing a $2048×512$ image. 
It uses 50 DDIM steps, guidance scale 7.5, `w1=1.0`, and `w2=0.001`. 
Generated images are saved as `outputs/wide_image/generated/<seed>.png`.

Use `--num-patches`, `--stride`, and `--num-images` to modify the output geometry or number of samples. 
The stride must be less than 512 and divisible by 8.

