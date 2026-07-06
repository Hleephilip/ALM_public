# 3D Mesh Texturing

## Environment for 3D Mesh Texturing Task 

Create the pinned `ALM-mesh` Conda environment:

```bash
conda env create -f environment-mesh.yml
conda activate ALM-mesh
```

The environment uses Python 3.8, PyTorch 2.0.0 with CUDA 11.7, Diffusers 0.19.3, PyTorch3D 0.7.3, CuPy, and xatlas. 
The PyTorch3D wheel is specific to Linux, Python 3.8, CUDA 11.7, and PyTorch 2.0.0. 




### Model Access

The pipeline uses [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and [depth-conditioned ControlNet](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth).
Authenticate with the Hugging Face Hub if requested:

```bash
huggingface-cli login
```





## Running Code

Run the example texturing command:

```bash
conda activate ALM-mesh
bash scripts/run_mesh_texturing.sh
```

The equivalent module command is:

```bash
python -m alm.mesh_texturing \
  --mesh data/meshes/suitcase.obj \
  --prompt "a suitcase" \
  --output-dir outputs/mesh_texturing/
```

The default configuration uses 30 sampling steps, guidance scale 15.5, `w1=0.5`, and `w2=0.001`. 
The ALM weights remain configurable through `--w1` and `--w2`.
OBJ meshes without a usable texture map are automatically unwrapped with xatlas.



## Outputs

Each sample produces only the final artifacts:

```text
textured.obj
textured.mtl
textured.png
textured_views_rgb.png
```



