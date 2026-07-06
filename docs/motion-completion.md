# Human Motion Completion

## Environment for Human Motion Completion Task

Create the pinned `ALM-motion` Conda environment:

```bash
conda env create -f environment-motion.yml
conda activate ALM-motion
```

The environment follows the [CondMDI](https://github.com/setarehc/diffusion-motion-inbetweening) setup with Python 3.9, PyTorch 1.13.1, and CUDA 11.7. 



### Assets

Download the GloVe vectors and the T2M and KIT evaluator archives used by the original CondMDI preparation scripts:

```bash
conda activate ALM-motion
python -m alm.motion_completion prepare --download-support
```





### HumanML3D

Clone HumanML3D, extract its text annotations, and copy the dataset into the canonical repository location:

```bash
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip HumanML3D/HumanML3D/texts.zip -d HumanML3D/HumanML3D/
mkdir -p data/motion
cp -r HumanML3D/HumanML3D data/motion/HumanML3D
cp -a tools/motion/HumanML3D_abs/. data/motion/HumanML3D/
```

ALM uses CondMDI's absolute-root HumanML3D representation. 
Follow the upstream HumanML3D preprocessing instructions, then run the supplied `motion_representation.ipynb` and `cal_mean_variance.ipynb` notebooks from `data/motion/HumanML3D/`. 
The prepared directory must contain:

```text
Mean.npy
Std.npy
Mean_abs_3d.npy
Std_abs_3d.npy
test.txt
new_joint_vecs/
new_joint_vecs_abs_3d/
texts/
```


### Checkpoint and SMPL Files

Download the unconditional CondMDI checkpoint used by ALM:

```bash
python -m alm.motion_completion prepare --download-checkpoint
```

The command installs:

```text
checkpoints/motion/condmdi_uncond/args.json
checkpoints/motion/condmdi_uncond/model000500000.pt
```

Download the SMPL files required for mesh conversion and validate the complete runtime setup:

```bash
python -m alm.motion_completion prepare --download-smpl
python -m alm.motion_completion prepare --check
```

The validation command checks the canonical data files, GloVe vectors, T2M evaluator, SMPL files, checkpoint, and checkpoint configuration.






## Reproduction of Qualitative Results

Run the following commands:

```bash
conda activate ALM-motion
bash scripts/run_motion_completion.sh
```

The equivalent module command is:

```bash
python -m alm.motion_completion complete \
  --edit-mode first_half \
  --transition-length 2 \
  --num-samples 64 \
  --num-repetitions 3 \
  --w_1 1 \
  --w_2 0.005
```

Following paths are used by default:

```text
data/motion/HumanML3D
data/motion/glove
checkpoints/motion/condmdi_uncond/model000500000.pt
outputs/motion_completion
```

Override these locations with `--data-root`, `--glove-root`, `--checkpoint`, or `--output-dir`. 



The paper evaluates `first_half`, `middle_half`, and `last_half`. 
These modes generate the named half of the motion while conditioning on the complementary frames. 
`--transition-length` does not affect the three half-completion modes.
Motion completion uses the checkpoint's complete 1,000-step cosine DDPM schedule. 
The defaults are `w_1=1.0` and `w_2=0.005`.



The command writes output MP4 files under `outputs/motion_completion/`.





## Quantitative Evaluation

Run the evaluation on HumanML3D dataset:

```bash
bash scripts/run_motion_evaluation.sh
```

Above configuration evaluates 1,000 samples over ten replications and may require several hours on a single GPU. 
Change `--edit-mode` to select another mode. 



## Human Mesh Rendering

Convert one generated video and its associated `results.npy` file into per-frame SMPL meshes:

```bash
bash scripts/render_motion_mesh.sh \
  outputs/motion_completion/condmdi_uncond/<run>/sample00_rep00.mp4
```

The equivalent module command is:

```bash
python -m alm.motion_completion render \
  --input outputs/motion_completion/condmdi_uncond/<run>/sample00_rep00.mp4
```


