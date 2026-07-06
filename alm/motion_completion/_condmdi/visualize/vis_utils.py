import os

import numpy as np
import torch
import trimesh
from trimesh import Trimesh

from alm.motion_completion._condmdi.model.rotation2xyz import Rotation2xyz
from alm.motion_completion._condmdi.visualize.simplify_loc2rot import joints2smpl


class npy2obj:
    """Convert one sample/repetition from results.npy to an SMPL motion."""

    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        results = np.load(npy_path, allow_pickle=True)
        if npy_path.endswith(".npz"):
            results = results["arr_0"]
        self.results = results[None][0]

        motions = self.results["motion"]
        num_samples = int(self.results["num_samples"])
        flat_index = rep_idx * num_samples + sample_idx
        if motions.ndim == 5:
            input_motion = motions[rep_idx, sample_idx]
            lengths = np.asarray(self.results["lengths"])
            real_num_frames = lengths[rep_idx, sample_idx]
        elif motions.ndim == 4:
            input_motion = motions[flat_index]
            real_num_frames = np.asarray(self.results["lengths"])[flat_index]
        else:
            raise ValueError(f"Unexpected motion array shape: {motions.shape}")

        texts = np.asarray(self.results["text"], dtype=object).reshape(-1)
        self.text = texts[flat_index]
        self.real_num_frames = int(real_num_frames)
        self.num_frames = input_motion.shape[-1]
        self.rot2xyz = Rotation2xyz(device="cpu")
        self.faces = self.rot2xyz.smpl_model.faces

        if input_motion.shape[1] == 3:
            print(
                f"Running SMPLify for sample [{sample_idx}], repetition "
                f"[{rep_idx}]. This may take a few minutes."
            )
            converter = joints2smpl(
                num_frames=self.num_frames,
                device_id=device,
                cuda=cuda,
            )
            motion, _ = converter.joint2smpl(input_motion.transpose(2, 0, 1))
            self.motion = motion.cpu().numpy()
        elif input_motion.shape[1] == 6:
            self.motion = input_motion[None]
        else:
            raise ValueError(f"Unexpected motion feature count: {input_motion.shape[1]}")

        self.vertices = self.rot2xyz(
            torch.from_numpy(self.motion),
            mask=None,
            pose_rep="rot6d",
            translation=True,
            glob=True,
            jointstype="vertices",
            vertstrans=True,
        )

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(
            vertices=self.get_vertices(sample_i, frame_i),
            faces=self.faces,
        )

    def get_traj_sphere(self, mesh):
        root_position = np.copy(mesh.vertices).mean(0)
        root_position[1] = self.vertices.numpy().min(axis=(0, 1, 3))[1] + 0.1
        return trimesh.primitives.Sphere(
            radius=0.05,
            center=root_position,
            transform=None,
            subdivisions=1,
        )

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        ground_sphere = self.get_traj_sphere(mesh)
        location_name = os.path.splitext(os.path.basename(save_path))[0]
        location_path = os.path.join(
            os.path.dirname(save_path),
            "loc",
            f"{location_name}_ground_loc.obj",
        )
        with open(save_path, "w") as output:
            mesh.export(output, "obj")
        with open(location_path, "w") as output:
            ground_sphere.export(output, "obj")
        return save_path

    def save_npy(self, save_path):
        data = {
            "motion": self.motion[0, :, :, : self.real_num_frames],
            "thetas": self.motion[0, :-1, :, : self.real_num_frames],
            "root_translation": self.motion[0, -1, :3, : self.real_num_frames],
            "faces": self.faces,
            "vertices": self.vertices[0, :, :, : self.real_num_frames],
            "text": self.text,
            "length": self.real_num_frames,
        }
        np.save(save_path, data)
