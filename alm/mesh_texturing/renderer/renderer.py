"""PyTorch3D projection and texture-baking utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xatlas
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.renderer import (
    AmbientLights,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesUV,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

from .geometry import HardGeometryShader
from .shader import HardNChannelFlatShader
from .voronoi import voronoi_fill


class UVProjection:
    """Render mesh views and map them to and from a shared UV texture."""

    def __init__(
        self,
        *,
        texture_size: int,
        render_size: int,
        sampling_mode: str,
        channels: int,
        device: torch.device,
    ) -> None:
        self.channels = channels
        self.device = device
        self.lights = AmbientLights(ambient_color=((1.0,) * channels,), device=device)
        self.target_size = (texture_size, texture_size)
        self.render_size = render_size
        self.sampling_mode = sampling_mode

    def load_mesh(
        self,
        mesh_path: str | Path,
        *,
        scale_factor: float,
        autouv: bool,
    ) -> None:
        mesh = load_objs_as_meshes([str(mesh_path)], device=self.device)
        vertices = mesh.verts_packed()
        faces = mesh.faces_packed()
        vertices = vertices - vertices.mean(dim=0)
        scale = torch.max(torch.norm(vertices, p=2, dim=1))
        vertices = vertices / scale
        vertices = vertices * scale_factor
        mesh = Meshes(verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0), textures=mesh.textures)
        if autouv or mesh.textures is None:
            mesh = self._unwrap_uv(mesh)
        self.mesh = mesh

    def _unwrap_uv(self, mesh: Meshes) -> Meshes:
        vertices = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices.cpu().numpy(), faces.int().cpu().numpy())
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 4
        atlas.generate(chart_options=chart_options)
        _, face_uvs, vertex_uvs = atlas[0]

        vertex_uvs = torch.from_numpy(vertex_uvs.astype(np.float32)).to(device=mesh.device, dtype=vertices.dtype)
        face_uvs = torch.from_numpy(face_uvs.astype(np.int64)).to(device=mesh.device, dtype=faces.dtype)
        empty_texture = torch.zeros(self.target_size + (self.channels,), device=mesh.device)
        mesh.textures = TexturesUV([empty_texture], [face_uvs], [vertex_uvs], sampling_mode=self.sampling_mode)
        return mesh

    def save_mesh(self, path: str | Path, texture: torch.Tensor) -> None:
        save_obj(
            str(path),
            self.mesh.verts_list()[0],
            self.mesh.faces_list()[0],
            verts_uvs=self.mesh.textures.verts_uvs_list()[0],
            faces_uvs=self.mesh.textures.faces_uvs_list()[0],
            texture_map=texture,
        )

    def set_texture_map(self, texture: torch.Tensor) -> None:
        texture_map = texture.permute(1, 2, 0).to(self.device)
        self.mesh.textures = TexturesUV([texture_map], self.mesh.textures.faces_uvs_padded().to(self.device), self.mesh.textures.verts_uvs_padded().to(self.device), sampling_mode=self.sampling_mode)

    def set_noise_texture(self) -> None:
        noise = torch.normal(0, 1, (self.channels,) + self.target_size, device=self.device)
        self.set_texture_map(noise)

    def set_cameras_and_render_settings(
        self,
        camera_poses: Sequence[tuple[int, int]],
        *,
        centers: Sequence[tuple[int, int, int]],
        camera_distance: float,
    ) -> None:
        elevations = torch.tensor([pose[0] for pose in camera_poses], dtype=torch.float32)
        azimuths = torch.tensor([pose[1] for pose in camera_poses], dtype=torch.float32)
        rotation, translation = look_at_view_transform(dist=camera_distance, elev=elevations, azim=azimuths, at=centers)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=rotation, T=translation)
        self._setup_renderer()
        self._disconnect_faces()
        self._construct_uv_mesh()
        self._calculate_texture_gradients()
        self._calculate_visible_triangle_masks()
        _, _, _, cosine_maps, _, _ = self.render_geometry()
        self.calculate_cosine_weights(cosine_maps)

    def _setup_renderer(self) -> None:
        raster_settings = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=True,
            max_faces_per_bin=30000,
            bin_size=-1,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=HardNChannelFlatShader(device=self.device, cameras=self.cameras, lights=self.lights, channels=self.channels),
        )

    def _disconnect_faces(self) -> None:
        vertices = self.mesh.verts_list()[0]
        faces = self.mesh.faces_list()[0]
        vertex_uvs = self.mesh.textures.verts_uvs_list()[0]
        face_uvs = self.mesh.textures.faces_uvs_list()[0]
        disconnected = torch.zeros((vertex_uvs.shape[0], 3), dtype=vertices.dtype, device=vertices.device)
        disconnected[face_uvs] = vertices[faces]
        if self.mesh.has_verts_normals():
            raise ValueError("Meshes with vertex normals are not supported.")
        self.mesh_d = Meshes([disconnected], [face_uvs], self.mesh.textures)

    def _construct_uv_mesh(self) -> None:
        vertices = self.mesh_d.verts_list()[0].clone()
        vertex_uvs = self.mesh_d.textures.verts_uvs_list()[0]
        vertices[..., :2] = vertex_uvs
        vertices = (vertices - 0.5) * 2
        self.mesh_uv = Meshes([vertices], self.mesh_d.faces_list(), self.mesh_d.textures.clone())

    @torch.enable_grad()
    def _calculate_texture_gradients(self) -> None:
        gradient_maps = []
        mesh = self.mesh.clone()
        for index in range(len(self.cameras)):
            texture = self._trainable_texture(self.channels)
            optimizer = torch.optim.SGD([texture], lr=1, momentum=0)
            optimizer.zero_grad()
            mesh.textures = self._textures_uv(texture)
            prediction = self.renderer(mesh, cameras=self.cameras[index], lights=self.lights)
            torch.sum((1 - prediction) ** 2).backward()
            optimizer.step()
            gradient_maps.append(texture.detach())
        self.gradient_maps = gradient_maps

    @torch.no_grad()
    def _calculate_visible_triangle_masks(self) -> None:
        visible_faces = []
        original_size = self.renderer.rasterizer.raster_settings.image_size
        self.renderer.rasterizer.raster_settings.image_size = (512, 512)
        for index in range(len(self.cameras)):
            fragments = self.renderer.rasterizer(self.mesh_d, cameras=self.cameras[index])
            visible_faces.append(torch.unique(fragments.pix_to_face))
        self.renderer.rasterizer.raster_settings.image_size = original_size

        raster_settings = RasterizationSettings(
            image_size=self.target_size,
            blur_radius=0,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False,
            max_faces_per_bin=30000,
        )
        rotation, translation = look_at_view_transform(dist=2, elev=0, azim=0)
        uv_camera = FoVOrthographicCameras(device=self.device, R=rotation, T=translation)
        uv_fragments = MeshRasterizer(cameras=uv_camera, raster_settings=raster_settings)(self.mesh_uv)
        uv_faces = uv_fragments.pix_to_face[0]

        masks = []
        for face_ids in visible_faces:
            face_ids = face_ids[face_ids >= 0]
            mask = torch.isin(uv_faces, face_ids, assume_unique=False)
            triangle_mask = torch.zeros(self.target_size + (1,), device=self.device)
            triangle_mask[mask] = 1
            triangle_mask[:, 1:][triangle_mask[:, :-1] > 0] = 1
            triangle_mask[:, :-1][triangle_mask[:, 1:] > 0] = 1
            triangle_mask[1:, :][triangle_mask[:-1, :] > 0] = 1
            triangle_mask[:-1, :][triangle_mask[1:, :] > 0] = 1
            masks.append(triangle_mask)
        self.visible_triangles = masks

    def _trainable_texture(self, channels: int) -> torch.Tensor:
        return torch.zeros(self.target_size + (channels,), device=self.device, requires_grad=True)

    def _textures_uv(self, texture: torch.Tensor) -> TexturesUV:
        return TexturesUV([texture], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)

    @torch.enable_grad()
    def calculate_cosine_weights(
        self,
        cosine_angles: torch.Tensor,
        *,
        fill_unobserved: bool = True,
        disable_voronoi: bool = False,
    ) -> None:
        cosine_maps = []
        mesh = self.mesh.clone()
        for index in range(len(self.cameras)):
            texture = self._trainable_texture(self.channels)
            optimizer = torch.optim.SGD([texture], lr=1, momentum=0)
            optimizer.zero_grad()
            mesh.textures = self._textures_uv(texture)
            prediction = self.renderer(mesh, cameras=self.cameras[index], lights=self.lights)
            torch.sum((cosine_angles[index, ..., :1] - prediction) ** 2).backward()
            optimizer.step()
            cosine_map = texture.detach() / (self.gradient_maps[index] + 1e-8)
            if fill_unobserved:
                cosine_map = voronoi_fill(cosine_map, self.gradient_maps[index][..., 0], disabled=disable_voronoi)
            cosine_maps.append(cosine_map)
        self.cos_maps = cosine_maps

    @torch.no_grad()
    def render_geometry(self, image_size: int | None = None):
        original_size = self.renderer.rasterizer.raster_settings.image_size
        if image_size is not None:
            self.renderer.rasterizer.raster_settings.image_size = image_size
        original_shader = self.renderer.shader
        self.renderer.shader = HardGeometryShader(device=self.device, cameras=self.cameras[0], lights=self.lights)
        result = self.renderer(self.mesh.clone().extend(len(self.cameras)), cameras=self.cameras, lights=self.lights)
        self.renderer.shader = original_shader
        self.renderer.rasterizer.raster_settings.image_size = original_size
        return result

    @staticmethod
    @torch.no_grad()
    def decode_normalized_depth(depths: torch.Tensor) -> torch.Tensor:
        depth, mask = depths.unbind(-1)
        inverse_depth = 1 / (depth * mask + 100 * (1 - mask))
        masked_inverse_depth = inverse_depth * mask + 100 * (1 - mask)
        maximum = inverse_depth.amax(dim=(1, 2), keepdim=True)
        minimum = masked_inverse_depth.amin(dim=(1, 2), keepdim=True)
        normalized = ((inverse_depth - minimum) / (maximum - minimum)).clamp(0, 1)
        return normalized[..., None].repeat(1, 1, 1, 3)

    @torch.no_grad()
    def render_textured_views(self) -> list[torch.Tensor]:
        images = self.renderer(self.mesh.extend(len(self.cameras)), cameras=self.cameras, lights=self.lights)
        return [image.permute(2, 0, 1) for image in images]

    @torch.enable_grad()
    def bake_texture(
        self,
        views: Sequence[torch.Tensor],
        *,
        exponent: int,
        fill_unobserved: bool,
        disable_voronoi: bool,
    ) -> torch.Tensor:
        """Back-project rendered views and blend them in UV space."""

        channel_last_views = [view.permute(1, 2, 0) for view in views]
        mesh = self.mesh
        bake_maps = [self._trainable_texture(channel_last_views[0].shape[-1]) for _ in channel_last_views]
        optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
        optimizer.zero_grad()
        loss = torch.zeros((), device=self.device)
        for index, (view, bake_map) in enumerate(zip(channel_last_views, bake_maps)):
            mesh.textures = self._textures_uv(bake_map)
            prediction = self.renderer(mesh, cameras=self.cameras[index], lights=self.lights)
            loss = loss + ((prediction[..., :-1] - view) ** 2).sum()
        loss.backward()
        optimizer.step()

        total_weights = torch.zeros_like(bake_maps[0])
        baked = torch.zeros_like(bake_maps[0])
        for index, bake_map in enumerate(bake_maps):
            normalized = bake_map.detach() / (self.gradient_maps[index] + 1e-8)
            normalized = voronoi_fill(normalized, self.gradient_maps[index][..., 0], disabled=disable_voronoi)
            weight = self.visible_triangles[index] * self.cos_maps[index] ** exponent
            total_weights += weight
            baked += normalized * weight
        baked /= total_weights + 1e-8

        if fill_unobserved:
            baked = voronoi_fill(baked, total_weights[..., 0], disabled=disable_voronoi)

        result = baked.permute(2, 0, 1)
        self.set_texture_map(result)
        return result

    def to(self, device: torch.device) -> None:
        self.device = device
        for mesh_name in ("mesh", "mesh_d", "mesh_uv"):
            setattr(self, mesh_name, getattr(self, mesh_name).to(device))
        for map_name in ("visible_triangles", "cos_maps", "gradient_maps"):
            maps = getattr(self, map_name)
            setattr(self, map_name, [value.to(device) for value in maps])

