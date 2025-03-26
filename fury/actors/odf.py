import math

import numpy as np

from fury.io import load_wgsl
from fury.lib import (
    BaseShader,
    Binding,
    Buffer,
    BufferUsage,
    CullMode,
    Mesh,
    MeshPhongMaterial,
    MeshPhongShader,
    PrimitiveTopology,
    RenderMask,
    WorldObject,
    enable_wgpu_features,
    register_wgpu_render_function,
    visibility_all,
)


class OdfSlicer(Mesh):
    def __init__(
        self,
        geometry=None,
        material=None,
        data=None,
        B_matrix=None,
        *args,
        **kwargs,
    ):
        super().__init__(geometry, material, *args, **kwargs)
        self.n_coefficients = data.shape[-1]
        self.data_shape = data.shape[:3]
        self.sh_coef = data.reshape(-1).astype(np.float32)
        self.n_vertices_per_voxel = B_matrix.shape[0]
        self.sf_func = B_matrix.reshape(-1).astype(np.float32)
        self.radii = np.ones(geometry.positions.data.shape[0], dtype=np.float32)
        enable_wgpu_features("float32-filterable", "vertex-writable-storage")


class OdfSlicerMaterial(MeshPhongMaterial):
    pass


class OdfSlicerComputeShader(BaseShader):
    type = "compute"

    def __init__(self, wobject):
        super().__init__(wobject)

    def get_bindings(self, wobject, shared):
        self["vertices_per_voxel"] = wobject.n_vertices_per_voxel
        self["n_vertices"] = wobject.geometry.positions.data.shape[0]
        self["n_coefficients"] = wobject.n_coefficients
        self["n_indices"] = wobject.geometry.indices.data.shape[0]
        # wobject.geometry.normals._wgpu_usage = BufferUsage.STORAGE
        # print(wobject.sh_coef.shape)

        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
            3: Binding(
                "s_indices",
                "buffer/read_only_storage",
                wobject.geometry.indices,
                visibility=visibility_all,
            ),
            4: Binding(
                "s_positions",
                "buffer/read_only_storage",
                wobject.geometry.positions,
                visibility=visibility_all,
            ),
            5: Binding(
                "s_normals",
                "buffer/storage",
                wobject.geometry.normals,
                visibility=visibility_all,
            ),
        }

        bindings1 = {
            0: Binding(
                "s_sf_func",
                "buffer/read_only_storage",
                Buffer(wobject.sf_func),
                visibility=visibility_all,
            ),
            1: Binding(
                "s_sh_coef",
                "buffer/read_only_storage",
                Buffer(wobject.sh_coef),
                visibility=visibility_all,
            ),
            2: Binding(
                "s_radii",
                "buffer/storage",
                Buffer(wobject.radii),
                visibility=visibility_all,
            ),
        }
        self.define_bindings(0, bindings)
        self.define_bindings(1, bindings1)
        return {0: bindings, 1: bindings1}

    def get_render_info(self, wobject, _shared):
        geometry = wobject.geometry

        # Determine number of vertices
        n = 3 * geometry.positions.nitems / np.power(128, 3)
        n = math.ceil(n)

        return {
            "indices": (128, 1, 1),
        }

    def get_pipeline_info(self, _wobject, _shared):
        # We draw triangles, no culling
        return {}

    def get_code(self):
        # print(load_wgsl("odf_compute_slicer.wgsl", package_name="fury.shaders"))
        return load_wgsl("odf_compute_slicer.wgsl", package_name="fury.shaders")


@register_wgpu_render_function(OdfSlicer, OdfSlicerMaterial)
def render_pass(wobject):
    return OdfSlicerComputeShader(wobject), MeshPhongShader(wobject)
