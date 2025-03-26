from typing import TypeAlias

import pygfx as gfx
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas

from fury.optpkg import optional_package

jupyter_pckg_msg = (
    "You do not have jupyter-rfb installed. The jupyter widget will not work for "
    "you. Please install or upgrade jupyter-rfb using pip install -U jupyter-rfb"
)

jupyter_rfb, have_jupyter_rfb, _ = optional_package(
    "jupyter_rfb", trip_msg=jupyter_pckg_msg
)

if have_jupyter_rfb:
    from wgpu.gui.jupyter import WgpuCanvas as JupyterWgpuCanvas

Texture = gfx.Texture
AmbientLight = gfx.AmbientLight
Background = gfx.Background
BackgroundSkyboxMaterial = gfx.BackgroundSkyboxMaterial

# Classes that needed to be written as types
Camera: TypeAlias = gfx.Camera
Controller: TypeAlias = gfx.Controller
Scene: TypeAlias = gfx.Scene
Viewport: TypeAlias = gfx.Viewport

DirectionalLight = gfx.DirectionalLight
OrbitController = gfx.OrbitController
PerspectiveCamera = gfx.PerspectiveCamera
Renderer = gfx.WgpuRenderer
run = run
Canvas = WgpuCanvas
OffscreenCanvas = OffscreenWgpuCanvas
BaseShader = gfx.renderers.wgpu.BaseShader
MeshPhongShader = gfx.renderers.wgpu.shaders.meshshader.MeshPhongShader
Buffer = gfx.Buffer
Binding = gfx.renderers.wgpu.Binding
PrimitiveTopology = wgpu.PrimitiveTopology
CullMode = wgpu.CullMode
RenderMask = gfx.renderers.wgpu.RenderMask
Geometry = gfx.Geometry
register_wgpu_render_function = gfx.renderers.wgpu.register_wgpu_render_function
Mesh = gfx.Mesh
WorldObject = gfx.WorldObject
Material = gfx.Material
MeshPhongMaterial = gfx.MeshPhongMaterial
enable_wgpu_features = gfx.renderers.wgpu.enable_wgpu_features
# wgpu.FeatureName.depth_clip_control
visibility_all = (
    wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
)
# visibility_render = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
BufferUsage = wgpu.BufferUsage
if have_jupyter_rfb:
    JupyterCanvas = JupyterWgpuCanvas
else:
    JupyterCanvas = jupyter_rfb
