from dipy.io.image import load_nifti
from fury.window import ShowManager, Scene
from fury.lib import OrthographicCamera, Texture, PanZoomController
import pygfx as gfx
import numpy as np


data, _ = load_nifti("~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii")
data = data.astype("float32")

data_z = np.rot90(data, 1)
data = np.swapaxes(data, 1, 2)
data_y = np.rot90(data, 1)
data = np.swapaxes(data, 1, 2)
data = np.swapaxes(data, 0, 2)
data_x = np.rot90(data, 1)
data = np.swapaxes(data, 0, 2)

nslices_x = data.shape[0]
nslices_y = data.shape[1]
nslices_z = data.shape[2]

index_x = nslices_x // 2
index_y = nslices_y // 2
index_z = nslices_z // 2

im_x = data_x[..., index_x] / np.max(data_x[..., index_x])
im_y = data_y[..., index_y] / np.max(data_y[..., index_y])
im_z = data_z[..., index_z] / np.max(data_z[..., index_z])
tex_x = Texture(im_x, dim=2)
tex_y = Texture(im_y, dim=2)
tex_z = Texture(im_z, dim=2)
geo = gfx.plane_geometry(400, 400, 12, 12)
mat_x = gfx.MeshBasicMaterial(map=tex_x)
mat_y = gfx.MeshBasicMaterial(map=tex_y)
mat_z = gfx.MeshBasicMaterial(map=tex_z)
plane_x = gfx.Mesh(geo, mat_x)
plane_y = gfx.Mesh(geo, mat_y)
plane_z = gfx.Mesh(geo, mat_z)

scene_x = Scene()
scene_y = Scene()
scene_z = Scene()
scene_x.add(plane_x)
scene_y.add(plane_y)
scene_z.add(plane_z)
camera = OrthographicCamera(200, 200)
controller = PanZoomController(camera, enabled=False)

show_m = ShowManager(
    scene=[scene_x, scene_y, scene_z],
    camera=camera,
    controller=controller,
    screen_config=[1, 1, 1],
    size=(1200, 400),
)


def handle_event(event):
    global index_x, index_y, index_z
    index_x = index_x + int(event.dy / 90)
    index_y = index_y + int(event.dy / 90)
    index_z = index_z + int(event.dy / 90)
    index_x = max(0, min(nslices_x - 1, index_x))
    index_y = max(0, min(nslices_y - 1, index_y))
    index_z = max(0, min(nslices_z - 1, index_z))
    im_x = data_x[..., index_x] / np.max(data_x[..., index_x])
    im_y = data_y[..., index_y] / np.max(data_y[..., index_y])
    im_z = data_z[..., index_z] / np.max(data_z[..., index_z])
    tex_x = Texture(im_x, dim=2)
    tex_y = Texture(im_y, dim=2)
    tex_z = Texture(im_z, dim=2)
    mat_x.map = tex_x
    mat_y.map = tex_y
    mat_z.map = tex_z
    show_m.window.request_draw()


show_m.renderer.add_event_handler(handle_event, "wheel")

if __name__ == "__main__":
    show_m.start()
