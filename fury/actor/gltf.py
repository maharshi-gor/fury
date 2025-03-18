import pygfx as gfx
from fury.v2.actor.materials import _create_line_material


def glTF(path, cube_map=None):
    """
    Load a glTF model and return the first child of the scene.

    Parameters
    ----------

    path : str
        The path to the glTF file.

    cube_map : Texture
        The cube map to use as the environment map.
    """

    _gltf = gfx.load_gltf(path, quiet=False)

    _gltf.scene.children[0].geometry.texcoords1 = _gltf.scene.children[0].geometry.texcoords

    if cube_map is not None:
        _gltf.scene.children[0].material.env_map = cube_map

    return _gltf.scene


