import fury.primitive as fp
import numpy as np
from fury.v2.actor.materials import _create_mesh_material
from fury.v2.actor.geometry import buffer_to_geometry, create_mesh


def arrow(
    centers,
    directions,
    colors,
    *,
    heights=1.0,
    resolution=10,
    tip_length=0.35,
    tip_radius=0.1,
    shaft_radius=0.03,
    scales=1,
    material='phong',
    enable_picking=True
):
    """Visualize one or many arrows with same features but different positions,
    colors and directions, and scales.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Arrow positions.
    directions : ndarray, shape (N, 3)
        The orientation vector of the arrow.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    heights : float
        The height of the arrow.
    resolution : int
        The resolution of the arrow.
    tip_length : float
        The tip size of the arrow (default: 0.35)
    tip_radius : float
        the tip radius of the arrow (default: 0.1)
    shaft_radius : float
        The shaft radius of the arrow (default: 0.03)
    scales : float
        The scale of the arrows.
    material : str
        The material of the arrow. Default is 'phong'.
    enable_picking : bool
        Enable picking of the arrow actor.

    Returns
    -------
    arrow_actor: Actor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> directions = np.random.rand(5, 3)
    >>> heights = np.random.rand(5)
    >>> arrow_actor = actor.arrow(centers, directions, (1, 1, 1), heights=heights)
    >>> scene.add(arrow_actor)
    >>> # window.show(scene)

    """
    vertices, faces = fp.prim_arrow(height=heights, tip_length=tip_length,
                                    tip_radius=tip_radius, shaft_radius=shaft_radius,
                                    resolution=resolution)
    res = fp.repeat_primitive(
        vertices,
        faces,
        directions=directions,
        centers=centers,
        colors=colors,
        scales=scales,
    )
    big_verts, big_faces, big_colors, _ = res

    geo =  buffer_to_geometry(
        positions=big_verts.astype('float32'),
        indices=big_faces.astype('int32'),
        colors=np.array(big_colors, dtype='float32')/255.,
        texcoords=big_verts.astype('float32')
    )

    mat = _create_mesh_material(material=material, enable_picking=enable_picking)
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]
    obj.prim_count = len(centers)

    return obj


def box(centers, *, directions=(1, 0, 0), colors=(1, 0, 0), scales=(1, 2, 3)):
    """Visualize one or many boxes with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Box positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the box.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        Box size on each direction (x, y), default(1)

    Returns
    -------
    box_actor: Actor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> dirs = np.random.rand(5, 3)
    >>> box_actor = actor.box(centers, directions=dirs, colors=(1, 1, 1))
    >>> scene.add(box_actor)
    >>> # window.show(scene)

    """
    verts, faces = fp.prim_box()
    res = fp.repeat_primitive(
        verts,
        faces,
        directions=directions,
        centers=centers,
        colors=colors,
        scales=scales,
    )

    big_verts, big_faces, big_colors, _ = res

    prim_count = len(centers)

    geo = buffer_to_geometry(
        indices=big_faces.astype('int32'),
        positions=big_verts.astype('float32'),
        texcoords=big_verts.astype('float32'),
        colors=np.array(big_colors, dtype='float32')/255.,
    )

    mat = _create_mesh_material(material='phong', enable_picking=True)
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]
    obj.prim_count = prim_count

    return obj


def cylinder(
    centers,
    directions,
    colors,
    *,
    radius=0.05,
    heights=1,
    capped=False,
    resolution=8,
    repeat_primitive=True,
):
    """Visualize one or many cylinder with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Cylinder positions.
    directions : ndarray, shape (N, 3)
        The orientation vector of the cylinder.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    radius : float
        cylinder radius.
    heights : ndarray, shape (N)
        The height of the cylinder.
    capped : bool
        Turn on/off whether to cap cylinder with polygons. Default (False).
    resolution: int
        Number of facets/sectors used to define cylinder.
    vertices : ndarray, shape (N, 3)
        The point cloud defining the sphere.
    faces : ndarray, shape (M, 3)
        If faces is None then a sphere is created based on theta
        and phi angles.
        If not then a sphere is created with the provided vertices and faces.
    repeat_primitive: bool
        If True, cylinder will be generated with primitives
        If False,
        repeat_sources will be invoked to use VTK filters for cylinder.

    Returns
    -------
    cylinder_actor: Actor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> dirs = np.random.rand(5, 3)
    >>> heights = np.random.rand(5)
    >>> actor = actor.cylinder(centers, dirs, (1, 1, 1), heights=heights)
    >>> scene.add(actor)
    >>> # window.show(scene)

    """
    if resolution < 8:
        # Sectors parameter should be greater than 7 in fp.prim_cylinder()
        raise ValueError("resolution parameter should be greater than 7")

    verts, faces = fp.prim_cylinder(
        radius=radius,
        sectors=resolution,
        capped=capped,
    )
    res = fp.repeat_primitive(
        verts,
        faces,
        centers=centers,
        directions=directions,
        colors=colors,
        scales=heights,
    )

    big_verts, big_faces, big_colors, _ = res

    prim_count = len(centers)

    geo = buffer_to_geometry(
        indices=big_faces.astype('int32'),
        positions=big_verts.astype('float32'),
        texcoords=big_verts.astype('float32'),
        colors=np.array(big_colors, dtype='float32')/255.,
    )

    mat = _create_mesh_material(material='phong', enable_picking=True)
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]
    obj.prim_count = prim_count

    return obj



def sphere(
    centers,
    colors,
    *,
    radii=1.0,
    phi=16,
    theta=16,
    vertices=None,
    faces=None,
    opacity=1,
    material='phong',
    enable_picking=True
):


    scales = radii
    directions = (1, 0, 0)


    if faces is None and vertices is None:
        vertices, faces = fp.prim_sphere(phi=phi, theta=theta)

    res = fp.repeat_primitive(
        vertices,
        faces,
        directions=directions,
        centers=centers,
        colors=colors,
        scales=scales,
    )
    big_verts, big_faces, big_colors, _ = res
    print(big_colors)

    prim_count = len(centers)

    geo = buffer_to_geometry(
        indices=big_faces.astype('int32'),
        positions=big_verts.astype('float32'),
        texcoords=big_verts.astype('float32'),
        colors=np.array(big_colors, dtype='float32')/255.,
    )

    mat = _create_mesh_material(material=material, enable_picking=enable_picking)
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]
    obj.prim_count = prim_count
    return obj


def axis(centers, scales):
    """Visualize one or many axes with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Axis positions.
    scales : ndarray, shape (N, 3)
        Axis scales.

    Returns
    -------
    axis_actor: Actor

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3)
    >>> scales = np.random.rand(5, 3)
    >>> actor = actor.axis(centers, scales)
    >>> scene.add(actor)
    >>> # window.show(scene)

    """
    vertices, faces = fp.prim_arrow()
    centers = np.repeat(centers, 3, axis=0)
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    scales = np.array([scales, scales, scales])
    res = fp.repeat_primitive(
        vertices,
        faces,
        directions=directions,
        centers=centers,
        colors=colors,
        scales=scales,
    )
    big_verts, big_faces, big_colors, _ = res

    prim_count = len(centers)

    geo = buffer_to_geometry(
        indices=big_faces.astype('int32'),
        positions=big_verts.astype('float32'),
        texcoords=big_verts.astype('float32'),
        colors=np.array(big_colors, dtype='float32')/255.,
    )

    mat = _create_mesh_material(material='phong', enable_picking=True)
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]

    obj.prim_count = prim_count
    return obj
