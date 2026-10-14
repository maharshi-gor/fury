import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory as InTemporaryDirectory

import numpy as np
import numpy.testing as npt
import polyxios as px

from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import (
    get_extension,
    load_cube_map_texture,
    load_image,
    load_image_as_wgpu_texture_view,
    load_image_texture,
    load_network,
    read_lines,
    read_mesh,
    read_points,
    save_image,
    save_network,
)
from fury.lib import Texture, wgpu
from fury.testing import assert_greater
from fury.window import ShowManager


def test_get_extension():
    npt.assert_equal(get_extension("image.png"), "png")
    npt.assert_equal(get_extension("archive.tar.gz"), "gz")
    npt.assert_equal(get_extension("/path/to/folder/file.jpg"), "jpg")
    npt.assert_equal(get_extension(".hidden"), "")
    npt.assert_equal(get_extension("noextension"), "")


def test_load_cube_map_texture():
    fetch_viz_cubemaps()
    texture_files = read_viz_cubemap("skybox")
    texture = load_cube_map_texture(texture_files)

    npt.assert_equal(type(texture), Texture)


def test_load_image_texture():
    fetch_viz_cubemaps()
    texture_files = read_viz_cubemap("skybox")
    texture = load_image_texture(texture_files[0])

    npt.assert_equal(type(texture), Texture)


def test_load_image_as_wgpu_texture_view_uses_show_manager_device():
    with InTemporaryDirectory() as tdir:
        data = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
        fname_path = pjoin(tdir, "temp.png")
        save_image(data, fname_path, compression_quality=100)

        show_m = ShowManager(window_type="offscreen")
        try:
            texture_view = load_image_as_wgpu_texture_view(fname_path, show_m.device)
        finally:
            show_m.close()

    assert isinstance(texture_view, wgpu.GPUTextureView)
    npt.assert_array_equal(texture_view.texture.size, (8, 8, 1))
    assert texture_view.texture._device is show_m.device


def test_save_and_load_network():
    formats = ["gexf", "gml", "xnet"]
    fname = "temp-network"

    nodes_xyz = np.random.rand(10, 3).astype(np.float32)
    edges_indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    colors = np.random.rand(10, 4).astype(np.float32)
    network_data = (nodes_xyz, edges_indices, colors)

    for fmt in formats:
        with InTemporaryDirectory() as odir:
            fname_path = pjoin(odir, f"{fname}.{fmt}")

            save_network(network_data, fname_path)

            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

            out_data = load_network(fname_path)

            npt.assert_array_almost_equal(network_data[0], out_data[0], decimal=4)
            npt.assert_array_equal(network_data[1], out_data[1])
            npt.assert_array_almost_equal(network_data[2], out_data[2], decimal=2)

    # Test invalid format
    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, f"{fname}.invalid")
        npt.assert_raises(
            ValueError,
            save_network,
            network_data,
            fname_path,
            format="invalid",
        )
        with open(fname_path, "w") as f:
            f.write("invalid data")
        npt.assert_raises(
            ValueError,
            load_network,
            fname_path,
            format="invalid",
        )


def test_read_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)

    poly = px.make_polydata(
        vertices,
        [("triangle", faces)],
        vertex_attrs={"colors": colors},
    )

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-mesh.vtk")
        px.write(poly, fname_path)

        out_vertices, out_faces, out_colors = read_mesh(fname_path)

    npt.assert_equal(out_vertices.dtype, np.float32)
    npt.assert_equal(out_faces.dtype, np.int32)
    npt.assert_array_almost_equal(out_vertices, vertices)
    npt.assert_array_equal(out_faces, faces)
    npt.assert_array_almost_equal(out_colors, colors)


def test_read_points():
    vertices = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    connectivity = np.array([[0], [1], [2], [3]], dtype=np.int32)

    poly = px.make_polydata(vertices, [("vertex", connectivity)])

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-points.vtk")
        px.write(poly, fname_path)

        out_points, out_colors = read_points(fname_path)

    npt.assert_equal(out_points.dtype, np.float32)
    npt.assert_equal(out_points.shape, (4, 3))
    npt.assert_array_almost_equal(out_points, vertices)
    npt.assert_equal(out_colors, None)


def test_read_lines():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    line = np.array([[0, 1, 2, 3]], dtype=np.int32)

    poly = px.make_polydata(vertices, [("poly_line", line)])

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-lines.vtk")
        px.write(poly, fname_path)

        out_lines, out_colors = read_lines(fname_path)

    npt.assert_equal(len(out_lines), 1)
    npt.assert_equal(out_lines[0].dtype, np.float32)
    npt.assert_array_almost_equal(out_lines[0], vertices)
    npt.assert_equal(out_colors, None)


def test_read_lines_without_line_elements():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    poly = px.make_polydata(vertices, [("triangle", faces)])

    with InTemporaryDirectory() as odir:
        fname_path = pjoin(odir, "temp-no-lines.vtk")
        px.write(poly, fname_path)

        out_lines, out_colors = read_lines(fname_path)

    npt.assert_equal(out_lines, [])
    npt.assert_equal(out_colors, None)


def test_save_load_image():
    l_ext = ["png", "jpeg", "jpg", "bmp", "tiff"]
    fury_logo_link = (
        "https://raw.githubusercontent.com/fury-gl/"
        "fury-communication-assets/main/fury-logo.png"
    )

    invalid_link = "https://picsum.photos/200"
    fname = "temp-io"

    for ext in l_ext:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            url_image = load_image(fury_logo_link)

            url_fname_path = pjoin(odir, f"fury_logo.{ext}")
            fname_path = pjoin(odir, f"{fname}.{ext}")

            save_image(data, fname_path, compression_quality=100)
            save_image(url_image, url_fname_path, compression_quality=100)

            npt.assert_equal(os.path.isfile(fname_path), True)
            npt.assert_equal(os.path.isfile(url_fname_path), True)

            assert_greater(os.stat(fname_path).st_size, 0)
            assert_greater(os.stat(url_fname_path).st_size, 0)

            out_image = load_image(fname_path)
            if ext not in ["jpeg", "jpg", "tiff"]:
                npt.assert_array_equal(data[..., 0], out_image[..., 0])
            else:
                npt.assert_array_almost_equal(
                    data[..., 0], out_image[..., 0], decimal=0
                )

    npt.assert_raises(
        IOError,
        load_image,
        invalid_link,
    )
    npt.assert_raises(
        IOError,
        load_image,
        "test.vtk",
    )
    npt.assert_raises(
        IOError,
        save_image,
        np.random.randint(0, 255, size=(50, 3)),
        "test.vtk",
        compression_quality=75,
        compression_type="deflation",
        dpi=(72, 72),
    )
    npt.assert_raises(
        IOError,
        save_image,
        np.random.randint(0, 255, size=(50, 3)),
        "test.vtk",
        compression_quality=75,
        compression_type="deflation",
        dpi=(72, 72),
    )
    npt.assert_raises(
        IOError,
        save_image,
        np.random.randint(0, 255, size=(50, 3, 1, 1)),
        "test.png",
        compression_quality=75,
        compression_type="deflation",
        dpi=(72, 72),
    )

    # Test valid TIFF compression types produce files
    for ct in [None, "lzw", "deflation"]:
        with InTemporaryDirectory() as odir:
            data = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)
            fname_path = pjoin(odir, f"{fname}.tif")

            save_image(data, fname_path, compression_type=ct)
            npt.assert_equal(os.path.isfile(fname_path), True)
            assert_greater(os.stat(fname_path).st_size, 0)

    # Test that compression actually changes file output
    with InTemporaryDirectory() as odir:
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        path_none = pjoin(odir, "none.tiff")
        path_lzw = pjoin(odir, "lzw.tiff")
        path_deflate = pjoin(odir, "deflate.tiff")

        save_image(data, path_none, compression_type=None)
        save_image(data, path_lzw, compression_type="lzw")
        save_image(data, path_deflate, compression_type="deflation")

        size_none = os.stat(path_none).st_size
        size_lzw = os.stat(path_lzw).st_size
        size_deflate = os.stat(path_deflate).st_size

        # Compressed files should be smaller than uncompressed for uniform data
        assert size_lzw < size_none
        assert size_deflate < size_none

    # Test invalid compression type raises OSError
    npt.assert_raises(
        OSError,
        save_image,
        np.random.randint(0, 255, size=(50, 3), dtype=np.uint8),
        "test.tiff",
        compression_type="invalid",
    )
