import itertools
import os
from tempfile import TemporaryDirectory as InTemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from fury.actor import sphere
from fury.animation import Animation  # , Timeline
from fury.decorators import skip_osx, skip_win
from fury.lib import (
    AmbientLight,
    DirectionalLight,
    OffscreenCanvas,
    OrbitController,
    PerspectiveCamera,
    Renderer,
    Texture,
)
from fury.testing import assert_greater, assert_less_equal, assert_true, captured_output
from fury.window import Scene, create_screen

##############################################################################
# Temporary variable until we fix the tests and module import
ImageData, numpy_support = None, None
remove_observer_from_actor = None
shaders = None
Timeline = None

# skip all the test in this module
# pytest.skip(allow_module_level=True)
##############################################################################


def test_scene_initialization_default():
    """Test Scene initialization with default parameters."""
    scene = Scene()
    assert scene.background == (0, 0, 0, 1)
    assert len(scene.lights) == 1
    assert isinstance(scene.lights[0], AmbientLight)


def test_scene_initialization_custom_background():
    """Test Scene initialization with custom background."""
    scene = Scene(background=(1, 1, 1, 1))
    assert scene.background == (1, 1, 1, 1)


def test_scene_initialization_with_lights():
    """Test Scene initialization with custom lights."""
    lights = [AmbientLight(), DirectionalLight()]
    scene = Scene(lights=lights)
    assert len(scene.lights) == 2
    assert isinstance(scene.lights[0], AmbientLight)
    assert isinstance(scene.lights[1], DirectionalLight)


def test_scene_initialization_with_skybox():
    """Test Scene initialization with a skybox."""
    skybox = Texture(
        np.zeros((100, 100, 3), dtype=np.uint8), dim=2, size=(100, 100)
    )  # Mock skybox
    scene = Scene(skybox=skybox)
    assert scene._bg_actor is not None


def test_scene_set_skybox():
    """Test setting a skybox."""
    scene = Scene()
    skybox = Texture(
        np.zeros((100, 100, 3), dtype=np.uint8), dim=2, size=(100, 100)
    )  # Mock skybox
    scene.set_skybox(skybox)
    assert scene._bg_actor is not None


def test_scene_clear():
    """Test clearing the scene. Should only remove the actors."""
    scene = Scene()
    actor = sphere(np.zeros((1, 3)))
    scene.add(actor)
    assert len(scene.children) == 3  # Background + actor + AmbientLight
    scene.clear()
    assert len(scene.children) == 2  # Background + AmbientLight


def test_screen_initialization_default():
    """Test Screen initialization with default parameters."""
    renderer = Renderer(OffscreenCanvas())
    screen = create_screen(renderer)
    assert screen.size == (640, 480)  # Default size of pygfx
    assert screen.position == (0, 0)  # Default position of pygfx
    assert isinstance(screen.camera, PerspectiveCamera)
    assert isinstance(screen.controller, OrbitController)
    # Background + AmbientLight + Camera
    assert len(screen.scene.children) == 3
    # Directional Light
    assert len(screen.camera.children) == 1


def test_screen_initialization_custom():
    """Test Screen initialization with custom parameters."""
    renderer = Renderer(OffscreenCanvas())
    scene = Scene()
    camera = PerspectiveCamera(75)
    controller = OrbitController(camera)
    screen = create_screen(renderer, scene=scene, camera=camera, controller=controller)
    assert screen.scene == scene
    assert screen.camera == camera
    assert screen.controller == controller
    # Background + AmbientLight
    assert len(screen.scene.children) == 2


def test_screen_bounding_box():
    """Test setting and getting the bounding box."""
    renderer = Renderer(OffscreenCanvas())
    screen = create_screen(renderer)
    screen.bounding_box = (100, 100, 600, 600)
    assert screen.bounding_box == (100, 100, 600, 600)


# def test_active_camera():
#     scene = window.Scene()
#     scene.add(actor.axes(scale=(1, 1, 1)))

#     scene.reset_camera()
#     scene.reset_clipping_range()

#     direction = scene.camera_direction()
#     position, focal_point, view_up = scene.get_camera()

#     scene.set_camera(
#         position=(0.0, 0.0, 1.0), focal_point=(0.0, 0.0, 0), view_up=view_up
#     )

#     position, focal_point, view_up = scene.get_camera()
#     npt.assert_almost_equal(np.dot(direction, position), -1)

#     scene.zoom(1.5)

#     new_position, _, _ = scene.get_camera()

#     npt.assert_array_almost_equal(position, new_position)

#     scene.zoom(1)

#     # rotate around focal point
#     scene.azimuth(90)

#     position, _, _ = scene.get_camera()

#     npt.assert_almost_equal(position, (1.0, 0.0, 0))

#     arr = window.snapshot(scene)
#     report = window.analyze_snapshot(arr, colors=[(255, 0, 0)])
#     npt.assert_equal(report.colors_found, [True])

#     # rotate around camera's center
#     scene.yaw(90)

#     arr = window.snapshot(scene)
#     report = window.analyze_snapshot(arr, colors=[(0, 0, 0)])
#     npt.assert_equal(report.colors_found, [True])

#     scene.yaw(-90)
#     scene.elevation(90)

#     arr = window.snapshot(scene)
#     report = window.analyze_snapshot(arr, colors=(0, 255, 0))
#     npt.assert_equal(report.colors_found, [True])

#     scene.set_camera(
#         position=(0.0, 0.0, 1.0), focal_point=(0.0, 0.0, 0), view_up=view_up
#     )

#     # vertical rotation of the camera around the focal point
#     scene.pitch(10)
#     scene.pitch(-10)

#     # rotate around the direction of projection
#     scene.roll(90)

#     # inverted normalized distance from focal point along the direction
#     # of the camera

#     position, _, _ = scene.get_camera()
#     scene.dolly(0.5)
#     new_position, focal_point, view_up = scene.get_camera()
#     npt.assert_almost_equal(position[2], 0.5 * new_position[2])

#     cam = scene.camera()
#     npt.assert_equal(new_position, cam.GetPosition())
#     npt.assert_equal(focal_point, cam.GetFocalPoint())
#     npt.assert_equal(view_up, cam.GetViewUp())


# def test_parallel_projection():
#     scene = window.Scene()
#     axes = actor.axes()
#     axes2 = actor.axes()
#     axes2.SetPosition((2, 0, 0))

#     # Add both axes.
#     scene.add(axes, axes2)

#     # Put the camera on a angle so that the
#     # camera can show the difference between perspective
#     # and parallel projection
#     scene.set_camera(position=(1.5, 1.5, 1.5))
#     scene.GetActiveCamera().Zoom(2)

#     # window.show(scene, reset_camera=True)
#     scene.reset_camera()
#     arr = window.snapshot(scene)

#     scene.projection(proj_type="parallel")
#     # window.show(scene, reset_camera=False)
#     arr2 = window.snapshot(scene)
#     # Because of the parallel projection the two axes
#     # will have the same size and therefore occupy more
#     # pixels rather than in perspective projection were
#     # the axes being further will be smaller.
#     npt.assert_equal(np.sum(arr2 > 0) > np.sum(arr > 0), True)
#     scene.projection(proj_type="perspective")
#     arr2 = window.snapshot(scene)
#     npt.assert_equal(np.sum(arr2 > 0), np.sum(arr > 0))


# def test_order_transparent():
#     scene = window.Scene()

#     red_cube = actor.cube(
#         centers=np.array([[0.0, 0.0, 2]]),
#         directions=np.array([[0, 1.0, 0]]),
#         colors=np.array([[1, 0.0, 0]]),
#     )

#     green_cube = actor.cube(
#         centers=np.array([[0.0, 0.0, -2]]),
#         directions=np.array([[0, 1.0, 0]]),
#         colors=np.array([[0, 1.0, 0]]),
#     )

#     red_cube.GetProperty().SetOpacity(0.2)
#     green_cube.GetProperty().SetOpacity(0.2)

#     scene.add(red_cube)
#     scene.add(green_cube)

#     scene.reset_camera()
#     scene.reset_clipping_range()

#     # without order_transparency the green will look stronger
#     # when looked from behind the red cube
#     arr = window.snapshot(scene, fname=None, offscreen=True, order_transparent=False)

#     # check if flags are set as expected (here no order transparency)
#     npt.assert_equal(scene.GetLastRenderingUsedDepthPeeling(), 0)

#     green_stronger = arr[150, 150, 1]

#     arr = window.snapshot(scene, fname=None, offscreen=True, order_transparent=True)

#     # # check if flags are set as expected (here with order transparency)
#     npt.assert_equal(scene.GetLastRenderingUsedDepthPeeling(), 1)

#     # when order transparency is True green should be weaker
#     green_weaker = arr[150, 150, 1]

#     assert_greater(green_stronger, green_weaker)


# def test_skybox():
#     # Test scene created without skybox
#     scene = window.Scene()
#     npt.assert_equal(scene.GetAutomaticLightCreation(), 1)
#     npt.assert_equal(scene.GetUseImageBasedLighting(), False)
#     npt.assert_equal(scene.GetUseSphericalHarmonics(), True)
#     npt.assert_equal(scene.GetEnvironmentTexture(), None)
#     report = window.analyze_scene(scene)
#     npt.assert_equal(report.actors, 0)
#     npt.assert_warns(UserWarning, scene.skybox)
#     # Test scene created with skybox
#     test_tex = Texture()
#     test_tex.CubeMapOn()
#     checker_arr = np.array([[1, 1], [1, 1]], dtype=np.uint8) * 255
#     for i in range(6):
#         vtk_img = ImageData()
#         vtk_img.SetDimensions(2, 2, 1)
#         img_arr = np.zeros((2, 2, 3), dtype=np.uint8)
#         img_arr[:, :, i // 2] = checker_arr
#         vtk_arr = numpy_support.numpy_to_vtk(img_arr.reshape((-1, 3), order="F"))
#         vtk_arr.SetName("Image")
#         img_point_data = vtk_img.GetPointData()
#         img_point_data.AddArray(vtk_arr)
#         img_point_data.SetActiveScalars("Image")
#         test_tex.SetInputDataObject(i, vtk_img)
#     test_tex.InterpolateOn()
#     test_tex.MipmapOn()
#     scene = window.Scene(skybox=test_tex)
#     npt.assert_equal(scene.GetAutomaticLightCreation(), 0)
#     npt.assert_equal(scene.GetUseImageBasedLighting(), True)
#     npt.assert_equal(scene.GetUseSphericalHarmonics(), False)
#     skybox_tex = scene.GetEnvironmentTexture()
#     npt.assert_equal(skybox_tex.GetCubeMap(), True)
#     npt.assert_equal(skybox_tex.GetMipmap(), True)
#     npt.assert_equal(skybox_tex.GetInterpolate(), 1)
#     npt.assert_equal(skybox_tex.GetNumberOfInputPorts(), 6)
#     npt.assert_equal(skybox_tex.GetInputDataObject(0, 0).GetDimensions(), (2, 2, 1))
#     report = window.analyze_scene(scene)
#     npt.assert_equal(report.actors, 1)
#     # Test removing automatically shown skybox
#     scene.skybox(visible=False)
#     report = window.analyze_scene(scene)
#     npt.assert_equal(report.actors, 0)


# def test_save_screenshot():
#     xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
#     colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1.0, 1]])
#     sphere_actor = actor.sphere(
#         centers=xyzr[:, :3], colors=colors[:], radii=xyzr[:, 3], phi=10, theta=30
#     )
#     scene = window.Scene()
#     scene.add(sphere_actor)

#     window_sz = (400, 400)
#     show_m = window.ShowManager(scene=scene, size=window_sz)

#     with InTemporaryDirectory():
#         fname = "test.png"
#         # Basic test
#         show_m.save_screenshot(fname)
#         npt.assert_equal(os.path.exists(fname), True)
#         data = io.load_image(fname)
#         report = window.analyze_snapshot(data, colors=[(0, 255, 0), (255, 0, 0)])
#         npt.assert_equal(report.objects, 3)
#         npt.assert_equal(report.colors_found, (True, True))
#         # Test size
#         ss_sz = (200, 200)
#         show_m.save_screenshot(fname, size=ss_sz)
#         npt.assert_equal(os.path.isfile(fname), True)
#         data = io.load_image(fname)
#         npt.assert_equal(data.shape[:2], ss_sz)
#         # Test magnification
#         magnification = 2
#         show_m.save_screenshot(fname, magnification=magnification)
#         npt.assert_equal(os.path.isfile(fname), True)
#         data = io.load_image(fname)
#         desired_sz = tuple(np.array(window_sz) * magnification)
#         npt.assert_equal(data.shape[:2], desired_sz)


# @pytest.mark.skipif(
#     skip_win, reason="This test does not work on Windows. Need to be introspected"
# )
# def test_stereo():
#     scene = window.Scene()

#     lines = [
#         np.array([[-1, 0, 0.0], [1, 0, 0.0]]),
#         np.array([[-1, 1, 0.0], [1, 1, 0.0]]),
#     ]
#     colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#     stream_actor = actor.streamtube(lines, colors=colors, linewidth=0.3, opacity=0.5)

#     scene.add(stream_actor)

#     # green in front
#     scene.elevation(90)
#     scene.camera().OrthogonalizeViewUp()
#     scene.reset_clipping_range()

#     scene.reset_camera()

#     mono = window.snapshot(
#         scene, fname="stereo_off.png", offscreen=True, size=(300, 300), stereo="off"
#     )

#     with npt.assert_warns(UserWarning):
#         stereo = window.snapshot(
#             scene,
#             fname="stereo_horizontal.png",
#             offscreen=True,
#             size=(300, 300),
#             stereo="On",
#         )

#     # mono render should have values in the center
#     # horizontal split stereo render should be empty in the center
#     npt.assert_raises(AssertionError, npt.assert_array_equal, mono[150, 150], [0, 0, 0])
#     npt.assert_array_equal(stereo[150, 150], [0, 0, 0])


# def test_frame_rate():
#     xyz = 1000 * np.random.rand(10, 3)
#     colors = np.random.rand(10, 4)
#     radii = np.random.rand(10) * 50 + 0.5
#     scene = window.Scene()
#     sphere_actor = actor.sphere(centers=xyz, colors=colors, radii=radii)
#     scene.add(sphere_actor)

#     showm = window.ShowManager(
#         scene=scene, size=(900, 768), reset_camera=False, order_transparent=True
#     )

#     counter = itertools.count()
#     frame_rates = []
#     render_times = []
#     showm.render()

#     def timer_callback(_obj, _event):
#         cnt = next(counter)
#         frame_rates.append(showm.frame_rate)

#         showm.scene.azimuth(0.05 * cnt)
#         sphere_actor.GetProperty().SetOpacity(cnt / 100.0)

#         render_times.append(scene.last_render_time)

#         if cnt > 100:
#             showm.exit()

#     showm.add_timer_callback(True, 10, timer_callback)
#     showm.start()

#     assert_greater(len(frame_rates), 0)
#     assert_greater(len(render_times), 0)

#     actual_fps = sum(frame_rates) / len(frame_rates)
#     ideal_fps = 1 / (sum(render_times) / len(render_times))

#     assert_greater(actual_fps, 0)
#     assert_greater(ideal_fps, 0)
#     # assert_al(ideal_fps, actual_fps) this is very imprecise


# @pytest.mark.skipif(True, reason="See TODO in the code")
# def test_record():
#     xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
#     colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1.0, 1]])
#     sphere_actor = actor.sphere(
#         centers=xyzr[:, :3],
#         colors=colors[:],
#         radii=xyzr[:, 3],
#         phi=30,
#         theta=30,
#         use_primitive=True,
#     )
#     scene = window.Scene()
#     scene.add(sphere_actor)

#     def test_content(filename="fury.png", colors_found=(True, True)):
#         npt.assert_equal(os.path.isfile(filename), True)
#         arr = io.load_image(filename)
#         report = window.analyze_snapshot(arr, colors=[(0, 255, 0), (0, 0, 255)])
#         npt.assert_equal(report.objects, 3)
#         npt.assert_equal(report.colors_found, colors_found)
#         return arr

#     # Basic test
#     with InTemporaryDirectory():
#         window.record(scene)
#         test_content()

#     # test out_path and path_numbering, n_frame
#     with InTemporaryDirectory():
#         filename = "tmp_snapshot.png"
#         window.record(scene, out_path=filename)
#         test_content(filename)
#         window.record(scene, out_path=filename, path_numbering=True)
#         test_content(filename + "000000.png")
#         window.record(scene, out_path=filename, path_numbering=True, n_frames=3)
#         test_content(filename + "000000.png")
#         test_content(filename + "000001.png")
#         test_content(filename + "000002.png")
#         npt.assert_equal(os.path.isfile(filename + "000003.png"), False)

#     # test verbose
#     with captured_output() as (out, _):
#         window.record(scene, verbose=True)

#     npt.assert_equal(
#         out.getvalue().strip(),
#         "Camera Position (315.32, 0.00, 536.73)\n"
#         "Camera Focal Point (119.97, 0.00, 0.00)\n"
#         "Camera View Up (0.00, 1.00, 0.00)",
#     )
#     # test camera option
#     with InTemporaryDirectory():
#         window.record(
#             scene, cam_pos=(310, 0, 530), cam_focal=(120, 0, 0), cam_view=(0, 0, 1)
#         )
#         test_content()

#     # test size and clipping
#     # Skip it on Mac mainly due to offscreen case on Travis. It works well
#     # with a display. Need to check if screen_clip works. Need to check if
#     # ReadFrontBufferOff(), ShouldRerenderOn() could improved this OSX case.
#     if not skip_osx:
#         with InTemporaryDirectory():
#             window.record(
#                 scene, out_path="fury_1.png", size=(1000, 1000), magnification=5
#             )
#             npt.assert_equal(os.path.isfile("fury_1.png"), True)
#             arr = io.load_image("fury_1.png")

#             npt.assert_equal(arr.shape, (5000, 5000, 3))

#             window.record(
#                 scene, out_path="fury_2.png", size=(5000, 5000), screen_clip=True
#             )
#             npt.assert_equal(os.path.isfile("fury_2.png"), True)
#             arr = io.load_image("fury_2.png")

#             assert_less_equal(arr.shape[0], 5000)
#             assert_less_equal(arr.shape[1], 5000)


# # @pytest.mark.skipif(True, reason="See TODO in the code")
# def test_opengl_state_simple():
#     for gl_state in [
#         window.gl_reset_blend,
#         window.gl_enable_depth,
#         window.gl_disable_depth,
#         window.gl_enable_blend,
#         window.gl_disable_blend,
#         window.gl_set_additive_blending,
#         window.gl_set_normal_blending,
#         window.gl_set_multiplicative_blending,
#         window.gl_set_subtractive_blending,
#         window.gl_set_additive_blending_white_background,
#     ]:
#         scene = window.Scene()
#         centers = np.array([[0, 0, 0], [-0.1, 0, 0], [0.1, 0, 0]])
#         colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

#         actors = actor.markers(
#             centers,
#             marker="s",
#             colors=colors,
#             marker_opacity=0.5,
#             scales=0.2,
#         )
#         showm = window.ShowManager(
#             scene=scene, size=(900, 768), reset_camera=False, order_transparent=False
#         )

#         scene.add(actors)
#         # single effect
#         shaders.shader_apply_effects(showm.window, actors, effects=gl_state)
#         showm.render()

#         # THIS HELPED BUT STILL ...
#         showm.exit()


# @pytest.mark.skipif(True, reason="See TODO in the code")
# def test_opengl_state_add_remove_and_check():
#     scene = window.Scene()
#     centers = np.array([[0, 0, 0], [-0.1, 0, 0], [0.1, 0, 0]])
#     colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

#     actor_no_depth_test = actor.markers(
#         centers,
#         marker="s",
#         colors=colors,
#         marker_opacity=0.5,
#         scales=0.2,
#     )
#     showm = window.ShowManager(
#         scene=scene, size=(900, 768), reset_camera=False, order_transparent=False
#     )

#     scene.add(actor_no_depth_test)

#     showm.render()
#     state = window.gl_get_current_state(showm.window.GetState())
#     before_depth_test = state["GL_DEPTH_TEST"]
#     npt.assert_equal(before_depth_test, True)
#     # TODO: we are getting bad request for enum status
#     # it seems we are not provide the correct values
#     # vtkOpenGLState.cxx:1299  WARN| Bad request for enum status
#     id_observer = shaders.shader_apply_effects(
#         showm.window,
#         actor_no_depth_test,
#         effects=[
#             window.gl_reset_blend,
#             window.gl_disable_blend,
#             window.gl_disable_depth,
#         ],
#     )

#     showm.render()
#     state = window.gl_get_current_state(showm.window.GetState())
#     # print('type', type(showm.window.GetState()))
#     after_depth_test = state["GL_DEPTH_TEST"]
#     npt.assert_equal(after_depth_test, False)
#     # removes the no_depth_test effect
#     remove_observer_from_actor(actor_no_depth_test, id_observer)
#     showm.render()
#     state = window.gl_get_current_state(showm.window.GetState())
#     after_remove_depth_test_observer = state["GL_DEPTH_TEST"]
#     npt.assert_equal(after_remove_depth_test_observer, True)


# def test_add_animation_to_show_manager():
#     showm = window.ShowManager()
#     showm.initialize()

#     cube = actor.cube(np.array([[2, 2, 3]]))

#     timeline = Timeline(playback_panel=True)
#     animation = Animation(actors=cube)
#     timeline.add_animation(animation)
#     showm.add_animation(timeline)

#     npt.assert_equal(len(showm._timelines), 1)
#     assert_true(showm._animation_callback is not None)

#     actors = showm.scene.GetActors()
#     assert_true(cube in actors)
#     actors_2d = showm.scene.GetActors2D()

#     [assert_true(act in actors_2d) for act in animation.static_actors]
#     showm.remove_animation(timeline)

#     actors = showm.scene.GetActors()
#     actors_2d = showm.scene.GetActors2D()

#     [assert_true(act not in actors) for act in animation.static_actors]
#     assert_true(cube not in actors)
#     assert_true(showm._animation_callback is None)
#     assert_true(showm.timelines == [])
#     assert_true(list(actors_2d) == [])

#     showm.add_animation(animation)
#     assert_true(cube in showm.scene.GetActors())

#     showm.remove_animation(animation)
#     assert_true(cube not in showm.scene.GetActors())
#     assert_true(showm.animations == [])
#     assert_true(list(showm.scene.GetActors()) == [])
