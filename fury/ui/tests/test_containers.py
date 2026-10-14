"""Test containers module."""

from os.path import join as pjoin

import numpy as np
import numpy.testing as npt

from fury import ui, window
from fury.data import DATA_DIR
from fury.testing import VisualTest


def test_panel2d_initialization_default():
    """Test Panel2D initialization and default property values."""
    panel = ui.Panel2D(size=(100, 100))

    npt.assert_equal(panel.size, (100, 100))
    npt.assert_equal(panel.has_border, False)
    npt.assert_array_almost_equal(panel.color, [0.1, 0.1, 0.1])
    npt.assert_almost_equal(panel.opacity, 0.7)
    assert panel.background in panel._elements


def test_panel2d_initialization_with_border():
    """
    Test Panel2D initialization ensures border elements are created and
    sized.
    """
    panel = ui.Panel2D(size=(300, 150), has_border=True, border_width=10)

    assert panel.has_border is True
    assert isinstance(panel.borders["left"], ui.Rectangle2D)
    assert len(panel.borders) == 4

    assert len(panel._elements) == 5

    npt.assert_equal(panel.borders["left"].size, (10, 160))
    npt.assert_equal(panel.borders["right"].size, (10, 160))
    npt.assert_equal(panel.borders["top"].size, (310, 10))
    npt.assert_equal(panel.borders["bottom"].size, (310, 10))


def test_panel2d_color_opacity_properties():
    """Test Panel2D color and opacity setters/getters proxy to background."""
    color = (0.9, 0.5, 0.1)
    opacity = 0.8

    panel = ui.Panel2D(size=(10, 10), color=color, opacity=opacity)

    npt.assert_array_almost_equal(panel.color, color)
    npt.assert_array_almost_equal(panel.background.color, color)

    npt.assert_almost_equal(panel.opacity, opacity)
    npt.assert_almost_equal(panel.background.opacity, opacity)

    new_color = (0.2, 0.4, 0.6)
    new_opacity = 0.2

    panel.color = new_color
    npt.assert_array_almost_equal(panel.color, new_color)
    npt.assert_array_almost_equal(panel.background.color, new_color)

    panel.opacity = new_opacity
    npt.assert_almost_equal(panel.opacity, new_opacity)
    npt.assert_almost_equal(panel.background.opacity, new_opacity)


def test_panel2d_color_normalization():
    """Test Panel2D normalizes hex / [0, 255] colors for body and border."""
    panel = ui.Panel2D(
        size=(10, 10), color="#FF0000", has_border=True, border_color=(0, 255, 0)
    )
    npt.assert_array_almost_equal(panel.color, [1, 0, 0])
    npt.assert_array_almost_equal(panel.background.color, [1, 0, 0])
    npt.assert_array_almost_equal(panel.borders["top"].color, [0, 1, 0])

    panel.color = (0, 0, 255)
    npt.assert_array_almost_equal(panel.color, [0, 0, 1])

    panel.border_color = ("left", "#FFFFFF")
    npt.assert_array_almost_equal(panel.borders["left"].color, [1, 1, 1])


def test_panel2d_resize_and_border_sync():
    """Test resizing updates background and correctly sizes border elements."""
    size = (100, 50)
    panel = ui.Panel2D(size=size, has_border=True, border_width=5)

    npt.assert_equal(panel.size, size)
    npt.assert_equal(panel.background.size, size)

    npt.assert_equal(panel.borders["left"].size, (5, 55))
    npt.assert_equal(panel.borders["right"].size, (5, 55))
    npt.assert_equal(panel.borders["top"].size, (105, 5))
    npt.assert_equal(panel.borders["bottom"].size, (105, 5))

    new_size = (300, 150)
    panel.resize(new_size)

    npt.assert_equal(panel.size, new_size)
    npt.assert_equal(panel.background.size, new_size)

    npt.assert_equal(panel.borders["left"].size, (5, 155))
    npt.assert_equal(panel.borders["right"].size, (5, 155))
    npt.assert_equal(panel.borders["top"].size, (305, 5))
    npt.assert_equal(panel.borders["bottom"].size, (305, 5))


def test_panel2d_add_remove_element():
    """Test add_element handles normalization and removes correctly."""
    panel = ui.Panel2D(size=(100, 100))
    mock_child = ui.Rectangle2D(size=(10, 10))

    initial_elements_count = len(panel._elements)

    panel.add_element(mock_child, coords=(0.5, 0.75))

    assert mock_child in panel._elements
    assert mock_child in panel._children
    assert len(panel._elements) == initial_elements_count + 1

    mock_child_2 = ui.Rectangle2D(size=(5, 5))
    panel.add_element(mock_child_2, coords=(10, 20))
    assert mock_child_2 in panel._elements
    assert mock_child_2 in panel._children

    panel.remove_element(mock_child)
    assert mock_child not in panel._elements
    assert mock_child not in panel._children

    with npt.assert_raises(ValueError):
        panel.remove_element(mock_child)


def test_panel2d_no_border_actors_and_scene():
    """
    Test Panel2D without borders can return actors and be added to a
    scene.
    """
    panel = ui.Panel2D(size=(200, 200))

    # _get_actors should work without borders
    actors = panel.actors
    assert len(actors) > 0

    # _update_actors_position should work without borders
    panel.set_position((10, 10))

    # Adding to a scene should not raise
    scene = window.Scene()
    scene.add(panel)

    # Adding child elements should also work
    child = ui.Rectangle2D(size=(20, 20))
    panel.add_element(child, (0.5, 0.5))
    panel._update_actors_position()

    # Border properties should return empty when has_border=False
    assert panel.border_color == []
    assert panel.border_width == []
    panel.update_border_coords()
    panel.resize((300, 300))


def test_panel2d_z_ordering():
    """
    Test that internal Z-ordering is set correctly by
    _update_actors_position.
    """
    panel = ui.Panel2D(size=(100, 100), has_border=True, border_width=1)

    custom_element = ui.Rectangle2D(size=(5, 5))
    panel.add_element(custom_element, coords=(0, 0))

    panel._update_actors_position()

    assert panel.background.z_order == 0
    assert all(border.z_order == 1 for border in panel.borders.values())
    assert custom_element.z_order == 2


def test_panel2d_visual_snapshot():
    """
    Visual test for Panel2D: Verifies composite rendering and color
    checks.
    """
    panel_size = (300, 150)
    panel_pos = (100, 100)

    panel = ui.Panel2D(
        size=panel_size,
        position=panel_pos,
        color=(0.1, 0.1, 0.1),
        has_border=True,
        border_width=5,
        border_color=(1.0, 0.0, 1.0),
    )

    disk = ui.Disk2D(outer_radius=25, color=(1, 1, 0))
    panel.add_element(disk, coords=(0.5, 0.5), anchor="center")

    scene = window.Scene()
    scene.add(panel)

    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    assert report.objects >= 1
    assert np.sum(arr) > 0

    mean_r, mean_g, mean_b, _ = np.mean(arr.reshape(-1, arr.shape[2]), axis=0)
    assert mean_r > mean_g
    assert mean_b > mean_g

    scene = window.Scene()
    scene.add(panel)

    panel.set_visibility(False)
    arr_hidden = window.snapshot(scene=scene, fname=None, return_array=True)
    report_hidden = window.analyze_snapshot(arr_hidden, find_objects=True)
    assert report_hidden.objects == 0
    npt.assert_almost_equal(np.mean(arr_hidden[..., :3]), 0, decimal=0)


def test_tab_panel2d_properties_and_content():
    """Test TabPanel2D title proxies and content element management."""
    tab_panel = ui.TabPanel2D(size=(120, 30), title="Tab 1")

    npt.assert_equal(tab_panel.size, (120, 30))
    npt.assert_equal(tab_panel.title, "Tab 1")
    npt.assert_equal(tab_panel.title_bold, False)
    npt.assert_equal(tab_panel.title_italic, False)
    npt.assert_equal(tab_panel.title_font_size, 18)
    npt.assert_equal(tab_panel.actors, [])
    assert not tab_panel.content_panel.actors[0].visible

    tab_panel_with_content = ui.TabPanel2D(
        size=(120, 30), content_size=(200, 180), content_visible=True
    )
    npt.assert_equal(tab_panel_with_content.content_panel.size, (200, 180))
    assert tab_panel_with_content.content_panel.actors[0].visible

    tab_panel.title = "Updated"
    tab_panel.title_bold = True
    tab_panel.title_italic = True
    tab_panel.title_color = (1, 0, 0)
    tab_panel.title_font_size = 12

    npt.assert_equal(tab_panel.title, "Updated")
    npt.assert_equal(tab_panel.title_bold, True)
    npt.assert_equal(tab_panel.title_italic, True)
    npt.assert_array_almost_equal(tab_panel.title_color, (1, 0, 0))
    npt.assert_equal(tab_panel.title_font_size, 12)

    element = ui.Rectangle2D(size=(10, 10))
    tab_panel.add_element(element, (0.5, 0.5))
    assert element in tab_panel.content_panel._elements
    assert not element.actors[0].visible

    tab_panel.update_element(element, (10, 10))
    tab_panel.remove_element(element)
    assert element not in tab_panel.content_panel._elements


def test_tab_panel2d_color_normalization():
    """Test TabPanel2D normalizes hex / [0, 255] colors."""
    tab_panel = ui.TabPanel2D(size=(120, 30), color="#FF0000")
    npt.assert_array_almost_equal(tab_panel.color, [1, 0, 0])
    npt.assert_array_almost_equal(tab_panel.panel.color, [1, 0, 0])

    tab_panel.color = (0, 0, 255)
    npt.assert_array_almost_equal(tab_panel.color, [0, 0, 1])

    tab_panel.title_color = "#00FF00"
    npt.assert_array_almost_equal(tab_panel.title_color, [0, 1, 0])


def test_tab_ui_initialization_and_selection():
    """Test TabUI initialization, element visibility, and tab selection."""
    tab_ui = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        font_size=14,
        startup_tab_id=1,
    )

    npt.assert_equal(tab_ui.size, (300, 300))
    npt.assert_equal(len(tab_ui.tabs), 3)
    npt.assert_equal(tab_ui.actors, [])
    npt.assert_equal(tab_ui.tabs[0].title, "Tab 1")
    npt.assert_equal(tab_ui.tabs[1].title, "Tab 2")
    npt.assert_equal(tab_ui.tabs[2].title, "Tab 3")
    npt.assert_equal(tab_ui.tabs[0].title_font_size, 14)
    npt.assert_equal(tab_ui.tabs[1].title_font_size, 14)
    npt.assert_equal(tab_ui.tabs[2].title_font_size, 14)
    npt.assert_equal(tab_ui.active_tab_idx, 1)
    npt.assert_equal(tab_ui.collapsed, False)

    assert not tab_ui.tabs[0].content_panel.actors[0].visible
    assert tab_ui.tabs[1].content_panel.actors[0].visible
    assert not tab_ui.tabs[2].content_panel.actors[0].visible

    element = ui.Rectangle2D(size=(20, 20))
    tab_ui.add_element(1, element, (0.5, 0.5))
    assert element.actors[0].visible

    changes = []
    collapses = []
    tab_ui.on_change = lambda ui_obj: changes.append(ui_obj.active_tab_idx)
    tab_ui.on_collapse = lambda ui_obj: collapses.append(ui_obj.collapsed)

    tab_ui.select_tab(2)
    npt.assert_equal(tab_ui.active_tab_idx, 2)
    npt.assert_equal(changes, [2])
    assert not tab_ui.tabs[1].content_panel.actors[0].visible
    assert tab_ui.tabs[2].content_panel.actors[0].visible

    tab_ui.select_tab(2)
    npt.assert_equal(tab_ui.active_tab_idx, 2)
    npt.assert_equal(tab_ui.collapsed, False)
    npt.assert_equal(changes, [2, 2])
    npt.assert_equal(collapses, [])
    assert not tab_ui.tabs[2].content_panel.actors[0].visible

    tab_ui.collapse_tab_ui()
    npt.assert_equal(tab_ui.active_tab_idx, None)
    npt.assert_equal(tab_ui.collapsed, True)
    npt.assert_equal(collapses, [True])


def test_tab_ui_layout_and_invalid_index():
    """Test TabUI layout and invalid index checks."""
    tab_ui_top = ui.TabUI(
        position=(50, 50), size=(300, 300), tab_titles=["Tab 1", "Tab 2", "Tab 3"]
    )
    tab_ui_bottom = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="bottom",
    )
    tab_ui_left = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="left",
    )
    tab_ui_right = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="right",
    )
    tab_ui_accordion = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        startup_tab_id=1,
        tab_bar_pos="accordion",
    )
    tab_ui_invalid = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="side",
    )

    npt.assert_equal(tab_ui_top.tabs[0].size, (100, 30))
    npt.assert_equal(tab_ui_top.tabs[0].content_panel.size, (300, 270))
    npt.assert_array_equal(tab_ui_top.tabs[0].get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_top.tabs[0].content_panel.get_position(), (50, 80))

    npt.assert_array_equal(tab_ui_bottom.tabs[0].content_panel.get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_bottom.tabs[0].get_position(), (50, 320))

    npt.assert_equal(tab_ui_left.tabs[0].size, (30, 100))
    npt.assert_equal(tab_ui_left.tabs[0].content_panel.size, (270, 300))
    npt.assert_array_equal(tab_ui_left.tabs[0].get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_left.tabs[1].get_position(), (50, 150))
    npt.assert_array_equal(tab_ui_left.tabs[0].content_panel.get_position(), (80, 50))

    npt.assert_equal(tab_ui_right.tabs[0].size, (30, 100))
    npt.assert_equal(tab_ui_right.tabs[0].content_panel.size, (270, 300))
    npt.assert_array_equal(tab_ui_right.tabs[0].content_panel.get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_right.tabs[0].get_position(), (320, 50))
    npt.assert_array_equal(tab_ui_right.tabs[1].get_position(), (320, 150))

    npt.assert_equal(tab_ui_accordion.tabs[0].size, (300, 30))
    npt.assert_equal(tab_ui_accordion.tabs[0].content_panel.size, (300, 210))
    npt.assert_array_equal(tab_ui_accordion.tabs[0].get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_accordion.tabs[1].get_position(), (50, 80))
    npt.assert_array_equal(
        tab_ui_accordion.tabs[1].content_panel.get_position(), (50, 110)
    )
    npt.assert_array_equal(tab_ui_accordion.tabs[2].get_position(), (50, 320))

    npt.assert_equal(tab_ui_invalid.tab_bar_pos, "top")
    npt.assert_equal(tab_ui_invalid.tabs[0].size, (100, 30))
    npt.assert_array_equal(
        tab_ui_invalid.tabs[0].content_panel.get_position(), (50, 80)
    )

    with npt.assert_raises(IndexError):
        tab_ui_top.add_element(3, ui.Rectangle2D(size=(10, 10)), (0.5, 0.5))

    with npt.assert_raises(IndexError):
        tab_ui_top.remove_element(3, ui.Rectangle2D(size=(10, 10)))

    with npt.assert_raises(IndexError):
        tab_ui_top.update_element(3, ui.Rectangle2D(size=(10, 10)), (0.5, 0.5))

    with npt.assert_raises(ValueError):
        ui.TabUI(position=(50, 50), size=(300, 300), tab_titles=[])

    with npt.assert_raises(ValueError):
        ui.TabUI(position=(50, 50), size=(300, 300), tab_titles="Tab")

    with npt.assert_raises(ValueError):
        ui.TabUI(position=(50, 50), size=(300, 300), tab_titles=["Tab", 2])


def test_ui_panel2d_visual(recording=False):
    panel = ui.Panel2D(
        size=(200, 200), color=(0.2, 0.3, 0.4), opacity=0.8, position=(300, 300)
    )

    event_counter = VisualTest()
    event_counter.monitor(panel)
    event_counter.record_or_test(
        pjoin(DATA_DIR, "test_ui_panel2d_container.log.gz"),
        pjoin(DATA_DIR, "test_ui_panel2d_container.json"),
        size=(600, 600),
        title="Panel2D Visual Test",
        recording=recording,
    )


def test_ui_tabui_visual(recording=False):
    tab_ui = ui.TabUI(
        size=(400, 300), tab_titles=["Home", "Settings"], position=(100, 100)
    )

    event_counter = VisualTest()
    event_counter.monitor(tab_ui)
    event_counter.record_or_test(
        pjoin(DATA_DIR, "test_ui_tabui_container.log.gz"),
        pjoin(DATA_DIR, "test_ui_tabui_container.json"),
        size=(600, 600),
        title="TabUI Visual Test",
        recording=recording,
    )
