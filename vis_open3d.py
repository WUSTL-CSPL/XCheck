import open3d as o3d
import numpy as np
import matplotlib as mpl
from open3d.visualization import rendering, gui
from point_cloud_funcs import *
from matplotlib import cm


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"
    TRANSPARENT = "defaultLitTransparency"
    SSR = "defaultLitSSR"


    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(0.4, 0.364, 0.356)
        self.use_cmap = False

        self.apply_material = True  # clear to False after processing
       
        self._voxel_material = rendering.MaterialRecord()
        self._reg_material = rendering.MaterialRecord()
        self._ray_material = rendering.MaterialRecord()
        self._ray_mesh_material = rendering.MaterialRecord()
        
        self._voxel_material.base_color = [1, 1, 1, 1]
        self._voxel_material.shader = Settings.LIT

        self._reg_material.base_color = [0, 0, 0, 1]
        self._reg_material.shader = Settings.LIT

        self._ray_material.base_color = [0.8, 0, 0, 1]
        self._ray_material.shader = Settings.LIT

        self._ray_mesh_material.base_color = [1, 1, 1, 1]
        self._ray_mesh_material.shader = Settings.SSR

        self._ray_mesh_material.base_roughness = 0.0
        self._ray_mesh_material.base_reflectance = 0.0
        self._ray_mesh_material.base_clearcoat = 1.0
        self._ray_mesh_material.thickness = 1.0
        self._ray_mesh_material.transmission = 1.0
        self._ray_mesh_material.absorption_distance = 10

class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    def __init__(self, width, height, src, tgt, src_dist, tgt_dist, alpha_shape):

        self.src = src
        self.tgt = tgt
        self.src_dist = src_dist
        self.tgt_dist = tgt_dist
        self.alpha_shape = alpha_shape
        self.dist = self.src_dist
        self.obj = self.src

        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window("XCheck", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(0, gui.Margins(0.05 * em, 0.05 * em, 0.05 * em, 0.05 * em))

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed.
        mode_settings = gui.CollapsableVert("Mode", 0, gui.Margins(em, 0, 0, 0))
        material_settings = gui.CollapsableVert("Colormap", 0, gui.Margins(em, 0, 0, 0))


        self.upper_bound = max(self.src_dist)
        self.lower_bound = min(self.src_dist)

        #======= Mode Settings ============
        self._show_registration = gui.Checkbox("Registration")
        self._show_registration.set_on_checked(self._on_show_registration)

        self._show_ray = gui.Checkbox("Ray-based")
        self._show_ray.set_on_checked(self._on_show_ray)

        self._show_added_voxel = gui.Checkbox("Added Voxel")
        self._show_added_voxel.set_on_checked(self._on_show_added_voxel)

        self._show_missing_voxel = gui.Checkbox("Missing Voxel")
        self._show_missing_voxel.set_on_checked(self._on_show_missing_voxel)

        grid = gui.VGrid(2, 0.05 * em)
        grid.add_child(self._show_registration)
        grid.add_child(self._show_ray)
        grid.add_child(self._show_added_voxel)
        grid.add_child(self._show_missing_voxel)
        mode_settings.add_child(grid)

        #======= Colormap Settings ============
        self._use_cmap = gui.Checkbox("Colormap")
        self._use_cmap.set_on_checked(self._on_use_cmap)

        self._opacity = gui.Slider(gui.Slider.DOUBLE)
        self._opacity.set_limits(self.lower_bound, self.upper_bound)
        self._opacity.double_value = self.lower_bound
        self._opacity.set_on_value_changed(self._on_opacity)

        self._cmap_max = gui.Slider(gui.Slider.DOUBLE)
        self._cmap_max.set_limits(self.lower_bound, self.upper_bound)
        self._cmap_max.double_value = self.upper_bound
        self._cmap_max.set_on_value_changed(self._on_cmap_max)

        self._cmap_min = gui.Slider(gui.Slider.DOUBLE)
        self._cmap_min.set_limits(self.lower_bound, self.upper_bound)
        self._cmap_min.double_value = self.lower_bound
        self._cmap_min.set_on_value_changed(self._on_cmap_min)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.05 * em)
        grid.add_child(self._use_cmap)
        grid.add_child(self._opacity)
        grid.add_child(gui.Label("Opacity - Divergence Isolation"))
        grid.add_child(self._opacity)
        grid.add_child(gui.Label("Colormap - Upper Bound"))
        grid.add_child(self._cmap_max)
        grid.add_child(gui.Label("Colormap - Lower Bound"))
        grid.add_child(self._cmap_min)
        grid.add_child(gui.Label("Voxel size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(mode_settings)
        self._settings_panel.add_child(material_settings)
        
        # ----

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Show settings",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("Settings", settings_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings._voxel_material)
            self.settings.apply_material = False

        self._show_added_voxel.checked = True
        self._on_show_added_voxel(True)


    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.    
        r = self.window.content_rect
        self._scene.frame = r
        width = 25 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)


    def _on_show_registration(self, use):

        self.src.paint_uniform_color([0.5, 0.5, 0.5])
        self.tgt.paint_uniform_color([0.5, 0.5, 0.5])

        self._use_cmap.checked = False
        self._show_added_voxel.checked = False
        self._show_missing_voxel.checked = False
        self._show_ray.checked = False
        self._use_cmap.enabled = False
        self._opacity.enabled = False
        self._cmap_max.enabled = False
        self._cmap_min.enabled = False
        self._point_size.enabled = False

        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__model1__", self.src, self.settings._voxel_material)
        self._scene.scene.add_geometry("__model2__", self.tgt, self.settings._reg_material)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())


    def _on_show_ray(self, use):
        
        self._on_use_cmap(False)

        self._show_added_voxel.checked = False
        self._show_missing_voxel.checked = False
        self._show_registration.checked = False
        self._use_cmap.enabled = False
        self._opacity.enabled = False
        self._cmap_max.enabled = False
        self._cmap_min.enabled = False
        self._point_size.enabled = False

        self._scene.scene.clear_geometry()
        for i in range(len(self.alpha_shape)-1):
            self._scene.scene.add_geometry("__model__"+str(i), self.alpha_shape[i], self.settings._ray_material)
        
        self._scene.scene.add_geometry("__model__", self.alpha_shape[-1], self.settings._ray_mesh_material)


        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_show_added_voxel(self, use):

        self.src.paint_uniform_color([0.5, 0.5, 0.5])

        self._show_ray.checked = False
        self._show_registration.checked = False
        self._show_missing_voxel.checked = False
        self._use_cmap.enabled = True
        self._opacity.enabled = True
        self._cmap_max.enabled = True
        self._cmap_min.enabled = True
        self._point_size.enabled = True

        self.dist = self.src_dist
        self.obj = self.src

        self.upper_bound = max(self.src_dist)
        self.lower_bound = min(self.src_dist)
        self._cmap_max.set_limits(self.lower_bound, self.upper_bound)
        self._cmap_max.double_value = self.upper_bound
        self._cmap_min.set_limits(self.lower_bound, self.upper_bound)
        self._cmap_min.double_value = self.lower_bound
        
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__model__", self.src, self.settings._voxel_material)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())
        self._point_size.double_value = self.settings._voxel_material.point_size
        self._use_cmap.checked = self.settings.use_cmap 

    def _on_show_missing_voxel(self, use):

        self.tgt.paint_uniform_color([0.5, 0.5, 0.5])

        self._show_ray.checked = False
        self._show_registration.checked = False
        self._show_added_voxel.checked = False
        self._use_cmap.enabled = True
        self._opacity.enabled = True
        self._cmap_max.enabled = True
        self._cmap_min.enabled = True
        self._point_size.enabled = True

        self.dist = self.tgt_dist
        self.obj = self.tgt

        self.upper_bound = max(self.tgt_dist)
        self.lower_bound = min(self.tgt_dist)
        self._cmap_max.set_limits(self.lower_bound, self.upper_bound)
        self._cmap_max.double_value = self.upper_bound
        self._cmap_min.set_limits(self.lower_bound, self.upper_bound)
        self._cmap_min.double_value = self.lower_bound
        
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__model__", self.tgt, self.settings._voxel_material)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())
        self._point_size.double_value = self.settings._voxel_material.point_size
        self._use_cmap.checked = self.settings.use_cmap 


    def _on_use_cmap(self, use):
        self._use_cmap.checked = use
        self.settings.use_cmap = self._use_cmap.checked

        cmap = cm.jet

        if (self._use_cmap.checked):
            norm = mpl.colors.Normalize(vmin=min(self.dist), vmax=max(self.dist))
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            colors = np.delete( m.to_rgba(self.dist), -1, axis=1)
            self.obj.colors = o3d.utility.Vector3dVector(colors)
        else:
            self.obj.paint_uniform_color([0.5, 0.5, 0.5])

        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__model__", self.obj, self.settings._voxel_material)
        
    def _on_opacity(self, thres):

        filt = self.obj.select_by_index( np.where(self.dist > thres)[0] )

        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__model__", filt, self.settings._voxel_material)

    def _on_cmap_max(self, upper_bound):
        cmap = cm.jet

        if (self._use_cmap.checked and upper_bound > self.lower_bound):
            self.upper_bound = upper_bound
            norm = mpl.colors.Normalize(vmin=self.lower_bound, vmax=self.upper_bound)
        
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            colors = np.delete( m.to_rgba(self.dist), -1, axis=1)
            self.obj.colors = o3d.utility.Vector3dVector(colors)

            self._scene.scene.clear_geometry()
            self._scene.scene.add_geometry("__model__", self.obj, self.settings._voxel_material)

        self._cmap_min.set_limits(min(self.dist), upper_bound)

    def _on_cmap_min(self, lower_bound):
        cmap = cm.jet

        if (self._use_cmap.checked and lower_bound < self.upper_bound):
            self.lower_bound = lower_bound
            norm = mpl.colors.Normalize(vmin=self.lower_bound, vmax=self.upper_bound)
            
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            colors = np.delete( m.to_rgba(self.dist), -1, axis=1)
            self.obj.colors = o3d.utility.Vector3dVector(colors)

            self._scene.scene.clear_geometry()
            self._scene.scene.add_geometry("__model__", self.obj, self.settings._voxel_material)

        self._cmap_max.set_limits(lower_bound, max(self.dist))
    
    def _on_point_size(self, size):
        self.settings._voxel_material.point_size = int(size)
        self.settings.apply_material = True     

        self._point_size.double_value = self.settings._voxel_material.point_size       

        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__model__", self.obj, self.settings._voxel_material)
    

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("XCheck"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def visualizer_app(src, tgt, src_dist, tgt_dist, alpha_shape):

    # App Interface
    gui.Application.instance.initialize()
    app = AppWindow(1024, 640, src, tgt, src_dist, tgt_dist, alpha_shape)

    gui.Application.instance.run()