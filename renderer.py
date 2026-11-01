from vispy import scene
from vispy.scene import visuals
import numpy as np

class SimulationRenderer:
    def __init__(self, title="Cancer ABM Tensor Visualization"):
        # Dark 3D Viewport Window
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            title=title,
            show=True,
            bgcolor='#0A0A0C',
            size=(1280, 720)
        )

        # 3D Viewport Camera Setup
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.center = (75, 75, 75)
        self.view.camera.distance = 340
        self.view.camera.fov = 45

        # 3D Scatter Markers
        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        # Spatial Reference Grid Box (150x150x150)
        self.bounds_box = visuals.Box(
            width=150, height=150, depth=150,
            color=(1, 1, 1, 0.03),
            edge_color=(1, 1, 1, 0.25),
            parent=self.view.scene
        )
        self.bounds_box.transform = scene.transforms.STTransform(translate=(75, 75, 75))

        # 2D Interactive Overlay
        self.overlay = self.canvas.central_widget.add_view()
        self.overlay.camera = scene.PanZoomCamera(rect=(0, 0, 1280, 720))
        self.overlay.camera.interactive = False

        # ---------------------------------------------------------------------
        # LEGEND OVERLAY (Top-Left Corner)
        # ---------------------------------------------------------------------
        visuals.Text(
            text="CELL PHENOTYPES", parent=self.overlay.scene, color='#FFFFFF',
            font_size=11, bold=True, anchor_x='left', anchor_y='center',
            pos=(20, 35)
        )

        legend_items = [
            ("● Agent A (CXCR4-ve Cancer)", "#FF3333"),
            ("● Agent B (CXCR4+ve Cancer)", "#FF8000"),
            ("● Agent C (Stromal Cell)", "#33CC33"),
            ("● Agent D (Tcyto Naive)", "#3399FF"),
            ("● Agent D (Tcyto Activated)", "#00FFFF"),
            ("● Agent D (Tcyto Exhausted)", "#9933FF"),
            ("● Agent E (Treg Cell)", "#FFFF00")
        ]

        start_x, start_y, spacing = 20, 60, 22
        for i, (label, color_hex) in enumerate(legend_items):
            visuals.Text(
                text=label, parent=self.overlay.scene, color=color_hex,
                font_size=10, bold=False, anchor_x='left', anchor_y='center',
                pos=(start_x, start_y + (i * spacing))
            )

        # ---------------------------------------------------------------------
        # DYNAMIC ANALYTICS HUD (Top-Right Corner)
        # ---------------------------------------------------------------------
        visuals.Text(
            text="SIMULATION RUNTIME METRICS", parent=self.overlay.scene, color='#FFFFFF',
            font_size=11, bold=True, anchor_x='right', anchor_y='center',
            pos=(1260, 35)
        )

        self.hud_day = visuals.Text(
            text="Day: 0.00 / 40.00", parent=self.overlay.scene, color='#E0E0E0',
            font_size=11, anchor_x='right', anchor_y='center', pos=(1260, 60)
        )
        self.hud_cancer_vol = visuals.Text(
            text="Cancer Vol: 0.0000 mm³", parent=self.overlay.scene, color='#FF5555',
            font_size=11, bold=True, anchor_x='right', anchor_y='center', pos=(1260, 85)
        )
        self.hud_total_vol = visuals.Text(
            text="Total Vol: 0.0000 mm³", parent=self.overlay.scene, color='#FF9955',
            font_size=10, anchor_x='right', anchor_y='center', pos=(1260, 110)
        )

        self.canvas.events.resize.connect(self.on_resize)

    def on_resize(self, event):
        """Re-anchors 2D HUD text and legend during window resizing."""
        w, h = event.size
        self.overlay.camera.rect = (0, 0, w, h)
        self.hud_day.pos = (w - 20, 60)
        self.hud_cancer_vol.pos = (w - 20, 85)
        self.hud_total_vol.pos = (w - 20, 110)

    def update_visuals(self, positions, colors, day_num, total_days, cancer_vol_mm3, total_vol_mm3):
        """Pushes position buffers and updates HUD volume metrics."""
        if len(positions) > 0:
            self.scatter.set_data(
                pos=positions,
                face_color=colors,
                edge_color=None,
                size=5.0
            )

        self.hud_day.text = f"Day: {day_num:.2f} / {total_days:.0f}"
        self.hud_cancer_vol.text = f"Cancer Vol: {cancer_vol_mm3:.4f} mm³"
        self.hud_total_vol.text = f"Total Vol: {total_vol_mm3:.4f} mm³"
        self.canvas.update()