//! 3D viewport widget for egui.
//!
//! Embeds a wgpu-rendered 3D scene inside an egui panel, showing particles as
//! a point cloud, mesh geometry as wireframe, a ground grid, and coordinate
//! axes. Supports orbit camera controls via mouse drag, scroll, and
//! shift-drag.

use egui::Ui;

use crate::camera::OrbitCamera;
use crate::renderer_3d::{SceneCallback, Vertex3D};

// ---------------------------------------------------------------------------
// Viewport3D widget
// ---------------------------------------------------------------------------

/// A 3D viewport that can be embedded in any egui panel.
pub struct Viewport3D {
    /// The orbit camera controlling the view.
    pub camera: OrbitCamera,
    /// Whether to draw the ground grid.
    pub show_grid: bool,
    /// Whether to draw the coordinate axes.
    pub show_axes: bool,
    /// Half-extent of the ground grid.
    pub grid_size: f32,
    /// Number of grid divisions per axis.
    pub grid_divisions: u32,
}

impl Default for Viewport3D {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            show_grid: true,
            show_axes: true,
            grid_size: 10.0,
            grid_divisions: 20,
        }
    }
}

/// A colored edge: start position, end position, start RGBA, end RGBA.
pub type ColoredEdge = ([f64; 3], [f64; 3], [f32; 4], [f32; 4]);

impl Viewport3D {
    /// Render the 3D viewport.
    ///
    /// - `particles`: optional slice of `(x, y, z, speed)` tuples.
    /// - `mesh_edges`: optional slice of `(start, end)` position pairs for
    ///   wireframe.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        particles: Option<&[(f64, f64, f64, f64)]>,
        mesh_edges: Option<&[([f64; 3], [f64; 3])]>,
    ) {
        // Convert uniform-color edges to colored edges.
        let colored: Vec<ColoredEdge> = mesh_edges
            .map(|edges| {
                let color = [0.6_f32, 0.8, 1.0, 0.7];
                edges.iter().map(|(s, e)| (*s, *e, color, color)).collect()
            })
            .unwrap_or_default();

        let colored_ref = if mesh_edges.is_some() {
            Some(colored.as_slice())
        } else {
            None
        };

        self.show_impl(ui, particles, colored_ref);
    }

    /// Render the 3D viewport with per-vertex colored edges.
    ///
    /// Each edge carries individual start/end RGBA colours for field
    /// visualization on mesh wireframes.
    pub fn show_colored(
        &mut self,
        ui: &mut Ui,
        particles: Option<&[(f64, f64, f64, f64)]>,
        colored_edges: Option<&[ColoredEdge]>,
    ) {
        self.show_impl(ui, particles, colored_edges);
    }

    fn show_impl(
        &mut self,
        ui: &mut Ui,
        particles: Option<&[(f64, f64, f64, f64)]>,
        colored_edges: Option<&[ColoredEdge]>,
    ) {
        // Controls strip above the viewport.
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_grid, "Grid");
            ui.checkbox(&mut self.show_axes, "Axes");
            if ui.button("Reset Camera").clicked() {
                self.camera = OrbitCamera::default();
            }
            ui.label(
                egui::RichText::new("LMB: rotate  |  Shift+LMB: pan  |  Scroll: zoom")
                    .small()
                    .weak(),
            );
        });

        // Allocate the viewport rectangle.
        let available = ui.available_size();
        let size = egui::vec2(available.x, available.y.max(300.0).min(600.0));
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());

        // --- Input handling ---

        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            if ui.input(|i| i.modifiers.shift) {
                self.camera.pan(delta.x, delta.y);
            } else {
                self.camera.rotate(delta.x, delta.y);
            }
        }

        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            self.camera.pan(delta.x, delta.y);
        }

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.01 {
                self.camera.zoom(scroll * 0.01);
            }
        }

        // --- Build scene geometry ---

        let aspect = rect.width() / rect.height();
        let view_proj = self.camera.view_projection(aspect);

        let mut line_vertices = Vec::new();
        let mut point_vertices = Vec::new();

        // Ground grid.
        if self.show_grid {
            build_grid(&mut line_vertices, self.grid_size, self.grid_divisions);
        }

        // Coordinate axes.
        if self.show_axes {
            build_axes(&mut line_vertices, self.grid_size * 0.5);
        }

        // Mesh wireframe (colored edges).
        if let Some(edges) = colored_edges {
            for &(start, end, color_s, color_e) in edges {
                line_vertices.push(Vertex3D {
                    position: [start[0] as f32, start[1] as f32, start[2] as f32],
                    color: color_s,
                });
                line_vertices.push(Vertex3D {
                    position: [end[0] as f32, end[1] as f32, end[2] as f32],
                    color: color_e,
                });
            }
        }

        // Particle point cloud.
        if let Some(particles) = particles {
            let max_speed = particles
                .iter()
                .map(|p| p.3)
                .fold(0.0_f64, f64::max);
            for &(x, y, z, speed) in particles {
                let t = if max_speed > 1e-10 {
                    (speed / max_speed) as f32
                } else {
                    0.5
                };
                point_vertices.push(Vertex3D {
                    position: [x as f32, y as f32, z as f32],
                    color: velocity_color(t),
                });
            }
        }

        // --- Submit paint callback ---

        ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
            rect,
            SceneCallback {
                view_proj,
                point_vertices,
                line_vertices,
            },
        ));
    }
}

// ---------------------------------------------------------------------------
// Geometry builders
// ---------------------------------------------------------------------------

/// Build a flat grid on the XZ plane at y=0.
fn build_grid(verts: &mut Vec<Vertex3D>, size: f32, divisions: u32) {
    let half = size / 2.0;
    let step = size / divisions as f32;
    let color = [0.3, 0.3, 0.3, 0.5];

    for i in 0..=divisions {
        let t = -half + i as f32 * step;
        // Lines parallel to Z.
        verts.push(Vertex3D {
            position: [t, 0.0, -half],
            color,
        });
        verts.push(Vertex3D {
            position: [t, 0.0, half],
            color,
        });
        // Lines parallel to X.
        verts.push(Vertex3D {
            position: [-half, 0.0, t],
            color,
        });
        verts.push(Vertex3D {
            position: [half, 0.0, t],
            color,
        });
    }
}

/// Build RGB coordinate axes emanating from the origin.
fn build_axes(verts: &mut Vec<Vertex3D>, length: f32) {
    // X — red.
    verts.push(Vertex3D {
        position: [0.0, 0.0, 0.0],
        color: [1.0, 0.2, 0.2, 1.0],
    });
    verts.push(Vertex3D {
        position: [length, 0.0, 0.0],
        color: [1.0, 0.2, 0.2, 1.0],
    });
    // Y — green.
    verts.push(Vertex3D {
        position: [0.0, 0.0, 0.0],
        color: [0.2, 1.0, 0.2, 1.0],
    });
    verts.push(Vertex3D {
        position: [0.0, length, 0.0],
        color: [0.2, 1.0, 0.2, 1.0],
    });
    // Z — blue.
    verts.push(Vertex3D {
        position: [0.0, 0.0, 0.0],
        color: [0.2, 0.2, 1.0, 1.0],
    });
    verts.push(Vertex3D {
        position: [0.0, 0.0, length],
        color: [0.2, 0.2, 1.0, 1.0],
    });
}

/// Map a 0..1 speed ratio to a cool-to-warm colour ramp
/// (blue -> cyan -> green -> yellow -> red).
fn velocity_color(t: f32) -> [f32; 4] {
    let r = (2.0 * t - 0.5).clamp(0.0, 1.0);
    let g = (1.0 - (2.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    let b = (1.0 - 2.0 * t).clamp(0.0, 1.0);
    [r, g, b, 1.0]
}
