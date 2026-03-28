//! Mesh Viewer UI panel.
//!
//! Load or generate meshes, inspect quality metrics, visualize scalar fields
//! on the wireframe, and perform midpoint refinement — all with interactive
//! 3D orbit-camera controls.

use egui::Ui;

use simucad_mesh::field::{
    colorize_nodes, colormap_coolwarm_rgba, colormap_jet_rgba, colormap_viridis_rgba,
    ColormapFn, MeshScalarField,
};
use simucad_mesh::io::{GmshLoader, MeshLoader, ObjLoader, StlLoader};
use simucad_mesh::quality::{self, MeshQualityReport};
use simucad_mesh::types::Mesh;

use crate::viewport_3d::{ColoredEdge, Viewport3D};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Source of the mesh data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshSource {
    #[default]
    File,
    GenerateRect,
    GenerateDisk,
    GenerateBox,
}

/// Scalar function to evaluate on mesh nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarFunctionChoice {
    #[default]
    X,
    Y,
    Z,
    Distance,
}

/// Colormap selection for field visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshColormapChoice {
    #[default]
    Viridis,
    Coolwarm,
    Jet,
}

impl MeshColormapChoice {
    fn function(&self) -> ColormapFn {
        match self {
            Self::Viridis => colormap_viridis_rgba,
            Self::Coolwarm => colormap_coolwarm_rgba,
            Self::Jet => colormap_jet_rgba,
        }
    }
}

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

/// State for the mesh viewer panel.
pub struct MeshPanel {
    // -- Source --
    pub mesh_source: MeshSource,
    pub mesh_path: String,

    // -- Rectangle generator --
    pub rect_x_min: f64,
    pub rect_x_max: f64,
    pub rect_y_min: f64,
    pub rect_y_max: f64,
    pub rect_nx: usize,
    pub rect_ny: usize,

    // -- Disk generator --
    pub disk_cx: f64,
    pub disk_cy: f64,
    pub disk_radius: f64,
    pub disk_rings: usize,
    pub disk_sectors: usize,

    // -- Box generator --
    pub box_x_min: f64,
    pub box_x_max: f64,
    pub box_y_min: f64,
    pub box_y_max: f64,
    pub box_z_min: f64,
    pub box_z_max: f64,
    pub box_nx: usize,
    pub box_ny: usize,
    pub box_nz: usize,

    // -- Current mesh --
    pub mesh: Option<Mesh>,
    pub quality_report: Option<MeshQualityReport>,
    pub status: String,

    // -- Field viz --
    pub field_function: ScalarFunctionChoice,
    pub colormap_choice: MeshColormapChoice,
    pub show_field_coloring: bool,

    // -- Cached render data --
    colored_edges: Vec<ColoredEdge>,

    // -- Viewport --
    pub viewport: Viewport3D,
}

impl Default for MeshPanel {
    fn default() -> Self {
        Self {
            mesh_source: MeshSource::GenerateRect,
            mesh_path: String::new(),

            rect_x_min: 0.0,
            rect_x_max: 1.0,
            rect_y_min: 0.0,
            rect_y_max: 1.0,
            rect_nx: 10,
            rect_ny: 10,

            disk_cx: 0.0,
            disk_cy: 0.0,
            disk_radius: 1.0,
            disk_rings: 5,
            disk_sectors: 16,

            box_x_min: 0.0,
            box_x_max: 1.0,
            box_y_min: 0.0,
            box_y_max: 1.0,
            box_z_min: 0.0,
            box_z_max: 1.0,
            box_nx: 3,
            box_ny: 3,
            box_nz: 3,

            mesh: None,
            quality_report: None,
            status: String::new(),

            field_function: ScalarFunctionChoice::X,
            colormap_choice: MeshColormapChoice::Viridis,
            show_field_coloring: true,

            colored_edges: Vec::new(),

            viewport: Viewport3D::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

impl MeshPanel {
    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("Mesh Viewer");
        ui.add_space(8.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            self.show_source_controls(ui);
            ui.add_space(4.0);

            if self.mesh.is_some() {
                self.show_mesh_info(ui);
                ui.add_space(4.0);
                self.show_quality_section(ui);
                ui.add_space(4.0);
                self.show_field_controls(ui);
                ui.add_space(4.0);
            }

            if !self.status.is_empty() {
                ui.label(&self.status);
                ui.add_space(4.0);
            }

            // 3D viewport.
            if !self.colored_edges.is_empty() {
                self.viewport
                    .show_colored(ui, None, Some(&self.colored_edges));
            } else if self.mesh.is_some() {
                self.viewport.show_colored(ui, None, Some(&[]));
            }
        });
    }

    fn show_source_controls(&mut self, ui: &mut Ui) {
        ui.collapsing("Mesh Source", |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mesh_source, MeshSource::File, "File");
                ui.selectable_value(&mut self.mesh_source, MeshSource::GenerateRect, "Rectangle");
                ui.selectable_value(&mut self.mesh_source, MeshSource::GenerateDisk, "Disk");
                ui.selectable_value(&mut self.mesh_source, MeshSource::GenerateBox, "Box");
            });

            match self.mesh_source {
                MeshSource::File => {
                    ui.horizontal(|ui| {
                        ui.label("Path:");
                        ui.text_edit_singleline(&mut self.mesh_path);
                    });
                    ui.label(
                        egui::RichText::new("Supports: .msh (Gmsh), .stl, .obj")
                            .small()
                            .weak(),
                    );
                }
                MeshSource::GenerateRect => {
                    ui.horizontal(|ui| {
                        ui.label("x:");
                        ui.add(egui::DragValue::new(&mut self.rect_x_min).speed(0.1).prefix("min "));
                        ui.add(egui::DragValue::new(&mut self.rect_x_max).speed(0.1).prefix("max "));
                    });
                    ui.horizontal(|ui| {
                        ui.label("y:");
                        ui.add(egui::DragValue::new(&mut self.rect_y_min).speed(0.1).prefix("min "));
                        ui.add(egui::DragValue::new(&mut self.rect_y_max).speed(0.1).prefix("max "));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Divisions:");
                        ui.add(egui::DragValue::new(&mut self.rect_nx).speed(1.0).prefix("nx ").range(1..=200));
                        ui.add(egui::DragValue::new(&mut self.rect_ny).speed(1.0).prefix("ny ").range(1..=200));
                    });
                }
                MeshSource::GenerateDisk => {
                    ui.horizontal(|ui| {
                        ui.label("Center:");
                        ui.add(egui::DragValue::new(&mut self.disk_cx).speed(0.1).prefix("cx "));
                        ui.add(egui::DragValue::new(&mut self.disk_cy).speed(0.1).prefix("cy "));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Radius:");
                        ui.add(egui::DragValue::new(&mut self.disk_radius).speed(0.1).range(0.01..=100.0));
                    });
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.disk_rings).speed(1.0).prefix("rings ").range(1..=50));
                        ui.add(egui::DragValue::new(&mut self.disk_sectors).speed(1.0).prefix("sectors ").range(3..=64));
                    });
                }
                MeshSource::GenerateBox => {
                    ui.horizontal(|ui| {
                        ui.label("x:");
                        ui.add(egui::DragValue::new(&mut self.box_x_min).speed(0.1).prefix("min "));
                        ui.add(egui::DragValue::new(&mut self.box_x_max).speed(0.1).prefix("max "));
                    });
                    ui.horizontal(|ui| {
                        ui.label("y:");
                        ui.add(egui::DragValue::new(&mut self.box_y_min).speed(0.1).prefix("min "));
                        ui.add(egui::DragValue::new(&mut self.box_y_max).speed(0.1).prefix("max "));
                    });
                    ui.horizontal(|ui| {
                        ui.label("z:");
                        ui.add(egui::DragValue::new(&mut self.box_z_min).speed(0.1).prefix("min "));
                        ui.add(egui::DragValue::new(&mut self.box_z_max).speed(0.1).prefix("max "));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Divisions:");
                        ui.add(egui::DragValue::new(&mut self.box_nx).speed(1.0).prefix("nx ").range(1..=30));
                        ui.add(egui::DragValue::new(&mut self.box_ny).speed(1.0).prefix("ny ").range(1..=30));
                        ui.add(egui::DragValue::new(&mut self.box_nz).speed(1.0).prefix("nz ").range(1..=30));
                    });
                }
            }

            ui.horizontal(|ui| {
                if ui
                    .button(if self.mesh_source == MeshSource::File {
                        "Load"
                    } else {
                        "Generate"
                    })
                    .clicked()
                {
                    self.load_or_generate();
                }

                if self.mesh.is_some() && ui.button("Subdivide").clicked() {
                    self.subdivide_current();
                }
            });
        });
    }

    fn show_mesh_info(&self, ui: &mut Ui) {
        if let Some(ref mesh) = self.mesh {
            ui.collapsing("Mesh Info", |ui| {
                ui.label(format!(
                    "Nodes: {}  |  Elements: {}  |  Dimension: {}D",
                    mesh.node_count(),
                    mesh.element_count(),
                    mesh.dimension,
                ));
                let bb = mesh.bounding_box();
                ui.label(format!(
                    "Bounds: x [{:.3}, {:.3}]  y [{:.3}, {:.3}]  z [{:.3}, {:.3}]",
                    bb.min.x, bb.max.x, bb.min.y, bb.max.y, bb.min.z, bb.max.z,
                ));
                ui.label(format!("Edges (wireframe): {}", self.colored_edges.len()));
            });
        }
    }

    fn show_quality_section(&self, ui: &mut Ui) {
        if let Some(ref report) = self.quality_report {
            ui.collapsing("Quality Report", |ui| {
                ui.label(format!("Mean aspect ratio: {:.3}", report.mean_aspect_ratio));
                ui.label(format!("Mean skewness: {:.4}", report.mean_skewness));
                ui.label(format!(
                    "Total area/volume: {:.6}",
                    report.total_area_or_volume
                ));
                if report.degenerate_count > 0 {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        format!("Degenerate elements: {}", report.degenerate_count),
                    );
                } else {
                    ui.label("No degenerate elements");
                }
            });
        }
    }

    fn show_field_controls(&mut self, ui: &mut Ui) {
        ui.collapsing("Field Visualization", |ui| {
            ui.checkbox(&mut self.show_field_coloring, "Color wireframe by field");

            if self.show_field_coloring {
                let prev_fn = self.field_function;
                let prev_cm = self.colormap_choice;

                ui.horizontal(|ui| {
                    ui.label("Function:");
                    ui.selectable_value(&mut self.field_function, ScalarFunctionChoice::X, "X");
                    ui.selectable_value(&mut self.field_function, ScalarFunctionChoice::Y, "Y");
                    ui.selectable_value(&mut self.field_function, ScalarFunctionChoice::Z, "Z");
                    ui.selectable_value(
                        &mut self.field_function,
                        ScalarFunctionChoice::Distance,
                        "Distance",
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Colormap:");
                    ui.selectable_value(
                        &mut self.colormap_choice,
                        MeshColormapChoice::Viridis,
                        "Viridis",
                    );
                    ui.selectable_value(
                        &mut self.colormap_choice,
                        MeshColormapChoice::Coolwarm,
                        "Cool-Warm",
                    );
                    ui.selectable_value(
                        &mut self.colormap_choice,
                        MeshColormapChoice::Jet,
                        "Jet",
                    );
                });

                // Recompute if settings changed.
                if self.field_function != prev_fn || self.colormap_choice != prev_cm {
                    self.recompute_edges();
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------------

    fn load_or_generate(&mut self) {
        let result = match self.mesh_source {
            MeshSource::File => self.load_from_file(),
            MeshSource::GenerateRect => {
                let m = simucad_mesh::generator::generate_rectangle_mesh(
                    self.rect_x_min,
                    self.rect_x_max,
                    self.rect_y_min,
                    self.rect_y_max,
                    self.rect_nx,
                    self.rect_ny,
                );
                Ok(m)
            }
            MeshSource::GenerateDisk => {
                let m = simucad_mesh::generator::generate_disk_mesh(
                    self.disk_cx,
                    self.disk_cy,
                    self.disk_radius,
                    self.disk_rings,
                    self.disk_sectors,
                );
                Ok(m)
            }
            MeshSource::GenerateBox => {
                let m = simucad_mesh::generator::generate_box_mesh(
                    self.box_x_min,
                    self.box_x_max,
                    self.box_y_min,
                    self.box_y_max,
                    self.box_z_min,
                    self.box_z_max,
                    self.box_nx,
                    self.box_ny,
                    self.box_nz,
                );
                Ok(m)
            }
        };

        match result {
            Ok(mesh) => {
                self.status = format!(
                    "Loaded mesh: {} nodes, {} elements",
                    mesh.node_count(),
                    mesh.element_count(),
                );

                // Compute quality.
                self.quality_report = quality::compute_mesh_quality(&mesh);

                // Fit camera.
                let bb = mesh.bounding_box();
                self.viewport.camera.fit_to_bounds(
                    [bb.min.x as f32, bb.min.y as f32, bb.min.z as f32],
                    [bb.max.x as f32, bb.max.y as f32, bb.max.z as f32],
                );

                self.mesh = Some(mesh);
                self.recompute_edges();
            }
            Err(e) => {
                self.status = format!("Error: {e}");
                self.mesh = None;
                self.quality_report = None;
                self.colored_edges.clear();
            }
        }
    }

    fn load_from_file(&self) -> Result<Mesh, String> {
        let path = std::path::Path::new(&self.mesh_path);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        match ext.as_str() {
            "msh" => GmshLoader::load(path).map_err(|e| e.to_string()),
            "stl" => StlLoader::load(path).map_err(|e| e.to_string()),
            "obj" => ObjLoader::load(path).map_err(|e| e.to_string()),
            _ => Err(format!("Unsupported format: .{ext} (use .msh, .stl, or .obj)")),
        }
    }

    fn subdivide_current(&mut self) {
        if let Some(ref mesh) = self.mesh {
            let refined = simucad_mesh::refine::subdivide_midpoint(mesh);
            self.status = format!(
                "Subdivided: {} nodes, {} elements",
                refined.node_count(),
                refined.element_count(),
            );
            self.quality_report = quality::compute_mesh_quality(&refined);
            self.mesh = Some(refined);
            self.recompute_edges();
        }
    }

    fn recompute_edges(&mut self) {
        let Some(ref mesh) = self.mesh else {
            self.colored_edges.clear();
            return;
        };

        let edge_pairs = mesh.edges();

        if self.show_field_coloring {
            let scalar_fn: fn(&simucad_core::types::Vec3) -> f64 = match self.field_function {
                ScalarFunctionChoice::X => |v| v.x,
                ScalarFunctionChoice::Y => |v| v.y,
                ScalarFunctionChoice::Z => |v| v.z,
                ScalarFunctionChoice::Distance => {
                    |v| (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
                }
            };

            let field = MeshScalarField::from_function(mesh, scalar_fn);
            let colors = colorize_nodes(&field, self.colormap_choice.function());

            self.colored_edges = edge_pairs
                .iter()
                .map(|&(a, b)| {
                    let pa = &mesh.nodes[a];
                    let pb = &mesh.nodes[b];
                    (
                        [pa.x, pa.y, pa.z],
                        [pb.x, pb.y, pb.z],
                        colors[a],
                        colors[b],
                    )
                })
                .collect();
        } else {
            let uniform = [0.6_f32, 0.8, 1.0, 0.7];
            self.colored_edges = edge_pairs
                .iter()
                .map(|&(a, b)| {
                    let pa = &mesh.nodes[a];
                    let pb = &mesh.nodes[b];
                    (
                        [pa.x, pa.y, pa.z],
                        [pb.x, pb.y, pb.z],
                        uniform,
                        uniform,
                    )
                })
                .collect();
        }

        // Warn if we are approaching vertex limits.
        if self.colored_edges.len() > 200_000 {
            self.status = format!(
                "Warning: {} edges ({} line vertices) — approaching GPU buffer limit",
                self.colored_edges.len(),
                self.colored_edges.len() * 2,
            );
        }
    }
}
