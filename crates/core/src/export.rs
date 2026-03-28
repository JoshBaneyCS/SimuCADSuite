use std::path::Path;

use rust_xlsxwriter::{Format, Workbook};
use serde::{Deserialize, Serialize};

use crate::error::SimuError;
use crate::types::Trajectory;

// ---------------------------------------------------------------------------
// DataColumn / DataTable — tabular simulation output
// ---------------------------------------------------------------------------

/// A named column of data for tabular export (CSV, Excel).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataColumn {
    pub name: String,
    pub unit: String,
    pub values: Vec<f64>,
}

/// A table of simulation results ready for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTable {
    pub title: String,
    pub columns: Vec<DataColumn>,
}

impl DataTable {
    /// Create an empty table with the given title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            columns: Vec::new(),
        }
    }

    /// Append a column of data to this table.
    pub fn add_column(
        &mut self,
        name: impl Into<String>,
        unit: impl Into<String>,
        values: Vec<f64>,
    ) {
        self.columns.push(DataColumn {
            name: name.into(),
            unit: unit.into(),
            values,
        });
    }

    /// The number of rows, determined by the longest column (0 if empty).
    pub fn row_count(&self) -> usize {
        self.columns.iter().map(|c| c.values.len()).max().unwrap_or(0)
    }

    /// Serialize the table to a CSV string.
    ///
    /// The header row contains `"name (unit)"` for each column.
    ///
    /// # Errors
    /// Returns `SimuError::Config` if the table has no columns.
    pub fn to_csv(&self) -> Result<String, SimuError> {
        if self.columns.is_empty() {
            return Err(SimuError::Config(
                "DataTable has no columns; cannot produce CSV".into(),
            ));
        }

        let rows = self.row_count();
        // Pre-allocate a reasonable buffer.
        let mut buf = String::with_capacity(rows * self.columns.len() * 12);

        // Header
        for (i, col) in self.columns.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            buf.push_str(&format!("{} ({})", col.name, col.unit));
        }
        buf.push('\n');

        // Data rows
        for row in 0..rows {
            for (i, col) in self.columns.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                if row < col.values.len() {
                    buf.push_str(&col.values[row].to_string());
                }
            }
            buf.push('\n');
        }

        Ok(buf)
    }

    /// Write the table to a CSV file at the given path.
    pub fn write_csv(&self, path: &Path) -> Result<(), SimuError> {
        let content = self.to_csv()?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Write the table to an XLSX (Excel) file at the given path.
    pub fn write_xlsx(&self, path: &Path) -> Result<(), SimuError> {
        if self.columns.is_empty() {
            return Err(SimuError::Config(
                "DataTable has no columns; cannot produce XLSX".into(),
            ));
        }

        let mut workbook = Workbook::new();
        let worksheet = workbook.add_worksheet();
        worksheet
            .set_name(&self.title)
            .map_err(|e| SimuError::Config(e.to_string()))?;

        let header_fmt = Format::new().set_bold();

        // Write header row.
        for (col_idx, col) in self.columns.iter().enumerate() {
            let header = format!("{} ({})", col.name, col.unit);
            worksheet
                .write_string_with_format(0, col_idx as u16, &header, &header_fmt)
                .map_err(|e| SimuError::Config(e.to_string()))?;
        }

        // Write data rows.
        let rows = self.row_count();
        for row in 0..rows {
            for (col_idx, col) in self.columns.iter().enumerate() {
                if row < col.values.len() {
                    worksheet
                        .write_number((row + 1) as u32, col_idx as u16, col.values[row])
                        .map_err(|e| SimuError::Config(e.to_string()))?;
                }
            }
        }

        workbook
            .save(path)
            .map_err(|e| SimuError::Config(format!("Failed to write XLSX: {e}")))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Multi-table export (multiple sheets in one XLSX)
// ---------------------------------------------------------------------------

/// Write multiple [`DataTable`]s to a single XLSX file, one sheet per table.
pub fn write_tables_xlsx(tables: &[DataTable], path: &Path) -> Result<(), SimuError> {
    if tables.is_empty() {
        return Err(SimuError::Config("No tables to export".into()));
    }

    let mut workbook = Workbook::new();
    let header_fmt = Format::new().set_bold();

    for table in tables {
        let worksheet = workbook.add_worksheet();
        worksheet
            .set_name(&table.title)
            .map_err(|e| SimuError::Config(e.to_string()))?;

        for (col_idx, col) in table.columns.iter().enumerate() {
            let header = format!("{} ({})", col.name, col.unit);
            worksheet
                .write_string_with_format(0, col_idx as u16, &header, &header_fmt)
                .map_err(|e| SimuError::Config(e.to_string()))?;
        }

        let rows = table.row_count();
        for row in 0..rows {
            for (col_idx, col) in table.columns.iter().enumerate() {
                if row < col.values.len() {
                    worksheet
                        .write_number((row + 1) as u32, col_idx as u16, col.values[row])
                        .map_err(|e| SimuError::Config(e.to_string()))?;
                }
            }
        }
    }

    workbook
        .save(path)
        .map_err(|e| SimuError::Config(format!("Failed to write XLSX: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Trajectory -> DataTable conversion
// ---------------------------------------------------------------------------

/// Convert a [`Trajectory`] to a [`DataTable`] suitable for CSV export.
///
/// Columns: time (s), x (m), y (m), vx (m/s), vy (m/s), speed (m/s).
pub fn trajectory_to_table(trajectory: &Trajectory, label: &str) -> DataTable {
    let n = trajectory.points.len();

    let mut time = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut vx = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    let mut speed = Vec::with_capacity(n);

    for pt in &trajectory.points {
        time.push(pt.time);
        x.push(pt.position.x);
        y.push(pt.position.y);
        vx.push(pt.velocity.x);
        vy.push(pt.velocity.y);
        speed.push(pt.speed);
    }

    let mut table = DataTable::new(label);
    table.add_column("time", "s", time);
    table.add_column("x", "m", x);
    table.add_column("y", "m", y);
    table.add_column("vx", "m/s", vx);
    table.add_column("vy", "m/s", vy);
    table.add_column("speed", "m/s", speed);
    table
}

// ---------------------------------------------------------------------------
// Generic JSON export for any Serialize type
// ---------------------------------------------------------------------------

/// Write any serializable value to a JSON file.
pub fn write_json<T: Serialize>(value: &T, path: &Path) -> Result<(), SimuError> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| SimuError::Config(format!("JSON serialization error: {e}")))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Read a JSON file into a deserializable value.
pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, SimuError> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content)
        .map_err(|e| SimuError::Config(format!("JSON parse error: {e}")))
}

// ---------------------------------------------------------------------------
// Project file — serializable snapshot of simulation parameters
// ---------------------------------------------------------------------------

/// A project file that captures simulation parameters across all panels.
///
/// This allows users to save their work and resume later without re-entering
/// all parameters. Only parameters are saved, not computed results (which
/// can be recomputed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFile {
    /// Format version for forward compatibility.
    pub version: u32,
    /// Optional project name.
    pub name: String,
    /// Kinematics panel parameters.
    pub kinematics: Option<KinematicsProject>,
    /// Fluid dynamics panel parameters.
    pub fluid: Option<FluidProject>,
    /// Calculator panel parameters.
    pub calculator: Option<CalculatorProject>,
}

impl Default for ProjectFile {
    fn default() -> Self {
        Self {
            version: 1,
            name: String::new(),
            kinematics: None,
            fluid: None,
            calculator: None,
        }
    }
}

impl ProjectFile {
    /// Save this project to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), SimuError> {
        write_json(self, path)
    }

    /// Load a project from a JSON file.
    pub fn load(path: &Path) -> Result<Self, SimuError> {
        read_json(path)
    }
}

/// Saved kinematics parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicsProject {
    pub velocity: f64,
    pub angle_deg: f64,
    pub gravity: f64,
    pub mass: f64,
    pub area: f64,
    pub initial_height: f64,
    pub drag_shape: String,
    pub integrator: String,
}

/// Saved fluid dynamics parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidProject {
    pub mesh_path: String,
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub velocity_z: f64,
    pub particle_count: usize,
    pub num_steps: usize,
    pub use_gpu: bool,
}

/// Saved calculator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculatorProject {
    pub expression: String,
    pub variable: String,
    pub plot_x_min: f64,
    pub plot_x_max: f64,
    pub plot_samples: usize,
    pub angle_mode: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TrajectoryPoint, Vec2};

    #[test]
    fn empty_table_row_count() {
        let t = DataTable::new("test");
        assert_eq!(t.row_count(), 0);
    }

    #[test]
    fn add_columns_and_row_count() {
        let mut t = DataTable::new("test");
        t.add_column("a", "m", vec![1.0, 2.0, 3.0]);
        t.add_column("b", "s", vec![4.0, 5.0]);
        assert_eq!(t.row_count(), 3);
    }

    #[test]
    fn to_csv_basic() {
        let mut t = DataTable::new("test");
        t.add_column("x", "m", vec![1.0, 2.0]);
        t.add_column("y", "m", vec![3.0, 4.0]);
        let csv = t.to_csv().unwrap();
        assert!(csv.starts_with("x (m),y (m)\n"));
        assert!(csv.contains("1,3\n") || csv.contains("1,3\n"));
    }

    #[test]
    fn to_csv_empty_table_errors() {
        let t = DataTable::new("test");
        assert!(t.to_csv().is_err());
    }

    #[test]
    fn to_csv_ragged_columns() {
        let mut t = DataTable::new("test");
        t.add_column("a", "m", vec![1.0, 2.0, 3.0]);
        t.add_column("b", "s", vec![10.0]);
        let csv = t.to_csv().unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4); // header + 3 data rows
        // Row 2 and 3 should have missing b values (empty after comma)
        assert!(lines[2].ends_with(','));
    }

    #[test]
    fn trajectory_to_table_roundtrip() {
        let traj = Trajectory {
            points: vec![
                TrajectoryPoint {
                    time: 0.0,
                    position: Vec2::new(0.0, 0.0),
                    velocity: Vec2::new(10.0, 10.0),
                    speed: 14.142_135_623_730_951,
                },
                TrajectoryPoint {
                    time: 1.0,
                    position: Vec2::new(10.0, 5.0),
                    velocity: Vec2::new(10.0, 0.0),
                    speed: 10.0,
                },
            ],
            max_height: 5.0,
            range: 20.0,
            flight_time: 2.0,
        };

        let table = trajectory_to_table(&traj, "Test Trajectory");
        assert_eq!(table.title, "Test Trajectory");
        assert_eq!(table.columns.len(), 6);
        assert_eq!(table.row_count(), 2);

        // Verify CSV generation works end-to-end
        let csv = table.to_csv().unwrap();
        assert!(csv.contains("time (s)"));
        assert!(csv.contains("speed (m/s)"));
    }

    #[test]
    fn write_csv_file() {
        let mut t = DataTable::new("test");
        t.add_column("x", "m", vec![1.0, 2.0]);
        t.add_column("y", "m", vec![3.0, 4.0]);

        let dir = std::env::temp_dir().join("simucad_test_csv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");

        t.write_csv(&path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("x (m),y (m)\n"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_xlsx_file() {
        let mut t = DataTable::new("Sheet1");
        t.add_column("x", "m", vec![1.0, 2.0, 3.0]);
        t.add_column("y", "m", vec![4.0, 5.0, 6.0]);

        let dir = std::env::temp_dir().join("simucad_test_xlsx");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.xlsx");

        t.write_xlsx(&path).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn project_save_load_roundtrip() {
        let project = ProjectFile {
            version: 1,
            name: "Test Project".into(),
            kinematics: Some(KinematicsProject {
                velocity: 50.0,
                angle_deg: 45.0,
                gravity: 9.81,
                mass: 1.0,
                area: 0.01,
                initial_height: 0.0,
                drag_shape: "Sphere".into(),
                integrator: "RK4".into(),
            }),
            fluid: None,
            calculator: Some(CalculatorProject {
                expression: "sin(x)".into(),
                variable: "x".into(),
                plot_x_min: -10.0,
                plot_x_max: 10.0,
                plot_samples: 500,
                angle_mode: "Radians".into(),
            }),
        };

        let dir = std::env::temp_dir().join("simucad_test_project");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.simucad");

        project.save(&path).unwrap();
        let loaded = ProjectFile::load(&path).unwrap();

        assert_eq!(loaded.name, "Test Project");
        assert_eq!(loaded.version, 1);

        let kin = loaded.kinematics.unwrap();
        assert_eq!(kin.velocity, 50.0);
        assert_eq!(kin.angle_deg, 45.0);

        let calc = loaded.calculator.unwrap();
        assert_eq!(calc.expression, "sin(x)");

        assert!(loaded.fluid.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_json_roundtrip() {
        let data = vec![1.0, 2.0, 3.0];
        let dir = std::env::temp_dir().join("simucad_test_json");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.json");

        write_json(&data, &path).unwrap();
        let loaded: Vec<f64> = read_json(&path).unwrap();
        assert_eq!(loaded, data);

        std::fs::remove_dir_all(&dir).ok();
    }
}
