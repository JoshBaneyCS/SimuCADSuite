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
}
