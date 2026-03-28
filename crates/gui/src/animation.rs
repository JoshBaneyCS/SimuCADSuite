//! Animation frame export: PNG sequence and animated GIF.
//!
//! Generates heatmap frames from 2D grid data (e.g. PDE solutions) and
//! encodes them as individual PNG files or a single animated GIF.

use std::path::Path;

use egui::Color32;

use crate::plotting::colormaps;

// ---------------------------------------------------------------------------
// Frame generation from 2D grid data
// ---------------------------------------------------------------------------

/// A single animation frame: RGBA pixel data at a given resolution.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Generate a heatmap frame from a 1D spatial slice (single time step).
///
/// Renders a horizontal bar of height `bar_height` pixels, width = data.len().
pub fn frame_from_slice(
    data: &[f64],
    bar_height: u32,
    colormap: fn(f64) -> Color32,
    min_val: f64,
    max_val: f64,
) -> Frame {
    let width = data.len() as u32;
    let height = bar_height;
    let range = (max_val - min_val).max(1e-12);

    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for _row in 0..height {
        for &val in data {
            let t = if val.is_finite() {
                (val - min_val) / range
            } else {
                0.0
            };
            let c = colormap(t);
            rgba.push(c.r());
            rgba.push(c.g());
            rgba.push(c.b());
            rgba.push(255);
        }
    }

    Frame {
        width,
        height,
        rgba,
    }
}

/// Generate a full heatmap frame from a 2D grid (rows × cols).
pub fn frame_from_grid(
    data: &[Vec<f64>],
    colormap: fn(f64) -> Color32,
    min_val: f64,
    max_val: f64,
) -> Frame {
    let rows = data.len();
    let cols = if rows > 0 { data[0].len() } else { 0 };
    let range = (max_val - min_val).max(1e-12);

    let mut rgba = Vec::with_capacity(cols * rows * 4);
    for row_idx in 0..rows {
        let y = rows - 1 - row_idx; // flip so row 0 is bottom
        for col_idx in 0..cols {
            let val = data[y][col_idx];
            let t = if val.is_finite() {
                (val - min_val) / range
            } else {
                0.0
            };
            let c = colormap(t);
            rgba.push(c.r());
            rgba.push(c.g());
            rgba.push(c.b());
            rgba.push(255);
        }
    }

    Frame {
        width: cols as u32,
        height: rows as u32,
        rgba,
    }
}

// ---------------------------------------------------------------------------
// PNG export
// ---------------------------------------------------------------------------

/// Save a single frame as a PNG file.
pub fn save_frame_png(frame: &Frame, path: &Path) -> Result<(), String> {
    let img = image::RgbaImage::from_raw(frame.width, frame.height, frame.rgba.clone())
        .ok_or("Failed to create image buffer")?;
    img.save(path)
        .map_err(|e| format!("PNG save error: {e}"))
}

/// Export a sequence of 1D spatial slices (e.g. PDE time steps) as PNG frames.
///
/// Files are named `{prefix}_0000.png`, `{prefix}_0001.png`, etc.
pub fn export_pde_frames_png(
    data: &[Vec<f64>],
    output_dir: &Path,
    prefix: &str,
    bar_height: u32,
    colormap: fn(f64) -> Color32,
) -> Result<usize, String> {
    if data.is_empty() {
        return Err("No data to export".into());
    }

    // Find global min/max for consistent color mapping.
    let (min_val, max_val) = global_min_max(data);

    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output directory: {e}"))?;

    for (i, slice) in data.iter().enumerate() {
        let frame = frame_from_slice(slice, bar_height, colormap, min_val, max_val);
        let filename = format!("{prefix}_{i:04}.png");
        let path = output_dir.join(filename);
        save_frame_png(&frame, &path)?;
    }

    Ok(data.len())
}

// ---------------------------------------------------------------------------
// GIF export
// ---------------------------------------------------------------------------

/// Export a sequence of 1D spatial slices as an animated GIF.
///
/// `frame_delay` is in hundredths of a second (e.g. 5 = 50ms per frame).
pub fn export_pde_gif(
    data: &[Vec<f64>],
    output_path: &Path,
    bar_height: u32,
    frame_delay: u16,
    colormap: fn(f64) -> Color32,
) -> Result<usize, String> {
    if data.is_empty() {
        return Err("No data to export".into());
    }

    let width = data[0].len() as u16;
    let height = bar_height as u16;

    let (min_val, max_val) = global_min_max(data);

    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("Failed to create GIF file: {e}"))?;

    let mut encoder = gif::Encoder::new(file, width, height, &[])
        .map_err(|e| format!("GIF encoder error: {e}"))?;

    encoder
        .set_repeat(gif::Repeat::Infinite)
        .map_err(|e| format!("GIF repeat error: {e}"))?;

    for slice in data {
        let frame = frame_from_slice(slice, bar_height, colormap, min_val, max_val);

        // Convert RGBA to RGB for the GIF encoder.
        let rgb: Vec<u8> = frame
            .rgba
            .chunks(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect();

        let mut gif_frame = gif::Frame::from_rgb(width, height, &rgb);
        gif_frame.delay = frame_delay;
        encoder
            .write_frame(&gif_frame)
            .map_err(|e| format!("GIF write error: {e}"))?;
    }

    Ok(data.len())
}

/// Export a full 2D heatmap as a single PNG.
pub fn export_heatmap_png(
    data: &[Vec<f64>],
    output_path: &Path,
    colormap: fn(f64) -> Color32,
) -> Result<(), String> {
    if data.is_empty() || data[0].is_empty() {
        return Err("No data to export".into());
    }

    let (min_val, max_val) = global_min_max(data);
    let frame = frame_from_grid(data, colormap, min_val, max_val);
    save_frame_png(&frame, output_path)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the global min and max across all rows.
fn global_min_max(data: &[Vec<f64>]) -> (f64, f64) {
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for row in data {
        for &v in row {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
    }
    (min_val, max_val)
}

// ---------------------------------------------------------------------------
// Color map selection helper
// ---------------------------------------------------------------------------

/// Named colormap options for the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColormapChoice {
    #[default]
    Viridis,
    Inferno,
    CoolWarm,
}

impl ColormapChoice {
    pub fn label(&self) -> &'static str {
        match self {
            ColormapChoice::Viridis => "Viridis",
            ColormapChoice::Inferno => "Inferno",
            ColormapChoice::CoolWarm => "Cool-Warm",
        }
    }

    pub fn function(&self) -> fn(f64) -> Color32 {
        match self {
            ColormapChoice::Viridis => colormaps::colormap_viridis,
            ColormapChoice::Inferno => colormaps::colormap_inferno,
            ColormapChoice::CoolWarm => colormaps::colormap_coolwarm,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_from_slice_dimensions() {
        let data = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let frame = frame_from_slice(&data, 10, colormaps::colormap_viridis, 0.0, 1.0);
        assert_eq!(frame.width, 5);
        assert_eq!(frame.height, 10);
        assert_eq!(frame.rgba.len(), 5 * 10 * 4);
    }

    #[test]
    fn frame_from_grid_dimensions() {
        let data = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
        ];
        let frame = frame_from_grid(&data, colormaps::colormap_viridis, 0.0, 5.0);
        assert_eq!(frame.width, 3);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.rgba.len(), 3 * 2 * 4);
    }

    #[test]
    fn global_min_max_works() {
        let data = vec![
            vec![-1.0, 3.0],
            vec![0.0, 7.0],
        ];
        let (min, max) = global_min_max(&data);
        assert_eq!(min, -1.0);
        assert_eq!(max, 7.0);
    }

    #[test]
    fn export_png_roundtrip() {
        let data = vec![
            vec![0.0, 0.5, 1.0],
            vec![1.0, 0.5, 0.0],
        ];
        let dir = std::env::temp_dir().join("simucad_test_heatmap_png");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.png");

        export_heatmap_png(&data, &path, colormaps::colormap_viridis).unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn export_gif_roundtrip() {
        let data = vec![
            vec![0.0, 0.5, 1.0, 0.5, 0.0],
            vec![0.1, 0.6, 0.9, 0.6, 0.1],
            vec![0.2, 0.7, 0.8, 0.7, 0.2],
        ];
        let dir = std::env::temp_dir().join("simucad_test_gif");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.gif");

        let count = export_pde_gif(&data, &path, 20, 5, colormaps::colormap_viridis).unwrap();
        assert_eq!(count, 3);
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn colormap_choice_label() {
        assert_eq!(ColormapChoice::Viridis.label(), "Viridis");
        assert_eq!(ColormapChoice::CoolWarm.label(), "Cool-Warm");
    }
}
