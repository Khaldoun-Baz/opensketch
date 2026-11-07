#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! OpenSketch — a SketchUp-style push/pull 3D modeler written in Rust.

mod annotation;
mod app;
mod camera;
mod document;
mod guide;
mod inference;
mod io;
mod model;
mod render;
mod solid;
mod sun;
mod texture;
mod units;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("OpenSketch"),
        depth_buffer: 24,
        multisampling: 4,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native("OpenSketch", options, Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
}
