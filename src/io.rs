//! File I/O: native .osk (JSON), OBJ import/export and STL export.

use crate::document::Document;
use crate::model::{newell, Id, Material, Model};
use glam::Vec3;
use std::fmt::Write as _;
use std::path::Path;

pub fn save(doc: &Document, path: &Path) -> Result<(), String> {
    let s = serde_json::to_string(doc).map_err(|e| e.to_string())?;
    std::fs::write(path, s).map_err(|e| e.to_string())
}

/// Loads a document; files from before groups existed (a bare mesh) still open.
pub fn load(path: &Path) -> Result<Document, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut doc = match serde_json::from_str::<Document>(&s) {
        Ok(d) => d,
        Err(e) => match serde_json::from_str::<Model>(&s) {
            Ok(m) => Document::from_model(m),
            Err(_) => return Err(e.to_string()),
        },
    };
    for d in doc.defs.values_mut() {
        d.mesh.recompute_normals();
    }
    Ok(doc)
}

fn mtl_name(materials: &[Material], m: Option<usize>) -> String {
    match m.and_then(|i| materials.get(i)) {
        Some(mat) => mat.name.replace(char::is_whitespace, "_"),
        None => "Default".to_string(),
    }
}

/// Converts Z-up model space to the Y-up convention used by OBJ consumers.
fn to_y_up(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, -p.y)
}

/// Writes a (flattened) mesh as OBJ + MTL.
/// Writes an OBJ + MTL pair. Textured materials also get a PNG next to the
/// OBJ (`map_Kd`) and their faces get texture coordinates.
pub fn export_obj(model: &Model, materials: &[Material], path: &Path) -> Result<(), String> {
    let mtl_path = path.with_extension("mtl");
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    let mut mtl = String::new();
    let _ = writeln!(mtl, "newmtl Default\nKd 0.96 0.96 0.94\n");
    let used: std::collections::HashSet<usize> = model.faces.values().filter_map(|f| f.material).collect();
    for (i, m) in materials.iter().enumerate() {
        let c = m.color;
        let name = mtl_name(materials, Some(i));
        let _ = writeln!(mtl, "newmtl {name}\nKd {} {} {}", c[0], c[1], c[2]);
        if let (Some(t), true) = (&m.texture, used.contains(&i)) {
            let file = format!("{stem}_{name}.png");
            let png = crate::texture::png_bytes(&t.pixels()?)?;
            std::fs::write(path.with_file_name(&file), png).map_err(|e| e.to_string())?;
            let _ = writeln!(mtl, "map_Kd {file}");
        }
        mtl.push('\n');
    }
    std::fs::write(&mtl_path, mtl).map_err(|e| e.to_string())?;
    let tex_size = |f: &crate::model::Face| f.material.and_then(|i| materials.get(i)).and_then(|m| m.texture.as_ref()).map(|t| t.size);
    let mut vt_count = 0usize;

    let mut out = String::new();
    let _ = writeln!(out, "# Exported by OpenSketch");
    let mtl_file = mtl_path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let _ = writeln!(out, "mtllib {mtl_file}\no OpenSketch");
    let mut index = std::collections::HashMap::<Id, usize>::new();
    let mut count = 0usize;
    for (&id, &p) in &model.vertices {
        let q = to_y_up(p);
        let _ = writeln!(out, "v {} {} {}", q.x, q.y, q.z);
        count += 1;
        index.insert(id, count);
    }
    for (&fid, face) in &model.faces {
        let _ = writeln!(out, "usemtl {}", mtl_name(materials, face.material));
        let size = tex_size(face);
        // Texture coordinates for the given points; returns "/vt" suffixes.
        let mut vts = |out: &mut String, pts: &[glam::Vec3]| -> Vec<String> {
            match size {
                Some(s) => pts
                    .iter()
                    .map(|p| {
                        let uv = crate::texture::uv(*p, face.normal, s);
                        let _ = writeln!(out, "vt {} {}", uv[0], uv[1]);
                        vt_count += 1;
                        format!("/{vt_count}")
                    })
                    .collect(),
                None => vec![String::new(); pts.len()],
            }
        };
        if face.holes.is_empty() {
            let suffix = vts(&mut out, &model.loop_points(&face.outer));
            let idx: Vec<String> = face.outer.iter().zip(&suffix).map(|(v, s)| format!("{}{s}", index[v])).collect();
            let _ = writeln!(out, "f {}", idx.join(" "));
        } else {
            // OBJ has no holes: write triangles with their own vertices.
            for t in model.triangulate(fid) {
                for p in t {
                    let q = to_y_up(p);
                    let _ = writeln!(out, "v {} {} {}", q.x, q.y, q.z);
                }
                let suffix = vts(&mut out, &t);
                let _ = writeln!(out, "f {}{} {}{} {}{}", count + 1, suffix[0], count + 2, suffix[1], count + 3, suffix[2]);
                count += 3;
            }
        }
    }
    std::fs::write(path, out).map_err(|e| e.to_string())
}

pub fn export_stl(model: &Model, path: &Path) -> Result<(), String> {
    let mut out = String::from("solid opensketch\n");
    for &fid in model.faces.keys() {
        for t in model.triangulate(fid) {
            let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
            let _ = writeln!(out, " facet normal {} {} {}\n  outer loop", n.x, n.y, n.z);
            for p in t {
                let q = p * 1000.0; // STL is conventionally millimeters
                let _ = writeln!(out, "   vertex {} {} {}", q.x, q.y, q.z);
            }
            let _ = writeln!(out, "  endloop\n endfacet");
        }
    }
    out.push_str("endsolid opensketch\n");
    std::fs::write(path, out).map_err(|e| e.to_string())
}

/// Imports polygons from an OBJ file (Y-up converted to Z-up). Returns face count.
pub fn import_obj(model: &mut Model, path: &Path) -> Result<usize, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut verts: Vec<Vec3> = Vec::new();
    let mut ids: Vec<Option<Id>> = Vec::new();
    let mut faces = 0;
    for line in s.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f32> = it.take(3).filter_map(|x| x.parse().ok()).collect();
                if c.len() == 3 {
                    verts.push(Vec3::new(c[0], -c[2], c[1]));
                    ids.push(None);
                }
            }
            Some("f") => {
                let mut lp = Vec::new();
                for tok in it {
                    let Some(Ok(i)) = tok.split('/').next().map(|x| x.parse::<i64>()) else { continue };
                    let idx = if i < 0 { verts.len() as i64 + i } else { i - 1 };
                    if idx < 0 || idx as usize >= verts.len() {
                        continue;
                    }
                    let idx = idx as usize;
                    let id = match ids[idx] {
                        Some(id) => id,
                        None => {
                            let id = model.add_vertex(verts[idx]);
                            ids[idx] = Some(id);
                            id
                        }
                    };
                    lp.push(id);
                }
                if lp.len() >= 3 {
                    let n = newell(&model.loop_points(&lp));
                    if model.add_face_loops(lp, vec![], n, None).is_some() {
                        faces += 1;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{DefKind, ROOT};

    #[test]
    fn opens_pre_groups_files_and_roundtrips() {
        let dir = std::env::temp_dir().join(format!("opensketch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A file saved by the first version: a bare mesh with materials.
        let mut legacy = Model::new();
        legacy.materials = crate::model::default_materials();
        legacy.add_polygon(&[Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y], None);
        let old = dir.join("old.osk");
        std::fs::write(&old, serde_json::to_string(&legacy).unwrap()).unwrap();
        let doc = load(&old).unwrap();
        assert_eq!(doc.defs[&ROOT].mesh.faces.len(), 1);
        assert_eq!(doc.materials.len(), legacy.materials.len());

        // New format round-trips groups.
        let mut doc = doc;
        let mut active = doc.load(&[]);
        let faces: Vec<_> = active.faces.keys().copied().collect();
        let g = doc.make_group(&[], &mut active, &faces, &[], &[], DefKind::Group).unwrap();
        doc.sync(&[], &active);
        let new = dir.join("new.osk");
        save(&doc, &new).unwrap();
        let back = load(&new).unwrap();
        assert!(back.instance(&[], g).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn guides_are_saved() {
        let mut doc = Document::new();
        doc.guides.push(crate::guide::Guide::line(glam::Vec3::ZERO, glam::Vec3::Y).unwrap());
        doc.guides.push(crate::guide::Guide::Point { p: glam::Vec3::ONE });
        let json = serde_json::to_string(&doc).unwrap();
        let back: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(back.guides, doc.guides);
        let empty = serde_json::to_string(&Document::new()).unwrap();
        assert!(!empty.contains("guides"), "files without guides stay unchanged");
    }

    #[test]
    fn annotations_are_saved() {
        use crate::annotation::Annotation;
        let mut doc = Document::new();
        doc.annotations.push(Annotation::Dimension { a: Vec3::ZERO, b: Vec3::X, offset: Vec3::Y, text: Some("<> wide".into()) });
        doc.annotations.push(Annotation::Text { anchor: Vec3::ZERO, at: Vec3::ONE, text: "Door".into() });
        doc.annotations.push(Annotation::ScreenText { pos: [0.1, 0.2], text: "Plan".into() });
        let back: Document = serde_json::from_str(&serde_json::to_string(&doc).unwrap()).unwrap();
        assert_eq!(back.annotations, doc.annotations);
        assert!(!serde_json::to_string(&Document::new()).unwrap().contains("annotations"));
    }

    #[test]
    fn textured_obj_export() {
        let dir = std::env::temp_dir().join("opensketch_obj_tex");
        std::fs::create_dir_all(&dir).unwrap();
        let mats = crate::model::default_materials();
        let brick = mats.iter().position(|m| m.name == "Brick").unwrap();
        let mut m = Model::new();
        let f = m
            .add_polygon(&[Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0)], Some(-Vec3::Y))
            .unwrap();
        m.faces.get_mut(&f).unwrap().material = Some(brick);
        let path = dir.join("wall.obj");
        export_obj(&m, &mats, &path).unwrap();
        let obj = std::fs::read_to_string(&path).unwrap();
        let mtl = std::fs::read_to_string(dir.join("wall.mtl")).unwrap();
        assert_eq!(obj.lines().filter(|l| l.starts_with("vt ")).count(), 4);
        assert!(obj.lines().any(|l| l.starts_with("f ") && l.contains('/')));
        assert!(mtl.contains("map_Kd wall_Brick.png"));
        assert!(dir.join("wall_Brick.png").exists());
        assert!(!mtl.contains("map_Kd wall_Wood"), "unused textures are not written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
