//! Material textures: built-in procedural patterns (brick, wood, …) and
//! images loaded from disk. Images are embedded in the model file (base64), so
//! `.osk` files stay self-contained. Faces get texture coordinates by
//! projecting onto their plane, scaled to the texture's real-world size.

use crate::model::drawing_axes;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pattern {
    Brick,
    Wood,
    Shingles,
    Grass,
    Stone,
    Tiles,
    /// Dressed limestone blocks in courses (ashlar).
    Ashlar,
    /// Orange clay roof tiles.
    ClayTiles,
}

impl Pattern {
    pub const ALL: [Pattern; 8] = [
        Pattern::Brick,
        Pattern::Wood,
        Pattern::Shingles,
        Pattern::Grass,
        Pattern::Stone,
        Pattern::Tiles,
        Pattern::Ashlar,
        Pattern::ClayTiles,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Pattern::Brick => "Brick",
            Pattern::Wood => "Wood planks",
            Pattern::Shingles => "Roof shingles",
            Pattern::Grass => "Grass",
            Pattern::Stone => "Cobblestone",
            Pattern::Tiles => "Floor tiles",
            Pattern::Ashlar => "Limestone blocks",
            Pattern::ClayTiles => "Clay roof tiles",
        }
    }

    /// Real-world size (meters) one repeat of the pattern covers.
    pub fn size(self) -> [f32; 2] {
        match self {
            Pattern::Brick => [0.45, 0.30],
            Pattern::Wood => [1.2, 0.6],
            Pattern::Shingles => [0.6, 0.6],
            Pattern::Grass => [1.0, 1.0],
            Pattern::Stone => [1.0, 1.0],
            Pattern::Tiles => [1.2, 1.2],
            Pattern::Ashlar => [1.4, 0.7],
            Pattern::ClayTiles => [0.8, 0.8],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Source {
    Builtin {
        pattern: Pattern,
    },
    /// Image file bytes (PNG or JPEG), base64-encoded.
    Image {
        name: String,
        data: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Texture {
    pub source: Source,
    /// Width and height (meters) that one copy of the image covers.
    pub size: [f32; 2],
}

/// Decoded RGBA8 pixels, top row first.
#[derive(Clone, Debug, PartialEq)]
pub struct Pixels {
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

impl Pixels {
    pub fn average(&self) -> [f32; 3] {
        let mut sum = [0f64; 3];
        for px in self.rgba.as_chunks::<4>().0 {
            for k in 0..3 {
                sum[k] += px[k] as f64;
            }
        }
        let n = (self.rgba.len() / 4).max(1) as f64 * 255.0;
        [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32]
    }
}

impl Texture {
    pub fn builtin(p: Pattern) -> Texture {
        Texture { source: Source::Builtin { pattern: p }, size: p.size() }
    }

    pub fn name(&self) -> String {
        match &self.source {
            Source::Builtin { pattern } => pattern.label().to_string(),
            Source::Image { name, .. } => name.clone(),
        }
    }

    /// Identifies the image (for caches); the size does not matter.
    pub fn key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        match &self.source {
            Source::Builtin { pattern } => pattern.hash(&mut h),
            Source::Image { data, .. } => data.hash(&mut h),
        }
        h.finish()
    }

    pub fn pixels(&self) -> Result<Pixels, String> {
        match &self.source {
            Source::Builtin { pattern } => Ok(generate(*pattern)),
            Source::Image { data, .. } => {
                let bytes = base64_decode(data).ok_or("corrupt texture data")?;
                let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
                let img = if img.width().max(img.height()) > 2048 { img.thumbnail(2048, 2048) } else { img };
                let rgba = img.to_rgba8();
                Ok(Pixels { w: rgba.width(), h: rgba.height(), rgba: rgba.into_raw() })
            }
        }
    }

    /// Loads an image file as a texture 1 m wide. Large images are scaled
    /// down to 1024 px so the model file stays reasonable.
    /// Returns the texture and the image's average color.
    pub fn from_file(path: &std::path::Path) -> Result<(Texture, [f32; 3]), String> {
        let mut bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let img = image::load_from_memory(&bytes).map_err(|e| format!("not a PNG/JPEG image: {e}"))?;
        let (w, h) = (img.width().max(1), img.height().max(1));
        if w.max(h) > 1024 {
            let small = img.thumbnail(1024, 1024);
            let mut out = std::io::Cursor::new(Vec::new());
            small.write_to(&mut out, image::ImageFormat::Png).map_err(|e| e.to_string())?;
            bytes = out.into_inner();
        }
        let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "Image".into());
        let tex = Texture { source: Source::Image { name, data: base64_encode(&bytes) }, size: [1.0, h as f32 / w as f32] };
        let avg = tex.pixels()?.average();
        Ok((tex, avg))
    }
}

/// Texture coordinates of a point on a face with normal `n` (both in the
/// face's own coordinates): planar projection, horizontal u, upward v on walls.
pub fn uv(p: Vec3, n: Vec3, size: [f32; 2]) -> [f32; 2] {
    let (u, v) = drawing_axes(n);
    [p.dot(u) / size[0].max(1e-4), p.dot(v) / size[1].max(1e-4)]
}

/// PNG bytes of the texture's pixels (for exporting).
pub fn png_bytes(px: &Pixels) -> Result<Vec<u8>, String> {
    let img = image::RgbaImage::from_raw(px.w, px.h, px.rgba.clone()).ok_or("bad pixel buffer")?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(out.into_inner())
}

// ---------------------------------------------------------------------------
// Procedural patterns (256 x 256, seamlessly tileable)
// ---------------------------------------------------------------------------

const N: u32 = 256;

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374_761_393) ^ (y as u32).wrapping_mul(668_265_263) ^ seed.wrapping_mul(2_246_822_519);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) & 0xffff) as f32 / 65535.0
}

/// Tileable value noise with `cells` cells across the image.
fn noise(x: f32, y: f32, cells: i32, seed: u32) -> f32 {
    let (fx, fy) = (x * cells as f32, y * cells as f32);
    let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - ix as f32, fy - iy as f32);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let g = |a: i32, b: i32| hash(a.rem_euclid(cells), b.rem_euclid(cells), seed);
    let top = g(ix, iy) + (g(ix + 1, iy) - g(ix, iy)) * s(tx);
    let bot = g(ix, iy + 1) + (g(ix + 1, iy + 1) - g(ix, iy + 1)) * s(tx);
    top + (bot - top) * s(ty)
}

fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    (noise(x, y, 4, seed) * 0.5 + noise(x, y, 8, seed + 1) * 0.3 + noise(x, y, 16, seed + 2) * 0.2).clamp(0.0, 1.0)
}

fn mixc(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn scale(c: [f32; 3], k: f32) -> [f32; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

pub fn generate(p: Pattern) -> Pixels {
    let mut rgba = Vec::with_capacity((N * N * 4) as usize);
    for py in 0..N {
        for px in 0..N {
            let (x, y) = (px as f32 / N as f32, py as f32 / N as f32);
            let c = shade(p, x, y);
            rgba.extend(c.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8));
            rgba.push(255);
        }
    }
    Pixels { w: N, h: N, rgba }
}

fn shade(p: Pattern, x: f32, y: f32) -> [f32; 3] {
    let grain = fbm(x, y, 7);
    match p {
        Pattern::Brick => {
            // 4 courses, 2 bricks per course, running bond.
            let row = (y * 4.0).floor();
            let shift = if row as i32 % 2 == 1 { 0.25 } else { 0.0 };
            let bx = (x + shift).rem_euclid(1.0) * 2.0;
            let (fx, fy) = (bx.fract(), (y * 4.0).fract());
            let mortar = fx < 0.04 || fy < 0.12;
            if mortar {
                scale([0.78, 0.76, 0.72], 0.9 + 0.15 * grain)
            } else {
                let id = hash(bx.floor() as i32, row as i32, 3);
                let base = mixc([0.66, 0.27, 0.20], [0.78, 0.40, 0.28], id);
                scale(base, 0.85 + 0.3 * grain)
            }
        }
        Pattern::Wood => {
            // 4 planks; grain runs along x.
            let plank = (y * 4.0).floor();
            let fy = (y * 4.0).fract();
            let id = hash(plank as i32, 0, 11);
            let rings = ((y * 40.0 + noise(x, y, 4, 13 + plank as u32) * 6.0) * std::f32::consts::TAU).sin() * 0.5 + 0.5;
            let base = mixc([0.55, 0.36, 0.20], [0.72, 0.52, 0.32], id);
            let c = scale(base, 0.82 + 0.18 * rings + 0.1 * grain);
            if !(0.04..=0.97).contains(&fy) {
                scale(c, 0.55)
            } else {
                c
            }
        }
        Pattern::Shingles => {
            // 4 rows of tiles, staggered, darker towards each row's lower edge.
            let row = (y * 4.0).floor();
            let shift = if row as i32 % 2 == 1 { 0.125 } else { 0.0 };
            let tx = ((x + shift).rem_euclid(1.0) * 4.0).fract();
            let fy = (y * 4.0).fract();
            let id = hash(((x + shift).rem_euclid(1.0) * 4.0) as i32, row as i32, 5);
            let base = mixc([0.42, 0.18, 0.14], [0.55, 0.25, 0.18], id);
            let edge = if tx < 0.03 { 0.6 } else { 1.0 };
            scale(base, (0.65 + 0.35 * (1.0 - fy)) * edge * (0.9 + 0.2 * grain))
        }
        Pattern::Grass => {
            let blades = noise(x, y, 64, 21);
            let base = mixc([0.26, 0.48, 0.18], [0.45, 0.66, 0.27], fbm(x, y, 23));
            scale(base, 0.8 + 0.35 * blades)
        }
        Pattern::Stone => {
            // Jittered 5 x 5 cells (Worley noise): stones with dark joints.
            let cells = 5;
            let (fx, fy) = (x * cells as f32, y * cells as f32);
            let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
            let (mut d1, mut d2, mut id) = (9.0f32, 9.0f32, 0.0);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (cx, cy) = (ix + dx, iy + dy);
                    let (wx, wy) = (cx.rem_euclid(cells), cy.rem_euclid(cells));
                    let px = cx as f32 + 0.15 + 0.7 * hash(wx, wy, 31);
                    let py = cy as f32 + 0.15 + 0.7 * hash(wx, wy, 37);
                    let d = ((fx - px).powi(2) + (fy - py).powi(2)).sqrt();
                    if d < d1 {
                        d2 = d1;
                        d1 = d;
                        id = hash(wx, wy, 41);
                    } else if d < d2 {
                        d2 = d;
                    }
                }
            }
            if d2 - d1 < 0.08 {
                scale([0.35, 0.34, 0.32], 0.9 + 0.2 * grain)
            } else {
                scale(mixc([0.52, 0.50, 0.47], [0.66, 0.63, 0.58], id), 0.85 + 0.3 * grain)
            }
        }
        Pattern::Ashlar => {
            // 2 courses, 2 blocks per course, running bond; thin pale joints
            // and a slightly darker, weathered rim on every block.
            let row = (y * 2.0).floor();
            let shift = if row as i32 % 2 == 1 { 0.25 } else { 0.0 };
            let bx = (x + shift).rem_euclid(1.0) * 2.0;
            let (fx, fy) = (bx.fract(), (y * 2.0).fract());
            if fx < 0.012 || fy < 0.025 {
                return scale([0.93, 0.91, 0.86], 0.95 + 0.08 * grain);
            }
            let id = hash(bx.floor() as i32, row as i32, 61);
            let base = mixc([0.89, 0.84, 0.71], [0.97, 0.93, 0.82], id);
            let rim = (fx.min(1.0 - fx) * 9.0).min(fy.min(1.0 - fy) * 4.5).min(1.0);
            let pits = noise(x, y, 48, 67);
            scale(base, (0.80 + 0.2 * rim) * (0.9 + 0.16 * grain) * (0.94 + 0.08 * pits))
        }
        Pattern::ClayTiles => {
            // 4 rows of rounded tiles, staggered; each tile is lit across its width.
            let row = (y * 4.0).floor();
            let shift = if row as i32 % 2 == 1 { 0.125 } else { 0.0 };
            let tx = ((x + shift).rem_euclid(1.0) * 4.0).fract();
            let fy = (y * 4.0).fract();
            let id = hash(((x + shift).rem_euclid(1.0) * 4.0) as i32, row as i32, 71);
            let base = mixc([0.74, 0.33, 0.17], [0.86, 0.45, 0.23], id);
            let round = 0.72 + 0.28 * (tx * std::f32::consts::PI).sin();
            let lap = if fy < 0.1 { 0.7 } else { 1.0 };
            scale(base, round * lap * (0.9 + 0.2 * grain))
        }
        Pattern::Tiles => {
            let (fx, fy) = ((x * 4.0).fract(), (y * 4.0).fract());
            if fx < 0.03 || fy < 0.03 {
                [0.62, 0.60, 0.56]
            } else {
                let id = hash((x * 4.0) as i32, (y * 4.0) as i32, 51);
                scale(mixc([0.86, 0.84, 0.79], [0.93, 0.91, 0.86], id), 0.95 + 0.08 * grain)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// base64 (standard alphabet, padded)
// ---------------------------------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(B64[((n >> (18 - 6 * k)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut val = [255u8; 256];
    for (i, c) in B64.iter().enumerate() {
        val[*c as usize] = i as u8;
    }
    let bytes: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        let mut pad = 0;
        for &c in chunk {
            n <<= 6;
            if c == b'=' {
                pad += 1;
            } else {
                let v = val[c as usize];
                if v == 255 || pad > 0 {
                    return None;
                }
                n |= v as u32;
            }
        }
        let b = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&b[..3 - pad.min(2)]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        for data in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            assert_eq!(base64_decode(&base64_encode(data)).unwrap(), data);
        }
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert!(base64_decode("Zm9v!mFy").is_none());
    }

    #[test]
    fn patterns_are_tileable_and_deterministic() {
        for p in Pattern::ALL {
            let a = generate(p);
            assert_eq!((a.w, a.h, a.rgba.len()), (N, N, (N * N * 4) as usize));
            assert_eq!(a, generate(p));
        }
        // The noise wraps around, so the image repeats without a seam.
        for (x, y) in [(0.0, 0.3), (0.37, 0.0), (0.999, 0.5)] {
            assert!((noise(x, y, 8, 1) - noise(x + 1.0, y, 8, 1)).abs() < 1e-4);
            assert!((fbm(x, y, 3) - fbm(x, y + 1.0, 3)).abs() < 1e-3);
        }
    }

    #[test]
    fn wall_uvs_run_along_and_up() {
        let n = -Vec3::Y; // a wall facing -Y
        let a = uv(Vec3::new(0.0, 0.0, 0.0), n, [0.5, 0.25]);
        let b = uv(Vec3::new(1.0, 0.0, 0.5), n, [0.5, 0.25]);
        assert!((b[0] - a[0] - 2.0).abs() < 1e-5 && (b[1] - a[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn image_files_load_and_embed() {
        let dir = std::env::temp_dir().join("opensketch_tex_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checker.png");
        let mut img = image::RgbaImage::new(8, 4);
        for (x, _, p) in img.enumerate_pixels_mut() {
            *p = if x % 2 == 0 { image::Rgba([255, 0, 0, 255]) } else { image::Rgba([0, 0, 255, 255]) };
        }
        img.save(&path).unwrap();
        let (t, avg) = Texture::from_file(&path).unwrap();
        assert_eq!(t.size, [1.0, 0.5], "keeps the aspect ratio");
        assert!((avg[0] - 0.5).abs() < 0.01 && avg[1] < 0.01 && (avg[2] - 0.5).abs() < 0.01);
        let px = t.pixels().unwrap();
        assert_eq!((px.w, px.h), (8, 4));
        assert_eq!(t.name(), "checker");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
