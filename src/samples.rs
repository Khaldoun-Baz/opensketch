//! Sample models built in code (File ▸ Samples, or `opensketch --sample <name>`).
//!
//! They show what OpenSketch can make and give new users something to orbit,
//! paint and take apart. Walls are made the way a person would make them:
//! a box pulled up from a rectangle, with shapes drawn on its faces and pushed
//! in for windows. Small parts (frames, railings, columns) are put together
//! from simple solids, each kind in its own group.

use crate::camera::Camera;
use crate::document::{DefKind, Document, Scene, SceneStyle, ROOT};
use crate::model::{newell, Id, Material, Model};
use crate::sun::Sun;
use crate::texture::{Pattern, Texture};
use glam::Vec3;
use std::f32::consts::PI;

/// Makes one sample document.
pub type Builder = fn() -> Document;

/// Name used on the command line, menu label, builder.
pub const SAMPLES: &[(&str, &str, Builder)] = &[("stone-house", "Stone house with arches", stone_house)];

pub fn by_name(name: &str) -> Option<Document> {
    SAMPLES.iter().find(|s| s.0.eq_ignore_ascii_case(name)).map(|s| (s.2)())
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

/// A solid made by sliding a flat outline (with optional holes) along `dir`.
/// Faces are added directly, so separate solids in one mesh never merge.
fn prism(m: &mut Model, outer: &[Vec3], holes: &[Vec<Vec3>], dir: Vec3, mat: Option<usize>) {
    let du = dir.normalize();
    let ids = |m: &mut Model, pts: &[Vec3]| -> Vec<Id> { pts.iter().map(|p| m.add_vertex(*p)).collect() };
    let moved = |pts: &[Vec3]| -> Vec<Vec3> { pts.iter().map(|p| *p + dir).collect() };
    let ob = ids(m, outer);
    let ot = ids(m, &moved(outer));
    let hb: Vec<Vec<Id>> = holes.iter().map(|h| ids(m, h)).collect();
    let ht: Vec<Vec<Id>> = holes.iter().map(|h| ids(m, &moved(h))).collect();
    m.add_face_loops(ob.clone(), hb.clone(), -du, mat);
    m.add_face_loops(ot.clone(), ht.clone(), du, mat);
    // Side walls face away from the solid: out of the outline, into the holes.
    let mut sides = |b: &[Id], t: &[Id], pts: &[Vec3], hole: bool| {
        let ccw = newell(pts).dot(du) > 0.0;
        let sign = if ccw != hole { 1.0 } else { -1.0 };
        for i in 0..b.len() {
            let j = (i + 1) % b.len();
            let e = pts[j] - pts[i];
            m.add_face_loops(vec![b[i], b[j], t[j], t[i]], vec![], e.cross(du) * sign, mat);
        }
    };
    sides(&ob, &ot, outer, false);
    for (k, h) in holes.iter().enumerate() {
        sides(&hb[k], &ht[k], h, true);
    }
}

/// An axis-aligned box.
fn cuboid(m: &mut Model, lo: Vec3, hi: Vec3, mat: Option<usize>) {
    let b = [Vec3::new(lo.x, lo.y, lo.z), Vec3::new(hi.x, lo.y, lo.z), Vec3::new(hi.x, hi.y, lo.z), Vec3::new(lo.x, hi.y, lo.z)];
    prism(m, &b, &[], Vec3::Z * (hi.z - lo.z), mat);
}

/// A vertical round column of `sides` facets.
fn cylinder(m: &mut Model, base: Vec3, r: f32, h: f32, sides: usize, mat: Option<usize>) {
    let pts: Vec<Vec3> = (0..sides)
        .map(|i| {
            let a = i as f32 / sides as f32 * 2.0 * PI;
            base + Vec3::new(a.cos(), a.sin(), 0.0) * r
        })
        .collect();
    prism(m, &pts, &[], Vec3::Z * h, mat);
}

/// A low-poly ball (latitude/longitude), for tree crowns.
fn ball(m: &mut Model, c: Vec3, r: Vec3, rings: usize, segs: usize, mat: Option<usize>) {
    let p = |i: usize, j: usize| {
        let th = PI * i as f32 / rings as f32;
        let ph = 2.0 * PI * j as f32 / segs as f32;
        c + Vec3::new(th.sin() * ph.cos() * r.x, th.sin() * ph.sin() * r.y, th.cos() * r.z)
    };
    for i in 0..rings {
        for j in 0..segs {
            let q = [p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)];
            let mut ids: Vec<Id> = q.iter().map(|x| m.add_vertex(*x)).collect();
            ids.dedup();
            if ids.first() == ids.last() {
                ids.pop();
            }
            let mid = q.iter().copied().sum::<Vec3>() / 4.0;
            m.add_face_loops(ids, vec![], mid - c, mat);
        }
    }
}

/// Hides the edges between faces that meet at less than `max_deg` (round
/// columns, arches, leaves), like SketchUp's Soften Edges. `max_deg` of 180
/// hides every edge.
fn soften(m: &mut Model, max_deg: f32) {
    let limit = max_deg.to_radians().cos();
    let ids: Vec<Id> = m.edges.keys().copied().collect();
    for e in ids {
        let faces = m.edge_faces(e);
        let smooth = max_deg >= 180.0 || (faces.len() == 2 && m.faces[&faces[0]].normal.dot(m.faces[&faces[1]].normal) > limit);
        if smooth {
            m.edges.get_mut(&e).unwrap().soft = true;
        }
    }
}

/// A flat wall seen from outside: point = `o` + `u`·s + Z·z, facing `n`.
#[derive(Clone, Copy)]
struct Wall {
    o: Vec3,
    u: Vec3,
    n: Vec3,
}

impl Wall {
    fn p(&self, s: f32, z: f32) -> Vec3 {
        self.o + self.u * s + Vec3::Z * z
    }

    /// The same wall moved `d` meters into the building.
    fn behind(&self, d: f32) -> Wall {
        Wall { o: self.o - self.n * d, ..*self }
    }

    fn pts(&self, sz: &[(f32, f32)]) -> Vec<Vec3> {
        sz.iter().map(|&(s, z)| self.p(s, z)).collect()
    }

    /// A box on the wall: across s0..s1, up z0..z1, from `d0` in front of
    /// the wall face to `d1` behind it (negative `d0` sticks out).
    fn block(&self, m: &mut Model, (s0, s1): (f32, f32), (z0, z1): (f32, f32), (d0, d1): (f32, f32), mat: Option<usize>) {
        let q = |s: f32, d: f32| self.o + self.u * s - self.n * d + Vec3::Z * z0;
        prism(m, &[q(s0, d0), q(s1, d0), q(s1, d1), q(s0, d1)], &[], Vec3::Z * (z1 - z0), mat);
    }
}

/// Outline of an opening in wall coordinates, grown by `e` on every side
/// (for the frame around it).
#[derive(Clone, Copy, Debug)]
enum Opening {
    Rect {
        s: f32,
        w: f32,
        z0: f32,
        z1: f32,
    },
    /// Round-headed arch springing at `zs`.
    Round {
        s: f32,
        w: f32,
        z0: f32,
        zs: f32,
    },
    /// Pointed (equilateral) arch springing at `zs`.
    Pointed {
        s: f32,
        w: f32,
        z0: f32,
        zs: f32,
    },
    Circle {
        s: f32,
        z: f32,
        r: f32,
    },
}

impl Opening {
    fn outline(&self, e: f32) -> Vec<(f32, f32)> {
        const N: usize = 10;
        match *self {
            Opening::Rect { s, w, z0, z1 } => vec![(s - e, z0 - e), (s + w + e, z0 - e), (s + w + e, z1 + e), (s - e, z1 + e)],
            Opening::Round { s, w, z0, zs } => {
                let (c, r) = (s + w / 2.0, w / 2.0 + e);
                let mut v = vec![(c - r, z0 - e), (c + r, z0 - e)];
                v.extend((0..=2 * N).map(|i| {
                    let a = PI * i as f32 / (2 * N) as f32;
                    (c + r * a.cos(), zs + r * a.sin())
                }));
                v
            }
            Opening::Pointed { s, w, z0, zs } => {
                // Two arcs of radius w (+e), each centred on the opposite
                // springing point, meeting at the apex above the middle.
                let r = w + e;
                let top = (w / (2.0 * r)).acos();
                let mut v = vec![(s - e, z0 - e), (s + w + e, z0 - e)];
                v.extend((0..=N).map(|i| {
                    let a = top * i as f32 / N as f32;
                    (s + r * a.cos(), zs + r * a.sin())
                }));
                v.extend((1..=N).map(|i| {
                    let a = PI - top + top * i as f32 / N as f32;
                    (s + w + r * a.cos(), zs + r * a.sin())
                }));
                v
            }
            Opening::Circle { s, z, r } => (0..24)
                .map(|i| {
                    let a = 2.0 * PI * i as f32 / 24.0;
                    (s + (r + e) * a.cos(), z + (r + e) * a.sin())
                })
                .collect(),
        }
    }

    /// (left, right, bottom, top) of the opening.
    fn bounds(&self) -> (f32, f32, f32, f32) {
        let mut b = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for (s, z) in self.outline(0.0) {
            b = (b.0.min(s), b.1.max(s), b.2.min(z), b.3.max(z));
        }
        b
    }
}

// ---------------------------------------------------------------------------
// The stone house
// ---------------------------------------------------------------------------

struct Mats {
    stone: usize,
    roof: usize,
    white: usize,
    glass: usize,
    leaves: usize,
    bark: usize,
    floor: usize,
}

fn add_materials(doc: &mut Document) -> Mats {
    let mut add = |name: &str, color: [f32; 3], pattern: Option<Pattern>| {
        let texture = pattern.map(Texture::builtin);
        doc.materials.push(Material { name: name.to_string(), color, texture });
        doc.materials.len() - 1
    };
    let stone = add("Limestone", [0.85, 0.80, 0.66], Some(Pattern::Ashlar));
    let roof = add("Clay Roof Tiles", [0.80, 0.40, 0.20], Some(Pattern::ClayTiles));
    let white = add("White Trim", [0.96, 0.96, 0.94], None);
    let glass = add("Window Glass", [0.64, 0.74, 0.80], None);
    let leaves = add("Leaves", [0.36, 0.62, 0.27], None);
    let bark = add("Bark", [0.42, 0.36, 0.30], None);
    let floor = doc.materials.iter().position(|m| m.name == "Floor Tiles").unwrap_or(stone);
    Mats { stone, roof, white, glass, leaves, bark, floor }
}

/// Paints a walled block: faces on a glass plane (normal, offset) get glass,
/// floors just above the ground (porches) get tiles, the rest is stone.
fn paint(m: &mut Model, glass_planes: &[(Vec3, f32)], glass: Option<usize>, stone: Option<usize>, floor: Option<usize>) {
    let ids: Vec<Id> = m.faces.keys().copied().collect();
    for f in ids {
        let face = &m.faces[&f];
        let p = m.pos(face.outer[0]);
        let on = |(n, d): &(Vec3, f32)| face.normal.dot(*n) > 0.999 && (p.dot(*n) - d).abs() < 1e-3;
        let mat = if glass_planes.iter().any(on) {
            glass
        } else if face.normal.z > 0.999 && p.z > 0.05 && p.z < 0.3 {
            floor
        } else {
            stone
        };
        m.faces.get_mut(&f).unwrap().material = mat;
    }
}

/// Glazing bars inside a window: a centre bar, plus cross bars.
fn glazing(m: &mut Model, w: &Wall, o: &Opening, depth: f32, mat: Option<usize>) {
    let (s0, s1, z0, z1) = o.bounds();
    let mid = (s0 + s1) / 2.0;
    let bar = 0.05;
    let d = (depth - 0.06, depth - 0.01);
    match *o {
        Opening::Rect { .. } => {
            w.block(m, (mid - bar / 2.0, mid + bar / 2.0), (z0, z1), d, mat);
            for k in [1.0 / 3.0, 2.0 / 3.0] {
                let z = z0 + (z1 - z0) * k;
                w.block(m, (s0, s1), (z - bar / 2.0, z + bar / 2.0), d, mat);
            }
        }
        Opening::Pointed { zs, .. } | Opening::Round { zs, .. } => {
            w.block(m, (mid - bar / 2.0, mid + bar / 2.0), (z0, zs), d, mat);
            w.block(m, (s0, s1), (zs - bar, zs), d, mat);
            w.block(m, (s0, s1), (z0 + (zs - z0) * 0.55, z0 + (zs - z0) * 0.55 + bar), d, mat);
        }
        Opening::Circle { s, z, r } => {
            w.block(m, (s - bar / 2.0, s + bar / 2.0), (z - r, z + r), d, mat);
            w.block(m, (s - r, s + r), (z - bar / 2.0, z + bar / 2.0), d, mat);
        }
    }
}

/// A railing along a path of floor points: posts, top and bottom rails, balusters.
fn railing(m: &mut Model, path: &[Vec3], height: f32, mat: Option<usize>) {
    let (rail, bal, gap) = (0.06, 0.028, 0.13);
    for seg in path.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        let flat = Vec3::new(b.x - a.x, b.y - a.y, 0.0);
        let side = Vec3::Z.cross(flat.normalize_or_zero()) * (rail / 2.0);
        for z in [height - rail, 0.08] {
            let base = a + Vec3::Z * z;
            let sq = [base - side, base + side, base + side + Vec3::Z * rail, base - side + Vec3::Z * rail];
            prism(m, &sq, &[], b - a, mat);
        }
        let n = (flat.length() / gap).floor().max(1.0) as usize;
        for i in 1..n {
            let p = a + (b - a) * (i as f32 / n as f32);
            let h = bal / 2.0;
            let lo = p + Vec3::new(-h, -h, 0.08 + rail);
            cuboid(m, lo, Vec3::new(p.x + h, p.y + h, p.z + height - rail), mat);
        }
    }
    for p in path {
        let h = 0.05;
        cuboid(m, *p + Vec3::new(-h, -h, 0.0), *p + Vec3::new(h, h, height + 0.03), mat);
    }
}

pub fn stone_house() -> Document {
    let mut doc = Document::new();
    let k = add_materials(&mut doc);
    let (stone, white, glass) = (Some(k.stone), Some(k.white), Some(k.glass));

    // Main block: 12 x 10 m, two storeys, walls up to 7.5 m.
    let (wx, wy, top) = (12.0, 10.0, 7.5);
    let front = Wall { o: Vec3::ZERO, u: Vec3::X, n: -Vec3::Y };
    let right = Wall { o: Vec3::new(wx, 0.0, 0.0), u: Vec3::Y, n: Vec3::X };
    let back = Wall { o: Vec3::new(wx, wy, 0.0), u: -Vec3::X, n: Vec3::Y };
    let left = Wall { o: Vec3::new(0.0, wy, 0.0), u: -Vec3::Y, n: -Vec3::X };

    let rect = |s: f32, w: f32, z0: f32, z1: f32| Opening::Rect { s, w, z0, z1 };
    let pointed = |s: f32| Opening::Pointed { s, w: 0.95, z0: 4.0, zs: 5.75 };
    let (low, high) = ((0.9, 2.6), (4.3, 6.0));
    let pair = |s: f32, (z0, z1): (f32, f32)| [rect(s, 0.9, z0, z1), rect(s + 1.3, 0.9, z0, z1)];

    let mut openings: Vec<(Wall, Opening)> = Vec::new();
    // Front: two windows each side on both floors, three pointed doors onto the
    // balcony, round windows up high and the arcaded loggia in the middle.
    for (s, z) in [(1.0, low), (8.8, low), (1.0, high), (8.8, high)] {
        openings.extend(pair(s, z).map(|o| (front, o)));
    }
    for i in 0..3 {
        openings.push((front, pointed(4.225 + i as f32 * 1.3)));
    }
    openings.push((front, Opening::Circle { s: 2.1, z: 6.85, r: 0.28 }));
    openings.push((front, Opening::Circle { s: 9.9, z: 6.85, r: 0.28 }));
    // Right side: pointed doors onto the terrace, windows behind it.
    for i in 0..3 {
        openings.push((right, pointed(1.6 + i as f32 * 1.3)));
    }
    openings.push((right, rect(7.6, 0.9, low.0, low.1)));
    openings.push((right, rect(7.6, 0.9, high.0, high.1)));
    openings.push((right, Opening::Circle { s: 8.05, z: 6.85, r: 0.28 }));
    // Back and left: plain windows.
    for (s, z) in [(1.4, low), (1.4, high), (8.4, low), (8.4, high)] {
        openings.extend(pair(s, z).map(|o| (back, o)));
    }
    openings.push((back, Opening::Round { s: 5.4, w: 1.2, z0: 0.3, zs: 2.2 }));
    for (s, z) in [(2.0, low), (2.0, high), (6.4, low), (6.4, high)] {
        openings.push((left, rect(s, 0.9, z.0, z.1)));
        openings.push((left, rect(s + 1.3, 0.9, z.0, z.1)));
    }

    // The walls: a rectangle on the ground pulled up, openings pushed in.
    let depth = 0.22;
    let mut walls = Model::new();
    let ground = [Vec3::ZERO, Vec3::new(wx, 0.0, 0.0), Vec3::new(wx, wy, 0.0), Vec3::new(0.0, wy, 0.0)];
    if let Some(f) = walls.add_polygon(&ground, None) {
        let up = if walls.faces[&f].normal.z < 0.0 { -top } else { top };
        walls.push_pull(f, up);
    }
    let push_in = |m: &mut Model, w: &Wall, outline: &[(f32, f32)], d: f32| {
        if let Some(f) = m.add_polygon(&w.pts(outline), Some(w.n)) {
            m.push_pull(f, -d);
        }
    };
    for (w, o) in &openings {
        push_in(&mut walls, w, &o.outline(0.0), depth);
    }
    // Loggia: an open porch 2 m deep behind the arcade, with a door at the back.
    let (lg0, lg1, lg_top) = (3.9, 8.1, 3.3);
    push_in(&mut walls, &front, &[(lg0, 0.15), (lg1, 0.15), (lg1, lg_top), (lg0, lg_top)], 2.0);
    let loggia_back = front.behind(2.0);
    let door = Opening::Round { s: 5.45, w: 1.1, z0: 0.3, zs: 2.2 };
    push_in(&mut walls, &loggia_back, &door.outline(0.0), depth);
    // Glass where a window recess ends, tiles on porch floors, stone everywhere else.
    let glass_planes: Vec<(Vec3, f32)> = [front, right, back, left, loggia_back].iter().map(|w| (w.n, w.o.dot(w.n) - depth)).collect();
    paint(&mut walls, &glass_planes, glass, stone, Some(k.floor));

    // Window frames and glazing bars.
    let mut frames = Model::new();
    for (w, o) in openings.iter().chain([(loggia_back, door)].iter()) {
        let e = if matches!(o, Opening::Circle { .. }) { 0.1 } else { 0.11 };
        prism(&mut frames, &w.pts(&o.outline(e)), &[w.pts(&o.outline(0.0))], w.n * 0.05, white);
        glazing(&mut frames, w, o, depth, white);
    }

    // The arcade across the loggia: three round arches on slender columns.
    let mut arcade = Model::new();
    let mut columns = Model::new();
    let (span, pier) = (1.05, 0.35);
    let end = (lg1 - lg0 - 3.0 * span - 2.0 * pier) / 2.0;
    let (z0, zs) = (0.15, 2.05);
    let mut outline = vec![(lg0, z0)];
    let mut col_at = vec![lg0 + end / 2.0];
    for i in 0..3 {
        let a = lg0 + end + i as f32 * (span + pier);
        outline.push((a, z0));
        let r = span / 2.0;
        outline.extend((0..=12).map(|j| {
            let t = PI * (1.0 - j as f32 / 12.0);
            (a + r + r * t.cos(), zs + r * t.sin())
        }));
        outline.push((a + span, z0));
        col_at.push(a + span + if i < 2 { pier / 2.0 } else { end / 2.0 });
    }
    outline.push((lg1, z0));
    outline.push((lg1, lg_top));
    outline.push((lg0, lg_top));
    prism(&mut arcade, &front.behind(0.3).pts(&outline), &[], front.n * 0.3, stone);
    // Arch trims.
    for i in 0..3 {
        let a = lg0 + end + i as f32 * (span + pier);
        let ring = |e: f32| Opening::Round { s: a, w: span, z0: zs, zs }.outline(e);
        let inner: Vec<(f32, f32)> = ring(0.0).into_iter().skip(2).collect();
        let outer: Vec<(f32, f32)> = ring(0.12).into_iter().skip(2).collect();
        let mut band = outer.clone();
        band.extend(inner.into_iter().rev());
        prism(&mut columns, &front.pts(&band), &[], front.n * 0.04, white);
    }
    for s in col_at {
        let base = front.p(s, z0) + front.n * 0.02;
        cylinder(&mut columns, base + Vec3::Z * 0.12, 0.11, zs - z0 - 0.24, 12, white);
        cuboid(&mut columns, base + Vec3::new(-0.15, -0.15, 0.0), base + Vec3::new(0.15, 0.15, 0.12), white);
        cuboid(&mut columns, base + Vec3::new(-0.16, -0.16, zs - z0 - 0.12), base + Vec3::new(0.16, 0.16, zs - z0), white);
    }
    // Balcony over the loggia: slab, brackets, railing.
    let mut balcony = Model::new();
    let (b0, b1, bd, bz) = (3.85, 8.15, 1.25, 3.85);
    cuboid(&mut balcony, Vec3::new(b0, -bd, bz), Vec3::new(b1, 0.0, bz + 0.15), white);
    for s in [b0 + 0.25, (b0 + b1) / 2.0, b1 - 0.25] {
        front.block(&mut balcony, (s - 0.12, s + 0.12), (bz - 0.45, bz), (-bd + 0.25, 0.0), white);
    }
    let mut rails = Model::new();
    let f = bz + 0.15;
    railing(
        &mut rails,
        &[
            Vec3::new(b0 + 0.06, -0.05, f),
            Vec3::new(b0 + 0.06, -bd + 0.06, f),
            Vec3::new(b1 - 0.06, -bd + 0.06, f),
            Vec3::new(b1 - 0.06, -0.05, f),
        ],
        1.0,
        white,
    );

    // Cornice and hipped roof.
    let mut cornice = Model::new();
    cuboid(&mut cornice, Vec3::new(-0.3, -0.3, top - 0.1), Vec3::new(wx + 0.3, wy + 0.3, top + 0.22), white);
    let mut roof = Model::new();
    let (ov, ze, rise) = (0.45, top + 0.22, 3.4);
    let e = [Vec3::new(-ov, -ov, ze), Vec3::new(wx + ov, -ov, ze), Vec3::new(wx + ov, wy + ov, ze), Vec3::new(-ov, wy + ov, ze)];
    let half = (wy + 2.0 * ov) / 2.0;
    let r0 = Vec3::new(-ov + half, wy / 2.0, ze + rise);
    let r1 = Vec3::new(wx + ov - half, wy / 2.0, ze + rise);
    let roof_faces: [(&[Vec3], Vec3); 5] = [
        (&[e[0], e[1], r1, r0], Vec3::new(0.0, -1.0, 1.0)),
        (&[e[1], e[2], r1], Vec3::new(1.0, 0.0, 1.0)),
        (&[e[2], e[3], r0, r1], Vec3::new(0.0, 1.0, 1.0)),
        (&[e[3], e[0], r0], Vec3::new(-1.0, 0.0, 1.0)),
        (&[e[0], e[1], e[2], e[3]], -Vec3::Z),
    ];
    for (pts, n) in roof_faces {
        let ids: Vec<Id> = pts.iter().map(|p| roof.add_vertex(*p)).collect();
        let mat = if n.z < 0.0 { white } else { Some(k.roof) };
        roof.add_face_loops(ids, vec![], n, mat);
    }

    // Side wing: an arcaded room with a terrace on top, reached by outside stairs.
    let (ax0, ax1, ay0, ay1, az) = (wx, wx + 3.0, 0.6, 6.4, 3.85);
    let mut wing = Model::new();
    let footprint = [Vec3::new(ax0, ay0, 0.0), Vec3::new(ax1, ay0, 0.0), Vec3::new(ax1, ay1, 0.0), Vec3::new(ax0, ay1, 0.0)];
    if let Some(f) = wing.add_polygon(&footprint, None) {
        let up = if wing.faces[&f].normal.z < 0.0 { -az } else { az };
        wing.push_pull(f, up);
    }
    let wing_side = Wall { o: Vec3::new(ax1, 0.0, 0.0), u: Vec3::Y, n: Vec3::X };
    let wing_front = Wall { o: Vec3::new(0.0, ay0, 0.0), u: Vec3::X, n: -Vec3::Y };
    let mut wing_arches = Vec::new();
    for i in 0..3 {
        wing_arches.push((wing_side, Opening::Round { s: 1.0 + i as f32 * 1.75, w: 1.3, z0: 0.15, zs: 2.2 }));
    }
    wing_arches.push((wing_front, Opening::Round { s: 12.3, w: 0.9, z0: 0.15, zs: 2.2 }));
    for (w, o) in &wing_arches {
        let d = if w.n.x > 0.5 { 1.6 } else { 1.0 };
        push_in(&mut wing, w, &o.outline(0.0), d);
    }
    paint(&mut wing, &[], glass, stone, Some(k.floor));
    for (w, o) in &wing_arches {
        let Opening::Round { s, w: span, zs, .. } = *o else { continue };
        let inner: Vec<(f32, f32)> = o.outline(0.0).into_iter().skip(2).collect();
        let outer: Vec<(f32, f32)> = Opening::Round { s, w: span, z0: zs, zs }.outline(0.12).into_iter().skip(2).collect();
        let mut band = outer;
        band.extend(inner.into_iter().rev());
        prism(&mut columns, &w.pts(&band), &[], w.n * 0.04, white);
        for c in [s - 0.08, s + span + 0.08] {
            let base = w.p(c, 0.0) + w.n * 0.1;
            cylinder(&mut columns, base + Vec3::Z * 0.12, 0.1, zs - 0.24, 12, white);
            cuboid(&mut columns, base + Vec3::new(-0.14, -0.14, 0.0), base + Vec3::new(0.14, 0.14, 0.12), white);
            cuboid(&mut columns, base + Vec3::new(-0.15, -0.15, zs - 0.12), base + Vec3::new(0.15, 0.15, zs), white);
        }
    }
    // Terrace coping and railing.
    cuboid(&mut cornice, Vec3::new(ax0, ay0 - 0.1, az - 0.08), Vec3::new(ax1 + 0.1, ay1 + 0.1, az + 0.06), white);
    let t = az + 0.06;
    let (sx0, sx1) = (ax1 - 1.15, ax1);
    railing(
        &mut rails,
        &[Vec3::new(ax0 + 0.06, ay0 - 0.04, t), Vec3::new(ax1 + 0.04, ay0 - 0.04, t), Vec3::new(ax1 + 0.04, ay1 + 0.04, t)],
        1.0,
        white,
    );
    railing(&mut rails, &[Vec3::new(ax0 + 0.06, ay1 + 0.04, t), Vec3::new(sx0, ay1 + 0.04, t)], 1.0, white);

    // Stairs from the terrace down to the garden.
    let mut stairs = Model::new();
    let steps = 19;
    let (rise_s, run) = (az / steps as f32, 0.28);
    let stair = Wall { o: Vec3::new(sx0, 0.0, 0.0), u: Vec3::Y, n: -Vec3::X };
    let mut profile = vec![(ay1, 0.0), (ay1, az - rise_s)];
    for i in 1..steps {
        let y = ay1 + i as f32 * run;
        profile.push((y, az - i as f32 * rise_s));
        profile.push((y, az - (i + 1) as f32 * rise_s));
    }
    profile.pop();
    profile.push((ay1 + (steps - 1) as f32 * run, 0.0));
    prism(&mut stairs, &stair.pts(&profile), &[], Vec3::X * (sx1 - sx0), stone);
    let slope = |x: f32| [Vec3::new(x, ay1 + 0.05, t), Vec3::new(x, ay1 + (steps - 1) as f32 * run, rise_s)];
    railing(&mut rails, &slope(sx0 + 0.06), 0.95, white);
    railing(&mut rails, &slope(sx1 - 0.04), 0.95, white);

    // A tree in front, on the left.
    let mut tree = Model::new();
    let tc = Vec3::new(-2.6, -1.8, 0.0);
    cylinder(&mut tree, tc, 0.16, 3.6, 10, Some(k.bark));
    for (off, r) in [
        (Vec3::new(0.0, 0.0, 4.6), Vec3::new(1.5, 1.5, 1.3)),
        (Vec3::new(0.7, 0.5, 5.6), Vec3::new(1.0, 1.0, 1.0)),
        (Vec3::new(-0.7, -0.3, 5.3), Vec3::new(1.1, 1.1, 1.0)),
        (Vec3::new(0.2, -0.8, 3.9), Vec3::new(0.9, 0.9, 0.8)),
        (Vec3::new(-0.3, 0.6, 6.4), Vec3::new(0.7, 0.7, 0.7)),
    ] {
        ball(&mut tree, tc + off, r, 8, 14, Some(k.leaves));
    }

    // Railings read as white lines, round things as round: no drawn edges there.
    soften(&mut rails, 180.0);
    soften(&mut tree, 60.0);
    soften(&mut columns, 35.0);
    soften(&mut frames, 35.0);

    // Groups, named for the Outliner; the tree and roof get their own tags.
    let roof_tag = doc.add_tag();
    let tree_tag = doc.add_tag();
    doc.tags.get_mut(&roof_tag).unwrap().name = "Roof".into();
    doc.tags.get_mut(&tree_tag).unwrap().name = "Vegetation".into();
    let parts: [(&str, Model, Option<u32>); 11] = [
        ("Walls", walls, None),
        ("Side wing", wing, None),
        ("Arcade", arcade, None),
        ("Columns and arches", columns, None),
        ("Window frames", frames, None),
        ("Balcony", balcony, None),
        ("Railings", rails, None),
        ("Stairs", stairs, None),
        ("Cornice and coping", cornice, None),
        ("Roof", roof, Some(roof_tag)),
        ("Tree", tree, Some(tree_tag)),
    ];
    for (name, mesh, tag) in parts {
        let Some(iid) = doc.add_world_mesh(&[], &mesh, DefKind::Group) else { continue };
        let inst = doc.defs.get_mut(&ROOT).unwrap().instances.get_mut(&iid).unwrap();
        inst.tag = tag;
        let def = inst.def;
        doc.defs.get_mut(&def).unwrap().name = name.to_string();
    }

    // Afternoon sun in early summer, from the front left.
    doc.sun = Sun { enabled: true, day: 160, hour: 15.0, latitude: 34.0, north: 15.0 };
    let camera = Camera {
        target: Vec3::new(6.5, 4.5, 4.2),
        distance: 34.0,
        yaw: -58f32.to_radians(),
        pitch: 20f32.to_radians(),
        ..Camera::default()
    };
    doc.scenes.push(Scene {
        name: "Street view".into(),
        camera,
        hidden_tags: vec![],
        style: SceneStyle { axes: false, sun: Some(doc.sun), ..SceneStyle::default() },
    });
    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stone_house_builds_closed_solids() {
        let doc = stone_house();
        let root = &doc.defs[&ROOT];
        assert_eq!(root.instances.len(), 11);
        for inst in root.instances.values() {
            let def = &doc.defs[&inst.def];
            assert!(!def.mesh.faces.is_empty(), "{} is empty", def.name);
            assert!(def.mesh.faces.keys().all(|f| def.mesh.face_area(*f) > 0.0), "{} has a flat face", def.name);
        }
        // The walls are one watertight solid with every window pushed in.
        let walls = root.instances.values().map(|i| &doc.defs[&i.def]).find(|d| d.name == "Walls").unwrap();
        let open: Vec<Id> = walls.mesh.edges.keys().copied().filter(|e| walls.mesh.edge_faces(*e).len() != 2).collect();
        assert!(open.is_empty(), "{} open edges", open.len());
        let glass = doc.materials.iter().position(|m| m.name == "Window Glass");
        let panes = walls.mesh.faces.values().filter(|f| f.material == glass).count();
        assert_eq!(panes, 37, "every window and door has its glass");
        let (lo, hi) = doc.world_bounds().unwrap();
        assert!(lo.z >= -1e-4 && hi.z > 10.0 && hi.z < 12.0, "{lo} {hi}");
        assert!(by_name("Stone-House").is_some() && by_name("castle").is_none());
    }

    #[test]
    fn stone_house_survives_save_and_open() {
        let doc = stone_house();
        let path = std::env::temp_dir().join(format!("opensketch-sample-{}.osk", std::process::id()));
        crate::io::save(&doc, &path).unwrap();
        let back = crate::io::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.defs.len(), doc.defs.len());
        assert_eq!(back.materials.len(), doc.materials.len());
        assert_eq!(back.scenes.len(), 1);
    }

    #[test]
    fn arch_outlines_close_neatly() {
        for o in [Opening::Pointed { s: 1.0, w: 0.95, z0: 4.0, zs: 5.75 }, Opening::Round { s: 1.0, w: 1.2, z0: 0.3, zs: 2.2 }] {
            for e in [0.0, 0.11] {
                let pts = o.outline(e);
                // Symmetric about the middle, and the apex is the highest point.
                let (s0, s1, _, z1) = o.bounds();
                let mid = (s0 + s1) / 2.0;
                for p in &pts {
                    assert!(pts.iter().any(|q| (q.0 - (2.0 * mid - p.0)).abs() < 1e-3 && (q.1 - p.1).abs() < 1e-3), "{o:?}");
                }
                let apex = pts.iter().fold(f32::MIN, |a, p| a.max(p.1));
                assert!(apex >= z1 - 1e-4);
                // No repeated points: every edge has a length.
                for i in 0..pts.len() {
                    let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                    assert!((a.0 - b.0).hypot(a.1 - b.1) > 1e-4, "{o:?} {e}");
                }
            }
        }
    }
}
