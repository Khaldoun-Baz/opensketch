//! Core geometry model: SketchUp-style "sticky" vertices, edges and faces.
//!
//! Faces are planar polygons described by an outer vertex loop (counter-clockwise
//! around the face normal) plus optional hole loops (clockwise). Edges are stored
//! explicitly so that free-standing lines can exist without faces.

use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

pub type Id = u32;

/// Geometric tolerance in model units (meters) — 0.1 mm.
pub const EPS: f32 = 1e-4;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub a: Id,
    pub b: Id,
    /// Edges drawn as one arc/circle share a curve id and select together.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve: Option<Id>,
    /// Soft edges are hidden and the faces around them are shaded smoothly.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub soft: bool,
}

impl Edge {
    pub fn has(&self, v: Id) -> bool {
        self.a == v || self.b == v
    }
    pub fn other(&self, v: Id) -> Id {
        if self.a == v {
            self.b
        } else {
            self.a
        }
    }
    pub fn same(&self, a: Id, b: Id) -> bool {
        (self.a == a && self.b == b) || (self.a == b && self.b == a)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Face {
    pub outer: Vec<Id>,
    #[serde(default)]
    pub holes: Vec<Vec<Id>>,
    pub normal: Vec3,
    #[serde(default)]
    pub material: Option<usize>,
}

impl Face {
    pub fn loops(&self) -> impl Iterator<Item = &Vec<Id>> {
        std::iter::once(&self.outer).chain(self.holes.iter())
    }
    pub fn vertices(&self) -> impl Iterator<Item = Id> + '_ {
        self.loops().flat_map(|l| l.iter().copied())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    /// Plain color (and the average color of the texture, if any).
    pub color: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<crate::texture::Texture>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub vertices: BTreeMap<Id, Vec3>,
    pub edges: BTreeMap<Id, Edge>,
    pub faces: BTreeMap<Id, Face>,
    /// Only used when loading files saved before groups existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<Material>,
    pub next_id: Id,
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free geometry helpers
// ---------------------------------------------------------------------------

/// Newell's method: un-normalized polygon normal (length = 2 * area).
pub fn newell(pts: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 0..pts.len() {
        let c = pts[i];
        let nx = pts[(i + 1) % pts.len()];
        n.x += (c.y - nx.y) * (c.z + nx.z);
        n.y += (c.z - nx.z) * (c.x + nx.x);
        n.z += (c.x - nx.x) * (c.y + nx.y);
    }
    n
}

/// Right-handed in-plane basis (u, v) with u x v = n.
pub fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let u = up.cross(n).normalize();
    let v = n.cross(u);
    (u, v)
}

/// Axis-aligned drawing axes for a plane (used by the rectangle tool).
pub fn drawing_axes(n: Vec3) -> (Vec3, Vec3) {
    if n.z.abs() > 0.9 {
        let u = Vec3::X;
        let v = n.cross(u).normalize();
        (u, v)
    } else {
        let u = Vec3::Z.cross(n).normalize();
        let v = n.cross(u);
        (u, v)
    }
}

pub fn point_in_poly(p: Vec2, poly: &[Vec2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Distance from point to segment, plus the segment parameter.
pub fn point_seg(p: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let ab = b - a;
    let l2 = ab.length_squared();
    if l2 < 1e-12 {
        return (p.distance(a), 0.0);
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    ((a + ab * t).distance(p), t)
}

pub fn ray_plane(o: Vec3, d: Vec3, p0: Vec3, n: Vec3) -> Option<Vec3> {
    let denom = d.dot(n);
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = (p0 - o).dot(n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(o + d * t)
}

/// Closest point on the infinite line (p0, dir) to the ray (o, d).
pub fn closest_on_line_to_ray(p0: Vec3, dir: Vec3, o: Vec3, d: Vec3) -> Vec3 {
    let w0 = p0 - o;
    let a = dir.dot(dir);
    let b = dir.dot(d);
    let c = d.dot(d);
    let dd = dir.dot(w0);
    let e = d.dot(w0);
    let den = a * c - b * b;
    let s = if den.abs() > 1e-12 { (b * e - c * dd) / den } else { 0.0 };
    p0 + dir * s
}

/// Offsets a closed loop (CCW around `n`) inward by `d` (negative = outward).
pub fn offset_loop(pts: &[Vec3], n: Vec3, d: f32) -> Vec<Vec3> {
    let len = pts.len();
    (0..len)
        .map(|i| {
            let p = pts[i];
            let prev = pts[(i + len - 1) % len];
            let next = pts[(i + 1) % len];
            let in1 = n.cross((p - prev).normalize_or_zero());
            let in2 = n.cross((next - p).normalize_or_zero());
            let denom = 1.0 + in1.dot(in2);
            if denom.abs() < 1e-4 {
                p + in1 * d
            } else {
                p + (in1 + in2) * (d / denom)
            }
        })
        .collect()
}

fn dedup_loop(lp: &mut Vec<Id>) {
    lp.dedup();
    while lp.len() > 1 && lp.first() == lp.last() {
        lp.pop();
    }
}

/// Ray (origin, direction) hits the triangle in front of the origin.
fn ray_hits_triangle(o: Vec3, d: Vec3, t: &[Vec3; 3]) -> bool {
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return false;
    }
    let inv = 1.0 / det;
    let s = o - t[0];
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    v >= 0.0 && u + v <= 1.0 && e2.dot(q) * inv > 1e-4
}

fn loop_edge_index(lp: &[Id], a: Id, b: Id) -> Option<usize> {
    let n = lp.len();
    (0..n).find(|&i| {
        let (x, y) = (lp[i], lp[(i + 1) % n]);
        (x == a && y == b) || (x == b && y == a)
    })
}

pub fn loop_has_edge(lp: &[Id], a: Id, b: Id) -> bool {
    loop_edge_index(lp, a, b).is_some()
}

pub fn default_materials() -> Vec<Material> {
    use crate::texture::{Pattern, Texture};
    let m = |name: &str, color: [f32; 3]| Material { name: name.to_string(), color, texture: None };
    let t = |name: &str, color: [f32; 3], p: Pattern| Material { name: name.to_string(), color, texture: Some(Texture::builtin(p)) };
    vec![
        t("Brick", [0.70, 0.32, 0.25], Pattern::Brick),
        t("Wood", [0.64, 0.45, 0.27], Pattern::Wood),
        m("Concrete", [0.62, 0.62, 0.60]),
        t("Stone", [0.55, 0.53, 0.49], Pattern::Stone),
        t("Roof Tile", [0.42, 0.19, 0.15], Pattern::Shingles),
        m("Glass", [0.62, 0.80, 0.90]),
        t("Grass", [0.33, 0.55, 0.22], Pattern::Grass),
        m("Water", [0.28, 0.48, 0.78]),
        m("Plaster", [0.93, 0.90, 0.82]),
        m("Charcoal", [0.18, 0.18, 0.20]),
        m("Sunflower", [0.95, 0.78, 0.22]),
        m("Teal", [0.20, 0.60, 0.58]),
        t("Floor Tiles", [0.86, 0.84, 0.79], Pattern::Tiles),
    ]
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

impl Model {
    pub fn new() -> Self {
        Self { vertices: BTreeMap::new(), edges: BTreeMap::new(), faces: BTreeMap::new(), materials: Vec::new(), next_id: 1 }
    }

    fn nid(&mut self) -> Id {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    pub fn pos(&self, v: Id) -> Vec3 {
        self.vertices.get(&v).copied().unwrap_or(Vec3::ZERO)
    }

    pub fn loop_points(&self, lp: &[Id]) -> Vec<Vec3> {
        lp.iter().map(|&v| self.pos(v)).collect()
    }

    // ----- vertices -------------------------------------------------------

    pub fn find_vertex(&self, p: Vec3) -> Option<Id> {
        self.vertices.iter().find(|(_, q)| q.distance(p) < EPS).map(|(i, _)| *i)
    }

    pub fn add_vertex(&mut self, p: Vec3) -> Id {
        if let Some(v) = self.find_vertex(p) {
            return v;
        }
        self.add_vertex_raw(p)
    }

    pub fn add_vertex_raw(&mut self, p: Vec3) -> Id {
        let id = self.nid();
        self.vertices.insert(id, p);
        id
    }

    /// Returns a vertex at `p`, re-using an existing vertex or splitting an edge
    /// that passes through `p` so the new geometry connects to the old.
    pub fn vertex_on_geometry(&mut self, p: Vec3) -> Id {
        if let Some(v) = self.find_vertex(p) {
            return v;
        }
        let hit = self
            .edges
            .iter()
            .find(|(_, e)| {
                let (d, t) = point_seg(p, self.pos(e.a), self.pos(e.b));
                d < EPS && t > 0.0 && t < 1.0
            })
            .map(|(i, _)| *i);
        match hit {
            Some(e) => self.split_edge(e, p),
            None => self.add_vertex_raw(p),
        }
    }

    pub fn vertex_edges(&self, v: Id) -> Vec<Id> {
        self.edges.iter().filter(|(_, e)| e.has(v)).map(|(i, _)| *i).collect()
    }

    pub fn neighbors(&self, v: Id) -> Vec<Id> {
        self.edges.values().filter(|e| e.has(v)).map(|e| e.other(v)).collect()
    }

    pub(crate) fn remove_vertex_if_orphan(&mut self, v: Id) {
        let used_by_edge = self.edges.values().any(|e| e.has(v));
        let used_by_face = self.faces.values().any(|f| f.vertices().any(|x| x == v));
        if !used_by_edge && !used_by_face {
            self.vertices.remove(&v);
        }
    }

    // ----- edges ----------------------------------------------------------

    pub fn find_edge(&self, a: Id, b: Id) -> Option<Id> {
        self.edges.iter().find(|(_, e)| e.same(a, b)).map(|(i, _)| *i)
    }

    pub fn add_edge(&mut self, a: Id, b: Id) -> Option<Id> {
        if a == b {
            return None;
        }
        if let Some(e) = self.find_edge(a, b) {
            return Some(e);
        }
        let id = self.nid();
        self.edges.insert(id, Edge { a, b, curve: None, soft: false });
        Some(id)
    }

    fn ensure_loop_edges(&mut self, lp: &[Id]) {
        for i in 0..lp.len() {
            self.add_edge(lp[i], lp[(i + 1) % lp.len()]);
        }
    }

    pub fn faces_with_edge(&self, a: Id, b: Id) -> Vec<Id> {
        self.faces.iter().filter(|(_, f)| f.loops().any(|l| loop_has_edge(l, a, b))).map(|(i, _)| *i).collect()
    }

    pub fn edge_faces(&self, e: Id) -> Vec<Id> {
        match self.edges.get(&e) {
            Some(edge) => self.faces_with_edge(edge.a, edge.b),
            None => vec![],
        }
    }

    pub fn edge_length(&self, e: Id) -> f32 {
        self.edges.get(&e).map(|e| self.pos(e.a).distance(self.pos(e.b))).unwrap_or(0.0)
    }

    pub fn split_edge(&mut self, e: Id, p: Vec3) -> Id {
        if !self.edges.contains_key(&e) {
            return self.add_vertex(p);
        }
        let v = self.add_vertex_raw(p);
        self.split_edge_at(e, v);
        v
    }

    /// Inserts vertex `v` into edge `e` and into every face loop using it.
    pub(crate) fn split_edge_at(&mut self, e: Id, v: Id) {
        let Some(old) = self.edges.get(&e).cloned() else { return };
        let (a, b) = (old.a, old.b);
        if v == a || v == b {
            return;
        }
        self.edges.remove(&e);
        for (x, y) in [(a, v), (v, b)] {
            if let Some(ne) = self.add_edge(x, y) {
                let edge = self.edges.get_mut(&ne).unwrap();
                edge.curve = old.curve;
                edge.soft = old.soft;
            }
        }
        for face in self.faces.values_mut() {
            for lp in std::iter::once(&mut face.outer).chain(face.holes.iter_mut()) {
                if let Some(i) = loop_edge_index(lp, a, b) {
                    lp.insert(i + 1, v);
                }
            }
        }
    }

    // ----- faces ----------------------------------------------------------

    fn orient(&self, lp: &mut [Id], n: Vec3, ccw: bool) {
        let nn = newell(&self.loop_points(lp));
        if (nn.dot(n) > 0.0) != ccw {
            lp.reverse();
        }
    }

    fn same_vertex_set(a: &[Id], b: &[Id]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut x = a.to_vec();
        let mut y = b.to_vec();
        x.sort_unstable();
        y.sort_unstable();
        x == y
    }

    pub fn find_face_with_loop(&self, lp: &[Id]) -> Option<Id> {
        self.faces.iter().find(|(_, f)| Self::same_vertex_set(&f.outer, lp)).map(|(i, _)| *i)
    }

    /// Low-level face insertion with explicit orientation (no inference).
    pub fn add_face_loops(&mut self, outer: Vec<Id>, holes: Vec<Vec<Id>>, normal: Vec3, material: Option<usize>) -> Option<Id> {
        self.insert_face(None, outer, holes, normal, material)
    }

    fn insert_face(
        &mut self,
        id: Option<Id>,
        mut outer: Vec<Id>,
        mut holes: Vec<Vec<Id>>,
        normal: Vec3,
        material: Option<usize>,
    ) -> Option<Id> {
        dedup_loop(&mut outer);
        if outer.len() < 3 {
            return None;
        }
        if newell(&self.loop_points(&outer)).length() < 1e-9 {
            return None;
        }
        if let Some(existing) = self.find_face_with_loop(&outer) {
            return Some(existing);
        }
        let n = normal.normalize_or_zero();
        self.orient(&mut outer, n, true);
        for h in holes.iter_mut() {
            dedup_loop(h);
            self.orient(h, n, false);
        }
        holes.retain(|h| h.len() >= 3);
        self.ensure_loop_edges(&outer.clone());
        for h in holes.clone() {
            self.ensure_loop_edges(&h);
        }
        let id = id.unwrap_or_else(|| self.nid());
        let normal = newell(&self.loop_points(&outer)).normalize_or_zero();
        self.faces.insert(id, Face { outer, holes, normal, material });
        Some(id)
    }

    /// High-level face creation used by drawing tools. Picks a sensible
    /// orientation and turns the new face into a hole when it lies inside an
    /// existing coplanar face.
    pub fn add_face(&mut self, mut lp: Vec<Id>, hint: Option<Vec3>) -> Option<Id> {
        dedup_loop(&mut lp);
        if lp.len() < 3 {
            return None;
        }
        let pts = self.loop_points(&lp);
        let nn = newell(&pts);
        if nn.length() < 1e-9 {
            return None;
        }
        let mut n = nn.normalize();
        match hint {
            Some(h) => {
                if n.dot(h) < 0.0 {
                    n = -n;
                }
            }
            None => {
                if n.z.abs() > 0.999 {
                    let on_ground = pts.iter().all(|p| p.z.abs() < EPS);
                    n = if on_ground { -Vec3::Z } else { Vec3::Z };
                }
            }
        }
        if self.find_face_with_loop(&lp).is_some() {
            return None;
        }

        // Hole detection: strictly inside an existing coplanar face.
        let set: HashSet<Id> = lp.iter().copied().collect();
        let host = self
            .faces
            .iter()
            .filter(|(_, f)| {
                f.normal.dot(n).abs() > 0.9999
                    && (pts[0] - self.pos(f.outer[0])).dot(f.normal).abs() < EPS * 10.0
                    && !f.vertices().any(|v| set.contains(&v))
            })
            .map(|(i, _)| *i)
            .find(|&fid| pts.iter().all(|&p| self.point_strictly_in_face(fid, p)));
        if let Some(host) = host {
            let hn = self.faces[&host].normal;
            let mut hole = lp.clone();
            self.orient(&mut hole, hn, false);
            self.faces.get_mut(&host).unwrap().holes.push(hole);
        }
        self.add_face_loops(lp, vec![], n, None)
    }

    /// Adds a closed polygon (rectangle, circle, ...) connecting to existing geometry.
    /// Overlaps with coplanar faces are resolved by rebuilding the plane.
    /// Returns the face now covering the polygon's interior.
    pub fn add_polygon(&mut self, pts: &[Vec3], hint: Option<Vec3>) -> Option<Id> {
        let lp: Vec<Id> = pts.iter().map(|&p| self.vertex_on_geometry(p)).collect();
        self.ensure_loop_edges(&lp);
        let f = self.add_face(lp.clone(), hint)?;
        let n = self.faces[&f].normal;
        let inside = self.interior_point(&lp, n);
        self.rebuild_planes_containing(pts);
        if self.faces.contains_key(&f) {
            return Some(f);
        }
        let ip = inside?;
        self.faces.iter().filter(|(_, x)| x.normal.dot(n).abs() > 0.9999).map(|(i, _)| *i).find(|&i| self.point_in_face(i, ip))
    }

    pub fn face_2d(&self, f: Id) -> Option<(Vec3, Vec3, Vec<Vec<Vec2>>)> {
        let face = self.faces.get(&f)?;
        let (u, v) = plane_basis(face.normal);
        let loops = face
            .loops()
            .map(|l| {
                l.iter()
                    .map(|&id| {
                        let p = self.pos(id);
                        Vec2::new(p.dot(u), p.dot(v))
                    })
                    .collect()
            })
            .collect();
        Some((u, v, loops))
    }

    pub fn point_in_face(&self, f: Id, p: Vec3) -> bool {
        let Some((u, v, loops)) = self.face_2d(f) else { return false };
        let q = Vec2::new(p.dot(u), p.dot(v));
        point_in_poly(q, &loops[0]) && !loops[1..].iter().any(|h| point_in_poly(q, h))
    }

    fn point_strictly_in_face(&self, f: Id, p: Vec3) -> bool {
        if !self.point_in_face(f, p) {
            return false;
        }
        let face = &self.faces[&f];
        face.loops().all(|l| (0..l.len()).all(|i| point_seg(p, self.pos(l[i]), self.pos(l[(i + 1) % l.len()])).0 > EPS))
    }

    pub fn point_in_loop(&self, lp: &[Id], p: Vec3, n: Vec3) -> bool {
        let (u, v) = plane_basis(n);
        let poly: Vec<Vec2> = lp
            .iter()
            .map(|&id| {
                let q = self.pos(id);
                Vec2::new(q.dot(u), q.dot(v))
            })
            .collect();
        point_in_poly(Vec2::new(p.dot(u), p.dot(v)), &poly)
    }

    pub fn face_area(&self, f: Id) -> f32 {
        let Some(face) = self.faces.get(&f) else { return 0.0 };
        let outer = newell(&self.loop_points(&face.outer)).length() * 0.5;
        let holes: f32 = face.holes.iter().map(|h| newell(&self.loop_points(h)).length() * 0.5).sum();
        outer - holes
    }

    pub fn recompute_normals(&mut self) {
        let normals: Vec<(Id, Vec3)> = self.faces.iter().map(|(i, f)| (*i, newell(&self.loop_points(&f.outer)))).collect();
        for (i, n) in normals {
            if n.length() > 1e-12 {
                self.faces.get_mut(&i).unwrap().normal = n.normalize();
            }
        }
    }

    // ----- triangulation --------------------------------------------------

    /// Triangles of a face as vertex ids, wound counter-clockwise around its normal.
    pub fn triangulate_ids(&self, f: Id) -> Vec<[Id; 3]> {
        let Some(face) = self.faces.get(&f) else { return vec![] };
        let n = face.normal;
        let (u, v) = plane_basis(n);
        let mut data: Vec<f64> = Vec::new();
        let mut hole_idx = Vec::new();
        let mut ids = Vec::new();
        for (i, lp) in face.loops().enumerate() {
            if i > 0 {
                hole_idx.push(ids.len());
            }
            for &id in lp {
                let p = self.pos(id);
                data.push(p.dot(u) as f64);
                data.push(p.dot(v) as f64);
                ids.push(id);
            }
        }
        let idx = earcut(&data, &hole_idx);
        idx.as_chunks::<3>()
            .0
            .iter()
            .map(|c| {
                let (a, mut b, mut c2) = (ids[c[0]], ids[c[1]], ids[c[2]]);
                let (pa, pb, pc) = (self.pos(a), self.pos(b), self.pos(c2));
                if (pb - pa).cross(pc - pa).dot(n) < 0.0 {
                    std::mem::swap(&mut b, &mut c2);
                }
                [a, b, c2]
            })
            .collect()
    }

    pub fn triangulate(&self, f: Id) -> Vec<[Vec3; 3]> {
        self.triangulate_ids(f).into_iter().map(|t| t.map(|v| self.pos(v))).collect()
    }

    /// Per (face, vertex) normals that blend across soft edges (smooth shading).
    pub fn shading_normals(&self) -> HashMap<(Id, Id), Vec3> {
        let mut out = HashMap::new();
        let soft_v: HashSet<Id> = self.edges.values().filter(|e| e.soft).flat_map(|e| [e.a, e.b]).collect();
        if soft_v.is_empty() {
            return out;
        }
        let mut vf: HashMap<Id, Vec<Vec3>> = HashMap::new();
        for f in self.faces.values() {
            for v in f.vertices().filter(|v| soft_v.contains(v)) {
                vf.entry(v).or_default().push(f.normal);
            }
        }
        for (&fid, f) in &self.faces {
            for v in f.vertices() {
                if let Some(ns) = vf.get(&v) {
                    let sum = ns.iter().filter(|n| n.dot(f.normal) > 0.8).fold(Vec3::ZERO, |acc, n| acc + *n);
                    out.insert((fid, v), sum.normalize_or(f.normal));
                }
            }
        }
        out
    }

    // ----- topology editing ----------------------------------------------

    /// Draws a line between two vertices, splitting faces or creating new ones.
    /// Returns true if a face was created or split.
    pub fn insert_line(&mut self, a: Id, b: Id) -> bool {
        if a == b || self.find_edge(a, b).is_some() {
            return false;
        }
        let r = self.insert_line_inner(a, b);
        let pts = [self.pos(a), self.pos(b)];
        self.rebuild_planes_containing(&pts);
        r
    }

    fn insert_line_inner(&mut self, a: Id, b: Id) -> bool {
        self.add_edge(a, b);
        let (pa, pb) = (self.pos(a), self.pos(b));
        let mid = (pa + pb) * 0.5;

        // 1. Split an existing face if the line cuts across it.
        let candidates: Vec<Id> = self.faces.keys().copied().collect();
        for f in candidates {
            let face = &self.faces[&f];
            if face.outer.contains(&a)
                && face.outer.contains(&b)
                && (mid - pa).dot(face.normal).abs() < EPS * 10.0
                && self.point_in_face(f, mid)
                && self.split_face(f, a, b).is_some()
            {
                return true;
            }
        }

        // 2. Close new loops.
        self.find_new_faces(a, b) > 0
    }

    fn split_face(&mut self, f: Id, a: Id, b: Id) -> Option<Id> {
        let face = self.faces.get(&f)?.clone();
        let lp = &face.outer;
        let ia = lp.iter().position(|&x| x == a)?;
        let ib = lp.iter().position(|&x| x == b)?;
        let (i, j) = (ia.min(ib), ia.max(ib));
        if j - i < 2 || (i == 0 && j == lp.len() - 1) {
            return None;
        }
        let l1: Vec<Id> = lp[i..=j].to_vec();
        let mut l2: Vec<Id> = lp[j..].to_vec();
        l2.extend_from_slice(&lp[..=i]);
        let (mut h1, mut h2) = (vec![], vec![]);
        for h in face.holes {
            if self.point_in_loop(&l1, self.pos(h[0]), face.normal) {
                h1.push(h);
            } else {
                h2.push(h);
            }
        }
        {
            let fm = self.faces.get_mut(&f).unwrap();
            fm.outer = l1;
            fm.holes = h1;
        }
        let id = self.nid();
        self.faces.insert(id, Face { outer: l2, holes: h2, normal: face.normal, material: face.material });
        Some(id)
    }

    fn interior_point(&self, lp: &[Id], n: Vec3) -> Option<Vec3> {
        let (u, v) = plane_basis(n);
        let pts = self.loop_points(lp);
        let data: Vec<f64> = pts.iter().flat_map(|p| [p.dot(u) as f64, p.dot(v) as f64]).collect();
        let idx = earcut(&data, &[]);
        if idx.len() < 3 {
            return None;
        }
        Some((pts[idx[0]] + pts[idx[1]] + pts[idx[2]]) / 3.0)
    }

    fn find_new_faces(&mut self, a: Id, b: Id) -> usize {
        let (pa, pb) = (self.pos(a), self.pos(b));
        let dir = pb - pa;
        let mut normals: Vec<Vec3> = Vec::new();
        for s in [a, b] {
            for c in self.neighbors(s) {
                if c == a || c == b {
                    continue;
                }
                let n = dir.cross(self.pos(c) - pa);
                if n.length() < 1e-9 {
                    continue;
                }
                let n = n.normalize();
                if !normals.iter().any(|m| m.dot(n).abs() > 0.9999) {
                    normals.push(n);
                }
            }
        }

        let mut created = 0;
        for n in normals {
            let on_plane = |m: &Model, v: Id| (m.pos(v) - pa).dot(n).abs() < EPS * 10.0;
            let mut prev: HashMap<Id, Id> = HashMap::new();
            let mut visited: HashSet<Id> = HashSet::from([b]);
            let mut queue = VecDeque::from([b]);
            let mut found = false;
            'bfs: while let Some(v) = queue.pop_front() {
                for w in self.neighbors(v) {
                    if v == b && w == a {
                        continue;
                    }
                    if visited.contains(&w) || !on_plane(self, w) {
                        continue;
                    }
                    prev.insert(w, v);
                    if w == a {
                        found = true;
                        break 'bfs;
                    }
                    visited.insert(w);
                    queue.push_back(w);
                }
            }
            if !found {
                continue;
            }
            let mut chain = vec![];
            let mut cur = a;
            while cur != b {
                cur = prev[&cur];
                chain.push(cur);
            }
            chain.reverse(); // b ... (vertex before a)
            let mut lp = vec![a];
            lp.extend(chain);
            if lp.len() < 3 || self.find_face_with_loop(&lp).is_some() {
                continue;
            }
            // Skip if this region is already covered by a coplanar face sharing the boundary.
            if let Some(ip) = self.interior_point(&lp, n) {
                let set: HashSet<Id> = lp.iter().copied().collect();
                let covered = self.faces.iter().any(|(fid, f)| {
                    f.normal.dot(n).abs() > 0.9999 && f.vertices().any(|v| set.contains(&v)) && self.point_in_face(*fid, ip)
                });
                if covered {
                    continue;
                }
            }
            if self.add_face(lp, None).is_some() {
                created += 1;
            }
        }
        created
    }

    /// Merges two coplanar faces sharing edge (a, b). Returns true on success.
    fn merge_faces(&mut self, f1: Id, f2: Id, a: Id, b: Id) -> bool {
        let (Some(fa), Some(fb)) = (self.faces.get(&f1).cloned(), self.faces.get(&f2).cloned()) else {
            return false;
        };
        let la = fa.outer.clone();
        let mut lb = fb.outer.clone();
        let Some(ia) = loop_edge_index(&la, a, b) else { return false };
        let (x, y) = (la[ia], la[(ia + 1) % la.len()]); // la has x -> y
                                                        // lb must traverse y -> x
        let ib = loop_edge_index(&lb, a, b);
        let Some(ib) = ib else { return false };
        if lb[ib] == x {
            lb.reverse();
        }
        let iy = la.iter().position(|&v| v == y).unwrap();
        let a_rot: Vec<Id> = la[iy..].iter().chain(la[..iy].iter()).copied().collect(); // y ... x
        let ix = lb.iter().position(|&v| v == x).unwrap();
        let b_rot: Vec<Id> = lb[ix..].iter().chain(lb[..ix].iter()).copied().collect(); // x ... y
        let mut merged = a_rot;
        merged.extend_from_slice(&b_rot[1..b_rot.len() - 1]);
        let unique: HashSet<Id> = merged.iter().copied().collect();
        if unique.len() != merged.len() {
            return false;
        }
        let mut holes = fa.holes.clone();
        holes.extend(fb.holes.clone());
        self.faces.remove(&f2);
        let f = self.faces.get_mut(&f1).unwrap();
        f.outer = merged;
        f.holes = holes;
        true
    }

    fn coplanar(&self, f1: Id, f2: Id) -> bool {
        let (a, b) = (&self.faces[&f1], &self.faces[&f2]);
        a.normal.dot(b.normal) > 0.9999 && (self.pos(b.outer[0]) - self.pos(a.outer[0])).dot(a.normal).abs() < EPS * 10.0
    }

    /// Removes a vertex that only joins two collinear edges.
    pub(crate) fn remove_collinear_vertex(&mut self, v: Id) {
        let es = self.vertex_edges(v);
        if es.len() != 2 {
            return;
        }
        let o1 = self.edges[&es[0]].other(v);
        let o2 = self.edges[&es[1]].other(v);
        let p = self.pos(v);
        let d1 = (p - self.pos(o1)).normalize_or_zero();
        let d2 = (self.pos(o2) - p).normalize_or_zero();
        if d1.dot(d2) < 0.99999 {
            return;
        }
        for face in self.faces.values_mut() {
            face.outer.retain(|&x| x != v);
            for h in face.holes.iter_mut() {
                h.retain(|&x| x != v);
            }
        }
        self.faces.retain(|_, f| f.outer.len() >= 3);
        self.edges.remove(&es[0]);
        self.edges.remove(&es[1]);
        self.vertices.remove(&v);
        self.add_edge(o1, o2);
    }

    /// Merge coplanar faces across the given edges and remove redundant vertices.
    pub fn heal(&mut self, pairs: &[(Id, Id)]) {
        for &(a, b) in pairs {
            let Some(e) = self.find_edge(a, b) else { continue };
            let fs = self.faces_with_edge(a, b);
            if fs.len() != 2 {
                continue;
            }
            let in_outer = fs.iter().all(|f| loop_has_edge(&self.faces[f].outer, a, b));
            if in_outer && self.coplanar(fs[0], fs[1]) && self.merge_faces(fs[0], fs[1], a, b) {
                self.edges.remove(&e);
                self.remove_collinear_vertex(a);
                self.remove_collinear_vertex(b);
            }
        }
    }

    pub fn delete_face(&mut self, f: Id) {
        self.faces.remove(&f);
    }

    /// Erases an edge: merges coplanar neighbours, otherwise removes faces bounded by it.
    pub fn delete_edge(&mut self, e: Id) {
        let Some(Edge { a, b, .. }) = self.edges.get(&e).cloned() else { return };
        let fs = self.faces_with_edge(a, b);
        // Holes bounded by this edge disappear.
        for &f in &fs {
            if let Some(face) = self.faces.get_mut(&f) {
                face.holes.retain(|h| !loop_has_edge(h, a, b));
            }
        }
        let outer_fs: Vec<Id> =
            fs.into_iter().filter(|f| self.faces.get(f).map(|x| loop_has_edge(&x.outer, a, b)).unwrap_or(false)).collect();
        if outer_fs.len() == 2 && self.coplanar(outer_fs[0], outer_fs[1]) && self.merge_faces(outer_fs[0], outer_fs[1], a, b) {
            self.edges.remove(&e);
            self.remove_collinear_vertex(a);
            self.remove_collinear_vertex(b);
            return;
        }
        for f in outer_fs {
            self.faces.remove(&f);
        }
        self.edges.remove(&e);
        self.remove_vertex_if_orphan(a);
        self.remove_vertex_if_orphan(b);
    }

    pub fn delete(&mut self, faces: &[Id], edges: &[Id]) {
        for &f in faces {
            self.delete_face(f);
        }
        for &e in edges {
            self.delete_edge(e);
        }
    }

    // ----- operations -----------------------------------------------------

    /// SketchUp Push/Pull: extrude face `f` by `d` along its normal.
    ///
    /// * When every face around `f` is perpendicular to it (a box's top, a
    ///   pocket's floor) the face simply moves and its sides stretch; pushing it
    ///   exactly flat collapses the volume and merges what is left.
    /// * Otherwise new side faces are created. Pushing towards the back of a
    ///   face with material behind it makes a pocket (and a hole once it
    ///   reaches the far side); with nothing behind, the face extrudes out of
    ///   its back instead, facing outward.
    pub fn push_pull(&mut self, f: Id, d: f32) -> bool {
        if d.abs() < EPS {
            return false;
        }
        let Some(face) = self.faces.get(&f).cloned() else { return false };
        if let Some(done) = self.push_pull_move(f, &face, d) {
            return done;
        }
        let n = face.normal.normalize();
        let off = n * d;
        let lone =
            face.loops().all(|lp| (0..lp.len()).all(|i| self.faces_with_edge(lp[i], lp[(i + 1) % lp.len()]).iter().all(|g| *g == f)));
        let recess = !lone && d < 0.0 && self.material_behind(f);
        self.faces.remove(&f);

        let mut new_loops = Vec::new();
        let mut old_edges = Vec::new();
        let mut rails = Vec::new();
        let mut sides = Vec::new();
        let loops: Vec<Vec<Id>> = face.loops().cloned().collect();
        for lp in &loops {
            let nl: Vec<Id> = lp
                .iter()
                .map(|&v| {
                    let p = self.pos(v) + off;
                    self.add_vertex(p)
                })
                .collect();
            for i in 0..lp.len() {
                let j = (i + 1) % lp.len();
                let (v0, v1, w0, w1) = (lp[i], lp[j], nl[i], nl[j]);
                old_edges.push((v0, v1));
                rails.push((v0, w0));
                let e = self.pos(v1) - self.pos(v0);
                let mut out = e.cross(n);
                if recess {
                    out = -out; // pocket walls face into the pocket
                }
                sides.extend(self.add_face_loops(vec![v0, v1, w1, w0], vec![], out, None));
            }
            new_loops.push(nl);
        }
        // The cap faces the way it moved, except a pocket floor, which keeps
        // facing out of the solid.
        let cap_n = if recess { n } else { n * d.signum() };
        let outer = new_loops.remove(0);
        let cap = self.add_face_loops(outer, new_loops, cap_n, face.material);
        if lone {
            self.add_face_loops(face.outer.clone(), face.holes.clone(), -n * d.signum(), face.material);
        } else {
            self.heal(&old_edges);
        }
        if let (true, Some(cap)) = (recess, cap) {
            self.punch_through(cap);
        }
        self.merge_overlaps(&sides);
        self.soften_pairs(&rails);
        true
    }

    /// True if a ray from inside face `f` going out of its back hits other geometry.
    fn material_behind(&self, f: Id) -> bool {
        let Some(face) = self.faces.get(&f) else { return false };
        let n = face.normal;
        let Some(t) = self.triangulate(f).into_iter().next() else { return false };
        let o = (t[0] + t[1] + t[2]) / 3.0;
        self.faces.keys().filter(|g| **g != f).flat_map(|g| self.triangulate(*g)).any(|tri| ray_hits_triangle(o, -n, &tri))
    }

    /// A pocket floor that reached a face on the far side opens a hole there.
    fn punch_through(&mut self, cap: Id) {
        let Some(c) = self.faces.get(&cap).cloned() else { return };
        let Some(t) = self.triangulate(cap).into_iter().next() else { return };
        let inside = (t[0] + t[1] + t[2]) / 3.0;
        let p0 = self.pos(c.outer[0]);
        let opposite = self.faces.iter().any(|(g, x)| {
            *g != cap
                && x.normal.dot(c.normal) < -0.9999
                && (self.pos(x.outer[0]) - p0).dot(c.normal).abs() < EPS * 10.0
                && self.point_in_face(*g, inside)
        });
        if opposite {
            let loops: Vec<Vec<Vec3>> = c.loops().map(|l| self.loop_points(l)).collect();
            self.subtract_loops(c.normal, p0, &loops[..1]);
        }
    }

    /// New faces that overlap existing coplanar faces are merged with them
    /// (e.g. a wall pulled up against a neighbouring building).
    fn merge_overlaps(&mut self, new_faces: &[Id]) {
        let mut done: Vec<(Vec3, f32)> = Vec::new();
        for &s in new_faces {
            let Some(face) = self.faces.get(&s) else { continue };
            let (n, d) = canonical_plane(face.normal, self.pos(face.outer[0]));
            if done.iter().any(|(m, e)| m.dot(n) > 0.9999 && (e - d).abs() < EPS * 10.0) {
                continue;
            }
            let others = self.faces.iter().any(|(g, x)| {
                !new_faces.contains(g)
                    && x.normal.dot(n).abs() > 0.9999
                    && x.outer.iter().all(|&v| (self.pos(v).dot(n) - d).abs() < EPS * 10.0)
            });
            if others {
                self.rebuild_plane(n, d);
                done.push((n, d));
            }
        }
    }

    /// Push/Pull of a face whose neighbours are all perpendicular to it: the
    /// face moves and stretches them. Returns None when that doesn't apply.
    fn push_pull_move(&mut self, f: Id, face: &Face, d: f32) -> Option<bool> {
        let n = face.normal.normalize();
        let mut sides: Vec<Id> = Vec::new();
        for lp in face.loops() {
            for i in 0..lp.len() {
                let others: Vec<Id> = self.faces_with_edge(lp[i], lp[(i + 1) % lp.len()]).into_iter().filter(|g| *g != f).collect();
                if others.len() != 1 || self.faces[&others[0]].normal.dot(n).abs() > 1e-3 {
                    return None;
                }
                if !sides.contains(&others[0]) {
                    sides.push(others[0]);
                }
            }
        }
        let verts: HashSet<Id> = face.vertices().collect();
        // Moving shared vertices must not bend any other face.
        if self.faces.iter().any(|(g, x)| *g != f && !sides.contains(g) && x.vertices().any(|v| verts.contains(&v))) {
            return None;
        }
        // Sides all behind the face (a box top) or all in front (a pocket floor).
        let p0 = self.pos(face.outer[0]);
        let reach = |g: Id, s: f32| self.faces[&g].vertices().map(|v| (self.pos(v) - p0).dot(n) * s).fold(0.0f32, f32::max);
        let behind = sides.iter().all(|g| reach(*g, 1.0) < EPS);
        let front = sides.iter().all(|g| reach(*g, -1.0) < EPS);
        let limit = match (behind, front, d < 0.0) {
            (true, _, true) => sides.iter().map(|g| reach(*g, -1.0)).fold(f32::MAX, f32::min),
            (_, true, false) => sides.iter().map(|g| reach(*g, 1.0)).fold(f32::MAX, f32::min),
            (true, _, false) | (_, true, true) => f32::MAX,
            _ => return None,
        };
        if d.abs() > limit + EPS {
            return None;
        }
        for v in &verts {
            if let Some(p) = self.vertices.get_mut(v) {
                *p += n * d;
            }
        }
        if (d.abs() - limit).abs() < EPS {
            self.collapse(f, &verts);
        } else {
            self.merge_overlaps(&sides);
        }
        Some(true)
    }

    /// After a face was pushed exactly flat onto other geometry: weld the
    /// vertices that met, drop the sides that vanished, and merge the face
    /// with what it now lies on.
    fn collapse(&mut self, f: Id, moved: &HashSet<Id>) {
        let mut remap: HashMap<Id, Id> = HashMap::new();
        for &v in moved {
            let p = self.pos(v);
            if let Some((&w, _)) = self.vertices.iter().find(|(id, q)| !moved.contains(id) && q.distance(p) < EPS) {
                remap.insert(v, w);
            }
        }
        let get = |v: Id| *remap.get(&v).unwrap_or(&v);
        for face in self.faces.values_mut() {
            for lp in std::iter::once(&mut face.outer).chain(face.holes.iter_mut()) {
                lp.iter_mut().for_each(|v| *v = get(*v));
                dedup_loop(lp);
            }
            face.holes.retain(|h| h.len() >= 3);
        }
        let mut seen: HashSet<(Id, Id)> = HashSet::new();
        let edges = std::mem::take(&mut self.edges);
        for (id, mut e) in edges {
            e.a = get(e.a);
            e.b = get(e.b);
            if e.a != e.b && seen.insert((e.a.min(e.b), e.a.max(e.b))) {
                self.edges.insert(id, e);
            }
        }
        for v in remap.keys() {
            self.vertices.remove(v);
        }
        let degenerate: Vec<Id> = self
            .faces
            .iter()
            .filter(|(_, x)| x.outer.len() < 3 || newell(&self.loop_points(&x.outer)).length() < 1e-9)
            .map(|(i, _)| *i)
            .collect();
        for g in degenerate {
            self.faces.remove(&g);
        }

        let affected: HashSet<Id> = moved.iter().map(|v| get(*v)).collect();
        if let Some(fc) = self.faces.get(&f).cloned() {
            let twin = self.faces.iter().find(|(g, x)| **g != f && Self::same_vertex_set(&x.outer, &fc.outer)).map(|(g, _)| *g);
            let host = self.faces.iter().find_map(|(g, x)| {
                (*g != f).then(|| x.holes.iter().position(|h| Self::same_vertex_set(h, &fc.outer)).map(|i| (*g, i))).flatten()
            });
            if let Some(g) = twin {
                // Pushed flat onto its opposite face: both disappear.
                self.faces.remove(&f);
                self.faces.remove(&g);
            } else if let Some((g, i)) = host {
                // Filled the hole it came out of.
                self.faces.remove(&f);
                self.faces.get_mut(&g).unwrap().holes.remove(i);
            } else {
                // Flush with coplanar neighbours: match their orientation and merge.
                let pairs: Vec<(Id, Id)> = (0..fc.outer.len()).map(|i| (fc.outer[i], fc.outer[(i + 1) % fc.outer.len()])).collect();
                let flip = pairs
                    .iter()
                    .any(|&(a, b)| self.faces_with_edge(a, b).iter().any(|g| *g != f && self.faces[g].normal.dot(fc.normal) < -0.9999));
                if flip {
                    let x = self.faces.get_mut(&f).unwrap();
                    x.outer.reverse();
                    x.holes.iter_mut().for_each(|h| h.reverse());
                    x.normal = -x.normal;
                }
                self.heal(&pairs);
            }
        }
        // Clean up edges and vertices nothing uses any more.
        let stray: Vec<Id> = self
            .edges
            .iter()
            .filter(|(_, e)| (affected.contains(&e.a) || affected.contains(&e.b)) && self.faces_with_edge(e.a, e.b).is_empty())
            .map(|(i, _)| *i)
            .collect();
        for e in stray {
            if let Some(Edge { a, b, .. }) = self.edges.remove(&e) {
                self.remove_vertex_if_orphan(a);
                self.remove_vertex_if_orphan(b);
            }
        }
        for v in affected {
            if self.vertices.contains_key(&v) {
                self.remove_collinear_vertex(v);
            }
        }
    }

    /// SketchUp Offset: copies the outer boundary of a face inward (d > 0) or outward (d < 0).
    pub fn offset_face(&mut self, f: Id, d: f32) -> bool {
        if d.abs() < EPS {
            return false;
        }
        let Some(face) = self.faces.get(&f).cloned() else { return false };
        let pts = self.loop_points(&face.outer);
        let new_pts = offset_loop(&pts, face.normal, d);
        let lp: Vec<Id> = new_pts.iter().map(|&p| self.add_vertex(p)).collect();
        self.ensure_loop_edges(&lp);
        if d > 0.0 {
            // add_face detects that the new loop lies inside `f` and turns it into a hole.
            self.add_face(lp, Some(face.normal)).is_some()
        } else {
            self.add_face_loops(lp, vec![face.outer.clone()], face.normal, face.material).is_some()
        }
    }

    /// Transforms the given vertices by `m`. Mirroring transforms flip the
    /// faces that move entirely, so they keep facing outward.
    pub fn apply_matrix(&mut self, verts: &HashSet<Id>, m: Mat4) {
        for v in verts {
            if let Some(p) = self.vertices.get_mut(v) {
                *p = m.transform_point3(*p);
            }
        }
        if m.determinant() < 0.0 {
            for face in self.faces.values_mut() {
                if face.vertices().all(|v| verts.contains(&v)) {
                    face.outer.reverse();
                    face.holes.iter_mut().for_each(|h| h.reverse());
                }
            }
        }
        self.recompute_normals();
    }

    /// True if every vertex of the face lies in one plane.
    fn is_flat(&self, face: &Face) -> bool {
        let pts = self.loop_points(&face.outer);
        let n = newell(&pts).normalize_or_zero();
        let c = pts.iter().copied().sum::<Vec3>() / pts.len().max(1) as f32;
        face.vertices().all(|v| (self.pos(v) - c).dot(n).abs() < 5e-4)
    }

    /// Auto-fold, as in SketchUp: faces that stopped being flat because some
    /// of their vertices (`moved`) were moved are split into triangles along
    /// new edges. Returns how many faces were folded.
    pub fn autofold(&mut self, moved: &HashSet<Id>) -> usize {
        let bent: Vec<Id> = self
            .faces
            .iter()
            .filter(|(_, f)| f.vertices().any(|v| moved.contains(&v)) && !f.vertices().all(|v| moved.contains(&v)))
            .filter(|(_, f)| !self.is_flat(f))
            .map(|(i, _)| *i)
            .collect();
        for f in &bent {
            let tris = self.triangulate_ids(*f);
            let Some(face) = self.faces.remove(f) else { continue };
            for t in tris {
                let n = newell(&self.loop_points(&t));
                let n = if n.dot(face.normal) < 0.0 { -n } else { n };
                self.add_face_loops(t.to_vec(), vec![], n, face.material);
            }
        }
        bent.len()
    }

    /// A copy of the whole mesh transformed by `m` (ids are preserved).
    pub fn transformed(&self, m: Mat4) -> Model {
        let mut out = self.clone();
        for p in out.vertices.values_mut() {
            *p = m.transform_point3(*p);
        }
        if m.determinant() < 0.0 {
            for face in out.faces.values_mut() {
                face.outer.reverse();
                face.holes.iter_mut().for_each(|h| h.reverse());
            }
        }
        out.recompute_normals();
        out
    }

    /// Copies the given faces and edges into a new, standalone mesh.
    pub fn extract(&self, faces: &[Id], edges: &[Id]) -> Model {
        let mut out = Model::new();
        let mut map: HashMap<Id, Id> = HashMap::new();
        let mut get = |out: &mut Model, v: Id| *map.entry(v).or_insert_with(|| out.add_vertex_raw(self.pos(v)));
        for e in edges {
            if let Some(edge) = self.edges.get(e) {
                let (a, b) = (get(&mut out, edge.a), get(&mut out, edge.b));
                out.add_edge(a, b);
            }
        }
        for f in faces {
            if let Some(face) = self.faces.get(f) {
                let outer: Vec<Id> = face.outer.iter().map(|v| get(&mut out, *v)).collect();
                let holes: Vec<Vec<Id>> = face.holes.iter().map(|h| h.iter().map(|v| get(&mut out, *v)).collect()).collect();
                out.add_face_loops(outer, holes, face.normal, face.material);
            }
        }
        out
    }

    /// Removes faces, plus the given edges that no remaining face uses.
    pub fn remove_geometry(&mut self, faces: &[Id], edges: &[Id]) {
        for f in faces {
            self.faces.remove(f);
        }
        for e in edges {
            let Some(Edge { a, b, .. }) = self.edges.get(e).cloned() else { continue };
            if self.faces_with_edge(a, b).is_empty() {
                self.edges.remove(e);
                self.remove_vertex_if_orphan(a);
                self.remove_vertex_if_orphan(b);
            }
        }
    }

    /// Merges another mesh into this one, welding coincident vertices.
    /// Returns the ids of the faces and edges that now represent it.
    pub fn merge_from(&mut self, other: &Model) -> (Vec<Id>, Vec<Id>) {
        let mut map: HashMap<Id, Id> = HashMap::new();
        for (&id, &p) in &other.vertices {
            map.insert(id, self.add_vertex(p));
        }
        let mut edges = Vec::new();
        for e in other.edges.values() {
            if let Some(ne) = self.add_edge(map[&e.a], map[&e.b]) {
                edges.push(ne);
            }
        }
        let mut faces = Vec::new();
        for face in other.faces.values() {
            let outer = face.outer.iter().map(|v| map[v]).collect();
            let holes = face.holes.iter().map(|h| h.iter().map(|v| map[v]).collect()).collect();
            if let Some(nf) = self.add_face_loops(outer, holes, face.normal, face.material) {
                faces.push(nf);
            }
        }
        (faces, edges)
    }

    /// Duplicates faces/edges with fresh vertices. Returns (new vertices, new faces, new edges).
    pub fn copy_geometry(&mut self, faces: &[Id], edges: &[Id]) -> (HashSet<Id>, Vec<Id>, Vec<Id>) {
        let mut map: HashMap<Id, Id> = HashMap::new();
        let mut used: Vec<Id> = Vec::new();
        for f in faces {
            if let Some(face) = self.faces.get(f) {
                used.extend(face.vertices());
            }
        }
        for e in edges {
            if let Some(edge) = self.edges.get(e) {
                used.push(edge.a);
                used.push(edge.b);
            }
        }
        for v in used {
            if let std::collections::hash_map::Entry::Vacant(slot) = map.entry(v) {
                let p = self.vertices[&v];
                slot.insert(self.add_vertex_raw(p));
            }
        }
        let mut new_edges = Vec::new();
        for e in edges {
            if let Some(edge) = self.edges.get(e).cloned() {
                if let Some(ne) = self.add_edge(map[&edge.a], map[&edge.b]) {
                    new_edges.push(ne);
                }
            }
        }
        let mut new_faces = Vec::new();
        for f in faces {
            if let Some(face) = self.faces.get(f).cloned() {
                let outer = face.outer.iter().map(|v| map[v]).collect();
                let holes = face.holes.iter().map(|h| h.iter().map(|v| map[v]).collect()).collect();
                if let Some(nf) = self.add_face_loops(outer, holes, face.normal, face.material) {
                    new_faces.push(nf);
                }
            }
        }
        // Edges bounding the copied faces become part of the copy too.
        for &nf in &new_faces {
            let face = self.faces[&nf].clone();
            for lp in face.loops() {
                for i in 0..lp.len() {
                    if let Some(e) = self.find_edge(lp[i], lp[(i + 1) % lp.len()]) {
                        if !new_edges.contains(&e) {
                            new_edges.push(e);
                        }
                    }
                }
            }
        }
        (map.values().copied().collect(), new_faces, new_edges)
    }

    // ----- planar arrangement & intersections -------------------------------

    /// Rebuilds the faces of every face plane that contains all `pts`.
    pub fn rebuild_planes_containing(&mut self, pts: &[Vec3]) {
        let mut planes: Vec<(Vec3, f32)> = Vec::new();
        for f in self.faces.values() {
            let (n, d) = canonical_plane(f.normal, self.pos(f.outer[0]));
            let contains = pts.iter().all(|p| (p.dot(n) - d).abs() < EPS * 10.0);
            if contains && !planes.iter().any(|(m, e)| m.dot(n) > 0.9999 && (e - d).abs() < EPS * 10.0) {
                planes.push((n, d));
            }
        }
        for (n, d) in planes {
            self.rebuild_plane(n, d);
        }
    }

    /// Splits every pair of crossing edges in the plane, and edges that pass
    /// through a vertex of the plane, then removes duplicate edges.
    fn split_plane_edges(&mut self, n: Vec3, d: f32) {
        let (u, v) = plane_basis(n);
        let to2 = |p: Vec3| Vec2::new(p.dot(u), p.dot(v));
        for _ in 0..10_000 {
            let on = |p: Vec3| (p.dot(n) - d).abs() < EPS * 10.0;
            let edges: Vec<(Id, Id, Id, Vec3, Vec3)> =
                self.edges.iter().map(|(&id, e)| (id, e.a, e.b, self.pos(e.a), self.pos(e.b))).filter(|x| on(x.3) && on(x.4)).collect();
            let verts: HashSet<Id> = edges.iter().flat_map(|x| [x.1, x.2]).collect();
            let mut split: Option<(Id, Vec3)> = None;

            // T-junctions: a vertex lying inside another edge.
            'outer: for &vid in &verts {
                let p = self.pos(vid);
                for &(eid, a, b, pa, pb) in &edges {
                    if vid == a || vid == b {
                        continue;
                    }
                    let (dist, t) = point_seg(p, pa, pb);
                    if dist < EPS && t > 1e-6 && t < 1.0 - 1e-6 {
                        split = Some((eid, p));
                        break 'outer;
                    }
                }
            }
            // Proper crossings.
            if split.is_none() {
                'cross: for i in 0..edges.len() {
                    let (e1, a1, b1, p1, q1) = edges[i];
                    let (s1, d1) = (to2(p1), to2(q1) - to2(p1));
                    for &(_, a2, b2, p2, q2) in &edges[i + 1..] {
                        if a1 == a2 || a1 == b2 || b1 == a2 || b1 == b2 {
                            continue;
                        }
                        let (s2, d2) = (to2(p2), to2(q2) - to2(p2));
                        let den = d1.perp_dot(d2);
                        if den.abs() < 1e-12 {
                            continue;
                        }
                        let w = s2 - s1;
                        let t = w.perp_dot(d2) / den;
                        let s = w.perp_dot(d1) / den;
                        let (l1, l2) = (d1.length(), d2.length());
                        let inside = |x: f32, l: f32| x * l > EPS && (1.0 - x) * l > EPS;
                        if inside(t, l1) && inside(s, l2) {
                            split = Some((e1, p1 + (q1 - p1) * t));
                            break 'cross;
                        }
                    }
                }
            }
            match split {
                Some((e, p)) => {
                    let v = self.find_vertex(p).unwrap_or_else(|| self.add_vertex_raw(p));
                    self.split_edge_at(e, v);
                }
                None => break,
            }
        }
        // Collinear overlaps leave duplicate edges between the same vertices.
        let mut seen: HashSet<(Id, Id)> = HashSet::new();
        let dupes: Vec<Id> = self.edges.iter().filter(|(_, e)| !seen.insert((e.a.min(e.b), e.a.max(e.b)))).map(|(i, _)| *i).collect();
        for e in dupes {
            self.edges.remove(&e);
        }
    }

    /// Recomputes the faces lying in plane (n, d) from the planar arrangement
    /// of all edges in that plane. Regions that were covered by a face before
    /// stay filled (keeping its material and orientation); new edges crossing
    /// them split them. This is what makes overlapping shapes, crossing lines
    /// and intersection lines divide faces the way SketchUp does.
    pub fn rebuild_plane(&mut self, n: Vec3, d: f32) {
        let on = |m: &Model, v: Id| (m.pos(v).dot(n) - d).abs() < EPS * 10.0;
        let plane_faces: Vec<Id> = self
            .faces
            .iter()
            .filter(|(_, f)| f.normal.dot(n).abs() > 0.9999 && f.outer.iter().all(|&v| on(self, v)))
            .map(|(i, _)| *i)
            .collect();
        if plane_faces.is_empty() {
            return;
        }
        let (u, v) = plane_basis(n);
        let to2 = |p: Vec3| Vec2::new(p.dot(u), p.dot(v));

        struct Orig {
            id: Id,
            loops: Vec<Vec<Vec2>>,
            normal: Vec3,
            material: Option<usize>,
        }
        let origs: Vec<Orig> = plane_faces
            .iter()
            .map(|&f| {
                let face = &self.faces[&f];
                Orig {
                    id: f,
                    loops: face.loops().map(|l| l.iter().map(|&x| to2(self.pos(x))).collect()).collect(),
                    normal: face.normal,
                    material: face.material,
                }
            })
            .collect();
        let covered_by =
            |p: Vec2| origs.iter().position(|o| point_in_poly(p, &o.loops[0]) && !o.loops[1..].iter().any(|h| point_in_poly(p, h)));

        self.split_plane_edges(n, d);

        // Planar graph of the plane's edges, neighbours sorted by angle.
        let pedges: Vec<(Id, Id)> = self.edges.values().filter(|e| on(self, e.a) && on(self, e.b)).map(|e| (e.a, e.b)).collect();
        let mut adj: HashMap<Id, Vec<Id>> = HashMap::new();
        for &(a, b) in &pedges {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        for (k, list) in adj.iter_mut() {
            let pk = to2(self.pos(*k));
            let ang = |x: &Id| {
                let dlt = to2(self.pos(*x)) - pk;
                dlt.y.atan2(dlt.x)
            };
            list.sort_by(|x, y| ang(x).total_cmp(&ang(y)));
            list.dedup();
        }

        // Connected components (for attaching islands as holes).
        let mut comp: HashMap<Id, usize> = HashMap::new();
        for &start in adj.keys() {
            if comp.contains_key(&start) {
                continue;
            }
            let c = comp.len();
            let mut stack = vec![start];
            while let Some(x) = stack.pop() {
                if comp.insert(x, c).is_none() {
                    stack.extend(adj[&x].iter().copied().filter(|y| !comp.contains_key(y)));
                }
            }
        }

        // Trace every directed half-edge; faces are on the left (CCW cycles).
        let mut visited: HashSet<(Id, Id)> = HashSet::new();
        let mut cycles: Vec<Vec<Id>> = Vec::new();
        for &(a, b) in &pedges {
            for (s, t) in [(a, b), (b, a)] {
                if visited.contains(&(s, t)) {
                    continue;
                }
                let mut cyc = Vec::new();
                let (mut x, mut y) = (s, t);
                loop {
                    visited.insert((x, y));
                    cyc.push(x);
                    let l = &adj[&y];
                    let i = l.iter().position(|w| *w == x).unwrap_or(0);
                    let w = l[(i + l.len() - 1) % l.len()];
                    x = y;
                    y = w;
                    if (x, y) == (s, t) || cyc.len() > 100_000 {
                        break;
                    }
                }
                clean_spikes(&mut cyc);
                if cyc.len() >= 3 {
                    cycles.push(cyc);
                }
            }
        }

        struct Cand {
            outer: Vec<Id>,
            holes: Vec<Vec<Id>>,
            poly: Vec<Vec2>,
            area: f32,
            owner: Option<usize>,
            comp: usize,
        }
        let mut cands: Vec<Cand> = Vec::new();
        let mut boundaries: Vec<(Vec<Id>, usize)> = Vec::new();
        for cyc in cycles {
            let poly: Vec<Vec2> = cyc.iter().map(|&x| to2(self.pos(x))).collect();
            let area = signed_area(&poly);
            let c = comp[&cyc[0]];
            if area > 1e-10 {
                cands.push(Cand { outer: cyc, holes: vec![], poly, area, owner: None, comp: c });
            } else if area < -1e-10 {
                boundaries.push((cyc, c));
            }
        }
        // Outer boundaries of islands become holes of the smallest enclosing region.
        for (cyc, c) in boundaries {
            let p = to2(self.pos(cyc[0]));
            let host = cands
                .iter()
                .enumerate()
                .filter(|(_, k)| k.comp != c && point_in_poly(p, &k.poly))
                .min_by(|a, b| a.1.area.total_cmp(&b.1.area))
                .map(|(i, _)| i);
            if let Some(h) = host {
                let ha = signed_area(&cyc.iter().map(|&x| to2(self.pos(x))).collect::<Vec<_>>()).abs();
                cands[h].area -= ha;
                cands[h].holes.push(cyc);
            }
        }
        // A region is kept if an original face covered it. The test point must
        // lie in the region itself, not in one of its holes.
        for c in cands.iter_mut() {
            let mut data: Vec<f64> = c.poly.iter().flat_map(|p| [p.x as f64, p.y as f64]).collect();
            let mut hole_idx = Vec::new();
            let mut pts = c.poly.clone();
            for h in &c.holes {
                hole_idx.push(pts.len());
                for &x in h {
                    let q = to2(self.pos(x));
                    data.extend([q.x as f64, q.y as f64]);
                    pts.push(q);
                }
            }
            let tri = earcut(&data, &hole_idx);
            let best = tri.as_chunks::<3>().0.iter().max_by(|a, b| {
                let ar = |t: &[usize; 3]| (pts[t[1]] - pts[t[0]]).perp_dot(pts[t[2]] - pts[t[0]]).abs();
                ar(a).total_cmp(&ar(b))
            });
            if let Some(t) = best {
                c.owner = covered_by((pts[t[0]] + pts[t[1]] + pts[t[2]]) / 3.0);
            }
        }
        let cands: Vec<Cand> = cands.into_iter().filter(|c| c.owner.is_some()).collect();

        // Each original face id lives on in its largest piece.
        let mut keep: HashMap<usize, Id> = HashMap::new();
        for (oi, o) in origs.iter().enumerate() {
            if let Some((ci, _)) =
                cands.iter().enumerate().filter(|(_, c)| c.owner == Some(oi)).max_by(|a, b| a.1.area.total_cmp(&b.1.area))
            {
                keep.insert(ci, o.id);
            }
        }
        for f in &plane_faces {
            self.faces.remove(f);
        }
        for (ci, c) in cands.into_iter().enumerate() {
            let Some(oi) = c.owner else { continue };
            let o = &origs[oi];
            self.insert_face(keep.get(&ci).copied(), c.outer, c.holes, o.normal, o.material);
        }
    }

    /// World-space polygons used as cutters by `intersect_faces`.
    pub fn cut_faces(&self, ids: &[Id]) -> Vec<CutFace> {
        ids.iter()
            .filter_map(|&f| {
                let face = self.faces.get(&f)?;
                let loops: Vec<Vec<Vec3>> = face.loops().map(|l| self.loop_points(l)).collect();
                let (mut lo, mut hi) = (loops[0][0], loops[0][0]);
                for p in &loops[0] {
                    lo = lo.min(*p);
                    hi = hi.max(*p);
                }
                Some(CutFace { id: Some(f), normal: face.normal, loops, lo, hi })
            })
            .collect()
    }

    /// SketchUp Intersect Faces: adds edges where `faces` cross the cutter
    /// polygons and splits the affected faces. Returns the number of new edges.
    pub fn intersect_faces(&mut self, faces: &[Id], cutters: &[CutFace]) -> usize {
        let targets = self.cut_faces(faces);
        let mut segs: Vec<(Vec3, Vec3)> = Vec::new();
        let mut planes: Vec<(Vec3, f32)> = Vec::new();
        for a in &targets {
            for b in cutters {
                if a.id.is_some() && a.id == b.id {
                    continue;
                }
                let grow = Vec3::splat(EPS * 10.0);
                if (a.lo - grow).cmpgt(b.hi).any() || (b.lo - grow).cmpgt(a.hi).any() {
                    continue;
                }
                let (n1, n2) = (a.normal, b.normal);
                let c = n1.dot(n2);
                if c.abs() > 0.9999 {
                    continue;
                }
                let dir = n1.cross(n2).normalize();
                let (d1, d2) = (n1.dot(a.loops[0][0]), n2.dot(b.loops[0][0]));
                let p0 = (n1 * (d1 - d2 * c) + n2 * (d2 - d1 * c)) / (1.0 - c * c);
                let ia = line_intervals(&a.loops, n1, p0, dir);
                let ib = line_intervals(&b.loops, n2, p0, dir);
                let mut hit = false;
                for &(s0, s1) in &ia {
                    for &(t0, t1) in &ib {
                        let (lo, hi) = (s0.max(t0), s1.min(t1));
                        if hi - lo > EPS {
                            segs.push((p0 + dir * lo, p0 + dir * hi));
                            hit = true;
                        }
                    }
                }
                if hit {
                    planes.push(canonical_plane(n1, a.loops[0][0]));
                    if b.id.is_some() {
                        planes.push(canonical_plane(n2, b.loops[0][0]));
                    }
                }
            }
        }
        let before = self.edges.len();
        for (p, q) in segs {
            let a = self.vertex_on_geometry(p);
            let b = self.vertex_on_geometry(q);
            self.add_edge(a, b);
        }
        let mut done: Vec<(Vec3, f32)> = Vec::new();
        for (n, d) in planes {
            if !done.iter().any(|(m, e)| m.dot(n) > 0.9999 && (e - d).abs() < EPS * 10.0) {
                self.rebuild_plane(n, d);
                done.push((n, d));
            }
        }
        self.edges.len().saturating_sub(before)
    }

    // ----- component openings ------------------------------------------------

    /// Outlines a cutting component cuts into the face it is glued to: the
    /// outer boundaries of its geometry on the local z = 0 plane (SketchUp's
    /// "cut opening" rule). Works for flat faces (a window drawn on a wall) and
    /// for rims left after the face was pushed in (a recessed window).
    pub fn opening_loops(&self) -> Vec<Vec<Vec3>> {
        let tol = EPS * 10.0;
        let on0 = |v: Id| self.pos(v).z.abs() < tol;
        let key = |a: Id, b: Id| (a.min(b), a.max(b));
        let mut usage: HashMap<(Id, Id), usize> = HashMap::new();
        for f in self.faces.values() {
            if f.normal.z.abs() < 0.9999 || !f.outer.iter().all(|&v| on0(v)) {
                continue;
            }
            for lp in f.loops() {
                for i in 0..lp.len() {
                    *usage.entry(key(lp[i], lp[(i + 1) % lp.len()])).or_default() += 1;
                }
            }
        }
        // Edges inside the z = 0 region (shared by two z = 0 faces) don't count.
        let mut adj: BTreeMap<Id, Vec<Id>> = BTreeMap::new();
        for e in self.edges.values() {
            if on0(e.a) && on0(e.b) && usage.get(&key(e.a, e.b)).copied().unwrap_or(0) != 2 {
                adj.entry(e.a).or_default().push(e.b);
                adj.entry(e.b).or_default().push(e.a);
            }
        }
        let mut used: HashSet<(Id, Id)> = HashSet::new();
        let mut loops: Vec<Vec<Vec3>> = Vec::new();
        for (&s, firsts) in &adj {
            for &first in firsts {
                if !used.insert(key(s, first)) {
                    continue;
                }
                let mut lp = vec![s];
                let (mut prev, mut cur) = (s, first);
                let mut closed = false;
                for _ in 0..100_000 {
                    if cur == s {
                        closed = true;
                        break;
                    }
                    lp.push(cur);
                    let next = adj[&cur].iter().copied().find(|&n| n != prev && !used.contains(&key(cur, n)));
                    let Some(n) = next else { break };
                    used.insert(key(cur, n));
                    prev = cur;
                    cur = n;
                }
                if closed && lp.len() >= 3 {
                    loops.push(lp.iter().map(|&v| self.pos(v)).collect());
                }
            }
        }
        // Keep only outermost outlines (a frame's inner edge is not an opening).
        let polys: Vec<Vec<Vec2>> = loops.iter().map(|l| l.iter().map(|p| p.truncate()).collect()).collect();
        let areas: Vec<f32> = polys.iter().map(|p| signed_area(p).abs()).collect();
        loops
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                let c = l.iter().fold(Vec3::ZERO, |a, p| a + *p).truncate() / l.len() as f32;
                !polys.iter().enumerate().any(|(j, q)| j != *i && areas[j] > areas[*i] && point_in_poly(c, q))
            })
            .map(|(_, l)| l.clone())
            .collect()
    }

    /// Cuts the polygons (projected onto the plane of normal `n` through `p0`)
    /// out of the faces lying in that plane. Returns true if a face was removed.
    pub fn subtract_loops(&mut self, n: Vec3, p0: Vec3, loops: &[Vec<Vec3>]) -> bool {
        let (n, d) = canonical_plane(n, p0);
        let (u, v) = plane_basis(n);
        let to2 = |p: Vec3| Vec2::new(p.dot(u), p.dot(v));
        let before: HashSet<Id> = self.edges.keys().copied().collect();
        let mut polys: Vec<Vec<Vec2>> = Vec::new();
        for lp in loops {
            let pts: Vec<Vec3> = lp.iter().map(|p| *p - n * (p.dot(n) - d)).collect();
            if pts.len() < 3 || newell(&pts).length() < 1e-9 {
                continue;
            }
            let ids: Vec<Id> = pts.iter().map(|&p| self.vertex_on_geometry(p)).collect();
            self.ensure_loop_edges(&ids);
            polys.push(pts.iter().map(|&p| to2(p)).collect());
        }
        if polys.is_empty() {
            return false;
        }
        self.rebuild_plane(n, d);
        let doomed: Vec<Id> = self
            .faces
            .iter()
            .filter(|(_, f)| f.normal.dot(n).abs() > 0.9999 && f.outer.iter().all(|&x| (self.pos(x).dot(n) - d).abs() < EPS * 10.0))
            .filter_map(|(&id, _)| {
                let t = *self.triangulate(id).first()?;
                let c = to2((t[0] + t[1] + t[2]) / 3.0);
                polys.iter().any(|p| point_in_poly(c, p)).then_some(id)
            })
            .collect();
        for f in &doomed {
            self.faces.remove(f);
        }
        // Parts of the outline outside every face (e.g. below a wall) are dropped.
        let stray: Vec<Id> = self
            .edges
            .iter()
            .filter(|(id, e)| !before.contains(id) && self.faces_with_edge(e.a, e.b).is_empty())
            .map(|(i, _)| *i)
            .collect();
        for e in stray {
            if let Some(Edge { a, b, .. }) = self.edges.remove(&e) {
                self.remove_vertex_if_orphan(a);
                self.remove_vertex_if_orphan(b);
            }
        }
        !doomed.is_empty()
    }

    /// Triangles of face `f` with the polygons cut out of it, leaving the mesh
    /// untouched. This is how a glued component shows its opening.
    pub fn cut_triangles(&self, f: Id, loops: &[Vec<Vec3>]) -> Vec<[Vec3; 3]> {
        let Some(face) = self.faces.get(&f) else { return vec![] };
        let mut tmp = Model::new();
        let copy = |tmp: &mut Model, lp: &[Id]| -> Vec<Id> { lp.iter().map(|&v| tmp.add_vertex_raw(self.pos(v))).collect() };
        let outer = copy(&mut tmp, &face.outer);
        let holes: Vec<Vec<Id>> = face.holes.iter().map(|h| copy(&mut tmp, h)).collect();
        tmp.insert_face(None, outer, holes, face.normal, None);
        tmp.subtract_loops(face.normal, self.pos(face.outer[0]), loops);
        tmp.faces.keys().flat_map(|&g| tmp.triangulate(g)).collect()
    }

    /// Fills the region bounded by `lp` (a loop of vertices in one face plane)
    /// and merges it into the coplanar faces around it, removing the edges in
    /// between. Used to heal a wall once its window becomes a cutting component.
    pub fn fill_and_merge(&mut self, lp: &[Id], normal: Vec3, material: Option<usize>) {
        if lp.len() < 3 || lp.iter().any(|v| !self.vertices.contains_key(v)) {
            return;
        }
        let (n, d) = canonical_plane(normal, self.pos(lp[0]));
        if self.add_face_loops(lp.to_vec(), vec![], normal, material).is_none() {
            return;
        }
        let on = |m: &Model, f: Id| {
            m.faces
                .get(&f)
                .is_some_and(|x| x.normal.dot(n).abs() > 0.9999 && x.outer.iter().all(|&v| (m.pos(v).dot(n) - d).abs() < EPS * 10.0))
        };
        let mut inner = Vec::new();
        for i in 0..lp.len() {
            let (a, b) = (lp[i], lp[(i + 1) % lp.len()]);
            let Some(e) = self.find_edge(a, b) else { continue };
            if self.faces_with_edge(a, b).into_iter().filter(|&f| on(self, f)).count() >= 2 {
                inner.push(e);
            }
        }
        for e in inner {
            self.edges.remove(&e);
        }
        self.rebuild_plane(n, d);
        for &v in lp {
            if !self.vertices.contains_key(&v) {
                continue;
            }
            if self.vertex_edges(v).is_empty() {
                self.remove_vertex_if_orphan(v);
            } else {
                self.remove_collinear_vertex(v);
            }
        }
    }

    // ----- curves, smoothing and Follow Me ---------------------------------

    /// Marks the edges between consecutive points as one curve.
    pub fn mark_curve(&mut self, pts: &[Vec3], closed: bool) {
        let ids: Vec<Option<Id>> = pts.iter().map(|p| self.find_vertex(*p)).collect();
        let curve = self.nid();
        let n = ids.len();
        let segs = if closed { n } else { n.saturating_sub(1) };
        for i in 0..segs {
            if let (Some(a), Some(b)) = (ids[i], ids[(i + 1) % n]) {
                if let Some(e) = self.find_edge(a, b) {
                    self.edges.get_mut(&e).unwrap().curve = Some(curve);
                }
            }
        }
    }

    /// All edges belonging to the same curve as `e` (or just `e`).
    pub fn curve_edges(&self, e: Id) -> Vec<Id> {
        match self.edges.get(&e).and_then(|x| x.curve) {
            Some(c) => self.edges.iter().filter(|(_, x)| x.curve == Some(c)).map(|(i, _)| *i).collect(),
            None => vec![e],
        }
    }

    /// Softens edges whose two faces meet at a shallow angle (curved surfaces).
    pub fn soften_pairs(&mut self, pairs: &[(Id, Id)]) {
        for &(a, b) in pairs {
            let Some(e) = self.find_edge(a, b) else { continue };
            let fs = self.faces_with_edge(a, b);
            if fs.len() == 2 {
                let d = self.faces[&fs[0]].normal.dot(self.faces[&fs[1]].normal);
                if d > 0.906 && d < 0.99999 {
                    self.edges.get_mut(&e).unwrap().soft = true;
                }
            }
        }
    }

    /// Draws a chain of edges as one curve (arcs). Like a line, the curve
    /// splits a face it crosses or closes new faces. Returns true if it did.
    pub fn insert_curve(&mut self, ids: &[Id]) -> bool {
        let r = self.insert_curve_inner(ids);
        let pts: Vec<Vec3> = ids.iter().map(|v| self.pos(*v)).collect();
        self.rebuild_planes_containing(&pts);
        r
    }

    fn insert_curve_inner(&mut self, ids: &[Id]) -> bool {
        let mut ids = ids.to_vec();
        ids.dedup();
        if ids.len() < 2 {
            return false;
        }
        if ids.len() == 2 {
            return self.insert_line(ids[0], ids[1]);
        }
        let curve = self.nid();
        for w in ids.windows(2) {
            if let Some(e) = self.add_edge(w[0], w[1]) {
                self.edges.get_mut(&e).unwrap().curve = Some(curve);
            }
        }
        let (a, b) = (ids[0], ids[ids.len() - 1]);
        if a == b {
            return false;
        }
        let interior = &ids[1..ids.len() - 1];
        let candidates: Vec<Id> = self.faces.keys().copied().collect();
        for f in candidates {
            let face = &self.faces[&f];
            let p0 = self.pos(face.outer[0]);
            let inside = interior.iter().all(|&v| {
                let p = self.pos(v);
                (p - p0).dot(face.normal).abs() < EPS * 10.0 && self.point_in_face(f, p)
            });
            if face.outer.contains(&a) && face.outer.contains(&b) && inside && self.split_face_chain(f, &ids) {
                return true;
            }
        }
        self.find_new_faces(ids[ids.len() - 2], b) > 0
    }

    /// Splits face `f` along a chain whose ends lie on its outer loop.
    fn split_face_chain(&mut self, f: Id, chain: &[Id]) -> bool {
        let Some(face) = self.faces.get(&f).cloned() else { return false };
        let lp = &face.outer;
        let (a, b) = (chain[0], chain[chain.len() - 1]);
        let (Some(ia), Some(ib)) = (lp.iter().position(|&x| x == a), lp.iter().position(|&x| x == b)) else {
            return false;
        };
        let walk = |from: usize, to: usize| {
            let mut out = vec![lp[from]];
            let mut i = from;
            while i != to {
                i = (i + 1) % lp.len();
                out.push(lp[i]);
            }
            out
        };
        let inner = &chain[1..chain.len() - 1];
        let mut l1 = walk(ia, ib);
        l1.extend(inner.iter().rev());
        let mut l2 = walk(ib, ia);
        l2.extend(inner.iter());
        let (mut h1, mut h2) = (vec![], vec![]);
        for h in face.holes {
            if self.point_in_loop(&l1, self.pos(h[0]), face.normal) {
                h1.push(h);
            } else {
                h2.push(h);
            }
        }
        {
            let fm = self.faces.get_mut(&f).unwrap();
            fm.outer = l1;
            fm.holes = h1;
        }
        let id = self.nid();
        self.faces.insert(id, Face { outer: l2, holes: h2, normal: face.normal, material: face.material });
        true
    }

    /// Orders a set of edges into one chain. Returns (vertices, closed).
    pub fn chain_from_edges(&self, edges: &[Id]) -> Option<(Vec<Id>, bool)> {
        let mut adj: BTreeMap<Id, Vec<Id>> = BTreeMap::new();
        for e in edges {
            let edge = self.edges.get(e)?;
            adj.entry(edge.a).or_default().push(edge.b);
            adj.entry(edge.b).or_default().push(edge.a);
        }
        if adj.is_empty() || adj.values().any(|v| v.len() > 2) {
            return None;
        }
        let ends: Vec<Id> = adj.iter().filter(|(_, v)| v.len() == 1).map(|(k, _)| *k).collect();
        let closed = ends.is_empty();
        if !closed && ends.len() != 2 {
            return None;
        }
        let start = if closed { *adj.keys().next()? } else { ends[0] };
        let mut chain = vec![start];
        let (mut prev, mut cur) = (None, start);
        while let Some(w) = adj[&cur].iter().copied().find(|w| Some(*w) != prev) {
            if w == start {
                break;
            }
            if chain.contains(&w) {
                return None;
            }
            chain.push(w);
            prev = Some(cur);
            cur = w;
        }
        (chain.len() == adj.len()).then_some((chain, closed))
    }

    /// SketchUp Follow Me: sweeps face `profile` along a path of points.
    /// Corners are mitred; a closed path makes a ring (or a lathe/revolve
    /// when the path circles an axis lying in the profile's plane).
    pub fn follow_me(&mut self, profile: Id, path: &[Vec3], closed: bool) -> bool {
        let Some(face) = self.faces.get(&profile).cloned() else { return false };
        let mut pts = path.to_vec();
        pts.dedup_by(|a, b| a.distance(*b) < EPS);
        if closed && pts.len() > 1 && pts[0].distance(pts[pts.len() - 1]) < EPS {
            pts.pop();
        }
        let m = pts.len();
        if m < 2 || (closed && m < 3) {
            return false;
        }
        // Start the sweep where the path meets the profile.
        let n_p = face.normal;
        let outer = self.loop_points(&face.outer);
        let c = outer.iter().copied().sum::<Vec3>() / outer.len() as f32;
        let score = |p: Vec3| (p - c).dot(n_p).abs() + 0.01 * p.distance(c);
        if closed {
            let k = (0..m).min_by(|&i, &j| score(pts[i]).total_cmp(&score(pts[j]))).unwrap();
            pts.rotate_left(k);
        } else if score(pts[m - 1]) < score(pts[0]) {
            pts.reverse();
        }

        let dir = |i: usize| (pts[(i + 1) % m] - pts[i % m]).normalize_or_zero();
        let miter = |k: usize| -> Vec3 {
            if closed {
                (dir((k + m - 1) % m) + dir(k)).normalize_or(dir(k))
            } else if k == 0 {
                dir(0)
            } else if k == m - 1 {
                dir(m - 2)
            } else {
                (dir(k - 1) + dir(k)).normalize_or(dir(k))
            }
        };
        let project = |p: Vec3, d: Vec3, at: Vec3, n: Vec3| {
            let denom = d.dot(n);
            if denom.abs() < 1e-6 {
                p
            } else {
                p + d * ((at - p).dot(n) / denom)
            }
        };

        let profile_loops: Vec<Vec<Vec3>> = face.loops().map(|l| self.loop_points(l)).collect();
        let mut rings: Vec<Vec<Vec<Vec3>>> = Vec::with_capacity(m);
        rings.push(if closed {
            profile_loops.iter().map(|l| l.iter().map(|&p| project(p, dir(0), pts[0], miter(0))).collect()).collect()
        } else {
            profile_loops.clone()
        });
        for k in 1..m {
            let (d, at, n) = (dir(k - 1), pts[k], miter(k));
            let next = rings[k - 1].iter().map(|l| l.iter().map(|&p| project(p, d, at, n)).collect()).collect();
            rings.push(next);
        }
        let mut ids: Vec<Vec<Vec<Id>>> = Vec::with_capacity(m);
        for (k, r) in rings.iter().enumerate() {
            if k == 0 && !closed {
                ids.push(face.loops().cloned().collect());
            } else {
                ids.push(r.iter().map(|l| l.iter().map(|&p| self.add_vertex(p)).collect()).collect());
            }
        }

        let s = if n_p.dot(dir(0)) >= 0.0 { 1.0 } else { -1.0 };
        let segs = if closed { m } else { m - 1 };
        let mut pairs = Vec::new();
        for seg in 0..segs {
            let (k0, k1) = (seg, (seg + 1) % m);
            let d = dir(seg);
            for (lp0, lp1) in ids[k0].iter().zip(&ids[k1]) {
                let len = lp0.len();
                for j in 0..len {
                    let j1 = (j + 1) % len;
                    let (a, b) = (lp0[j], lp0[j1]);
                    let (c2, d2) = (lp1[j1], lp1[j]);
                    let out = (self.pos(b) - self.pos(a)).cross(d * s);
                    self.add_face_loops(vec![a, b, c2, d2], vec![], out, face.material);
                    pairs.extend([(a, b), (a, d2), (b, c2), (d2, c2)]);
                }
            }
        }

        self.faces.remove(&profile);
        if !closed {
            let last = ids[m - 1].clone();
            self.add_face_loops(last[0].clone(), last[1..].to_vec(), dir(m - 2), face.material);
            self.add_face_loops(face.outer.clone(), face.holes.clone(), -dir(0), face.material);
        }
        self.soften_pairs(&pairs);
        true
    }

    // ----- queries --------------------------------------------------------

    pub fn vertices_of(&self, faces: &[Id], edges: &[Id]) -> HashSet<Id> {
        let mut set = HashSet::new();
        for f in faces {
            if let Some(face) = self.faces.get(f) {
                set.extend(face.vertices());
            }
        }
        for e in edges {
            if let Some(edge) = self.edges.get(e) {
                set.insert(edge.a);
                set.insert(edge.b);
            }
        }
        set
    }

    pub fn face_edges(&self, f: Id) -> Vec<Id> {
        let Some(face) = self.faces.get(&f) else { return vec![] };
        let mut out = Vec::new();
        for lp in face.loops() {
            for i in 0..lp.len() {
                if let Some(e) = self.find_edge(lp[i], lp[(i + 1) % lp.len()]) {
                    out.push(e);
                }
            }
        }
        out
    }

    /// All geometry connected to the given vertices.
    pub fn connected(&self, seeds: &[Id]) -> (Vec<Id>, Vec<Id>) {
        let mut visited: HashSet<Id> = seeds.iter().copied().collect();
        let mut queue: VecDeque<Id> = seeds.iter().copied().collect();
        let mut adj: HashMap<Id, Vec<Id>> = HashMap::new();
        for e in self.edges.values() {
            adj.entry(e.a).or_default().push(e.b);
            adj.entry(e.b).or_default().push(e.a);
        }
        while let Some(v) = queue.pop_front() {
            for &w in adj.get(&v).map(|x| x.as_slice()).unwrap_or(&[]) {
                if visited.insert(w) {
                    queue.push_back(w);
                }
            }
        }
        let faces = self.faces.iter().filter(|(_, f)| f.vertices().all(|v| visited.contains(&v))).map(|(i, _)| *i).collect();
        let edges = self.edges.iter().filter(|(_, e)| visited.contains(&e.a)).map(|(i, _)| *i).collect();
        (faces, edges)
    }

    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.vertices.values();
        let first = *it.next()?;
        let (mut lo, mut hi) = (first, first);
        for p in it {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        Some((lo, hi))
    }

    pub fn bounds_of(&self, verts: &HashSet<Id>) -> Option<(Vec3, Vec3)> {
        let mut it = verts.iter().map(|v| self.pos(*v));
        let first = it.next()?;
        let (mut lo, mut hi) = (first, first);
        for p in it {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        Some((lo, hi))
    }
}

/// A world-space polygon (outer loop + holes) used by Intersect Faces.
#[derive(Clone, Debug)]
pub struct CutFace {
    /// Face id when the polygon belongs to the edited mesh.
    pub id: Option<Id>,
    pub normal: Vec3,
    pub loops: Vec<Vec<Vec3>>,
    pub lo: Vec3,
    pub hi: Vec3,
}

/// Plane with a sign-normalized normal so equal planes compare equal.
pub fn canonical_plane(n: Vec3, p: Vec3) -> (Vec3, f32) {
    let mut n = n.normalize_or(Vec3::Z);
    let flip = if n.x.abs() > 1e-6 {
        n.x < 0.0
    } else if n.y.abs() > 1e-6 {
        n.y < 0.0
    } else {
        n.z < 0.0
    };
    if flip {
        n = -n;
    }
    (n, n.dot(p))
}

fn signed_area(poly: &[Vec2]) -> f32 {
    let n = poly.len();
    (0..n).map(|i| poly[i].perp_dot(poly[(i + 1) % n])).sum::<f32>() * 0.5
}

/// Removes back-and-forth spikes (x, y, x) left by dangling edges.
fn clean_spikes(c: &mut Vec<Id>) {
    loop {
        let n = c.len();
        if n < 3 {
            return;
        }
        let Some(i) = (0..n).find(|&i| c[i] == c[(i + 2) % n]) else { return };
        let (i1, i2) = ((i + 1) % n, (i + 2) % n);
        let (hi, lo) = (i1.max(i2), i1.min(i2));
        c.remove(hi);
        c.remove(lo);
    }
}

/// Parameter intervals where the line p0 + t·dir lies inside a planar polygon
/// with holes (even-odd rule).
fn line_intervals(loops: &[Vec<Vec3>], n: Vec3, p0: Vec3, dir: Vec3) -> Vec<(f32, f32)> {
    let (u, v) = plane_basis(n);
    let to2 = |p: Vec3| Vec2::new(p.dot(u), p.dot(v));
    let (q0, d) = (to2(p0), Vec2::new(dir.dot(u), dir.dot(v)));
    let side = Vec2::new(-d.y, d.x);
    let mut ts = Vec::new();
    for lp in loops {
        for i in 0..lp.len() {
            let (a, b) = (to2(lp[i]), to2(lp[(i + 1) % lp.len()]));
            // Vertices within tolerance of the line count as exactly on it, so
            // edges lying along the line are handled consistently (half-open rule).
            let snap = |x: f32| if x.abs() < EPS * side.length() { 0.0 } else { x };
            let (sa, sb) = (snap((a - q0).dot(side)), snap((b - q0).dot(side)));
            if (sa > 0.0) != (sb > 0.0) {
                let x = a + (b - a) * (sa / (sa - sb));
                ts.push((x - q0).dot(d) / d.length_squared());
            }
        }
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    ts.as_chunks::<2>().0.iter().map(|c| (c[0], c[1])).collect()
}

/// Thin wrapper around earcutr returning triangle indices (empty on failure).
fn earcut(data: &[f64], holes: &[usize]) -> Vec<usize> {
    earcutr::earcut(data, holes, 2).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(m: &mut Model, x0: f32, y0: f32, x1: f32, y1: f32, z: f32) -> Id {
        m.add_polygon(&[Vec3::new(x0, y0, z), Vec3::new(x1, y0, z), Vec3::new(x1, y1, z), Vec3::new(x0, y1, z)], None).unwrap()
    }

    #[test]
    fn ground_rect_faces_down() {
        let mut m = Model::new();
        let f = rect(&mut m, 0.0, 0.0, 2.0, 1.0, 0.0);
        assert!(m.faces[&f].normal.z < -0.99);
        assert_eq!(m.edges.len(), 4);
        assert!((m.face_area(f) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn moving_a_corner_folds_the_faces_around_it() {
        let mut m = Model::new();
        let f = rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        m.push_pull(f, -2.0);
        let corner = m.find_vertex(Vec3::new(2.0, 2.0, 2.0)).unwrap();
        let moved: HashSet<Id> = [corner].into_iter().collect();
        // Sliding the corner along an edge keeps every face flat: nothing folds.
        let mut slide = m.clone();
        slide.apply_matrix(&moved, Mat4::from_translation(Vec3::new(0.0, 0.0, -0.5)));
        assert_eq!(slide.autofold(&moved), 1, "only the top stops being flat");
        // Lifting it bends the three faces that meet there.
        m.apply_matrix(&moved, Mat4::from_translation(Vec3::new(0.5, 0.5, 1.0)));
        assert_eq!(m.autofold(&moved), 3);
        assert_eq!(m.faces.len(), 9);
        assert_eq!(m.edges.len(), 15);
        assert!(m.faces.values().all(|f| f.outer.len() == 3 || !f.vertices().any(|v| v == corner)));
        assert!(m.faces.values().all(|f| m.is_flat(f)));
        // Still a closed shell: every edge bounds exactly two faces.
        assert!(m.edges.keys().all(|e| m.edge_faces(*e).len() == 2));
    }

    #[test]
    fn push_pull_makes_closed_box() {
        let mut m = Model::new();
        let f = rect(&mut m, 0.0, 0.0, 2.0, 1.0, 0.0);
        assert!(m.push_pull(f, -3.0)); // ground faces point down: negative = up
        assert_eq!(m.faces.len(), 6);
        assert_eq!(m.edges.len(), 12);
        assert_eq!(m.vertices.len(), 8);
        let top = m.faces.values().find(|f| f.normal.z > 0.99).unwrap();
        assert!((m.pos(top.outer[0]).z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn push_pull_top_extends_box_and_heals() {
        let mut m = Model::new();
        let f = rect(&mut m, 0.0, 0.0, 1.0, 1.0, 0.0);
        m.push_pull(f, -1.0);
        let top = *m.faces.iter().find(|(_, f)| f.normal.z > 0.99).unwrap().0;
        m.push_pull(top, 1.0);
        assert_eq!(m.faces.len(), 6, "coplanar sides should merge");
        assert_eq!(m.edges.len(), 12);
    }

    #[test]
    fn line_splits_face() {
        let mut m = Model::new();
        rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        let a = m.vertex_on_geometry(Vec3::new(1.0, 0.0, 0.0));
        let b = m.vertex_on_geometry(Vec3::new(1.0, 2.0, 0.0));
        assert!(m.insert_line(a, b));
        assert_eq!(m.faces.len(), 2);
    }

    #[test]
    fn lines_close_into_face() {
        let mut m = Model::new();
        let p = [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        let ids: Vec<Id> = p.iter().map(|&x| m.vertex_on_geometry(x)).collect();
        assert!(!m.insert_line(ids[0], ids[1]));
        assert!(!m.insert_line(ids[1], ids[2]));
        assert!(!m.insert_line(ids[2], ids[3]));
        assert!(m.insert_line(ids[3], ids[0]));
        assert_eq!(m.faces.len(), 1);
    }

    #[test]
    fn inner_rect_becomes_hole_and_pocket() {
        let mut m = Model::new();
        let f = rect(&mut m, 0.0, 0.0, 4.0, 4.0, 0.0);
        m.push_pull(f, -1.0);
        let top = *m.faces.iter().find(|(_, f)| f.normal.z > 0.99).unwrap().0;
        let inner = rect(&mut m, 1.0, 1.0, 2.0, 2.0, 1.0);
        assert_eq!(m.faces[&top].holes.len(), 1);
        let n_before = m.faces.len();
        assert!(m.push_pull(inner, -0.5));
        assert_eq!(m.faces.len(), n_before + 4);
    }

    #[test]
    fn erase_edge_merges_coplanar() {
        let mut m = Model::new();
        rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        let a = m.vertex_on_geometry(Vec3::new(1.0, 0.0, 0.0));
        let b = m.vertex_on_geometry(Vec3::new(1.0, 2.0, 0.0));
        m.insert_line(a, b);
        let e = m.find_edge(a, b).unwrap();
        m.delete_edge(e);
        assert_eq!(m.faces.len(), 1);
        assert_eq!(m.edges.len(), 4);
    }

    #[test]
    fn offset_inward_creates_hole() {
        let mut m = Model::new();
        let f = rect(&mut m, 0.0, 0.0, 4.0, 4.0, 0.0);
        assert!(m.offset_face(f, 0.5));
        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.faces[&f].holes.len(), 1);
        assert!((m.face_area(f) - (16.0 - 9.0)).abs() < 1e-3);
    }

    fn arc_pts(c: Vec3, r: f32, a0: f32, a1: f32, n: usize) -> Vec<Vec3> {
        (0..=n)
            .map(|i| {
                let t = a0 + (a1 - a0) * i as f32 / n as f32;
                c + Vec3::new(t.cos() * r, t.sin() * r, 0.0)
            })
            .collect()
    }

    /// Signed volume (positive when every face points outward).
    fn volume(m: &Model) -> f32 {
        m.faces.keys().flat_map(|f| m.triangulate(*f)).map(|t| t[0].dot(t[1].cross(t[2])) / 6.0).sum()
    }

    fn total_area(m: &Model) -> f32 {
        m.faces.keys().map(|f| m.face_area(*f)).sum()
    }

    #[test]
    fn arc_closes_half_disc_as_one_curve() {
        let mut m = Model::new();
        let a = m.vertex_on_geometry(Vec3::new(1.0, 0.0, 0.0));
        let b = m.vertex_on_geometry(Vec3::new(-1.0, 0.0, 0.0));
        m.insert_line(a, b);
        let ids: Vec<Id> = arc_pts(Vec3::ZERO, 1.0, 0.0, std::f32::consts::PI, 12).iter().map(|p| m.vertex_on_geometry(*p)).collect();
        assert!(m.insert_curve(&ids));
        assert_eq!(m.faces.len(), 1);
        // 12-segment half polygon: 6 * sin(15°)
        assert!((total_area(&m) - 6.0 * 15f32.to_radians().sin()).abs() < 1e-3);
        let e = m.find_edge(ids[0], ids[1]).unwrap();
        assert_eq!(m.curve_edges(e).len(), 12);
    }

    #[test]
    fn arc_across_face_splits_it() {
        let mut m = Model::new();
        rect(&mut m, -2.0, -2.0, 2.0, 2.0, 0.0);
        let c = Vec3::new(0.0, -1.5, 0.0);
        let (a0, a1) = (1.5f32.atan2(2.0), 1.5f32.atan2(-2.0));
        let ids: Vec<Id> = arc_pts(c, 2.5, a0, a1, 8).iter().map(|p| m.vertex_on_geometry(*p)).collect();
        assert!(m.insert_curve(&ids));
        assert_eq!(m.faces.len(), 2);
        assert!((total_area(&m) - 16.0).abs() < 1e-3);
    }

    #[test]
    fn follow_me_straight_path_makes_box() {
        let mut m = Model::new();
        let f = m.add_polygon(&[Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 1.0)], None).unwrap();
        assert!(m.follow_me(f, &[Vec3::new(0.0, 0.5, 0.5), Vec3::new(3.0, 0.5, 0.5)], false));
        assert_eq!(m.faces.len(), 6);
        assert!((volume(&m) - 3.0).abs() < 1e-3, "volume {}", volume(&m));
    }

    #[test]
    fn follow_me_corner_is_mitred() {
        let mut m = Model::new();
        let f = m.add_polygon(&[Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 1.0)], None).unwrap();
        let path = [Vec3::new(0.0, 0.5, 0.5), Vec3::new(3.0, 0.5, 0.5), Vec3::new(3.0, 3.5, 0.5)];
        assert!(m.follow_me(f, &path, false));
        assert_eq!(m.faces.len(), 10);
        // Cross-section 1 x centerline 6 (exact for a mitred prism).
        assert!((volume(&m) - 6.0).abs() < 1e-3, "volume {}", volume(&m));
    }

    #[test]
    fn follow_me_closed_path_revolves() {
        let mut m = Model::new();
        let f = m
            .add_polygon(&[Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0)], None)
            .unwrap();
        let path: Vec<Vec3> = (0..24)
            .map(|i| {
                let t = i as f32 / 24.0 * std::f32::consts::TAU;
                Vec3::new(t.cos() * 1.5, t.sin() * 1.5, 0.0)
            })
            .collect();
        assert!(m.follow_me(f, &path, true));
        assert_eq!(m.faces.len(), 96, "profile removed, 24 x 4 sides");
        // Pappus: area 1 x 2π·1.5, within polygon approximation.
        let v = volume(&m);
        assert!((v - std::f32::consts::TAU * 1.5).abs() < 0.05 * v, "volume {v}");
        assert!(m.edges.values().filter(|e| e.soft).count() >= 48);
    }

    #[test]
    fn push_pull_cylinder_softens_sides() {
        let mut m = Model::new();
        let pts: Vec<Vec3> = (0..24)
            .map(|i| {
                let t = i as f32 / 24.0 * std::f32::consts::TAU;
                Vec3::new(t.cos(), t.sin(), 0.0)
            })
            .collect();
        let f = m.add_polygon(&pts, None).unwrap();
        m.push_pull(f, -2.0);
        assert_eq!(m.edges.values().filter(|e| e.soft).count(), 24);
        assert!(!m.shading_normals().is_empty());
    }

    #[test]
    fn chain_ordering() {
        let mut m = Model::new();
        let v: Vec<Id> = [Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y].iter().map(|p| m.add_vertex(*p)).collect();
        let e1 = m.add_edge(v[2], v[3]).unwrap();
        let e0 = m.add_edge(v[0], v[1]).unwrap();
        let e2 = m.add_edge(v[1], v[2]).unwrap();
        let (chain, closed) = m.chain_from_edges(&[e1, e0, e2]).unwrap();
        assert!(!closed);
        assert_eq!(chain.len(), 4);
        let e3 = m.add_edge(v[3], v[0]).unwrap();
        let (chain, closed) = m.chain_from_edges(&[e0, e1, e2, e3]).unwrap();
        assert!(closed && chain.len() == 4);
    }

    #[test]
    fn overlapping_rectangles_split_into_regions() {
        let mut m = Model::new();
        rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        rect(&mut m, 1.0, 1.0, 3.0, 3.0, 0.0);
        assert_eq!(m.faces.len(), 3, "A-only, overlap, B-only");
        assert!((total_area(&m) - 7.0).abs() < 1e-3);
        assert!(m.faces.values().all(|f| f.normal.z < -0.99), "ground faces keep facing down");
    }

    #[test]
    fn crossing_lines_quarter_a_face() {
        let mut m = Model::new();
        rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        let a = m.vertex_on_geometry(Vec3::new(1.0, 0.0, 0.0));
        let b = m.vertex_on_geometry(Vec3::new(1.0, 2.0, 0.0));
        m.insert_line(a, b);
        let c = m.vertex_on_geometry(Vec3::new(0.0, 1.0, 0.0));
        let d = m.vertex_on_geometry(Vec3::new(2.0, 1.0, 0.0));
        m.insert_line(c, d);
        assert_eq!(m.faces.len(), 4);
        assert!(m.faces.keys().all(|f| (m.face_area(*f) - 1.0).abs() < 1e-3));
        assert!(m.find_vertex(Vec3::new(1.0, 1.0, 0.0)).is_some(), "lines split at their crossing");
    }

    #[test]
    fn line_through_face_from_outside_splits_it() {
        let mut m = Model::new();
        rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        let a = m.vertex_on_geometry(Vec3::new(1.0, -1.0, 0.0));
        let b = m.vertex_on_geometry(Vec3::new(1.0, 3.0, 0.0));
        m.insert_line(a, b);
        assert_eq!(m.faces.len(), 2);
        assert!((total_area(&m) - 4.0).abs() < 1e-3);
    }

    #[test]
    fn rectangle_touching_boundary_divides_host() {
        let mut m = Model::new();
        let host = rect(&mut m, 0.0, 0.0, 4.0, 4.0, 0.0);
        let mat = Some(3);
        m.faces.get_mut(&host).unwrap().material = mat;
        rect(&mut m, 0.0, 1.0, 1.0, 2.0, 0.0); // shares part of the host's left edge
        assert_eq!(m.faces.len(), 2);
        assert!((total_area(&m) - 16.0).abs() < 1e-3, "no overlapping faces");
        assert!(m.faces.contains_key(&host), "host keeps its id");
        assert!(m.faces[&host].holes.is_empty());
        assert!(m.faces.values().all(|f| f.material == mat), "pieces inherit the host material");
    }

    #[test]
    fn intersect_faces_cuts_crossing_boxes() {
        let mut m = Model::new();
        let a = rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        m.push_pull(a, -2.0);
        // A second, separate box poking through the first one's +X and +Y sides.
        let b = m
            .add_polygon(
                &[Vec3::new(1.0, 1.0, 0.5), Vec3::new(3.0, 1.0, 0.5), Vec3::new(3.0, 3.0, 0.5), Vec3::new(1.0, 3.0, 0.5)],
                Some(Vec3::Z),
            )
            .unwrap();
        m.push_pull(b, 1.0);
        let faces_before = m.faces.len();
        let all: Vec<Id> = m.faces.keys().copied().collect();
        let cutters = m.cut_faces(&all);
        let added = m.intersect_faces(&all, &cutters);
        assert!(added >= 8, "added {added} edges");
        assert!(m.faces.len() > faces_before);
        // On the first box's +X side, the second box leaves a 1 x 1 imprint.
        let imprint = m
            .faces
            .iter()
            .any(|(f, face)| face.normal.x > 0.99 && (m.pos(face.outer[0]).x - 2.0).abs() < 1e-4 && (m.face_area(*f) - 1.0).abs() < 1e-3);
        assert!(imprint);
    }

    #[test]
    fn intersect_with_external_cutter() {
        // A wall face cut by a cylinder that is not part of the mesh (e.g. a group).
        let mut m = Model::new();
        let wall = m
            .add_polygon(&[Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 3.0)], Some(-Vec3::Y))
            .unwrap();
        let mut cutter = Model::new();
        let pts: Vec<Vec3> = (0..16)
            .map(|i| {
                let t = i as f32 / 16.0 * std::f32::consts::TAU;
                Vec3::new(2.0 + 0.5 * t.cos(), -1.0, 1.5 + 0.5 * t.sin())
            })
            .collect();
        let c = cutter.add_polygon(&pts, Some(Vec3::Y)).unwrap();
        cutter.push_pull(c, 2.0);
        let all: Vec<Id> = cutter.faces.keys().copied().collect();
        let mut cuts = cutter.cut_faces(&all);
        cuts.iter_mut().for_each(|c| c.id = None);
        m.intersect_faces(&[wall], &cuts);
        assert_eq!(m.faces.len(), 2, "wall gets a circular hole + inner disc");
        assert!(m.faces.values().any(|f| f.holes.len() == 1));
    }

    fn wall_xz(m: &mut Model) -> Id {
        m.add_polygon(&[Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 3.0)], Some(-Vec3::Y)).unwrap()
    }

    fn xz_rect(x0: f32, x1: f32, z0: f32, z1: f32) -> Vec<Vec3> {
        vec![Vec3::new(x0, 0.0, z0), Vec3::new(x1, 0.0, z0), Vec3::new(x1, 0.0, z1), Vec3::new(x0, 0.0, z1)]
    }

    #[test]
    fn opening_loop_is_frame_outline() {
        let mut m = Model::new();
        rect(&mut m, 0.0, 0.0, 3.0, 3.0, 0.0);
        rect(&mut m, 1.0, 1.0, 2.0, 2.0, 0.0); // glass hole + pane
        let loops = m.opening_loops();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 4);
        let a = newell(&loops[0]).length() * 0.5;
        assert!((a - 9.0).abs() < 1e-3);
    }

    #[test]
    fn subtract_window_and_door() {
        let mut m = Model::new();
        let w = wall_xz(&mut m);
        assert!(m.subtract_loops(-Vec3::Y, Vec3::ZERO, &[xz_rect(1.0, 2.0, 1.0, 2.0)]));
        assert_eq!(m.faces.len(), 1);
        assert_eq!(m.faces[&w].holes.len(), 1, "wall keeps its id and gets a hole");
        assert!((m.face_area(w) - 11.0).abs() < 1e-3);
        // A door outline reaching below the wall: the part outside is dropped.
        m.subtract_loops(-Vec3::Y, Vec3::ZERO, &[xz_rect(2.5, 3.5, -0.5, 2.0)]);
        assert!((total_area(&m) - 9.0).abs() < 1e-3);
        assert!(m.vertices.values().all(|p| p.z > -1e-4), "no stray edges below the wall");
    }

    #[test]
    fn cut_triangles_leave_mesh_untouched() {
        let mut m = Model::new();
        let w = wall_xz(&mut m);
        let tris = m.cut_triangles(w, &[xz_rect(1.0, 2.0, 1.0, 2.0), xz_rect(2.5, 3.5, 0.0, 2.0)]);
        let area: f32 = tris.iter().map(|t| (t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5).sum();
        assert!((area - 9.0).abs() < 1e-3);
        assert!(tris.iter().all(|t| (t[1] - t[0]).cross(t[2] - t[0]).dot(-Vec3::Y) > 0.0), "wound like the face");
        assert_eq!(m.faces.len(), 1);
        assert!(m.faces[&w].holes.is_empty());
    }

    #[test]
    fn drawing_on_a_face_with_holes_keeps_its_identity() {
        let mut m = Model::new();
        let ground = rect(&mut m, -7.0, -6.0, 7.0, 7.0, 0.0);
        m.faces.get_mut(&ground).unwrap().material = Some(6);
        for (x0, y0, x1, y1) in [(0.0, 0.0, 4.0, 3.0), (-4.5, -1.0, -2.5, 1.0)] {
            let f = rect(&mut m, x0, y0, x1, y1, 0.0);
            m.push_pull(f, -1.0);
        }
        let patch = rect(&mut m, -4.5, 2.5, -1.0, 5.0, 0.0);
        assert!((m.face_area(patch) - 3.5 * 2.5).abs() < 1e-3, "returned face is the new patch");
        assert!(m.faces[&patch].material.is_none());
        assert_eq!(m.faces[&ground].material, Some(6), "the ground keeps its id and paint");
        assert!((m.face_area(ground) - (14.0 * 13.0 - 12.0 - 4.0 - 3.5 * 2.5)).abs() < 1e-3);
    }

    // ----- push/pull workflows ----------------------------------------------------

    fn ground_box(m: &mut Model, x0: f32, y0: f32, x1: f32, y1: f32, h: f32) {
        let f =
            m.add_polygon(&[Vec3::new(x0, y0, 0.0), Vec3::new(x1, y0, 0.0), Vec3::new(x1, y1, 0.0), Vec3::new(x0, y1, 0.0)], None).unwrap();
        m.push_pull(f, -h); // ground faces point down: negative pulls up
    }

    fn face_at(m: &Model, n: Vec3, pred: impl Fn(Vec3) -> bool) -> Id {
        *m.faces.iter().find(|(_, f)| f.normal.dot(n) > 0.99 && pred(m.pos(f.outer[0]))).unwrap().0
    }

    fn solid_with(m: &Model, vol: f32, faces: usize) {
        assert!(crate::solid::is_solid(m), "not a closed solid ({} faces)", m.faces.len());
        assert!((crate::solid::volume(m) - vol).abs() < 1e-3, "volume {}", crate::solid::volume(m));
        assert_eq!(m.faces.len(), faces);
    }

    fn top_rect(m: &mut Model, x0: f32, y0: f32, x1: f32, y1: f32, z: f32) -> Id {
        m.add_polygon(&[Vec3::new(x0, y0, z), Vec3::new(x1, y0, z), Vec3::new(x1, y1, z), Vec3::new(x0, y1, z)], Some(Vec3::Z)).unwrap()
    }

    #[test]
    fn pushing_a_box_top_down_shortens_it() {
        let mut m = Model::new();
        ground_box(&mut m, 0.0, 0.0, 2.0, 2.0, 2.0);
        let top = face_at(&m, Vec3::Z, |p| p.z > 1.9);
        m.push_pull(top, -0.5);
        solid_with(&m, 6.0, 6);
        assert!(m.faces.contains_key(&top), "the top just moved");
        m.push_pull(top, -1.5);
        assert!(m.faces.is_empty() && m.edges.is_empty() && m.vertices.is_empty(), "pushed flat: nothing left");
    }

    #[test]
    fn step_pushed_back_flush_merges() {
        let mut m = Model::new();
        ground_box(&mut m, 0.0, 0.0, 4.0, 2.0, 2.0);
        let r = top_rect(&mut m, 0.0, 0.0, 2.0, 2.0, 2.0);
        m.push_pull(r, 1.0);
        solid_with(&m, 20.0, 8);
        let raised = face_at(&m, Vec3::Z, |p| p.z > 2.9);
        m.push_pull(raised, -1.0);
        solid_with(&m, 16.0, 6);
    }

    #[test]
    fn pocket_floor_pulled_flush_closes_it() {
        let mut m = Model::new();
        ground_box(&mut m, 0.0, 0.0, 4.0, 4.0, 2.0);
        let r = top_rect(&mut m, 1.0, 1.0, 3.0, 3.0, 2.0);
        m.push_pull(r, -1.0);
        solid_with(&m, 28.0, 11);
        let floor = face_at(&m, Vec3::Z, |p| p.z > 0.9 && p.z < 1.1);
        m.push_pull(floor, 1.0);
        solid_with(&m, 32.0, 6);
    }

    #[test]
    fn pushing_through_cuts_a_hole() {
        let mut m = Model::new();
        ground_box(&mut m, 0.0, 0.0, 4.0, 4.0, 2.0);
        let r = top_rect(&mut m, 1.0, 1.0, 3.0, 3.0, 2.0);
        m.push_pull(r, -2.0);
        solid_with(&m, 24.0, 10);
        assert_eq!(m.faces.values().filter(|f| f.holes.len() == 1).count(), 2, "top and bottom are rings");

        // A window through a 0.3 m thick wall.
        let mut w = Model::new();
        ground_box(&mut w, 0.0, 0.0, 4.0, 0.3, 3.0);
        let win = w
            .add_polygon(
                &[Vec3::new(1.0, 0.0, 1.0), Vec3::new(2.0, 0.0, 1.0), Vec3::new(2.0, 0.0, 2.0), Vec3::new(1.0, 0.0, 2.0)],
                Some(-Vec3::Y),
            )
            .unwrap();
        w.push_pull(win, -0.3);
        solid_with(&w, 3.6 - 0.3, 10);
    }

    #[test]
    fn box_pulled_from_a_ground_face_faces_outward() {
        let mut m = Model::new();
        rect(&mut m, -5.0, -5.0, 5.0, 5.0, 0.0);
        let f = rect(&mut m, 0.0, 0.0, 2.0, 2.0, 0.0);
        m.push_pull(f, -2.0);
        let top = m.faces.values().filter(|x| x.normal.z > 0.99 && x.outer.iter().all(|&v| m.pos(v).z > 1.9)).count();
        assert_eq!(top, 1, "the top faces up");
        for x in m.faces.values().filter(|x| x.normal.z.abs() < 0.1) {
            let c = m.loop_points(&x.outer).iter().fold(Vec3::ZERO, |a, p| a + *p) / 4.0;
            assert!((c - Vec3::new(1.0, 1.0, c.z)).dot(x.normal) > 0.0, "walls face outward");
        }
    }
}
