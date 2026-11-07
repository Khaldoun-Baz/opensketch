//! Solid tools: SketchUp's Outer Shell, Intersect, Union, Subtract, Trim and Split.
//!
//! A mesh is a solid when every edge bounds exactly two faces. Booleans work on
//! world-space copies of two solids: each is cut along the other (Intersect
//! Faces, plus imprinting where faces are coplanar), every face is classified
//! as inside, outside or on the other solid, and the faces to keep are welded
//! into a new mesh whose coplanar seams are cleaned up.

use crate::model::{canonical_plane, plane_basis, point_in_poly, point_seg, Id, Model, EPS};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoolOp {
    Union,
    Intersect,
    /// First operand minus the second.
    Subtract,
}

/// Every edge bounds exactly two faces (a closed, manifold shell).
pub fn is_solid(m: &Model) -> bool {
    if m.faces.len() < 4 {
        return false;
    }
    let mut uses: HashMap<(Id, Id), usize> = HashMap::new();
    for f in m.faces.values() {
        for lp in f.loops() {
            for i in 0..lp.len() {
                let (a, b) = (lp[i], lp[(i + 1) % lp.len()]);
                *uses.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }
    uses.len() == m.edges.len() && m.edges.values().all(|e| uses.get(&(e.a.min(e.b), e.a.max(e.b))) == Some(&2))
}

/// Enclosed volume (positive when faces point outward).
pub fn volume(m: &Model) -> f32 {
    m.faces.keys().flat_map(|&f| m.triangulate(f)).map(|t| t[0].dot(t[1].cross(t[2])) / 6.0).sum()
}

/// The boolean of two solids, as a new mesh (empty if nothing is left).
pub fn boolean(a: &Model, b: &Model, op: BoolOp) -> Model {
    let mut a = oriented(a);
    let mut b = oriented(b);
    let (ca, cb) = (cutters(&a), cutters(&b));
    let fa: Vec<Id> = a.faces.keys().copied().collect();
    a.intersect_faces(&fa, &cb);
    let fb: Vec<Id> = b.faces.keys().copied().collect();
    b.intersect_faces(&fb, &ca);
    imprint(&mut a, &b);
    imprint(&mut b, &a);

    let (ta, tb) = (triangles(&a), triangles(&b));
    let mut keep: Vec<Part> = Vec::new();
    for &f in a.faces.keys() {
        let c = classify(&a, f, &b, &tb);
        let k = match op {
            BoolOp::Union => matches!(c, Class::Outside | Class::Same),
            BoolOp::Intersect => matches!(c, Class::Inside | Class::Same),
            BoolOp::Subtract => matches!(c, Class::Outside | Class::Opposite),
        };
        if k {
            keep.push(part(&a, f, false));
        }
    }
    for &f in b.faces.keys() {
        let c = classify(&b, f, &a, &ta);
        match (op, c) {
            (BoolOp::Union, Class::Outside) | (BoolOp::Intersect, Class::Inside) => keep.push(part(&b, f, false)),
            // The cutter's faces inside the first solid line the new cavity.
            (BoolOp::Subtract, Class::Inside) => keep.push(part(&b, f, true)),
            _ => {}
        }
    }
    weld(keep)
}

// ---------------------------------------------------------------------------

type Part = (Vec<Vec<Vec3>>, Vec3, Option<usize>);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Inside,
    Outside,
    /// Lies on a face of the other solid that points the same way.
    Same,
    /// Lies on a face of the other solid that points the other way.
    Opposite,
}

fn oriented(m: &Model) -> Model {
    let mut m = m.clone();
    if volume(&m) < 0.0 {
        for f in m.faces.values_mut() {
            f.outer.reverse();
            f.holes.iter_mut().for_each(|h| h.reverse());
            f.normal = -f.normal;
        }
    }
    m
}

fn cutters(m: &Model) -> Vec<crate::model::CutFace> {
    let ids: Vec<Id> = m.faces.keys().copied().collect();
    let mut c = m.cut_faces(&ids);
    c.iter_mut().for_each(|x| x.id = None);
    c
}

fn triangles(m: &Model) -> Vec<[Vec3; 3]> {
    m.faces.keys().flat_map(|&f| m.triangulate(f)).collect()
}

fn part(m: &Model, f: Id, reverse: bool) -> Part {
    let face = &m.faces[&f];
    let mut loops: Vec<Vec<Vec3>> = face.loops().map(|l| m.loop_points(l)).collect();
    if reverse {
        loops.iter_mut().for_each(|l| l.reverse());
    }
    (loops, if reverse { -face.normal } else { face.normal }, face.material)
}

/// A point well inside a face: the centroid of its largest triangle.
fn sample(m: &Model, f: Id) -> Option<Vec3> {
    m.triangulate(f)
        .into_iter()
        .max_by(|x, y| {
            let ar = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).length();
            ar(x).total_cmp(&ar(y))
        })
        .map(|t| (t[0] + t[1] + t[2]) / 3.0)
}

fn classify(m: &Model, f: Id, other: &Model, other_tris: &[[Vec3; 3]]) -> Class {
    let n = m.faces[&f].normal;
    let Some(p) = sample(m, f) else { return Class::Outside };
    for (&g, face) in &other.faces {
        if face.normal.dot(n).abs() > 0.9999
            && (p - other.pos(face.outer[0])).dot(face.normal).abs() < EPS * 10.0
            && other.point_in_face(g, p)
        {
            return if face.normal.dot(n) > 0.0 { Class::Same } else { Class::Opposite };
        }
    }
    if inside(other_tris, p) {
        Class::Inside
    } else {
        Class::Outside
    }
}

/// Ray-parity point-in-solid test; three skewed rays vote, so a ray that
/// happens to graze an edge doesn't decide the answer alone.
fn inside(tris: &[[Vec3; 3]], p: Vec3) -> bool {
    const DIRS: [[f32; 3]; 3] = [[0.5413, 0.7351, 0.4084], [-0.6217, 0.2873, 0.7286], [0.3119, -0.8342, -0.4547]];
    let votes = DIRS
        .iter()
        .filter(|d| {
            let d = Vec3::from_array(**d).normalize();
            tris.iter().filter(|t| ray_tri(p, d, t)).count() % 2 == 1
        })
        .count();
    votes >= 2
}

fn ray_tri(o: Vec3, d: Vec3, t: &[Vec3; 3]) -> bool {
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
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    e2.dot(q) * inv > 1e-6
}

/// Adds to `a` the outlines of `b`'s faces that are coplanar with `a`'s faces
/// (Intersect Faces skips parallel faces), so that coincident regions become
/// faces of their own.
fn imprint(a: &mut Model, b: &Model) {
    let mut segs: Vec<(Vec3, Vec3)> = Vec::new();
    let mut planes: Vec<(Vec3, f32)> = Vec::new();
    for face in a.faces.values() {
        let n = face.normal;
        let pa = a.pos(face.outer[0]);
        let (u, v) = plane_basis(n);
        let to2 = |p: Vec3| Vec2::new(p.dot(u), p.dot(v));
        let loops2: Vec<Vec<Vec2>> = face.loops().map(|l| l.iter().map(|&x| to2(a.pos(x))).collect()).collect();
        let mut hit = false;
        for g in b.faces.values() {
            if g.normal.dot(n).abs() < 0.9999 || (b.pos(g.outer[0]) - pa).dot(n).abs() > EPS * 10.0 {
                continue;
            }
            for lp in g.loops() {
                for i in 0..lp.len() {
                    let (p, q) = (b.pos(lp[i]), b.pos(lp[(i + 1) % lp.len()]));
                    for (s, t) in clip_segment(to2(p), to2(q), &loops2) {
                        segs.push((p.lerp(q, s), p.lerp(q, t)));
                        hit = true;
                    }
                }
            }
        }
        if hit {
            let pl = canonical_plane(n, pa);
            if !planes.iter().any(|(m, d)| m.dot(pl.0) > 0.9999 && (d - pl.1).abs() < EPS * 10.0) {
                planes.push(pl);
            }
        }
    }
    for (p, q) in segs {
        let x = a.vertex_on_geometry(p);
        let y = a.vertex_on_geometry(q);
        a.add_edge(x, y);
    }
    for (n, d) in planes {
        a.rebuild_plane(n, d);
    }
}

/// Parameter intervals of segment p→q lying inside (or on) a polygon with holes.
fn clip_segment(p: Vec2, q: Vec2, loops: &[Vec<Vec2>]) -> Vec<(f32, f32)> {
    let d = q - p;
    let len = d.length();
    if len < EPS {
        return vec![];
    }
    let mut ts = vec![0.0f32, 1.0];
    for l in loops {
        for i in 0..l.len() {
            let (a, b) = (l[i], l[(i + 1) % l.len()]);
            let e = b - a;
            let den = d.perp_dot(e);
            if den.abs() > 1e-12 {
                let w = a - p;
                let t = w.perp_dot(e) / den;
                let s = w.perp_dot(d) / den;
                if (-1e-6..=1.0 + 1e-6).contains(&s) && t > 0.0 && t < 1.0 {
                    ts.push(t);
                }
            }
            // Loop vertices on the segment (collinear overlaps).
            let t = (a - p).dot(d) / (len * len);
            if t > 0.0 && t < 1.0 && (p + d * t - a).length() < EPS {
                ts.push(t);
            }
        }
    }
    ts.sort_by(f32::total_cmp);
    ts.dedup_by(|x, y| (*x - *y).abs() * len < EPS);
    let on_boundary = |m: Vec2| {
        loops.iter().any(|l| {
            (0..l.len()).any(|i| {
                let (a, b) = (l[i].extend(0.0), l[(i + 1) % l.len()].extend(0.0));
                point_seg(m.extend(0.0), a, b).0 < EPS
            })
        })
    };
    let mut out: Vec<(f32, f32)> = Vec::new();
    for w in ts.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        if (t1 - t0) * len < EPS {
            continue;
        }
        let m = p + d * ((t0 + t1) * 0.5);
        let inside = point_in_poly(m, &loops[0]) && !loops[1..].iter().any(|h| point_in_poly(m, h));
        if inside || on_boundary(m) {
            match out.last_mut() {
                Some(last) if (last.1 - t0).abs() < 1e-6 => last.1 = t1,
                _ => out.push((t0, t1)),
            }
        }
    }
    out
}

/// Builds a mesh from face outlines, welding shared vertices, splitting
/// T-junctions and merging coplanar neighbours left by the cuts.
fn weld(parts: Vec<Part>) -> Model {
    let mut m = Model::new();
    for (loops, n, mat) in parts {
        let ids: Vec<Vec<Id>> = loops.iter().map(|l| l.iter().map(|p| m.add_vertex(*p)).collect()).collect();
        m.add_face_loops(ids[0].clone(), ids[1..].to_vec(), n, mat);
    }
    fix_t_junctions(&mut m);
    merge_coplanar(&mut m);
    m
}

fn fix_t_junctions(m: &mut Model) {
    let verts: Vec<(Id, Vec3)> = m.vertices.iter().map(|(i, p)| (*i, *p)).collect();
    let edges: Vec<Id> = m.edges.keys().copied().collect();
    for e in edges {
        let Some(edge) = m.edges.get(&e).cloned() else { continue };
        let (pa, pb) = (m.pos(edge.a), m.pos(edge.b));
        let mut on: Vec<(f32, Id)> = verts
            .iter()
            .filter(|(v, _)| *v != edge.a && *v != edge.b)
            .filter_map(|(v, p)| {
                let (dist, t) = point_seg(*p, pa, pb);
                (dist < EPS && t > 1e-5 && t < 1.0 - 1e-5).then_some((t, *v))
            })
            .collect();
        on.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut cur = Some(e);
        for (_, v) in on {
            let Some(c) = cur else { break };
            m.split_edge_at(c, v);
            cur = m.find_edge(v, edge.b);
        }
    }
}

fn merge_coplanar(m: &mut Model) {
    let mut planes: Vec<(Vec3, f32)> = Vec::new();
    let mut doomed = Vec::new();
    for (&e, edge) in &m.edges {
        let fs = m.faces_with_edge(edge.a, edge.b);
        if fs.is_empty() {
            doomed.push(e);
            continue;
        }
        if fs.len() != 2 {
            continue;
        }
        let (f, g) = (&m.faces[&fs[0]], &m.faces[&fs[1]]);
        if f.material == g.material && f.normal.dot(g.normal) > 0.9999 {
            doomed.push(e);
            let pl = canonical_plane(f.normal, m.pos(edge.a));
            if !planes.iter().any(|(n, d)| n.dot(pl.0) > 0.9999 && (d - pl.1).abs() < EPS * 10.0) {
                planes.push(pl);
            }
        }
    }
    if doomed.is_empty() {
        return;
    }
    for e in doomed {
        m.edges.remove(&e);
    }
    for (n, d) in planes {
        m.rebuild_plane(n, d);
    }
    let vs: Vec<Id> = m.vertices.keys().copied().collect();
    for v in vs {
        if !m.vertices.contains_key(&v) {
            continue;
        }
        if m.vertex_edges(v).is_empty() {
            m.remove_vertex_if_orphan(v);
        } else {
            m.remove_collinear_vertex(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box with outward faces.
    fn cube(lo: Vec3, hi: Vec3) -> Model {
        let mut m = Model::new();
        let c = crate::document::box_corners(lo, hi).map(|p| m.add_vertex(p));
        let faces: [([usize; 4], Vec3); 6] = [
            ([0, 3, 2, 1], -Vec3::Z),
            ([4, 5, 6, 7], Vec3::Z),
            ([0, 1, 5, 4], -Vec3::Y),
            ([2, 3, 7, 6], Vec3::Y),
            ([1, 2, 6, 5], Vec3::X),
            ([3, 0, 4, 7], -Vec3::X),
        ];
        for (ix, n) in faces {
            m.add_face_loops(ix.iter().map(|&i| c[i]).collect(), vec![], n, None);
        }
        m
    }

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn check(m: &Model, vol: f32) {
        assert!(is_solid(m), "result is a solid ({} faces, {} edges)", m.faces.len(), m.edges.len());
        assert!((volume(m) - vol).abs() < 1e-3, "volume {} != {vol}", volume(m));
    }

    #[test]
    fn cube_is_solid() {
        let c = cube(Vec3::ZERO, v(1.0, 2.0, 3.0));
        check(&c, 6.0);
        let mut open = c.clone();
        let f = *open.faces.keys().next().unwrap();
        open.delete_face(f);
        assert!(!is_solid(&open));
    }

    #[test]
    fn overlapping_cubes() {
        let a = cube(Vec3::ZERO, Vec3::splat(2.0));
        let b = cube(Vec3::ONE, Vec3::splat(3.0));
        check(&boolean(&a, &b, BoolOp::Union), 15.0);
        let i = boolean(&a, &b, BoolOp::Intersect);
        check(&i, 1.0);
        assert_eq!(i.faces.len(), 6);
        check(&boolean(&a, &b, BoolOp::Subtract), 7.0);
        check(&boolean(&b, &a, BoolOp::Subtract), 7.0);
    }

    #[test]
    fn flush_pocket_and_union() {
        let a = cube(Vec3::ZERO, Vec3::splat(2.0));
        let b = cube(v(0.5, 0.5, 1.0), v(1.5, 1.5, 2.0)); // flush with a's top
        let pocket = boolean(&a, &b, BoolOp::Subtract);
        check(&pocket, 7.0);
        assert_eq!(pocket.faces.len(), 11, "6 sides with an open top ring + 5 pocket faces");
        let u = boolean(&a, &b, BoolOp::Union);
        check(&u, 8.0);
        assert_eq!(u.faces.len(), 6, "coplanar seams are merged away");
    }

    #[test]
    fn touching_cubes_union_into_one_box() {
        let a = cube(Vec3::ZERO, Vec3::ONE);
        let b = cube(v(1.0, 0.0, 0.0), v(2.0, 1.0, 1.0));
        let u = boolean(&a, &b, BoolOp::Union);
        check(&u, 2.0);
        assert_eq!(u.faces.len(), 6);
        assert_eq!(u.vertices.len(), 8);
        // Subtracting a neighbour that only touches changes nothing.
        check(&boolean(&a, &b, BoolOp::Subtract), 1.0);
    }

    #[test]
    fn cavity_and_disjoint() {
        let a = cube(Vec3::ZERO, Vec3::splat(3.0));
        let b = cube(Vec3::ONE, Vec3::splat(2.0));
        let hollow = boolean(&a, &b, BoolOp::Subtract);
        check(&hollow, 26.0);
        assert_eq!(hollow.faces.len(), 12);
        let far = cube(Vec3::splat(5.0), Vec3::splat(6.0));
        assert!(boolean(&a, &far, BoolOp::Intersect).faces.is_empty());
        check(&boolean(&a, &far, BoolOp::Union), 28.0);
    }

    #[test]
    fn tunnel_through_a_box() {
        let a = cube(Vec3::ZERO, v(4.0, 2.0, 2.0));
        let b = cube(v(1.0, -1.0, 0.5), v(2.0, 3.0, 1.5));
        check(&boolean(&a, &b, BoolOp::Subtract), 14.0);
        check(&boolean(&a, &b, BoolOp::Intersect), 2.0);
        check(&boolean(&a, &b, BoolOp::Union), 16.0 + 4.0 - 2.0);
    }

    #[test]
    fn rotated_overlap_is_consistent() {
        let a = cube(Vec3::ZERO, Vec3::splat(2.0));
        let b = cube(v(-1.0, -1.0, 0.5), v(1.0, 1.0, 1.5))
            .transformed(glam::Mat4::from_translation(v(1.8, 1.0, 0.0)) * glam::Mat4::from_rotation_z(0.5));
        let (u, i, s) = (boolean(&a, &b, BoolOp::Union), boolean(&a, &b, BoolOp::Intersect), boolean(&a, &b, BoolOp::Subtract));
        for m in [&u, &i, &s] {
            assert!(is_solid(m));
        }
        let (vu, vi, vs) = (volume(&u), volume(&i), volume(&s));
        assert!(vi > 0.1);
        assert!((vu - (8.0 + 4.0 - vi)).abs() < 1e-3, "union {vu}, intersect {vi}");
        assert!((vs - (8.0 - vi)).abs() < 1e-3, "subtract {vs}");
    }

    #[test]
    fn push_pulled_box_minus_cylinder() {
        let mut a = Model::new();
        let f = a.add_polygon(&[Vec3::ZERO, v(3.0, 0.0, 0.0), v(3.0, 3.0, 0.0), v(0.0, 3.0, 0.0)], None).unwrap();
        a.push_pull(f, -2.0);
        assert!(is_solid(&a));
        let mut b = Model::new();
        let pts: Vec<Vec3> = (0..24)
            .map(|i| {
                let t = i as f32 / 24.0 * std::f32::consts::TAU;
                v(3.0 + 1.2 * t.cos(), 1.5 + 1.2 * t.sin(), 0.5)
            })
            .collect();
        let c = b.add_polygon(&pts, Some(Vec3::Z)).unwrap();
        b.push_pull(c, 2.0);
        assert!(is_solid(&b) && b.edges.values().any(|e| e.soft));
        let vb = volume(&b);
        // The axis lies on the box's +X side and z 0.5..2 of its 0.5..2.5 is inside.
        let half_in = vb * 0.5 * 0.75;
        check(&boolean(&a, &b, BoolOp::Subtract), 18.0 - half_in);
        check(&boolean(&a, &b, BoolOp::Union), 18.0 + vb - half_in);
    }

    #[test]
    fn drill_a_hole_through_a_block() {
        let a = cube(Vec3::ZERO, v(4.0, 4.0, 1.0));
        // An octagonal prism through the block.
        let mut b = Model::new();
        let pts: Vec<Vec3> = (0..8)
            .map(|i| {
                let t = (i as f32 + 0.5) / 8.0 * std::f32::consts::TAU;
                v(2.0 + t.cos(), 2.0 + t.sin(), -1.0)
            })
            .collect();
        let f = b.add_polygon(&pts, Some(-Vec3::Z)).unwrap();
        b.push_pull(f, -3.0); // bottom faces down: negative goes up
        assert!(is_solid(&b));
        let oct = volume(&b) / 3.0;
        let r = boolean(&a, &b, BoolOp::Subtract);
        check(&r, 16.0 - oct);
        assert!(r.faces.values().any(|f| f.holes.len() == 1), "top and bottom get a hole");
    }

    #[test]
    fn groups_subtract_into_a_new_group() {
        use crate::document::{DefKind, Document, ROOT as ROOT_DEF};
        let mut doc = Document::new();
        let mut active = cube(Vec3::ZERO, Vec3::splat(2.0));
        active.merge_from(&cube(v(1.0, 1.0, 1.0), v(3.0, 3.0, 3.0)));
        let fs: Vec<Id> = active.faces.keys().copied().collect();
        let (first, second): (Vec<Id>, Vec<Id>) = fs.iter().partition(|f| {
            active.pos(active.faces[*f].outer[0]).max_element() > 2.0 + 1e-4
                || active.faces[*f].outer.iter().all(|&x| active.pos(x).min_element() > 1.0 - 1e-4)
        });
        assert_eq!((first.len(), second.len()), (6, 6));
        let b = doc.make_group(&[], &mut active, &first, &[], &[], DefKind::Group).unwrap();
        let a = doc.make_group(&[], &mut active, &second, &[], &[], DefKind::Group).unwrap();
        doc.sync(&[], &active);
        // Subtract the second cube (b) from the first (a).
        let (ma, mb) = (doc.instance_world_mesh(&[], a).unwrap(), doc.instance_world_mesh(&[], b).unwrap());
        assert!(is_solid(&ma) && is_solid(&mb));
        let r = boolean(&ma, &mb, BoolOp::Subtract);
        doc.remove_instances(&[], &[a, b]);
        let g = doc.add_world_mesh(&[], &r, DefKind::Group).unwrap();
        let back = doc.instance_world_mesh(&[], g).unwrap();
        check(&back, 7.0);
        assert_eq!(doc.defs[&ROOT_DEF].instances.len(), 1);
    }
}
