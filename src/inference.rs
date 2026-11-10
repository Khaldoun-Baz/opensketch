//! Inference engine: snaps the cursor to endpoints, midpoints, edges, faces and
//! the red/green/blue axes, like SketchUp's colored inference tooltips.
//!
//! Editable geometry (the active context mesh) is picked with ids; geometry in
//! groups/components and outside the edit context is snappable but anonymous.

use crate::camera::Camera;
use crate::guide::Guide;
use crate::model::{closest_on_line_to_ray, ray_plane, Id, Model};
use eframe::egui::{Pos2, Rect};
use glam::Vec3;
use std::collections::HashSet;

pub const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// What a pickable triangle belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Owner {
    /// A face of the mesh being edited.
    Face(Id),
    /// A group/component instance that is a direct child of the edit context.
    Instance(Id),
    /// Geometry outside the edit context (faded).
    Outside,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SnapKind {
    Endpoint,
    Midpoint,
    OnEdge,
    OnFace,
    Axis(usize),
    Ground,
    Free,
    GuidePoint,
    Intersection,
    OnGuide,
    /// Drawing parallel to an edge the cursor passed over.
    Parallel,
    /// Drawing perpendicular to an edge the cursor passed over.
    Perpendicular,
}

impl SnapKind {
    pub fn label(&self) -> &'static str {
        match self {
            SnapKind::Endpoint => "Endpoint",
            SnapKind::Midpoint => "Midpoint",
            SnapKind::OnEdge => "On Edge",
            SnapKind::OnFace => "On Face",
            SnapKind::Axis(0) => "On Red Axis",
            SnapKind::Axis(1) => "On Green Axis",
            SnapKind::Axis(_) => "On Blue Axis",
            SnapKind::Ground => "On Ground",
            SnapKind::Free => "",
            SnapKind::GuidePoint => "Guide Point",
            SnapKind::Intersection => "Intersection",
            SnapKind::OnGuide => "On Guide",
            SnapKind::Parallel => "Parallel to Edge",
            SnapKind::Perpendicular => "Perpendicular to Edge",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Snap {
    pub pos: Vec3,
    pub kind: SnapKind,
    pub vertex: Option<Id>,
    pub edge: Option<Id>,
    /// Editable face under the cursor (even when snapping to something else).
    pub face: Option<Id>,
    /// Group/component instance under the cursor.
    pub instance: Option<Id>,
    /// Normal of whatever surface is under the cursor.
    pub normal: Option<Vec3>,
    /// True if any surface is under the cursor.
    pub hit: bool,
    /// Guide snapped to (index into the document's guides).
    pub guide: Option<usize>,
    /// Direction of the edge or guide line snapped to.
    pub line_dir: Option<Vec3>,
    /// "From Point": the point this position lines up with.
    pub from_point: Option<Vec3>,
}

impl Snap {
    fn new(pos: Vec3, kind: SnapKind) -> Self {
        Self {
            pos,
            kind,
            vertex: None,
            edge: None,
            face: None,
            instance: None,
            normal: None,
            hit: false,
            guide: None,
            line_dir: None,
            from_point: None,
        }
    }
}

pub struct Picker<'a> {
    pub model: &'a Model,
    pub cam: &'a Camera,
    pub rect: Rect,
    pub tris: &'a [(Owner, [Vec3; 3])],
    /// Snappable vertices outside the editable mesh.
    pub points: &'a [Vec3],
    /// Snappable edges outside the editable mesh.
    pub segs: &'a [(Vec3, Vec3)],
    /// Inference axes of the edit context.
    pub axes: [Vec3; 3],
    /// Guides (empty when hidden).
    pub guides: &'a [Guide],
    /// Where guide lines cross each other and edges.
    pub guide_pts: &'a [Vec3],
    /// Vertices being moved: they, and the edges touching them, are not snapped to.
    pub skip: Option<&'a HashSet<Id>>,
    /// Points the cursor passed over lately ("From Point" inference).
    pub recent: &'a [Vec3],
    /// Directions of edges the cursor passed over lately (parallel / perpendicular).
    pub edge_dirs: &'a [Vec3],
}

fn ray_tri(o: Vec3, d: Vec3, t: &[Vec3; 3]) -> Option<f32> {
    let e1 = t[1] - t[0];
    let e2 = t[2] - t[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - t[0];
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let tt = e2.dot(q) * inv;
    (tt > 1e-6).then_some(tt)
}

/// Closest point of segment (a, b) to the ray (o, d).
fn seg_closest_to_ray(a: Vec3, b: Vec3, o: Vec3, d: Vec3) -> Vec3 {
    let u = b - a;
    let w0 = a - o;
    let (aa, bb, cc) = (u.dot(u), u.dot(d), d.dot(d));
    let (dd, ee) = (u.dot(w0), d.dot(w0));
    let den = aa * cc - bb * bb;
    let s = if den.abs() > 1e-12 { ((bb * ee - cc * dd) / den).clamp(0.0, 1.0) } else { 0.0 };
    a + u * s
}

pub struct FaceHit {
    pub owner: Owner,
    pub t: f32,
    pub normal: Vec3,
}

impl<'a> Picker<'a> {
    pub fn ray(&self, p: Pos2) -> (Vec3, Vec3) {
        self.cam.ray(self.rect, p)
    }

    pub fn pick_face(&self, o: Vec3, d: Vec3) -> Option<FaceHit> {
        let mut best: Option<(usize, f32)> = None;
        for (i, (_, tri)) in self.tris.iter().enumerate() {
            if let Some(t) = ray_tri(o, d, tri) {
                if best.is_none_or(|(_, bt)| t < bt) {
                    best = Some((i, t));
                }
            }
        }
        best.map(|(i, t)| {
            let (owner, tri) = self.tris[i];
            let normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
            FaceHit { owner, t, normal }
        })
    }

    fn visible(&self, p: Vec3, o: Vec3, d: Vec3, face_t: Option<f32>) -> bool {
        match face_t {
            Some(ft) => (p - o).dot(d) <= ft * 1.002 + 1e-3,
            None => true,
        }
    }

    fn screen_dist(&self, p: Vec3, cursor: Pos2) -> Option<f32> {
        self.cam.project(self.rect, p).map(|s| s.distance(cursor))
    }

    /// Nearest visible candidate within `tol` pixels.
    fn nearest<T: Copy>(
        &self,
        cands: impl Iterator<Item = (T, Vec3)>,
        cursor: Pos2,
        tol: f32,
        o: Vec3,
        d: Vec3,
        face_t: Option<f32>,
    ) -> Option<(T, Vec3)> {
        let mut best: Option<(T, Vec3, f32)> = None;
        for (id, p) in cands {
            if !self.visible(p, o, d, face_t) {
                continue;
            }
            if let Some(dist) = self.screen_dist(p, cursor) {
                if dist < tol && best.is_none_or(|b| dist < b.2) {
                    best = Some((id, p, dist));
                }
            }
        }
        best.map(|(i, p, _)| (i, p))
    }

    fn skipped(&self, v: Id) -> bool {
        self.skip.is_some_and(|s| s.contains(&v))
    }

    fn all_edges(&self) -> impl Iterator<Item = (Option<Id>, Vec3, Vec3)> + '_ {
        self.model
            .edges
            .iter()
            .filter(|(_, e)| !e.soft && !self.skipped(e.a) && !self.skipped(e.b))
            .map(|(&id, e)| (Some(id), self.model.pos(e.a), self.model.pos(e.b)))
            .chain(self.segs.iter().map(|&(a, b)| (None, a, b)))
    }

    pub fn pick_vertex(&self, cursor: Pos2, tol: f32, o: Vec3, d: Vec3, face_t: Option<f32>) -> Option<(Option<Id>, Vec3)> {
        let own = self.model.vertices.iter().filter(|(id, _)| !self.skipped(**id)).map(|(&id, &p)| (Some(id), p));
        self.nearest(own.chain(self.points.iter().map(|&p| (None, p))), cursor, tol, o, d, face_t)
    }

    pub fn pick_midpoint(&self, cursor: Pos2, tol: f32, o: Vec3, d: Vec3, face_t: Option<f32>) -> Option<(Option<Id>, Vec3)> {
        let cands = self.all_edges().map(|(id, a, b)| (id, (a + b) * 0.5));
        self.nearest(cands, cursor, tol, o, d, face_t)
    }

    /// Nearest edge: (editable id, point, edge direction).
    pub fn pick_edge(&self, cursor: Pos2, tol: f32, o: Vec3, d: Vec3, face_t: Option<f32>) -> Option<(Option<Id>, Vec3, Vec3)> {
        let cands = self.all_edges().map(|(id, a, b)| ((id, (b - a).normalize_or_zero()), seg_closest_to_ray(a, b, o, d)));
        self.nearest(cands, cursor, tol, o, d, face_t).map(|((id, dir), p)| (id, p, dir))
    }

    fn pick_guide_point(&self, cursor: Pos2, tol: f32, o: Vec3, d: Vec3, face_t: Option<f32>) -> Option<(usize, Vec3)> {
        let cands = self.guides.iter().enumerate().filter_map(|(i, g)| match g {
            Guide::Point { p } => Some((i, *p)),
            Guide::Line { .. } => None,
        });
        self.nearest(cands, cursor, tol, o, d, face_t)
    }

    fn pick_guide_line(&self, cursor: Pos2, tol: f32, o: Vec3, d: Vec3, face_t: Option<f32>) -> Option<(usize, Vec3)> {
        let cands = self.guides.iter().enumerate().filter_map(|(i, g)| match g {
            Guide::Line { .. } => Some((i, g.closest_to_ray(o, d))),
            Guide::Point { .. } => None,
        });
        self.nearest(cands, cursor, tol, o, d, face_t)
    }

    /// Guide under the cursor (for selection and the eraser).
    pub fn pick_guide(&self, cursor: Pos2, tol: f32) -> Option<usize> {
        let (o, d) = self.ray(cursor);
        let face_t = self.pick_face(o, d).map(|h| h.t);
        self.pick_guide_point(cursor, tol + 2.0, o, d, face_t).or_else(|| self.pick_guide_line(cursor, tol, o, d, face_t)).map(|x| x.0)
    }

    /// Editable edge under the cursor (for selection and the eraser).
    pub fn pick_model_edge(&self, cursor: Pos2, tol: f32) -> Option<Id> {
        let (o, d) = self.ray(cursor);
        let face_t = self.pick_face(o, d).map(|h| h.t);
        let cands = self
            .model
            .edges
            .iter()
            .filter(|(_, e)| !e.soft)
            .map(|(&id, e)| (Some(id), seg_closest_to_ray(self.model.pos(e.a), self.model.pos(e.b), o, d)));
        self.nearest(cands, cursor, tol, o, d, face_t).and_then(|x| x.0)
    }

    /// Main inference entry point.
    /// * `from`: the previous point (enables axis inference and sets the fallback plane height)
    /// * `lock`: locked axis index (arrow keys / Shift)
    pub fn snap(&self, cursor: Pos2, from: Option<Vec3>, lock: Option<usize>) -> Snap {
        let (o, d) = self.ray(cursor);
        let hit = self.pick_face(o, d);
        let face_t = hit.as_ref().map(|h| h.t);
        let decorate = |mut s: Snap| {
            if let Some(h) = &hit {
                s.hit = true;
                s.normal = Some(h.normal);
                match h.owner {
                    Owner::Face(f) => {
                        s.face = Some(f);
                        s.normal = self.model.faces.get(&f).map(|x| x.normal).or(s.normal);
                    }
                    Owner::Instance(i) => s.instance = Some(i),
                    Owner::Outside => {}
                }
            }
            s
        };

        // "From Point": a position on the line through `from` along `dir` that
        // lines up with a point the cursor passed over lately.
        let align = |p: Vec3, from: Vec3, dir: Vec3| -> (Vec3, Option<Vec3>) {
            let mut best: Option<(Vec3, Vec3, f32)> = None;
            for &r in self.recent {
                let q = from + dir * (r - from).dot(dir);
                if q.distance(from) < 1e-4 || q.distance(r) < 1e-4 {
                    continue;
                }
                if let Some(dist) = self.screen_dist(q, cursor) {
                    if dist < 9.0 && best.is_none_or(|b| dist < b.2) {
                        best = Some((q, r, dist));
                    }
                }
            }
            match best {
                Some((q, r, _)) => (q, Some(r)),
                None => (p, None),
            }
        };

        if let (Some(from), Some(ax)) = (from, lock) {
            let p = closest_on_line_to_ray(from, self.axes[ax], o, d);
            // A locked axis still snaps to real points: the position lines up with them.
            let target = self
                .pick_vertex(cursor, 10.0, o, d, face_t)
                .map(|x| x.1)
                .or_else(|| self.pick_midpoint(cursor, 9.0, o, d, face_t).map(|x| x.1));
            let (p, from_point) = match target {
                Some(t) if t.distance(from) > 1e-4 => (from + self.axes[ax] * (t - from).dot(self.axes[ax]), Some(t)),
                _ => align(p, from, self.axes[ax]),
            };
            let mut s = Snap::new(p, SnapKind::Axis(ax));
            s.from_point = from_point.filter(|t| t.distance(p) > 1e-4);
            return decorate(s);
        }

        if let Some((v, p)) = self.pick_vertex(cursor, 10.0, o, d, face_t) {
            let mut s = Snap::new(p, SnapKind::Endpoint);
            s.vertex = v;
            return decorate(s);
        }
        if let Some((g, p)) = self.pick_guide_point(cursor, 10.0, o, d, face_t) {
            let mut s = Snap::new(p, SnapKind::GuidePoint);
            s.guide = Some(g);
            return decorate(s);
        }
        if let Some((_, p)) = self.nearest(self.guide_pts.iter().map(|p| ((), *p)), cursor, 10.0, o, d, face_t) {
            return decorate(Snap::new(p, SnapKind::Intersection));
        }
        if let Some((e, p)) = self.pick_midpoint(cursor, 9.0, o, d, face_t) {
            let mut s = Snap::new(p, SnapKind::Midpoint);
            s.edge = e;
            return decorate(s);
        }
        if let Some(from) = from {
            let mut best: Option<(usize, Vec3, f32)> = None;
            for (i, ax) in self.axes.iter().enumerate() {
                let p = closest_on_line_to_ray(from, *ax, o, d);
                if p.distance(from) < 1e-5 {
                    continue;
                }
                if let Some(dist) = self.screen_dist(p, cursor) {
                    if dist < 12.0 && best.is_none_or(|b| dist < b.2) {
                        best = Some((i, p, dist));
                    }
                }
            }
            if let Some((i, p, _)) = best {
                let (p, from_point) = align(p, from, self.axes[i]);
                let mut s = Snap::new(p, SnapKind::Axis(i));
                s.from_point = from_point;
                return decorate(s);
            }
            // Parallel or perpendicular to an edge the cursor passed over.
            let mut best: Option<(SnapKind, Vec3, f32)> = None;
            for &e in self.edge_dirs {
                let perp = e.cross(self.axes[2]).normalize_or_zero();
                for (kind, dir) in [(SnapKind::Parallel, e), (SnapKind::Perpendicular, perp)] {
                    if dir == Vec3::ZERO || self.axes.iter().any(|a| a.dot(dir).abs() > 0.9999) {
                        continue;
                    }
                    let p = closest_on_line_to_ray(from, dir, o, d);
                    if p.distance(from) < 1e-5 {
                        continue;
                    }
                    if let Some(dist) = self.screen_dist(p, cursor) {
                        if dist < 9.0 && best.is_none_or(|b| dist < b.2) {
                            best = Some((kind, p, dist));
                        }
                    }
                }
            }
            if let Some((kind, p, _)) = best {
                return decorate(Snap::new(p, kind));
            }
        }
        if let Some((e, p, dir)) = self.pick_edge(cursor, 7.0, o, d, face_t) {
            let mut s = Snap::new(p, SnapKind::OnEdge);
            s.edge = e;
            s.line_dir = Some(dir);
            return decorate(s);
        }
        if let Some((g, p)) = self.pick_guide_line(cursor, 7.0, o, d, face_t) {
            let mut s = Snap::new(p, SnapKind::OnGuide);
            s.guide = Some(g);
            if let Some(Guide::Line { d: gd, .. }) = self.guides.get(g) {
                s.line_dir = Some(*gd);
            }
            return decorate(s);
        }
        if let Some(h) = &hit {
            return decorate(Snap::new(o + d * h.t, SnapKind::OnFace));
        }
        let z = from.map(|p| p.z).unwrap_or(0.0);
        if let Some(p) = ray_plane(o, d, Vec3::new(0.0, 0.0, z), Vec3::Z) {
            if p.distance(o) < self.cam.distance * 60.0 {
                return Snap::new(p, if z.abs() < 1e-6 { SnapKind::Ground } else { SnapKind::Free });
            }
        }
        // Fallback: a point in front of the camera at target depth.
        let depth = (self.cam.target - o).dot(d).max(0.1);
        Snap::new(o + d * depth, SnapKind::Free)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{pos2, vec2};

    fn snap_at(guides: &[Guide], pts: &[Vec3], target: Vec3) -> Snap {
        let model = Model::new();
        let cam = Camera::default();
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 600.0));
        let picker = Picker {
            model: &model,
            cam: &cam,
            rect,
            tris: &[],
            points: &[],
            segs: &[],
            axes: AXES,
            guides,
            guide_pts: pts,
            skip: None,
            recent: &[],
            edge_dirs: &[],
        };
        let cursor = cam.project(rect, target).unwrap() + vec2(2.0, 1.0);
        picker.snap(cursor, None, None)
    }

    /// Snaps the cursor placed over `target` while drawing from `from`.
    fn draw_snap(
        model: &Model,
        skip: Option<&HashSet<Id>>,
        recent: &[Vec3],
        dirs: &[Vec3],
        from: Option<Vec3>,
        lock: Option<usize>,
        target: Vec3,
    ) -> Snap {
        let cam = Camera::default();
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 600.0));
        let picker = Picker {
            model,
            cam: &cam,
            rect,
            tris: &[],
            points: &[],
            segs: &[],
            axes: AXES,
            guides: &[],
            guide_pts: &[],
            skip,
            recent,
            edge_dirs: dirs,
        };
        let cursor = cam.project(rect, target).unwrap() + vec2(2.0, 1.0);
        picker.snap(cursor, from, lock)
    }

    #[test]
    fn moving_geometry_is_not_snapped_to() {
        let mut m = Model::new();
        let a = m.add_vertex(Vec3::new(1.0, 1.0, 0.0));
        let b = m.add_vertex(Vec3::new(2.0, 1.0, 0.0));
        m.add_edge(a, b);
        let at = Vec3::new(1.0, 1.0, 0.0);
        assert_eq!(draw_snap(&m, None, &[], &[], None, None, at).vertex, Some(a));
        // While `a` is being moved, neither it nor its edge attract the cursor.
        let skip: HashSet<Id> = [a].into_iter().collect();
        let s = draw_snap(&m, Some(&skip), &[], &[], None, None, at);
        assert_eq!((s.kind, s.vertex), (SnapKind::Ground, None));
        let s = draw_snap(&m, Some(&skip), &[], &[], None, None, Vec3::new(1.5, 1.0, 0.0));
        assert_eq!(s.kind, SnapKind::Ground);
        // The far end of the edge stays put, so it still snaps.
        assert_eq!(draw_snap(&m, Some(&skip), &[], &[], None, None, Vec3::new(2.0, 1.0, 0.0)).vertex, Some(b));
    }

    #[test]
    fn lines_up_with_recent_points() {
        let m = Model::new();
        let from = Vec3::ZERO;
        let other = Vec3::new(2.0, 3.0, 0.0);
        // Drawing along the red axis, near where it passes the other point.
        let s = draw_snap(&m, None, &[other], &[], Some(from), None, Vec3::new(2.02, 0.0, 0.0));
        assert_eq!(s.kind, SnapKind::Axis(0));
        assert!(s.pos.distance(Vec3::new(2.0, 0.0, 0.0)) < 1e-4, "{:?}", s.pos);
        assert_eq!(s.from_point, Some(other));
        // Far from that spot it is a plain axis snap.
        let s = draw_snap(&m, None, &[other], &[], Some(from), None, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!((s.kind, s.from_point), (SnapKind::Axis(0), None));
    }

    #[test]
    fn locked_axis_lines_up_with_the_point_under_the_cursor() {
        let mut m = Model::new();
        let v = m.add_vertex(Vec3::new(3.0, 2.0, 0.0));
        let _ = v;
        // Locked to red, cursor on a vertex off the axis: stop level with it.
        let s = draw_snap(&m, None, &[], &[], Some(Vec3::ZERO), Some(0), Vec3::new(3.0, 2.0, 0.0));
        assert_eq!(s.kind, SnapKind::Axis(0));
        assert!(s.pos.distance(Vec3::new(3.0, 0.0, 0.0)) < 1e-4, "{:?}", s.pos);
        assert_eq!(s.from_point, Some(Vec3::new(3.0, 2.0, 0.0)));
    }

    #[test]
    fn parallel_and_perpendicular_to_a_recent_edge() {
        let m = Model::new();
        let e = Vec3::new(1.0, 1.0, 0.0).normalize();
        let s = draw_snap(&m, None, &[], &[e], Some(Vec3::ZERO), None, Vec3::new(2.0, 2.02, 0.0));
        assert_eq!(s.kind, SnapKind::Parallel);
        assert!(s.pos.normalize().dot(e) > 0.9999);
        let s = draw_snap(&m, None, &[], &[e], Some(Vec3::ZERO), None, Vec3::new(2.0, -2.02, 0.0));
        assert_eq!(s.kind, SnapKind::Perpendicular);
        assert!(s.pos.normalize().dot(e).abs() < 1e-3);
        // The main axes win over an edge that runs along one of them.
        let s = draw_snap(&m, None, &[], &[Vec3::X], Some(Vec3::ZERO), None, Vec3::new(2.0, 0.01, 0.0));
        assert_eq!(s.kind, SnapKind::Axis(0));
    }

    #[test]
    fn snaps_to_guides() {
        let line = Guide::line(Vec3::new(0.0, 2.0, 0.0), Vec3::X).unwrap();
        let s = snap_at(&[line], &[], Vec3::new(1.0, 2.0, 0.0));
        assert_eq!(s.kind, SnapKind::OnGuide);
        assert_eq!(s.guide, Some(0));
        assert!((s.pos.y - 2.0).abs() < 1e-3 && s.pos.z.abs() < 1e-3);
        assert!(s.line_dir.unwrap().dot(Vec3::X) > 0.999);

        let pt = Guide::Point { p: Vec3::new(1.0, 1.0, 0.0) };
        let s = snap_at(&[line, pt], &[], Vec3::new(1.0, 1.0, 0.0));
        assert_eq!((s.kind, s.guide), (SnapKind::GuidePoint, Some(1)));

        let x = Vec3::new(1.0, 2.0, 0.0);
        let s = snap_at(&[line], &[x], x);
        assert_eq!(s.kind, SnapKind::Intersection);
        assert!(s.pos.distance(x) < 1e-5);
    }
}
