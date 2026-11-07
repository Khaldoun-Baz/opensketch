//! Construction geometry: infinite guide lines and guide points, created by the
//! Tape Measure and Protractor tools. Guides are snapped to like edges, but
//! they are not part of the model (they are not exported).

use crate::model::EPS;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Guide {
    /// Infinite line through `p` along the unit direction `d`.
    Line { p: Vec3, d: Vec3 },
    Point { p: Vec3 },
}

impl Guide {
    pub fn line(p: Vec3, d: Vec3) -> Option<Guide> {
        let d = d.normalize_or_zero();
        (d != Vec3::ZERO).then_some(Guide::Line { p, d })
    }

    pub fn transformed(&self, m: Mat4) -> Guide {
        match *self {
            Guide::Line { p, d } => Guide::Line { p: m.transform_point3(p), d: m.transform_vector3(d).normalize_or(d) },
            Guide::Point { p } => Guide::Point { p: m.transform_point3(p) },
        }
    }

    /// Point of the guide closest to the ray (o, dir).
    pub fn closest_to_ray(&self, o: Vec3, dir: Vec3) -> Vec3 {
        match *self {
            Guide::Line { p, d } => crate::model::closest_on_line_to_ray(p, d, o, dir),
            Guide::Point { p } => p,
        }
    }
}

/// Closest points of two lines (directions need not be unit); `None` if parallel.
/// Returns the line parameters (s, t).
fn line_params(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> Option<(f32, f32)> {
    let w0 = p1 - p2;
    let (a, b, c) = (d1.dot(d1), d1.dot(d2), d2.dot(d2));
    let (d, e) = (d1.dot(w0), d2.dot(w0));
    let den = a * c - b * b;
    if den.abs() < 1e-9 * a * c {
        return None;
    }
    Some(((b * e - c * d) / den, (a * e - b * d) / den))
}

/// Where two guide lines cross.
pub fn line_line(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> Option<Vec3> {
    let (s, t) = line_params(p1, d1, p2, d2)?;
    let (q1, q2) = (p1 + d1 * s, p2 + d2 * t);
    (q1.distance(q2) < EPS * 10.0).then_some((q1 + q2) * 0.5)
}

/// Where a guide line crosses the segment a–b (endpoints included).
pub fn line_segment(p: Vec3, d: Vec3, a: Vec3, b: Vec3) -> Option<Vec3> {
    let u = b - a;
    if u.length() < EPS {
        return None;
    }
    let (s, t) = line_params(p, d, a, u)?;
    if !(-1e-4..=1.0 + 1e-4).contains(&t) {
        return None;
    }
    let (q1, q2) = (p + d * s, a + u * t);
    (q1.distance(q2) < EPS * 10.0).then_some(q2)
}

/// Snap points where guide lines cross each other and the given edges.
pub fn intersections(guides: &[Guide], segs: &[(Vec3, Vec3)]) -> Vec<Vec3> {
    let lines: Vec<(Vec3, Vec3)> =
        guides.iter().filter_map(|g| if let Guide::Line { p, d } = g { Some((*p, *d)) } else { None }).collect();
    let mut out = Vec::new();
    for (i, &(p, d)) in lines.iter().enumerate() {
        for &(q, e) in &lines[i + 1..] {
            out.extend(line_line(p, d, q, e));
        }
        for &(a, b) in segs {
            out.extend(line_segment(p, d, a, b));
        }
    }
    out
}

/// Short world-space dashes along a guide line, around `focus` (for drawing).
pub fn dashes(p: Vec3, d: Vec3, focus: Vec3, half_len: f32, dash: f32) -> Vec<(Vec3, Vec3)> {
    let c = p + d * (focus - p).dot(d);
    let n = ((half_len / dash) as i32).clamp(1, 4000);
    // Anchor dashes to the line, not the focus, so they don't crawl when panning.
    let k0 = ((c - p).dot(d) / dash).floor() as i32;
    (-n..n)
        .filter(|k| (k0 + k) % 2 == 0)
        .map(|k| {
            let s = (k0 + k) as f32 * dash;
            (p + d * s, p + d * (s + dash))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossings() {
        let x = line_line(Vec3::ZERO, Vec3::X, Vec3::new(2.0, -1.0, 0.0), Vec3::Y).unwrap();
        assert!(x.distance(Vec3::new(2.0, 0.0, 0.0)) < 1e-5);
        assert!(line_line(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 0.0, 1.0), Vec3::Y).is_none(), "skew lines");
        assert!(line_line(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::X).is_none(), "parallel");
        let s = line_segment(Vec3::new(0.5, -3.0, 0.0), Vec3::Y, Vec3::ZERO, Vec3::X).unwrap();
        assert!(s.distance(Vec3::new(0.5, 0.0, 0.0)) < 1e-5);
        assert!(line_segment(Vec3::new(1.5, -3.0, 0.0), Vec3::Y, Vec3::ZERO, Vec3::X).is_none(), "misses the segment");
        let g = [Guide::line(Vec3::ZERO, Vec3::X).unwrap(), Guide::line(Vec3::ONE, Vec3::Y).unwrap(), Guide::Point { p: Vec3::ONE }];
        assert_eq!(intersections(&g, &[]).len(), 0, "z differs: no crossing");
        let g2 = [Guide::line(Vec3::ZERO, Vec3::X).unwrap(), Guide::line(Vec3::new(1.0, 1.0, 0.0), Vec3::Y).unwrap()];
        assert_eq!(intersections(&g2, &[(Vec3::new(3.0, -1.0, 0.0), Vec3::new(3.0, 1.0, 0.0))]).len(), 2);
    }

    #[test]
    fn dashes_are_stable() {
        let a = dashes(Vec3::ZERO, Vec3::X, Vec3::new(0.3, 5.0, 0.0), 2.0, 0.5);
        let b = dashes(Vec3::ZERO, Vec3::X, Vec3::new(0.4, 5.0, 0.0), 2.0, 0.5);
        assert!(!a.is_empty());
        assert!(a.iter().all(|(s, _)| (s.x / 0.5).round() as i32 % 2 == 0));
        assert_eq!(a, b, "small pans keep the same dashes");
    }
}
