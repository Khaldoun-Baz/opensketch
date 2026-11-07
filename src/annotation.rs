//! Dimensions and text (SketchUp's Dimension and Text tools).
//!
//! Annotations live in world space next to the model and are drawn on top of
//! it. Labels are computed when drawn, so they follow the current units; a
//! custom text may contain `<>`, which stands for the measured value.

use crate::units::Units;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Annotation {
    /// Linear dimension between `a` and `b`, drawn displaced by `offset`.
    Dimension {
        a: Vec3,
        b: Vec3,
        offset: Vec3,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// Radius (arcs) or diameter (circles) dimension, label at `at`.
    Radial {
        center: Vec3,
        radius: f32,
        normal: Vec3,
        diameter: bool,
        at: Vec3,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// Text with a leader line pointing at `anchor`.
    Text { anchor: Vec3, at: Vec3, text: String },
    /// Text fixed on screen; `pos` is a fraction of the viewport (0..1).
    ScreenText { pos: [f32; 2], text: String },
}

impl Annotation {
    /// The value a dimension measures (meters), if it is one.
    pub fn measured(&self) -> Option<f32> {
        match self {
            Annotation::Dimension { a, b, .. } => Some(a.distance(*b)),
            Annotation::Radial { radius, diameter, .. } => Some(if *diameter { radius * 2.0 } else { *radius }),
            _ => None,
        }
    }

    /// The text shown, with `<>` replaced by the measured value.
    pub fn label(&self, units: Units) -> String {
        let value = match self {
            Annotation::Radial { diameter: true, .. } => self.measured().map(|v| format!("Ø {}", units.format(v))),
            Annotation::Radial { .. } => self.measured().map(|v| format!("R {}", units.format(v))),
            _ => self.measured().map(|v| units.format(v)),
        };
        match (self, value) {
            (Annotation::Text { text, .. } | Annotation::ScreenText { text, .. }, _) => text.clone(),
            (Annotation::Dimension { text: Some(t), .. } | Annotation::Radial { text: Some(t), .. }, Some(v)) => t.replace("<>", &v),
            (_, Some(v)) => v,
            _ => String::new(),
        }
    }

    /// Editable text: the custom text of a dimension, or the text itself.
    pub fn text_mut(&mut self) -> Option<&mut Option<String>> {
        match self {
            Annotation::Dimension { text, .. } | Annotation::Radial { text, .. } => Some(text),
            _ => None,
        }
    }

    pub fn transformed(&self, m: Mat4) -> Annotation {
        let p = |v: &Vec3| m.transform_point3(*v);
        match self {
            Annotation::Dimension { a, b, offset, text } => {
                let (na, nb) = (p(a), p(b));
                Annotation::Dimension { a: na, b: nb, offset: p(&(*a + *offset)) - na, text: text.clone() }
            }
            Annotation::Radial { center, radius, normal, diameter, at, text } => {
                let scale = m.transform_vector3(normal.any_orthonormal_vector()).length();
                Annotation::Radial {
                    center: p(center),
                    radius: radius * scale,
                    normal: m.transform_vector3(*normal).normalize_or(*normal),
                    diameter: *diameter,
                    at: p(at),
                    text: text.clone(),
                }
            }
            Annotation::Text { anchor, at, text } => Annotation::Text { anchor: p(anchor), at: p(at), text: text.clone() },
            s @ Annotation::ScreenText { .. } => s.clone(),
        }
    }

    /// World-space line segments to draw (screen text has none).
    pub fn segments(&self) -> Vec<(Vec3, Vec3)> {
        match self {
            Annotation::Dimension { a, b, offset, .. } => {
                let (pa, pb) = (*a + *offset, *b + *offset);
                // Extension lines overshoot the dimension line a little.
                let over = offset.normalize_or_zero() * (a.distance(*b) * 0.04).min(offset.length() * 0.3 + 0.05);
                vec![(*a, pa + over), (*b, pb + over), (pa, pb)]
            }
            Annotation::Radial { at, .. } => {
                let (rim, start) = self.radial_points().unwrap_or((*at, *at));
                vec![(start, rim), (rim, *at)]
            }
            Annotation::Text { anchor, at, .. } => vec![(*anchor, *at)],
            Annotation::ScreenText { .. } => vec![],
        }
    }

    /// For radial dimensions: (point on the circle towards the label, where the
    /// dimension line starts: the center, or the far rim for a diameter).
    pub fn radial_points(&self) -> Option<(Vec3, Vec3)> {
        let Annotation::Radial { center, radius, normal, diameter, at, .. } = self else { return None };
        let v = *at - *center;
        let mut dir = (v - *normal * v.dot(*normal)).normalize_or_zero();
        if dir == Vec3::ZERO {
            dir = normal.any_orthonormal_vector();
        }
        let rim = *center + dir * *radius;
        Some((rim, if *diameter { *center - dir * *radius } else { *center }))
    }

    /// Where the label is drawn (world space; `None` for screen text).
    pub fn label_point(&self) -> Option<Vec3> {
        match self {
            Annotation::Dimension { a, b, offset, .. } => Some((*a + *b) * 0.5 + *offset),
            Annotation::Radial { at, .. } | Annotation::Text { at, .. } => Some(*at),
            Annotation::ScreenText { .. } => None,
        }
    }
}

/// Center of the circle through three points (None if they are collinear).
pub fn circumcenter(a: Vec3, b: Vec3, c: Vec3) -> Option<Vec3> {
    let (ab, ac) = (b - a, c - a);
    let n = ab.cross(ac);
    let nn = n.length_squared();
    if nn < 1e-12 {
        return None;
    }
    Some(a + (n.cross(ab) * ac.length_squared() + ac.cross(n) * ab.length_squared()) / (2.0 * nn))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_follow_units_and_custom_text() {
        let mut d = Annotation::Dimension { a: Vec3::ZERO, b: Vec3::new(3.0, 4.0, 0.0), offset: Vec3::Z, text: None };
        assert_eq!(d.label(Units::Meters), "5.000 m");
        assert_eq!(d.label(Units::Millimeters), "5000 mm");
        *d.text_mut().unwrap() = Some("Width <>".into());
        assert_eq!(d.label(Units::Centimeters), "Width 500.0 cm");
        let r = Annotation::Radial { center: Vec3::ZERO, radius: 1.5, normal: Vec3::Z, diameter: true, at: Vec3::X * 3.0, text: None };
        assert_eq!(r.label(Units::Meters), "Ø 3.000 m");
        let (rim, start) = r.radial_points().unwrap();
        assert!(rim.distance(Vec3::X * 1.5) < 1e-5 && start.distance(-Vec3::X * 1.5) < 1e-5);
        let t = Annotation::Text { anchor: Vec3::ZERO, at: Vec3::ONE, text: "Door".into() };
        assert_eq!(t.label(Units::Meters), "Door");
    }

    #[test]
    fn transforms_move_and_scale() {
        let m = Mat4::from_translation(Vec3::X) * Mat4::from_scale(Vec3::splat(2.0));
        let d = Annotation::Dimension { a: Vec3::ZERO, b: Vec3::Y, offset: Vec3::X * 0.5, text: None }.transformed(m);
        assert!((d.measured().unwrap() - 2.0).abs() < 1e-5);
        assert_eq!(d.label_point().unwrap(), Vec3::new(2.0, 1.0, 0.0));
        let r = Annotation::Radial { center: Vec3::ZERO, radius: 1.0, normal: Vec3::Z, diameter: false, at: Vec3::X * 2.0, text: None }
            .transformed(m);
        assert!((r.measured().unwrap() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn circle_center() {
        let c = Vec3::new(1.0, 2.0, 3.0);
        let p = |t: f32| c + Vec3::new(t.cos(), 0.0, t.sin()) * 2.0;
        let got = circumcenter(p(0.1), p(1.3), p(2.9)).unwrap();
        assert!(got.distance(c) < 1e-4);
        assert!(circumcenter(Vec3::ZERO, Vec3::X, Vec3::X * 2.0).is_none());
    }
}
