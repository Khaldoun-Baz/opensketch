//! Sun position for shadows (like SketchUp's Shadows panel): time of day,
//! date and latitude give the sun direction; north is the green axis rotated
//! by the north angle.

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sun {
    /// Shadows are shown.
    pub enabled: bool,
    /// Day of the year, 1..=365.
    pub day: u32,
    /// Local solar time in hours (12 = noon).
    pub hour: f32,
    /// Degrees north (negative = south).
    pub latitude: f32,
    /// Clockwise angle (degrees) from the green axis to true north.
    pub north: f32,
}

impl Default for Sun {
    fn default() -> Self {
        // SketchUp's default: November 8, 1:30 PM.
        Self { enabled: false, day: 312, hour: 13.5, latitude: 40.0, north: 0.0 }
    }
}

const MONTHS: [(&str, u32); 12] = [
    ("Jan", 31),
    ("Feb", 28),
    ("Mar", 31),
    ("Apr", 30),
    ("May", 31),
    ("Jun", 30),
    ("Jul", 31),
    ("Aug", 31),
    ("Sep", 30),
    ("Oct", 31),
    ("Nov", 30),
    ("Dec", 31),
];

impl Sun {
    pub fn is_default(&self) -> bool {
        *self == Sun::default()
    }

    /// Unit vector pointing towards the sun (Z up).
    pub fn direction(&self) -> Vec3 {
        let decl = 23.44f32.to_radians() * ((284.0 + self.day as f32) / 365.0 * std::f32::consts::TAU).sin();
        let h = (15.0 * (self.hour - 12.0)).to_radians();
        let lat = self.latitude.to_radians();
        let east = -decl.cos() * h.sin();
        let north = lat.cos() * decl.sin() - lat.sin() * decl.cos() * h.cos();
        let up = lat.sin() * decl.sin() + lat.cos() * decl.cos() * h.cos();
        // North is +Y rotated clockwise (seen from above) by the north angle.
        let a = self.north.to_radians();
        let (n_dir, e_dir) = (Vec3::new(a.sin(), a.cos(), 0.0), Vec3::new(a.cos(), -a.sin(), 0.0));
        (n_dir * north + e_dir * east + Vec3::Z * up).normalize()
    }

    pub fn above_horizon(&self) -> bool {
        self.direction().z > 0.02
    }

    pub fn date_label(day: u32) -> String {
        let mut d = day.clamp(1, 365);
        for (name, len) in MONTHS {
            if d <= len {
                return format!("{name} {d}");
            }
            d -= len;
        }
        "Dec 31".into()
    }

    pub fn time_label(hour: f32) -> String {
        let mins = (hour * 60.0).round() as i32;
        format!("{:02}:{:02}", mins / 60, mins % 60)
    }
}

/// Orthographic light matrix covering the scene box `lo..hi` and the
/// shadows it throws onto the ground plane at `ground_z`. Returns the matrix
/// and the ground area (min, max xy corners) that can receive shadows.
pub fn light_matrix(sun: Vec3, lo: Vec3, hi: Vec3, ground_z: f32) -> (Mat4, Vec3, Vec3) {
    let corners = crate::document::box_corners(lo, hi);
    let mut pts: Vec<Vec3> = corners.to_vec();
    let mut glo = Vec3::new(lo.x, lo.y, ground_z);
    let mut ghi = Vec3::new(hi.x, hi.y, ground_z);
    if sun.z > 0.02 {
        for c in corners {
            let q = c - sun * ((c.z - ground_z) / sun.z);
            // Very low sun: cap the shadow length.
            let span = (hi - lo).length().max(1.0) * 6.0;
            let q = Vec3::new(q.x.clamp(lo.x - span, hi.x + span), q.y.clamp(lo.y - span, hi.y + span), ground_z);
            glo = glo.min(q);
            ghi = ghi.max(q);
        }
    }
    pts.extend([glo, Vec3::new(ghi.x, glo.y, ground_z), ghi, Vec3::new(glo.x, ghi.y, ground_z)]);
    let center = pts.iter().fold(Vec3::ZERO, |a, p| a + *p) / pts.len() as f32;
    let reach = pts.iter().map(|p| p.distance(center)).fold(1.0f32, f32::max);
    let up = if sun.z.abs() > 0.99 { Vec3::Y } else { Vec3::Z };
    let view = Mat4::look_at_rh(center + sun * reach * 2.0, center, up);
    let (mut a, mut b) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in &pts {
        let v = view.transform_point3(*p);
        a = a.min(v);
        b = b.max(v);
    }
    let m = reach * 0.02;
    let proj = Mat4::orthographic_rh_gl(a.x - m, b.x + m, a.y - m, b.y + m, -b.z - m, -a.z + m);
    (proj * view, glo, ghi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noon_and_morning() {
        let equinox_noon = Sun { day: 80, hour: 12.0, latitude: 0.0, ..Sun::default() };
        assert!(equinox_noon.direction().z > 0.99, "overhead at the equator");
        let summer = Sun { day: 172, hour: 12.0, latitude: 40.0, ..Sun::default() };
        let elev = summer.direction().z.asin().to_degrees();
        assert!((elev - 73.4).abs() < 0.6, "elevation {elev}");
        assert!(summer.direction().y < 0.0, "the noon sun is in the south");
        let morning = Sun { hour: 8.0, ..summer };
        assert!(morning.direction().x > 0.3, "morning sun in the east");
        let night = Sun { hour: 0.0, ..summer };
        assert!(!night.above_horizon());
        // Turning north by 90° moves the noon sun from -Y to -X.
        let turned = Sun { north: 90.0, ..summer };
        assert!(turned.direction().x < -0.2 && turned.direction().y.abs() < 1e-3);
    }

    #[test]
    fn labels() {
        assert_eq!(Sun::date_label(1), "Jan 1");
        assert_eq!(Sun::date_label(312), "Nov 8");
        assert_eq!(Sun::date_label(365), "Dec 31");
        assert_eq!(Sun::time_label(13.5), "13:30");
    }

    #[test]
    fn light_frustum_covers_scene_and_shadows() {
        let sun = Sun { day: 172, hour: 16.0, ..Sun::default() }.direction();
        let (lo, hi) = (Vec3::ZERO, Vec3::new(2.0, 1.0, 3.0));
        let (m, glo, ghi) = light_matrix(sun, lo, hi, 0.0);
        let inside = |p: Vec3| {
            let c = m.project_point3(p);
            c.abs().max_element() <= 1.0001
        };
        for c in crate::document::box_corners(lo, hi) {
            assert!(inside(c));
            let shadow = c - sun * (c.z / sun.z);
            assert!(inside(shadow));
            assert!(shadow.x >= glo.x - 1e-4 && shadow.y <= ghi.y + 1e-4);
        }
    }
}
