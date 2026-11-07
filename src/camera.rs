//! Orbit camera (Z-up, like SketchUp) with perspective / parallel projection.

use eframe::egui::{Pos2, Rect, Vec2};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub perspective: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::new(1.5, 1.5, 0.8),
            distance: 14.0,
            yaw: -60f32.to_radians(),
            pitch: 28f32.to_radians(),
            fov_y: 35f32.to_radians(),
            perspective: true,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let dir = Vec3::new(self.pitch.cos() * self.yaw.cos(), self.pitch.cos() * self.yaw.sin(), self.pitch.sin());
        self.target + dir * self.distance
    }

    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Z).normalize_or(Vec3::X)
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward())
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Z)
    }

    fn far(&self) -> f32 {
        self.distance * 100.0 + 200.0
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        let far = self.far();
        if self.perspective {
            let near = (self.distance * 0.01).max(0.002);
            Mat4::perspective_rh_gl(self.fov_y, aspect, near, far)
        } else {
            let h = self.distance * (self.fov_y * 0.5).tan();
            let w = h * aspect;
            Mat4::orthographic_rh_gl(-w, w, -h, h, -far, far)
        }
    }

    pub fn view_proj(&self, rect: Rect) -> Mat4 {
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.01);
        self.proj(aspect) * self.view()
    }

    /// World-space ray (origin, normalized direction) through a screen point.
    pub fn ray(&self, rect: Rect, pos: Pos2) -> (Vec3, Vec3) {
        let x = (pos.x - rect.left()) / rect.width() * 2.0 - 1.0;
        let y = 1.0 - (pos.y - rect.top()) / rect.height() * 2.0;
        let inv = self.view_proj(rect).inverse();
        let near = inv.project_point3(Vec3::new(x, y, -1.0));
        let far = inv.project_point3(Vec3::new(x, y, 1.0));
        (near, (far - near).normalize())
    }

    /// Projects a world point to screen space (None when behind the camera).
    pub fn project(&self, rect: Rect, p: Vec3) -> Option<Pos2> {
        let clip = self.view_proj(rect) * p.extend(1.0);
        if clip.w <= 1e-6 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Pos2::new(rect.left() + (ndc.x + 1.0) * 0.5 * rect.width(), rect.top() + (1.0 - ndc.y) * 0.5 * rect.height()))
    }

    pub fn orbit(&mut self, delta: Vec2) {
        self.yaw -= delta.x * 0.008;
        self.pitch = (self.pitch + delta.y * 0.008).clamp(-1.55, 1.55);
    }

    pub fn pan(&mut self, rect: Rect, delta: Vec2) {
        let world_per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / rect.height().max(1.0);
        self.target -= self.right() * delta.x * world_per_px;
        self.target += self.up() * delta.y * world_per_px;
    }

    /// Zooms by `factor` (<1 = closer) towards `anchor` if given.
    pub fn zoom(&mut self, factor: f32, anchor: Option<Vec3>) {
        let new_d = (self.distance * factor).clamp(0.05, 50_000.0);
        let f = new_d / self.distance;
        if let Some(a) = anchor {
            self.target = a + (self.target - a) * f;
        }
        self.distance = new_d;
    }

    pub fn fit(&mut self, lo: Vec3, hi: Vec3) {
        let center = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(0.5);
        self.target = center;
        self.distance = radius / (self.fov_y * 0.5).sin() * 1.15;
    }

    pub fn set_view(&mut self, yaw_deg: f32, pitch_deg: f32) {
        self.yaw = yaw_deg.to_radians();
        self.pitch = pitch_deg.to_radians();
    }

    /// In-between camera for animated scene transitions (`t` in 0..=1):
    /// the orbit angle takes the short way round and zoom eases geometrically.
    pub fn lerp(a: &Camera, b: &Camera, t: f32) -> Camera {
        let tau = std::f32::consts::TAU;
        let mut dyaw = (b.yaw - a.yaw) % tau;
        if dyaw > tau / 2.0 {
            dyaw -= tau;
        } else if dyaw < -tau / 2.0 {
            dyaw += tau;
        }
        Camera {
            target: a.target.lerp(b.target, t),
            distance: (a.distance.ln() + (b.distance.ln() - a.distance.ln()) * t).exp(),
            yaw: a.yaw + dyaw * t,
            pitch: a.pitch + (b.pitch - a.pitch) * t,
            fov_y: a.fov_y + (b.fov_y - a.fov_y) * t,
            perspective: if t < 0.5 { a.perspective } else { b.perspective },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_takes_the_short_way() {
        let a = Camera { yaw: 170f32.to_radians(), distance: 1.0, ..Camera::default() };
        let b = Camera { yaw: -170f32.to_radians(), distance: 100.0, target: Vec3::ONE, ..Camera::default() };
        let m = Camera::lerp(&a, &b, 0.5);
        assert!((m.yaw.to_degrees() - 180.0).abs() < 1e-3, "yaw {}", m.yaw.to_degrees());
        assert!((m.distance - 10.0).abs() < 1e-3);
        let end = Camera::lerp(&a, &b, 1.0);
        assert!(end.eye().distance(b.eye()) < 1e-3 && end.target == b.target);
        assert_eq!(Camera::lerp(&a, &b, 0.0).eye(), a.eye());
    }
}
