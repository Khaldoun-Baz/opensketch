//! Application shell: panels, tools, the Measurements box (VCB) and the 3D viewport.

use crate::ai;
use crate::annotation::{circumcenter, Annotation};
use crate::camera::Camera;
use crate::document::{box_corners, glue_frame, DefId, DefKind, Document, Scene, SceneStyle, ROOT};
use crate::guide::{self, Guide};
use crate::inference::{Owner, Picker, Snap, SnapKind, AXES};
use crate::io;
use crate::model::{closest_on_line_to_ray, drawing_axes, offset_loop, point_seg, ray_plane, Id, Material, Model, EPS};
use crate::render::{push_face_vertex, push_line, push_vertex, DrawData, Renderer, ShadowData, TexBatch, FLOATS};
use crate::solid::{self, BoolOp};
use crate::sun::{self, Sun};
use crate::texture::{self, Pattern, Pixels, Texture};
use crate::units::Units;
use eframe::egui::{self, Align2, Color32, FontId, Key, Modifiers, PointerButton, Pos2, Rect, RichText, Sense, Shape, Stroke, Vec2};
use eframe::glow;
use glam::{Mat3, Mat4, Vec3};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Color of the "special" inferences (parallel, perpendicular, half circle, square).
const MAGENTA: Color32 = Color32::from_rgb(200, 0, 160);

const NOT_SOLID: &str = "That is not a solid: every edge must bound exactly two faces.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Select,
    Eraser,
    Paint,
    Line,
    Arc,
    CenterArc,
    Pie,
    Rectangle,
    Circle,
    Polygon,
    PushPull,
    FollowMe,
    Move,
    Rotate,
    Scale,
    Offset,
    Tape,
    Protractor,
    Dimension,
    Text,
    Orbit,
    Pan,
    OuterShell,
    SolidIntersect,
    Union,
    Subtract,
    Trim,
    Split,
}

const TOOL_GROUPS: &[(&str, &[Tool])] = &[
    ("PRINCIPAL", &[Tool::Select, Tool::Eraser, Tool::Paint]),
    ("DRAW", &[Tool::Line, Tool::Arc, Tool::CenterArc, Tool::Pie, Tool::Rectangle, Tool::Circle, Tool::Polygon]),
    ("EDIT", &[Tool::PushPull, Tool::FollowMe, Tool::Move, Tool::Rotate, Tool::Scale, Tool::Offset]),
    ("CONSTRUCTION", &[Tool::Tape, Tool::Protractor, Tool::Dimension, Tool::Text]),
    ("SOLID TOOLS", &[Tool::OuterShell, Tool::SolidIntersect, Tool::Union, Tool::Subtract, Tool::Trim, Tool::Split]),
    ("CAMERA", &[Tool::Orbit, Tool::Pan]),
];

impl Tool {
    fn label(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Eraser => "Eraser",
            Tool::Paint => "Paint Bucket",
            Tool::Line => "Line",
            Tool::Arc => "2-Point Arc",
            Tool::CenterArc => "Arc",
            Tool::Pie => "Pie",
            Tool::Rectangle => "Rectangle",
            Tool::Circle => "Circle",
            Tool::Polygon => "Polygon",
            Tool::PushPull => "Push/Pull",
            Tool::FollowMe => "Follow Me",
            Tool::Move => "Move",
            Tool::Rotate => "Rotate",
            Tool::Scale => "Scale",
            Tool::Offset => "Offset",
            Tool::Tape => "Tape Measure",
            Tool::Protractor => "Protractor",
            Tool::Dimension => "Dimension",
            Tool::Text => "Text",
            Tool::Orbit => "Orbit",
            Tool::Pan => "Pan",
            Tool::OuterShell => "Outer Shell",
            Tool::SolidIntersect => "Intersect",
            Tool::Union => "Union",
            Tool::Subtract => "Subtract",
            Tool::Trim => "Trim",
            Tool::Split => "Split",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            Tool::Select => "⬉",
            Tool::Eraser => "🗑",
            Tool::Paint => "🎨",
            Tool::Line => "✏",
            Tool::Arc => "◠",
            Tool::CenterArc => "◜",
            Tool::Pie => "◔",
            Tool::Rectangle => "⬜",
            Tool::Circle => "⭕",
            Tool::Polygon => "⬟",
            Tool::PushPull => "⬆",
            Tool::FollowMe => "➰",
            Tool::Move => "✥",
            Tool::Rotate => "⟲",
            Tool::Scale => "⤢",
            Tool::Offset => "▣",
            Tool::Tape => "📏",
            Tool::Protractor => "📐",
            Tool::Dimension => "↔",
            Tool::Text => "💬",
            Tool::Orbit => "🌐",
            Tool::Pan => "✋",
            Tool::OuterShell => "◎",
            Tool::SolidIntersect => "◑",
            Tool::Union => "⊞",
            Tool::Subtract => "⊟",
            Tool::Trim => "✂",
            Tool::Split => "⊗",
        }
    }
    fn shortcut(self) -> &'static str {
        match self {
            Tool::Select => "Space",
            Tool::Eraser => "E",
            Tool::Paint => "B",
            Tool::Line => "L",
            Tool::Arc => "A",
            Tool::CenterArc | Tool::Pie | Tool::FollowMe | Tool::Protractor | Tool::Dimension | Tool::Text => "—",
            Tool::Rectangle => "R",
            Tool::Circle => "C",
            Tool::Polygon => "N",
            Tool::PushPull => "P",
            Tool::Move => "M",
            Tool::Rotate => "Q",
            Tool::Scale => "S",
            Tool::Offset => "F",
            Tool::Tape => "T",
            Tool::Orbit => "O",
            Tool::Pan => "H",
            Tool::OuterShell | Tool::SolidIntersect | Tool::Union | Tool::Subtract | Tool::Trim | Tool::Split => "—",
        }
    }
    fn is_solid_tool(self) -> bool {
        matches!(self, Tool::OuterShell | Tool::SolidIntersect | Tool::Union | Tool::Subtract | Tool::Trim | Tool::Split)
    }
}

/// Geometry and instances an edit tool acts on.
#[derive(Clone, Default)]
struct Targets {
    verts: HashSet<Id>,
    faces: Vec<Id>,
    edges: Vec<Id>,
    insts: Vec<Id>,
    guides: Vec<usize>,
    annots: Vec<usize>,
}

#[derive(Clone)]
enum State {
    Idle,
    Line {
        start: Vec3,
    },
    Rect {
        p0: Vec3,
        p1: Vec3,
        n: Vec3,
        u: Vec3,
        v: Vec3,
        on_face: bool,
    },
    Circle {
        center: Vec3,
        edge: Vec3,
        n: Vec3,
        on_face: bool,
    },
    /// 2-point arc: start placed, choosing the end.
    ArcEnd {
        p0: Vec3,
        p1: Vec3,
        n: Vec3,
    },
    /// 2-point arc: chord placed, choosing the bulge.
    ArcBulge {
        p0: Vec3,
        p1: Vec3,
        n: Vec3,
        bulge: f32,
    },
    /// Center arc / pie: center placed, choosing the start (radius).
    ArcStart {
        center: Vec3,
        edge: Vec3,
        n: Vec3,
        on_face: bool,
    },
    /// Center arc / pie: choosing the sweep angle.
    ArcSweep {
        center: Vec3,
        start: Vec3,
        n: Vec3,
        on_face: bool,
        angle: f32,
    },
    PushPull {
        face: Id,
        start: Vec3,
        n: Vec3,
        dist: f32,
    },
    Move {
        t: Targets,
        base: Vec3,
        delta: Vec3,
        copy: bool,
    },
    RotateCenter {
        t: Targets,
        center: Vec3,
        n: Vec3,
    },
    RotateAngle {
        t: Targets,
        center: Vec3,
        n: Vec3,
        ref_dir: Vec3,
        angle: f32,
    },
    Scale {
        t: Targets,
        center: Vec3,
        ref_dist: f32,
        factor: Vec3,
    },
    Offset {
        face: Id,
        dist: f32,
    },
    /// Tape Measure. `dir` is set when measuring from an edge or guide line
    /// (makes a parallel guide); `end` is where the guide will go.
    Tape {
        start: Vec3,
        dir: Option<Vec3>,
        end: Vec3,
    },
    /// Dimension: first point placed, choosing the second.
    DimSecond {
        a: Vec3,
    },
    /// Dimension: both points placed, dragging the dimension line out.
    DimOffset {
        a: Vec3,
        b: Vec3,
        offset: Vec3,
    },
    /// Radius/diameter dimension of an arc or circle: placing the label.
    DimRadial {
        center: Vec3,
        radius: f32,
        normal: Vec3,
        diameter: bool,
        at: Vec3,
    },
    /// Text: anchor picked, placing the text.
    TextPlace {
        anchor: Vec3,
        text: String,
        at: Vec3,
    },
    /// Protractor: center placed, choosing the baseline.
    ProtBase {
        center: Vec3,
        n: Vec3,
    },
    /// Protractor: choosing the guide's angle from the baseline.
    ProtAngle {
        center: Vec3,
        n: Vec3,
        ref_dir: Vec3,
        angle: f32,
    },
    BoxSelect {
        start: Pos2,
    },
    Erase,
    Place {
        def: DefId,
    },
    /// Solid tools: the first solid has been picked.
    SolidFirst {
        first: Id,
    },
}

/// A full undo point: the document plus the open edit context.
#[derive(Clone)]
struct Snapshot {
    doc: Document,
    path: Vec<Id>,
}

#[derive(Clone)]
enum XOp {
    Move(Vec3),
    Rotate { center: Vec3, n: Vec3, angle: f32 },
}

impl XOp {
    fn matrix(&self) -> Mat4 {
        match self {
            XOp::Move(d) => Mat4::from_translation(*d),
            XOp::Rotate { center, n, angle } => about(*center, Mat4::from_axis_angle(*n, *angle)),
        }
    }
}

/// The last completed operation, kept so a value typed afterwards can adjust it
/// (SketchUp lets you re-type the distance right after a Push/Pull or Move).
#[derive(Clone)]
enum LastOp {
    PushPull { before: Snapshot, face: Id, sign: f32 },
    Transform { before: Snapshot, t: Targets, op: XOp, copy: bool },
    Offset { before: Snapshot, face: Id, sign: f32 },
}

#[derive(Default, Clone)]
struct Selection {
    faces: BTreeSet<Id>,
    edges: BTreeSet<Id>,
    insts: BTreeSet<Id>,
    guides: BTreeSet<usize>,
    annots: BTreeSet<usize>,
}

impl Selection {
    fn is_empty(&self) -> bool {
        self.faces.is_empty() && self.edges.is_empty() && self.insts.is_empty() && self.guides.is_empty() && self.annots.is_empty()
    }
    fn clear(&mut self) {
        self.faces.clear();
        self.edges.clear();
        self.insts.clear();
        self.guides.clear();
        self.annots.clear();
    }
    fn guides(&self) -> Vec<usize> {
        self.guides.iter().copied().collect()
    }
    fn annots(&self) -> Vec<usize> {
        self.annots.iter().copied().collect()
    }
    fn faces(&self) -> Vec<Id> {
        self.faces.iter().copied().collect()
    }
    fn edges(&self) -> Vec<Id> {
        self.edges.iter().copied().collect()
    }
    fn insts(&self) -> Vec<Id> {
        self.insts.iter().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// Scene traversal (rendering + picking through groups/components)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SceneOut {
    tris: Vec<f32>,
    lines: Vec<f32>,
    pick: Vec<(Owner, [Vec3; 3])>,
    points: Vec<Vec3>,
    segs: Vec<(Vec3, Vec3)>,
    guide_pts: Vec<Vec3>,
    /// Textured faces, batched per material.
    tex: std::collections::BTreeMap<usize, Vec<f32>>,
}

/// Screen-space segments, arrow tips (tip, from) and label center of an annotation.
type ScreenParts = (Vec<(Pos2, Pos2)>, Vec<(Pos2, Pos2)>, Option<Pos2>);

/// SketchUp's back-face color, shown on the back of unpainted faces.
const BACK_COLOR: [f32; 3] = [0.62, 0.66, 0.78];

/// Where a face's triangles go: the plain batch or its material's texture batch.
fn tri_buf(out: &mut SceneOut, tex: Option<(usize, [f32; 2])>) -> &mut Vec<f32> {
    match tex {
        Some((i, _)) => out.tex.entry(i).or_default(),
        None => &mut out.tris,
    }
}

struct WalkOpts<'a> {
    active: &'a Model,
    /// World transform previewed on selected instances (Move/Rotate/Scale).
    op: Option<Mat4>,
    copy: bool,
    render: bool,
}

#[derive(Clone, Copy)]
struct Node {
    def: DefId,
    xf: Mat4,
    /// Some(k) while on the path to the edit context (k = depth).
    level: Option<usize>,
    /// Child instance of the edit context this geometry belongs to.
    owner: Option<Id>,
    mat: Option<usize>,
    faded: bool,
    selected: bool,
    hovered: bool,
    /// Effective tag (an instance's own, else its parent's), for Color by Tag.
    tag: Option<u32>,
}

/// Animated camera move to a scene.
struct CamAnim {
    from: Camera,
    to: Camera,
    t0: Option<f64>,
}

const BOX_EDGES: [(usize, usize); 12] = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn about(c: Vec3, m: Mat4) -> Mat4 {
    Mat4::from_translation(c) * m * Mat4::from_translation(-c)
}

fn grow(bb: Option<(Vec3, Vec3)>, p: Vec3) -> Option<(Vec3, Vec3)> {
    Some(match bb {
        Some((lo, hi)) => (lo.min(p), hi.max(p)),
        None => (p, p),
    })
}

fn axis_color(i: usize) -> Color32 {
    match i {
        0 => Color32::from_rgb(220, 40, 40),
        1 => Color32::from_rgb(30, 160, 40),
        _ => Color32::from_rgb(40, 70, 230),
    }
}

fn axis_rgb(i: usize) -> [f32; 3] {
    let c = axis_color(i);
    [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0]
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn to_color32(c: [f32; 3]) -> Color32 {
    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

fn in_plane(v: Vec3, n: Vec3) -> Vec3 {
    v - n * v.dot(n)
}

/// Axis index if the direction is (nearly) parallel to one of `axes`.
fn parallel_axis(d: Vec3, axes: &[Vec3; 3]) -> Option<usize> {
    let d = d.normalize_or_zero();
    (0..3).find(|&i| d.dot(axes[i]).abs() > 0.9999)
}

/// Points of a 2-point arc from p0 to p1 bulging `bulge` (sagitta) to the
/// left of the chord around normal `n`.
fn arc_points_2pt(p0: Vec3, p1: Vec3, n: Vec3, bulge: f32, segs: usize) -> Vec<Vec3> {
    let chord = p1 - p0;
    let l = chord.length();
    if bulge.abs() < EPS || l < EPS {
        return vec![p0, p1];
    }
    let m = n.cross(chord / l) * bulge.signum();
    let s = bulge.abs();
    let r = (s * s + l * l / 4.0) / (2.0 * s);
    let mid = (p0 + p1) * 0.5;
    let center = mid + m * (s - r);
    let half = ((l / 2.0) / r).clamp(-1.0, 1.0).asin();
    let sweep = if s > r { std::f32::consts::TAU - 2.0 * half } else { 2.0 * half };
    let apex = mid + m * s;
    let v0 = p0 - center;
    let rot = |a: f32| center + glam::Quat::from_axis_angle(n, a) * v0;
    let sign = if rot(sweep / 2.0).distance(apex) < rot(-sweep / 2.0).distance(apex) { 1.0 } else { -1.0 };
    let mut pts: Vec<Vec3> = (0..=segs).map(|i| rot(sign * sweep * i as f32 / segs as f32)).collect();
    *pts.last_mut().unwrap() = p1;
    pts
}

/// Bulges that make a 2-point arc from `p0` to `p1` (in the plane with normal
/// `n`) continue smoothly from an edge that ends at `p0`, or into one that
/// starts at `p1`.
fn tangent_bulges(model: &Model, p0: Vec3, p1: Vec3, n: Vec3) -> Vec<f32> {
    let chord = p1 - p0;
    let l = chord.length();
    if l < EPS {
        return Vec::new();
    }
    let c = chord / l;
    let m = n.cross(c);
    let mut out = Vec::new();
    for (p, at_start) in [(p0, true), (p1, false)] {
        let Some(v) = model.find_vertex(p) else { continue };
        for w in model.neighbors(v) {
            // Direction of travel through the vertex: into p0, or out of p1.
            let full = if at_start { p - model.pos(w) } else { model.pos(w) - p };
            let t = in_plane(full, n);
            if t.length() < full.length() * 0.9 || t.length() < EPS {
                continue;
            }
            let t = t.normalize();
            let a = if at_start { t.dot(m).atan2(t.dot(c)) } else { (-t.dot(m)).atan2(t.dot(c)) };
            if a.abs() > 0.02 && a.abs() < 3.0 {
                out.push(l / 2.0 * (a / 2.0).tan());
            }
        }
    }
    out
}

/// Points of an arc around `center` starting at `start`, sweeping `angle` around `n`.
fn arc_points_center(center: Vec3, start: Vec3, n: Vec3, angle: f32, segs: usize) -> Vec<Vec3> {
    let v0 = start - center;
    (0..=segs).map(|i| center + glam::Quat::from_axis_angle(n, angle * i as f32 / segs as f32) * v0).collect()
}

fn circle_points(center: Vec3, n: Vec3, radius_vec: Vec3, segs: usize) -> Vec<Vec3> {
    let r = radius_vec.length();
    let u = radius_vec.normalize_or_zero();
    let v = n.cross(u);
    (0..segs)
        .map(|i| {
            let t = i as f32 / segs as f32 * std::f32::consts::TAU;
            center + (u * t.cos() + v * t.sin()) * r
        })
        .collect()
}

/// Applies a world matrix to the loose geometry of `t` (or to a copy of it).
fn apply_to_model(m: &mut Model, t: &Targets, mat: Mat4, copy: bool) -> Option<(Vec<Id>, Vec<Id>)> {
    if copy {
        let (nv, nf, ne) = m.copy_geometry(&t.faces, &t.edges);
        m.apply_matrix(&nv, mat);
        Some((nf, ne))
    } else {
        m.apply_matrix(&t.verts, mat);
        m.autofold(&t.verts);
        None
    }
}

fn segs_intersect(a: Pos2, b: Pos2, c: Pos2, d: Pos2) -> bool {
    let cross = |o: Pos2, p: Pos2, q: Pos2| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    let d1 = cross(c, d, a);
    let d2 = cross(c, d, b);
    let d3 = cross(a, b, c);
    let d4 = cross(a, b, d);
    (d1 * d2 < 0.0) && (d3 * d4 < 0.0)
}

fn seg_hits_rect(a: Pos2, b: Pos2, r: Rect) -> bool {
    if r.contains(a) || r.contains(b) {
        return true;
    }
    let c = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()];
    (0..4).any(|i| segs_intersect(a, b, c[i], c[(i + 1) % 4]))
}

fn stroke(width: f32, color: Color32) -> Stroke {
    Stroke::new(width, color)
}

// ---------------------------------------------------------------------------
// AI Render window state
// ---------------------------------------------------------------------------

/// A clean frame of the viewport, filled in by the paint callback.
type Capture = Arc<Mutex<Option<image::RgbaImage>>>;

/// A render running on a background thread (its answer, and when it started).
type RenderJob = (std::sync::mpsc::Receiver<Result<Vec<u8>, String>>, std::time::Instant);

struct AiUi {
    open: bool,
    settings: ai::Settings,
    /// What is on disk, to write the settings file only when something changed.
    saved: ai::Settings,
    description: String,
    show_key: bool,
    /// Set while the viewport is being captured for a render.
    grab: Option<Capture>,
    /// The render in progress and when it started.
    job: Option<RenderJob>,
    /// The picture that was sent.
    source: Option<egui::TextureHandle>,
    result: Option<(egui::TextureHandle, image::RgbaImage)>,
    show_source: bool,
    error: Option<String>,
    note: Option<String>,
}

impl AiUi {
    fn new() -> Self {
        let settings = ai::settings_path().map(|p| ai::Settings::load_from(&p)).unwrap_or_default();
        Self {
            open: false,
            saved: settings.clone(),
            settings,
            description: String::new(),
            show_key: false,
            grab: None,
            job: None,
            source: None,
            result: None,
            show_source: false,
            error: None,
            note: None,
        }
    }

    /// The capture a paint callback should fill in, if one is wanted.
    fn pending_capture(&self) -> Option<Capture> {
        self.grab.clone().filter(|g| g.lock().unwrap().is_none())
    }

    fn save_settings(&mut self) {
        if self.settings != self.saved {
            if let Some(path) = ai::settings_path() {
                if self.settings.save_to(&path).is_ok() {
                    self.saved = self.settings.clone();
                }
            }
        }
    }
}

fn color_image(img: &image::RgbaImage) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw())
}

// ---------------------------------------------------------------------------
// Scripted input (self-test)
// ---------------------------------------------------------------------------

/// One line of an input script. Positions are window points.
#[derive(Clone, Debug)]
enum Step {
    Move(Pos2),
    Button(bool),
    Text(String),
    Key(Key, Modifiers),
    Wait(u32),
    /// Save a picture of the whole window.
    Shot(PathBuf),
    /// Write the model's counts and the tool state to a text file.
    Dump(PathBuf),
    Quit,
}

/// Plays mouse and keyboard input from a file, one step per frame, so tool
/// handling can be tested without a person at the mouse:
///
/// ```text
/// text r            # press R (Rectangle)
/// move 500 400
/// click             # = down, up
/// drag 700 500      # = down, move there in steps, up
/// key enter         # enter, escape, delete, up, down, left, right (+ctrl / +shift)
/// wait 5            # frames
/// shot out.png
/// dump out.txt
/// quit
/// ```
struct Script {
    steps: std::collections::VecDeque<Step>,
    pos: Pos2,
    wait: u32,
    /// A step that needs the app (shot, dump, quit), run from `update`.
    pending: Option<Step>,
    shot: Option<PathBuf>,
}

impl Script {
    fn load(path: &std::path::Path) -> Option<Script> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut steps = std::collections::VecDeque::new();
        let mut pos = Pos2::new(0.0, 0.0);
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let (cmd, arg) = line.split_once(' ').map(|(c, a)| (c, a.trim())).unwrap_or((line, ""));
            let xy = || {
                let mut it = arg.split_whitespace().filter_map(|v| v.parse::<f32>().ok());
                Some(Pos2::new(it.next()?, it.next()?))
            };
            match cmd {
                "move" => {
                    pos = xy()?;
                    steps.push_back(Step::Move(pos));
                }
                "down" => steps.push_back(Step::Button(true)),
                "up" => steps.push_back(Step::Button(false)),
                "click" => steps.extend([Step::Button(true), Step::Button(false)]),
                "drag" => {
                    let to = xy()?;
                    steps.push_back(Step::Button(true));
                    for k in 1..=4 {
                        steps.push_back(Step::Move(pos + (to - pos) * (k as f32 / 4.0)));
                    }
                    steps.push_back(Step::Button(false));
                    pos = to;
                }
                "text" => steps.push_back(Step::Text(arg.to_string())),
                "key" => {
                    let mut mods = Modifiers::NONE;
                    let mut key = None;
                    for part in arg.split('+') {
                        match part.trim() {
                            "ctrl" => mods.ctrl = true,
                            "shift" => mods.shift = true,
                            "enter" => key = Some(Key::Enter),
                            "escape" => key = Some(Key::Escape),
                            "delete" => key = Some(Key::Delete),
                            "up" => key = Some(Key::ArrowUp),
                            "down" => key = Some(Key::ArrowDown),
                            "left" => key = Some(Key::ArrowLeft),
                            "right" => key = Some(Key::ArrowRight),
                            other => key = Key::from_name(other),
                        }
                    }
                    steps.push_back(Step::Key(key?, mods));
                }
                "wait" => steps.push_back(Step::Wait(arg.parse().ok()?)),
                "shot" => steps.push_back(Step::Shot(PathBuf::from(arg))),
                "dump" => steps.push_back(Step::Dump(PathBuf::from(arg))),
                "quit" => steps.push_back(Step::Quit),
                _ => {}
            }
        }
        Some(Script { steps, pos: Pos2::new(0.0, 0.0), wait: 10, pending: None, shot: None })
    }

    /// Adds this frame's scripted input to what the window received.
    fn feed(&mut self, raw: &mut egui::RawInput) {
        if let Some(path) = &self.shot {
            let image = raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            });
            let Some(image) = image else { return };
            if let Some(img) = image::RgbaImage::from_raw(image.width() as u32, image.height() as u32, image.as_raw().to_vec()) {
                let _ = img.save(path);
            }
            self.shot = None;
        }
        if self.wait > 0 {
            self.wait -= 1;
            return;
        }
        if self.pending.is_some() {
            return;
        }
        let modifiers = raw.modifiers;
        match self.steps.pop_front() {
            Some(Step::Move(p)) => {
                self.pos = p;
                raw.events.push(egui::Event::PointerMoved(p));
            }
            Some(Step::Button(pressed)) => {
                raw.events.push(egui::Event::PointerButton { pos: self.pos, button: PointerButton::Primary, pressed, modifiers });
            }
            Some(Step::Text(t)) => raw.events.push(egui::Event::Text(t)),
            Some(Step::Key(key, m)) => {
                raw.modifiers = m;
                raw.events.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: m });
            }
            Some(Step::Wait(n)) => self.wait = n,
            Some(step) => {
                // Let the frame that follows the last input settle first.
                self.wait = 2;
                self.pending = Some(step);
            }
            None => {}
        }
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

pub struct App {
    doc: Document,
    /// Open edit context: nested instance ids from the root.
    path: Vec<Id>,
    ctx_xf: Mat4,
    /// Working mesh of the edit context, in world coordinates.
    model: Model,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    cam: Camera,
    renderer: Option<Arc<Mutex<Renderer>>>,
    tool: Tool,
    state: State,
    sel: Selection,
    vcb: String,
    measure: String,
    status: String,
    units: Units,
    material: Option<usize>,
    xray: bool,
    show_axes: bool,
    lock_axis: Option<usize>,
    shift_lock: Option<usize>,
    snap: Option<Snap>,
    hover_face: Option<Id>,
    hover_edge: Option<Id>,
    hover_inst: Option<Id>,
    last_op: Option<LastOp>,
    circle_segments: usize,
    polygon_sides: usize,
    arc_segments: usize,
    show_hidden: bool,
    file: Option<PathBuf>,
    dirty: bool,
    show_help: bool,
    prev_ctrl: bool,
    hover_guide: Option<usize>,
    show_guides: bool,
    color_by_tag: bool,
    show_annotations: bool,
    hover_annot: Option<usize>,
    /// Cursor position as a fraction of the viewport (for screen text).
    cursor_frac: Option<[f32; 2]>,
    current_scene: Option<usize>,
    cam_anim: Option<CamAnim>,
    /// Tape Measure creates guides (Ctrl toggles measure-only).
    tape_guides: bool,
    tape_result: Option<(Vec3, Vec3)>,
    title: String,
    /// Triangulated faces with component openings cut out, by geometry hash.
    cut_cache: RefCell<HashMap<u64, Rc<Vec<[Vec3; 3]>>>>,
    /// Decoded texture images by `Texture::key` (None = failed to decode).
    tex_cache: RefCell<HashMap<u64, Option<Arc<Pixels>>>>,
    /// Screenshot self-test (OPENSKETCH_SHOT=file.png): frames drawn so far
    /// and whether the image was written.
    shot: Option<(PathBuf, u32, Arc<std::sync::atomic::AtomicBool>)>,
    /// Where the left button went down in the viewport (drawing by dragging).
    press: Option<Pos2>,
    /// Where and when the last tool click landed, to tell a second click on
    /// the same spot (and a double-click) from a new one.
    last_click: Option<(Pos2, f64)>,
    /// The Move/Rotate/Scale in progress picked what was under the cursor
    /// itself, so the selection is dropped again when it ends.
    auto_sel: bool,
    /// Distance of the last Push/Pull (double-click repeats it).
    last_pp: Option<f32>,
    /// Points and edge directions the cursor passed over lately, for the
    /// "From Point" and parallel / perpendicular inferences.
    recent_pts: Vec<Vec3>,
    recent_dirs: Vec<Vec3>,
    /// The 2-point arc being drawn is tangent to an edge at one of its ends.
    arc_tangent: bool,
    /// Scripted input self-test (OPENSKETCH_SCRIPT=file).
    script: Option<Script>,
    ai: AiUi,
    /// AI Render self-test (OPENSKETCH_SHOT_AI=file.png): where to save the
    /// picture of the whole window, and how far the test has got.
    ai_test: Option<(PathBuf, u32)>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        let renderer = cc.gl.as_ref().map(|gl| Arc::new(Mutex::new(Renderer::new(gl))));
        let mut app = Self {
            doc: Document::new(),
            path: Vec::new(),
            ctx_xf: Mat4::IDENTITY,
            model: Model::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            cam: Camera::default(),
            renderer,
            tool: Tool::Rectangle,
            state: State::Idle,
            sel: Selection::default(),
            vcb: String::new(),
            measure: String::new(),
            status: "Welcome to OpenSketch — draw a rectangle on the ground, then Push/Pull it (P).".into(),
            units: Units::Meters,
            material: None,
            xray: false,
            show_axes: true,
            lock_axis: None,
            shift_lock: None,
            snap: None,
            hover_face: None,
            hover_edge: None,
            hover_inst: None,
            last_op: None,
            circle_segments: 24,
            polygon_sides: 6,
            arc_segments: 12,
            show_hidden: false,
            file: None,
            dirty: false,
            show_help: false,
            prev_ctrl: false,
            hover_guide: None,
            show_guides: true,
            color_by_tag: false,
            show_annotations: true,
            hover_annot: None,
            cursor_frac: None,
            current_scene: None,
            cam_anim: None,
            tape_guides: true,
            tape_result: None,
            title: String::new(),
            cut_cache: RefCell::new(HashMap::new()),
            tex_cache: RefCell::new(HashMap::new()),
            shot: None,
            press: None,
            last_click: None,
            auto_sel: false,
            last_pp: None,
            recent_pts: Vec::new(),
            recent_dirs: Vec::new(),
            arc_tangent: false,
            script: std::env::var_os("OPENSKETCH_SCRIPT").and_then(|p| Script::load(std::path::Path::new(&p))),
            ai: AiUi::new(),
            ai_test: None,
        };
        if let Some(path) = std::env::var_os("OPENSKETCH_SHOT") {
            app.doc = demo_document(std::env::var_os("OPENSKETCH_SHOT_NO_GROUND").is_none());
            app.load_context();
            app.cam = Camera { target: Vec3::new(0.5, 1.0, 1.0), distance: 17.0, ..Camera::default() };
            app.shot = Some((PathBuf::from(path), 0, Arc::new(std::sync::atomic::AtomicBool::new(false))));
        } else if std::env::var_os("OPENSKETCH_DEMO").is_some() || std::env::var_os("OPENSKETCH_SHOT_AI").is_some() {
            // Start with the demo scene instead of an empty model (handy for trying things out).
            app.doc = demo_document(true);
            app.load_context();
            app.cam = Camera { target: Vec3::new(0.5, 1.0, 1.0), distance: 17.0, ..Camera::default() };
            if let Some(path) = std::env::var_os("OPENSKETCH_SHOT_AI") {
                app.ai.open = true;
                app.ai.description = "a small brick house with a wooden roof in a garden".into();
                app.ai_test = Some((PathBuf::from(path), 0));
            }
        }
        app
    }

    // ----- document, context & undo ------------------------------------------

    fn ctx_def(&self) -> DefId {
        self.doc.ctx_def(&self.path)
    }

    /// Inference axes of the edit context (a rotated group has rotated axes).
    fn axes(&self) -> [Vec3; 3] {
        let m = self.ctx_xf;
        [m.x_axis.truncate().normalize_or(Vec3::X), m.y_axis.truncate().normalize_or(Vec3::Y), m.z_axis.truncate().normalize_or(Vec3::Z)]
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot { doc: self.doc.clone(), path: self.path.clone() }
    }

    fn push_undo(&mut self, s: Snapshot) {
        self.undo.push(s);
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.dirty = true;
    }

    /// Records an undo point before a document-level edit.
    fn checkpoint(&mut self) -> Snapshot {
        let s = self.snapshot();
        self.push_undo(s.clone());
        s
    }

    fn restore(&mut self, s: Snapshot) {
        self.doc = s.doc;
        self.path = s.path;
        self.load_context();
    }

    fn load_context(&mut self) {
        while !self.path.is_empty() && self.doc.ctx_def(&self.path) == ROOT {
            self.path.pop();
        }
        self.ctx_xf = self.doc.ctx_xf(&self.path);
        self.model = self.doc.load(&self.path);
        self.sel.clear();
        self.state = State::Idle;
        self.auto_sel = false;
        self.recent_pts.clear();
        self.recent_dirs.clear();
    }

    fn sync(&mut self) {
        self.doc.sync(&self.path, &self.model);
    }

    /// Runs an edit of the working mesh with undo support.
    fn commit(&mut self, f: impl FnOnce(&mut Model) -> bool) -> Option<Snapshot> {
        let before = self.snapshot();
        let backup = self.model.clone();
        if f(&mut self.model) {
            self.sync();
            self.push_undo(before.clone());
            Some(before)
        } else {
            self.model = backup;
            None
        }
    }

    fn undo(&mut self) {
        if let Some(s) = self.undo.pop() {
            let cur = self.snapshot();
            self.redo.push(cur);
            self.restore(s);
            self.last_op = None;
            self.dirty = true;
            self.status = "Undo".into();
        }
    }

    fn redo(&mut self) {
        if let Some(s) = self.redo.pop() {
            let cur = self.snapshot();
            self.undo.push(cur);
            self.restore(s);
            self.last_op = None;
            self.dirty = true;
            self.status = "Redo".into();
        }
    }

    // ----- generic actions ------------------------------------------------

    fn set_tool(&mut self, t: Tool) {
        self.end_auto_sel();
        self.tool = t;
        self.state = State::Idle;
        self.vcb.clear();
        self.measure.clear();
        self.last_op = None;
        self.lock_axis = None;
        self.tape_result = None;
        if matches!(t, Tool::OuterShell | Tool::Union | Tool::SolidIntersect) && self.sel.insts.len() >= 2 {
            self.solid_selection(t);
        }
    }

    fn escape(&mut self) {
        self.vcb.clear();
        self.lock_axis = None;
        if !matches!(self.state, State::Idle) {
            self.state = State::Idle;
            self.end_auto_sel();
            self.status = "Cancelled".into();
        } else if !self.sel.is_empty() {
            self.sel.clear();
        } else if !self.path.is_empty() {
            self.exit_context();
        }
    }

    /// Drops the selection a Move/Rotate/Scale made for itself.
    fn end_auto_sel(&mut self) {
        if std::mem::take(&mut self.auto_sel) {
            self.sel.clear();
        }
    }

    fn toggle_lock(&mut self, i: usize) {
        self.lock_axis = if self.lock_axis == Some(i) { None } else { Some(i) };
    }

    fn select_all(&mut self) {
        self.sel.faces = self.model.faces.keys().copied().collect();
        self.sel.edges = self.model.edges.keys().copied().collect();
        self.sel.insts =
            self.doc.defs[&self.ctx_def()].instances.iter().filter(|(_, i)| self.doc.tag_visible(i.tag)).map(|(k, _)| *k).collect();
        if self.show_guides {
            self.sel.guides = (0..self.doc.guides.len()).collect();
        }
        if self.show_annotations {
            self.sel.annots = (0..self.doc.annotations.len()).collect();
        }
    }

    fn delete_selection(&mut self) {
        if self.sel.is_empty() {
            return;
        }
        let (f, e, i) = (self.sel.faces(), self.sel.edges(), self.sel.insts());
        let (g, an) = (self.sel.guides(), self.sel.annots());
        self.checkpoint();
        self.model.delete(&f, &e);
        self.doc.remove_instances(&self.path, &i);
        self.remove_guides(&g);
        self.remove_annotations(&an);
        self.sync();
        self.sel.clear();
    }

    // ----- textures & shadows -------------------------------------------------------

    /// The material's texture (index, real-world size) if faces using it should
    /// be drawn textured.
    fn face_texture(&self, mat: Option<usize>, faded: bool) -> Option<(usize, [f32; 2])> {
        if self.color_by_tag || faded {
            return None;
        }
        let i = mat?;
        let t = self.doc.materials.get(i)?.texture.as_ref()?;
        Some((i, t.size))
    }

    fn texture_pixels(&self, mat: usize) -> Option<(u64, Arc<Pixels>)> {
        let t = self.doc.materials.get(mat)?.texture.as_ref()?;
        let key = t.key();
        let mut cache = self.tex_cache.borrow_mut();
        let px = cache.entry(key).or_insert_with(|| t.pixels().ok().map(Arc::new)).clone()?;
        Some((key, px))
    }

    fn set_texture(&mut self, i: usize, tex: Option<Texture>, color: Option<[f32; 3]>) {
        self.checkpoint();
        if let Some(m) = self.doc.materials.get_mut(i) {
            m.texture = tex;
            if let Some(c) = color {
                m.color = c;
            }
        }
    }

    /// Texture rows under the selected material in the Materials panel.
    fn texture_controls(&mut self, ui: &mut egui::Ui, i: usize) {
        let tex = self.doc.materials[i].texture.clone();
        let (mut remove, mut load, mut builtin) = (false, false, None);
        match &tex {
            Some(t) => {
                ui.horizontal(|ui| {
                    ui.label(format!("▦ {}", t.name()));
                    remove = ui.small_button("Remove texture").clicked();
                });
                let mut sz = t.size;
                ui.horizontal(|ui| {
                    ui.label("Size");
                    let a = ui.add(egui::DragValue::new(&mut sz[0]).speed(0.01).range(0.01..=100.0).suffix(" m")).changed();
                    ui.label("×");
                    let b = ui.add(egui::DragValue::new(&mut sz[1]).speed(0.01).range(0.01..=100.0).suffix(" m")).changed();
                    if a || b {
                        if let Some(t) = self.doc.materials[i].texture.as_mut() {
                            t.size = sz;
                            self.dirty = true;
                        }
                    }
                });
            }
            None => {
                ui.horizontal(|ui| {
                    load = ui.small_button("Load image…").on_hover_text("PNG or JPEG; embedded in the model file").clicked();
                    egui::ComboBox::from_id_salt("builtin_tex").selected_text("Built-in texture").show_ui(ui, |ui| {
                        for p in Pattern::ALL {
                            if ui.selectable_label(false, p.label()).clicked() {
                                builtin = Some(p);
                            }
                        }
                    });
                });
            }
        }
        if remove {
            self.set_texture(i, None, None);
        } else if let Some(p) = builtin {
            let avg = texture::generate(p).average();
            self.set_texture(i, Some(Texture::builtin(p)), Some(avg));
        } else if load {
            if let Some(path) = rfd::FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg"]).pick_file() {
                match Texture::from_file(&path) {
                    Ok((t, avg)) => {
                        self.status = format!("Loaded texture {}", t.name());
                        self.set_texture(i, Some(t), Some(avg));
                    }
                    Err(e) => self.status = format!("Could not load texture: {e}"),
                }
            }
        }
    }

    fn shadows_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Shadows");
        let mut s = self.doc.sun;
        ui.checkbox(&mut s.enabled, "Show shadows");
        egui::Grid::new("sun").num_columns(2).show(ui, |ui| {
            ui.label("Time");
            ui.add(egui::Slider::new(&mut s.hour, 4.0..=20.0).step_by(0.25).custom_formatter(|v, _| Sun::time_label(v as f32)));
            ui.end_row();
            ui.label("Date");
            ui.add(egui::Slider::new(&mut s.day, 1..=365).custom_formatter(|v, _| Sun::date_label(v as u32)));
            ui.end_row();
            ui.label("Latitude");
            ui.add(egui::DragValue::new(&mut s.latitude).range(-90.0..=90.0).speed(0.5).suffix("°"));
            ui.end_row();
            ui.label("North angle");
            ui.add(egui::DragValue::new(&mut s.north).range(0.0..=360.0).speed(1.0).suffix("°"));
            ui.end_row();
        });
        if s.enabled && !s.above_horizon() {
            ui.label(RichText::new("The sun is below the horizon at this time.").weak());
        }
        if s != self.doc.sun {
            self.doc.sun = s;
            self.dirty = true;
        }
    }

    // ----- dimensions & text -------------------------------------------------------

    fn add_annotation(&mut self, a: Annotation) {
        self.checkpoint();
        let what = if a.measured().is_some() { "Dimension" } else { "Text" };
        self.doc.annotations.push(a);
        self.show_annotations = true;
        self.sel.clear();
        self.sel.annots.insert(self.doc.annotations.len() - 1);
        self.state = State::Idle;
        self.status = format!("{what} added — edit its text in Entity Info");
    }

    fn remove_annotations(&mut self, idx: &[usize]) {
        let mut idx = idx.to_vec();
        idx.sort_unstable();
        for i in idx.into_iter().rev() {
            if i < self.doc.annotations.len() {
                self.doc.annotations.remove(i);
            }
        }
        self.sel.annots.clear();
        self.hover_annot = None;
    }

    /// Center, radius, plane normal and "is a full circle" of the arc/circle
    /// an edge belongs to.
    fn curve_circle(&self, e: Id) -> Option<(Vec3, f32, Vec3, bool)> {
        self.model.edges.get(&e)?.curve?;
        let es = self.model.curve_edges(e);
        if es.len() < 2 {
            return None;
        }
        let mut deg: HashMap<Id, usize> = HashMap::new();
        for x in &es {
            let edge = self.model.edges.get(x)?;
            *deg.entry(edge.a).or_default() += 1;
            *deg.entry(edge.b).or_default() += 1;
        }
        let closed = deg.values().all(|&n| n == 2);
        let pts: Vec<Vec3> = deg.keys().map(|v| self.model.pos(*v)).collect();
        let a = pts[0];
        let b = *pts.iter().max_by(|p, q| p.distance(a).total_cmp(&q.distance(a)))?;
        let c = *pts.iter().max_by(|p, q| p.distance(a).min(p.distance(b)).total_cmp(&q.distance(a).min(q.distance(b))))?;
        let center = circumcenter(a, b, c)?;
        let radius = center.distance(a);
        if pts.iter().any(|p| (p.distance(center) - radius).abs() > radius * 0.01) {
            return None;
        }
        Some((center, radius, (b - a).cross(c - a).normalize(), closed))
    }

    /// Offset of a dimension line: within the axis-aligned plane through a–b
    /// that faces the camera most.
    fn dim_offset(&self, a: Vec3, b: Vec3, o: Vec3, d: Vec3) -> Vec3 {
        let dir = (b - a).normalize_or_zero();
        let mut best: Option<(f32, Vec3)> = None;
        for ax in self.axes() {
            let side = ax - dir * ax.dot(dir);
            if side.length() < 0.1 {
                continue;
            }
            let side = side.normalize();
            let n = dir.cross(side).normalize();
            let score = n.dot(d).abs();
            if let Some(p) = ray_plane(o, d, a, n) {
                if best.is_none_or(|b| score > b.0) {
                    best = Some((score, side * (p - a).dot(side)));
                }
            }
        }
        best.map(|b| b.1).unwrap_or(Vec3::ZERO)
    }

    /// What the Text tool writes by default for the thing clicked.
    fn default_text(&self, s: &Snap) -> String {
        let u = self.units;
        let coords = |p: Vec3| format!("{}, {}, {}", u.format(p.x), u.format(p.y), u.format(p.z));
        match s.kind {
            SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::GuidePoint | SnapKind::Intersection => coords(s.pos),
            SnapKind::OnEdge => match s.edge {
                Some(e) => u.format(self.model.edge_length(e)),
                None => coords(s.pos),
            },
            _ => match (s.face, s.instance) {
                (Some(f), _) => u.format_area(self.model.face_area(f)),
                (None, Some(i)) => self.doc.instance(&self.path, i).map(|x| self.def_name(x.def)).unwrap_or_default(),
                _ => coords(s.pos),
            },
        }
    }

    /// Screen-space parts of an annotation: segments, arrow tips (tip, from)
    /// and the label center.
    fn annotation_screen(&self, rect: Rect, a: &Annotation) -> ScreenParts {
        let proj = |p: Vec3| self.cam.project(rect, p);
        let segs: Vec<(Pos2, Pos2)> = a.segments().into_iter().filter_map(|(p, q)| Some((proj(p)?, proj(q)?))).collect();
        let mut arrows = Vec::new();
        match a {
            Annotation::Dimension { a: pa, b: pb, offset, .. } => {
                if let (Some(x), Some(y)) = (proj(*pa + *offset), proj(*pb + *offset)) {
                    arrows.push((x, y));
                    arrows.push((y, x));
                }
            }
            Annotation::Radial { diameter, .. } => {
                if let Some((rim, start)) = a.radial_points() {
                    if let (Some(r), Some(st)) = (proj(rim), proj(start)) {
                        arrows.push((r, st));
                        if *diameter {
                            arrows.push((st, r));
                        }
                    }
                }
            }
            _ => {}
        }
        let center = match a {
            Annotation::ScreenText { pos, .. } => Some(rect.min + Vec2::new(pos[0] * rect.width(), pos[1] * rect.height())),
            _ => a.label_point().and_then(proj),
        };
        (segs, arrows, center)
    }

    /// Annotation under the cursor (labels and lines).
    fn pick_annotation(&self, rect: Rect, pos: Pos2) -> Option<usize> {
        if !self.show_annotations {
            return None;
        }
        let near = |p: Pos2, q: Pos2| {
            let d = q - p;
            let t = ((pos - p).dot(d) / d.length_sq().max(1e-6)).clamp(0.0, 1.0);
            (p + d * t).distance(pos) < 5.0
        };
        self.doc.annotations.iter().enumerate().rev().find_map(|(i, a)| {
            let (segs, _, center) = self.annotation_screen(rect, a);
            let label = a.label(self.units);
            let lines = label.lines().count().max(1) as f32;
            let width = label.lines().map(|l| l.chars().count()).max().unwrap_or(1) as f32 * 7.4 + 10.0;
            let on_label = center.is_some_and(|c| Rect::from_center_size(c, Vec2::new(width, lines * 16.0 + 6.0)).contains(pos));
            (on_label || segs.iter().any(|(p, q)| near(*p, *q))).then_some(i)
        })
    }

    fn draw_annotation(&self, painter: &egui::Painter, rect: Rect, a: &Annotation, col: Color32) {
        let (segs, arrows, center) = self.annotation_screen(rect, a);
        for (p, q) in segs {
            painter.line_segment([p, q], stroke(1.3, col));
        }
        for (tip, from) in arrows {
            let d = (tip - from).normalized();
            if d.is_finite() {
                let n = Vec2::new(-d.y, d.x);
                let base = tip - d * 10.0;
                painter.add(Shape::convex_polygon(vec![tip, base + n * 3.5, base - n * 3.5], col, Stroke::NONE));
            }
        }
        if let Annotation::Text { anchor, .. } = a {
            if let Some(p) = self.cam.project(rect, *anchor) {
                painter.circle_filled(p, 2.5, col);
            }
        }
        if let Some(c) = center {
            let galley = painter.layout_no_wrap(a.label(self.units), FontId::proportional(13.0), col);
            let r = Rect::from_center_size(c, galley.size()).expand2(Vec2::new(5.0, 3.0));
            painter.rect_filled(r, 3.0, Color32::from_white_alpha(235));
            painter.rect_stroke(r, 3.0, stroke(1.0, col.gamma_multiply(0.5)), egui::StrokeKind::Inside);
            painter.galley(r.min + Vec2::new(5.0, 3.0), galley, col);
        }
    }

    fn annotation_info(&mut self, ui: &mut egui::Ui) {
        let n = self.sel.annots.len();
        if n != 1 {
            ui.label(format!("{n} dimension(s)/text(s)"));
            return;
        }
        let i = *self.sel.annots.iter().next().unwrap();
        let Some(a) = self.doc.annotations.get(i).cloned() else { return };
        let mut focus = false;
        match a.measured() {
            Some(v) => {
                ui.label(RichText::new(if matches!(a, Annotation::Radial { .. }) { "Radial dimension" } else { "Dimension" }).strong());
                ui.label(format!("Measured: {}", self.units.format(v)));
                let mut t = match &a {
                    Annotation::Dimension { text, .. } | Annotation::Radial { text, .. } => text.clone().unwrap_or_default(),
                    _ => String::new(),
                };
                let r = ui.add(egui::TextEdit::singleline(&mut t).hint_text("Text (<> = measured value)"));
                focus = r.gained_focus();
                if r.changed() {
                    if let Some(slot) = self.doc.annotations[i].text_mut() {
                        *slot = (!t.is_empty()).then_some(t);
                    }
                }
            }
            None => {
                ui.label(RichText::new(if matches!(a, Annotation::ScreenText { .. }) { "Screen text" } else { "Text" }).strong());
                if let Annotation::Text { text, .. } | Annotation::ScreenText { text, .. } = &mut self.doc.annotations[i] {
                    let r = ui.add(egui::TextEdit::multiline(text).desired_rows(2));
                    focus = r.gained_focus();
                }
            }
        }
        if focus {
            // One undo step per editing session.
            let snap = self.snapshot();
            self.push_undo(snap);
        }
    }

    // ----- tags & scenes ------------------------------------------------------------

    fn tag_color(&self, tag: Option<u32>) -> [f32; 3] {
        tag.and_then(|t| self.doc.tags.get(&t)).map(|t| t.color).unwrap_or([0.86, 0.86, 0.86])
    }

    fn set_tag(&mut self, insts: &[Id], tag: Option<u32>) {
        let ctx = self.ctx_def();
        if let Some(d) = self.doc.defs.get_mut(&ctx) {
            for i in insts {
                if let Some(inst) = d.instances.get_mut(i) {
                    inst.tag = tag;
                }
            }
        }
    }

    fn style(&self) -> SceneStyle {
        SceneStyle {
            xray: self.xray,
            hidden_geometry: self.show_hidden,
            guides: self.show_guides,
            axes: self.show_axes,
            color_by_tag: self.color_by_tag,
            hide_annotations: !self.show_annotations,
            sun: Some(self.doc.sun),
        }
    }

    fn capture_scene(&self, name: String) -> Scene {
        Scene { name, camera: self.cam.clone(), hidden_tags: self.doc.hidden_tags(), style: self.style() }
    }

    fn add_scene(&mut self) {
        self.checkpoint();
        let sc = self.capture_scene(self.doc.next_scene_name());
        self.status = format!("Added {}", sc.name);
        self.doc.scenes.push(sc);
        self.current_scene = Some(self.doc.scenes.len() - 1);
    }

    fn update_scene(&mut self, i: usize) {
        let Some(name) = self.doc.scenes.get(i).map(|s| s.name.clone()) else { return };
        self.checkpoint();
        self.doc.scenes[i] = self.capture_scene(name.clone());
        self.current_scene = Some(i);
        self.status = format!("Updated {name}");
    }

    fn delete_scene(&mut self, i: usize) {
        if i >= self.doc.scenes.len() {
            return;
        }
        self.checkpoint();
        self.doc.scenes.remove(i);
        self.current_scene = None;
    }

    /// Switches to a scene: tags and display settings at once, camera animated.
    fn go_to_scene(&mut self, i: usize) {
        let Some(sc) = self.doc.scenes.get(i).cloned() else { return };
        self.cam_anim = Some(CamAnim { from: self.cam.clone(), to: sc.camera.clone(), t0: None });
        for (id, t) in self.doc.tags.iter_mut() {
            t.visible = !sc.hidden_tags.contains(id);
        }
        let st = sc.style;
        self.xray = st.xray;
        self.show_hidden = st.hidden_geometry;
        self.show_guides = st.guides;
        self.show_axes = st.axes;
        self.color_by_tag = st.color_by_tag;
        self.show_annotations = !st.hide_annotations;
        if let Some(sun) = st.sun {
            self.doc.sun = sun;
        }
        self.current_scene = Some(i);
        self.status = format!("Scene: {}", sc.name);
    }

    fn step_scene(&mut self, forward: bool) {
        let n = self.doc.scenes.len();
        if n == 0 {
            return;
        }
        let next = match (self.current_scene.filter(|c| *c < n), forward) {
            (Some(c), true) => (c + 1) % n,
            (Some(c), false) => (c + n - 1) % n,
            (None, _) => 0,
        };
        self.go_to_scene(next);
    }

    fn animate_camera(&mut self, ctx: &egui::Context) {
        let Some(a) = &mut self.cam_anim else { return };
        let now = ctx.input(|i| i.time);
        let t0 = *a.t0.get_or_insert(now);
        let u = (((now - t0) / 0.8) as f32).clamp(0.0, 1.0);
        self.cam = Camera::lerp(&a.from, &a.to, u * u * (3.0 - 2.0 * u));
        if u >= 1.0 {
            self.cam_anim = None;
        } else {
            ctx.request_repaint();
        }
    }

    fn scene_tabs(&mut self, ctx: &egui::Context) {
        if self.doc.scenes.is_empty() {
            return;
        }
        egui::TopBottomPanel::top("scene_tabs").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                let mut go = None;
                for (i, sc) in self.doc.scenes.iter().enumerate() {
                    if ui.selectable_label(self.current_scene == Some(i), &sc.name).on_hover_text("PageUp / PageDown").clicked() {
                        go = Some(i);
                    }
                }
                if let Some(i) = go {
                    self.go_to_scene(i);
                }
            });
        });
    }

    /// Entity Info row assigning a tag to the selected instances.
    fn tag_picker(&mut self, ui: &mut egui::Ui, insts: &[Id]) {
        let ctx = self.ctx_def();
        let tags: Vec<Option<u32>> =
            insts.iter().filter_map(|i| self.doc.defs.get(&ctx).and_then(|d| d.instances.get(i)).map(|x| x.tag)).collect();
        let Some(&first) = tags.first() else { return };
        let mixed = tags.iter().any(|t| *t != first);
        let name = |t: Option<u32>| t.and_then(|t| self.doc.tags.get(&t)).map(|t| t.name.clone()).unwrap_or("Untagged".into());
        let current = if mixed { "(mixed)".to_string() } else { name(first) };
        let options: Vec<(Option<u32>, String)> =
            std::iter::once((None, "Untagged".to_string())).chain(self.doc.tags.iter().map(|(i, t)| (Some(*i), t.name.clone()))).collect();
        let (mut pick, mut new_tag) = (None, false);
        ui.horizontal(|ui| {
            ui.label("Tag");
            egui::ComboBox::from_id_salt("tag_pick").selected_text(current).show_ui(ui, |ui| {
                for (t, n) in &options {
                    if ui.selectable_label(!mixed && *t == first, n).clicked() {
                        pick = Some(*t);
                    }
                }
            });
            new_tag = ui.small_button("➕").on_hover_text("New tag for the selection").clicked();
        });
        if new_tag {
            self.checkpoint();
            let t = self.doc.add_tag();
            self.set_tag(insts, Some(t));
        } else if let Some(t) = pick {
            self.checkpoint();
            self.set_tag(insts, t);
        }
    }

    fn tags_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Tags");
        ui.horizontal(|ui| {
            if ui.small_button("➕ Add Tag").clicked() {
                self.checkpoint();
                self.doc.add_tag();
            }
            ui.checkbox(&mut self.color_by_tag, "Color by tag");
        });
        let mut delete = None;
        egui::Grid::new("tags").num_columns(4).striped(true).show(ui, |ui| {
            let mut on = true;
            ui.add_enabled(false, egui::Checkbox::without_text(&mut on));
            ui.label("");
            ui.label("Untagged");
            ui.label("");
            ui.end_row();
            let ids: Vec<u32> = self.doc.tags.keys().copied().collect();
            for id in ids {
                let usage = self.doc.tag_usage(id);
                let Some(t) = self.doc.tags.get_mut(&id) else { continue };
                ui.checkbox(&mut t.visible, "").on_hover_text("Visible");
                ui.color_edit_button_rgb(&mut t.color);
                ui.add(egui::TextEdit::singleline(&mut t.name).desired_width(96.0));
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("×{usage}")).weak());
                    if ui.small_button("🗑").on_hover_text("Delete tag (its objects become untagged)").clicked() {
                        delete = Some(id);
                    }
                });
                ui.end_row();
            }
        });
        if let Some(id) = delete {
            self.checkpoint();
            self.doc.delete_tag(id);
        }
        if self.doc.tags.is_empty() {
            ui.label(RichText::new("Add a tag, then assign it to groups/components in Entity Info.").weak());
        }
    }

    fn scenes_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Scenes");
        if ui.small_button("➕ Add Scene").on_hover_text("Saves the camera, hidden tags and display settings").clicked() {
            self.add_scene();
        }
        let (mut go, mut update, mut delete) = (None, None, None);
        for i in 0..self.doc.scenes.len() {
            let current = self.current_scene == Some(i);
            ui.horizontal(|ui| {
                if ui.selectable_label(current, "🎬").on_hover_text("Go to scene").clicked() {
                    go = Some(i);
                }
                ui.add(egui::TextEdit::singleline(&mut self.doc.scenes[i].name).desired_width(96.0));
                if ui.small_button("⟳").on_hover_text("Update with the current view").clicked() {
                    update = Some(i);
                }
                if ui.small_button("🗑").on_hover_text("Delete scene").clicked() {
                    delete = Some(i);
                }
            });
        }
        if let Some(i) = go {
            self.go_to_scene(i);
        }
        if let Some(i) = update {
            self.update_scene(i);
        }
        if let Some(i) = delete {
            self.delete_scene(i);
        }
    }

    // ----- guides ---------------------------------------------------------------

    fn remove_guides(&mut self, idx: &[usize]) {
        let mut idx = idx.to_vec();
        idx.sort_unstable();
        for i in idx.into_iter().rev() {
            if i < self.doc.guides.len() {
                self.doc.guides.remove(i);
            }
        }
        self.sel.guides.clear();
        self.hover_guide = None;
    }

    fn delete_all_guides(&mut self) {
        if self.doc.guides.is_empty() {
            return;
        }
        self.checkpoint();
        let n = self.doc.guides.len();
        self.doc.guides.clear();
        self.sel.guides.clear();
        self.status = format!("Deleted {n} guide(s)");
    }

    fn add_guide(&mut self, g: Guide) {
        self.checkpoint();
        self.doc.guides.push(g);
        self.show_guides = true;
    }

    /// Finishes the Tape Measure at `end` (already resolved for parallel mode).
    fn finish_tape(&mut self, start: Vec3, dir: Option<Vec3>, end: Vec3) {
        self.state = State::Idle;
        let d = start.distance(end);
        self.tape_result = Some((start, end));
        self.measure = self.units.format(d);
        let made = match (self.tape_guides, dir) {
            (false, _) => None,
            // From an edge: a parallel guide line (on the edge itself if not moved).
            (true, Some(dir)) => Guide::line(end, dir),
            (true, None) if d > EPS => Some(Guide::Point { p: end }),
            _ => None,
        };
        self.status = match made {
            Some(g) => {
                self.add_guide(g);
                let what = if matches!(g, Guide::Line { .. }) { "guide line" } else { "guide point" };
                format!("Distance: {} — {what} added", self.units.format(d))
            }
            None => format!("Distance: {}", self.units.format(d)),
        };
    }

    fn finish_protractor(&mut self, angle_override: Option<f32>) {
        let State::ProtAngle { center, n, ref_dir, angle } = self.state.clone() else { return };
        let a = angle_override.unwrap_or(angle);
        self.state = State::Idle;
        let dir = glam::Quat::from_axis_angle(n, a) * ref_dir;
        if let Some(g) = Guide::line(center, dir) {
            self.add_guide(g);
            self.measure = format!("{:.1}°", a.to_degrees());
            self.status = format!("Guide line at {:.1}°", a.to_degrees());
        }
    }

    fn zoom_extents(&mut self) {
        self.cam_anim = None;
        match self.doc.world_bounds() {
            Some((lo, hi)) => self.cam.fit(lo, hi),
            None => self.cam = Camera { perspective: self.cam.perspective, ..Camera::default() },
        }
    }

    fn new_model(&mut self) {
        self.doc = Document::new();
        self.path.clear();
        self.load_context();
        self.undo.clear();
        self.redo.clear();
        self.file = None;
        self.dirty = false;
        self.set_tool(Tool::Rectangle);
        self.cam = Camera::default();
    }

    // ----- groups & components ---------------------------------------------

    fn def_name(&self, def: DefId) -> String {
        self.doc.defs.get(&def).map(|d| d.name.clone()).unwrap_or_default()
    }

    fn make_group(&mut self, kind: DefKind) {
        if self.sel.is_empty() {
            self.status = format!("Select geometry first, then make a {}.", kind.label().to_lowercase());
            return;
        }
        let (f, e, i) = (self.sel.faces(), self.sel.edges(), self.sel.insts());
        let before = self.checkpoint();
        match self.doc.make_group(&self.path, &mut self.model, &f, &e, &i, kind) {
            Some(inst) => {
                self.sync();
                self.sel.clear();
                self.sel.insts.insert(inst);
                let def = self.doc.instance(&self.path, inst).map(|x| x.def).unwrap_or(ROOT);
                let glued = self.doc.instance(&self.path, inst).and_then(|x| x.glued).is_some();
                self.status = if glued {
                    format!("Created {} — glued to the face, cuts an opening", self.def_name(def))
                } else {
                    format!("Created {}", self.def_name(def))
                };
            }
            None => {
                self.undo.pop();
                self.restore(before);
            }
        }
        self.last_op = None;
    }

    fn explode(&mut self) {
        let insts = self.sel.insts();
        if insts.is_empty() {
            return;
        }
        self.checkpoint();
        self.sel.clear();
        for iid in insts {
            if let Some((f, e, released)) = self.doc.explode(&self.path, &mut self.model, iid) {
                self.sel.faces.extend(f);
                self.sel.edges.extend(e);
                self.sel.insts.extend(released);
            }
        }
        self.sync();
        self.status = "Exploded".into();
    }

    fn make_unique(&mut self) {
        let insts = self.sel.insts();
        if insts.is_empty() {
            return;
        }
        self.checkpoint();
        for iid in insts {
            self.doc.make_unique(&self.path, iid);
        }
        self.status = "Made unique".into();
    }

    fn convert_to_component(&mut self, def: DefId) {
        self.checkpoint();
        let name = self.doc.next_name(DefKind::Component);
        if let Some(d) = self.doc.defs.get_mut(&def) {
            d.kind = DefKind::Component;
            d.name = name;
        }
    }

    fn set_instance_material(&mut self, insts: &[Id], mat: Option<usize>) {
        self.checkpoint();
        let ctx = self.ctx_def();
        if let Some(d) = self.doc.defs.get_mut(&ctx) {
            for i in insts {
                if let Some(inst) = d.instances.get_mut(i) {
                    inst.material = mat;
                }
            }
        }
    }

    fn enter_context(&mut self, iid: Id) {
        let Some(def) = self.doc.instance(&self.path, iid).map(|i| i.def) else { return };
        self.path.push(iid);
        self.load_context();
        self.last_op = None;
        self.status = format!("Editing {} — Esc or click outside to close", self.def_name(def));
    }

    fn exit_context(&mut self) {
        if let Some(i) = self.path.pop() {
            self.load_context();
            self.last_op = None;
            self.sel.insts.insert(i);
            self.status = "Closed edit context".into();
        }
    }

    /// Opens the context that contains `iid` and selects it (Outliner).
    fn navigate(&mut self, parent: Vec<Id>, iid: Id) {
        if self.path != parent {
            self.path = parent;
            self.load_context();
        }
        self.sel.clear();
        self.sel.insts.insert(iid);
    }

    fn start_place(&mut self, def: DefId) {
        if self.doc.contains_def(def, self.ctx_def()) {
            self.status = "A component can't be placed inside itself.".into();
            return;
        }
        self.state = State::Place { def };
        self.status = format!("Placing {}", self.def_name(def));
    }

    /// World placement for a component under the cursor: gluing components
    /// lie on the hovered face (blue axis out of it, towards the viewer).
    fn place_xf(&self, def: DefId, s: &Snap) -> Mat4 {
        let glue = self.doc.defs.get(&def).is_some_and(|d| d.glue);
        if let Some(face) = s.face.filter(|_| glue).and_then(|f| self.model.faces.get(&f)) {
            let p0 = self.model.pos(face.outer[0]);
            let mut n = face.normal;
            if n.dot(self.cam.eye() - s.pos) < 0.0 {
                n = -n;
            }
            let pos = s.pos - n * (s.pos - p0).dot(n);
            let ax = self.axes();
            return glue_frame(n, pos, ax[2], ax[0]);
        }
        let local = self.ctx_xf.inverse().transform_point3(s.pos);
        self.ctx_xf * Mat4::from_translation(local)
    }

    fn place_at(&mut self, def: DefId, world: Mat4) {
        let before = self.checkpoint();
        match self.doc.place(&self.path, def, world) {
            Some(i) => {
                self.sel.clear();
                self.sel.insts.insert(i);
            }
            None => {
                self.undo.pop();
                self.restore(before);
            }
        }
        self.state = State::Idle;
    }

    // ----- files ------------------------------------------------------------

    fn save(&mut self) {
        match self.file.clone() {
            Some(p) => self.save_to(p),
            None => self.save_as(),
        }
    }

    fn save_as(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("OpenSketch model", &["osk"]).set_file_name("model.osk").save_file() {
            self.save_to(p);
        }
    }

    fn save_to(&mut self, p: PathBuf) {
        self.sync();
        match io::save(&self.doc, &p) {
            Ok(()) => {
                self.status = format!("Saved {}", p.display());
                self.file = Some(p);
                self.dirty = false;
            }
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    fn open(&mut self) {
        let Some(p) = rfd::FileDialog::new().add_filter("OpenSketch model", &["osk"]).pick_file() else { return };
        match io::load(&p) {
            Ok(doc) => {
                self.doc = doc;
                self.path.clear();
                self.load_context();
                self.undo.clear();
                self.redo.clear();
                self.status = format!("Opened {}", p.display());
                self.file = Some(p);
                self.dirty = false;
                self.zoom_extents();
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    fn import_obj(&mut self) {
        let Some(p) = rfd::FileDialog::new().add_filter("Wavefront OBJ", &["obj"]).pick_file() else { return };
        let mut result = Err(String::new());
        self.commit(|m| {
            result = io::import_obj(m, &p);
            matches!(result, Ok(n) if n > 0)
        });
        self.status = match result {
            Ok(n) => format!("Imported {n} faces from {}", p.display()),
            Err(e) => format!("Import failed: {e}"),
        };
        self.zoom_extents();
    }

    fn export(&mut self, stl: bool) {
        let (name, ext) = if stl { ("STL", "stl") } else { ("Wavefront OBJ", "obj") };
        let Some(p) = rfd::FileDialog::new().add_filter(name, &[ext]).set_file_name(format!("model.{ext}")).save_file() else {
            return;
        };
        self.sync();
        let flat = self.doc.flatten();
        let r = if stl { io::export_stl(&flat, &p) } else { io::export_obj(&flat, &self.doc.materials, &p) };
        self.status = match r {
            Ok(()) => format!("Exported {}", p.display()),
            Err(e) => format!("Export failed: {e}"),
        };
    }

    // ----- keyboard & Measurements box --------------------------------------

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        let mods = ctx.input(|i| i.modifiers);
        if mods.ctrl && !self.prev_ctrl {
            if let State::Move { copy, .. } = &mut self.state {
                *copy = !*copy;
            } else if self.tool == Tool::Tape {
                self.tape_guides = !self.tape_guides;
                self.status = if self.tape_guides { "Tape Measure: creates guides" } else { "Tape Measure: measure only" }.into();
            }
        }
        self.prev_ctrl = mods.ctrl;
        if !mods.shift {
            self.shift_lock = None;
        } else if self.shift_lock.is_none() {
            if let Some(Snap { kind: SnapKind::Axis(i), .. }) = &self.snap {
                self.shift_lock = Some(*i);
            }
        }
        if ctx.wants_keyboard_input() {
            return;
        }
        let events = ctx.input(|i| i.events.clone());
        for ev in events {
            match ev {
                egui::Event::Key { key, pressed: true, modifiers, .. } => self.on_key(ctx, key, modifiers),
                egui::Event::Text(t) => t.chars().for_each(|c| self.on_char(c)),
                _ => {}
            }
        }
    }

    fn on_key(&mut self, ctx: &egui::Context, key: Key, m: Modifiers) {
        let cmd = m.command || m.ctrl;
        match key {
            Key::Z if cmd && m.shift => self.redo(),
            Key::Z if cmd => self.undo(),
            Key::Y if cmd => self.redo(),
            Key::S if cmd && m.shift => self.save_as(),
            Key::S if cmd => self.save(),
            Key::O if cmd => self.open(),
            Key::N if cmd => self.new_model(),
            Key::A if cmd => self.select_all(),
            Key::G if cmd => self.make_group(DefKind::Group),
            Key::Q if cmd => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            Key::Escape => self.escape(),
            Key::Enter => self.apply_vcb(),
            Key::Backspace => {
                self.vcb.pop();
            }
            Key::Delete => self.delete_selection(),
            Key::ArrowUp => self.toggle_lock(2),
            Key::ArrowRight => self.toggle_lock(0),
            Key::ArrowLeft => self.toggle_lock(1),
            Key::ArrowDown => self.lock_axis = None,
            Key::PageDown => self.step_scene(true),
            Key::PageUp => self.step_scene(false),
            Key::Space if !cmd => self.set_tool(Tool::Select),
            _ => {}
        }
    }

    fn on_char(&mut self, ch: char) {
        if ch == ' ' {
            return;
        }
        let numeric = ch.is_ascii_digit() || ".,;:-'\"/*".contains(ch);
        if numeric || (!self.vcb.is_empty() && ch.is_ascii_alphabetic()) {
            self.vcb.push(ch);
            return;
        }
        let tool = match ch {
            'Z' => {
                self.zoom_extents();
                return;
            }
            'g' => {
                self.make_group(DefKind::Component);
                return;
            }
            'l' => Tool::Line,
            'a' => Tool::Arc,
            'r' => Tool::Rectangle,
            'c' => Tool::Circle,
            'n' => Tool::Polygon,
            'p' => Tool::PushPull,
            'm' => Tool::Move,
            'q' => Tool::Rotate,
            's' => Tool::Scale,
            'f' => Tool::Offset,
            'e' => Tool::Eraser,
            't' => Tool::Tape,
            'b' => Tool::Paint,
            'o' => Tool::Orbit,
            'h' => Tool::Pan,
            _ => return,
        };
        self.set_tool(tool);
    }

    fn apply_vcb(&mut self) {
        let text = std::mem::take(&mut self.vcb).trim().to_string();
        if text.is_empty() {
            return;
        }
        let value = self.units.parse(&text);
        let parts: Vec<&str> = text.split([',', ';']).map(|s| s.trim()).collect();
        let parse_count = |t: &str| t.trim_end_matches('s').trim().parse::<usize>().ok();

        match self.state.clone() {
            State::Line { start } => {
                if let (Some(l), Some(s)) = (value, self.snap.clone()) {
                    let dir = (s.pos - start).normalize_or_zero();
                    if dir != Vec3::ZERO {
                        self.line_to(start, start + dir * l);
                    }
                }
            }
            State::Rect { p0, p1, n, u, v, on_face } => {
                if parts.len() == 2 {
                    if let (Some(w), Some(h)) = (self.units.parse(parts[0]), self.units.parse(parts[1])) {
                        let d = p1 - p0;
                        let su = if d.dot(u) < 0.0 { -1.0 } else { 1.0 };
                        let sv = if d.dot(v) < 0.0 { -1.0 } else { 1.0 };
                        self.finish_rect(p0, p0 + u * w * su + v * h * sv, n, u, v, on_face);
                    }
                }
            }
            State::Circle { center, edge, n, on_face } => {
                if text.ends_with('s') {
                    self.set_segments(parse_count(&text));
                } else if let Some(r) = value {
                    let mut dir = in_plane(edge - center, n).normalize_or_zero();
                    if dir == Vec3::ZERO {
                        dir = drawing_axes(n).0;
                    }
                    self.finish_circle(center, center + dir * r, n, on_face);
                }
            }
            State::ArcEnd { p0, p1, n } => {
                if let Some(l) = value {
                    let dir = (p1 - p0).normalize_or_zero();
                    if dir != Vec3::ZERO {
                        self.state = State::ArcBulge { p0, p1: p0 + dir * l, n, bulge: 0.0 };
                    }
                }
            }
            State::ArcBulge { p0, p1, n, bulge } => {
                let l = p0.distance(p1);
                let sign = if bulge < 0.0 { -1.0 } else { 1.0 };
                let segs = self.arc_segments.max(2);
                if text.ends_with('s') {
                    self.set_segments(parse_count(&text));
                } else if let Some(r) = text.strip_suffix('r').and_then(|t| self.units.parse(t)) {
                    if r + EPS >= l / 2.0 {
                        let s = r - (r * r - l * l / 4.0).max(0.0).sqrt();
                        self.finish_arc(arc_points_2pt(p0, p1, n, s * sign, segs));
                    } else {
                        self.status = "The radius must be at least half the chord length.".into();
                    }
                } else if let Some(b) = value {
                    self.finish_arc(arc_points_2pt(p0, p1, n, b * sign, segs));
                }
            }
            State::ArcStart { center, edge, n, on_face } => {
                if text.ends_with('s') {
                    self.set_segments(parse_count(&text));
                } else if let Some(r) = value {
                    let mut dir = in_plane(edge - center, n).normalize_or_zero();
                    if dir == Vec3::ZERO {
                        dir = drawing_axes(n).0;
                    }
                    self.state = State::ArcSweep { center, start: center + dir * r, n, on_face, angle: 0.0 };
                }
            }
            State::ArcSweep { angle, .. } => {
                if text.ends_with('s') {
                    self.set_segments(parse_count(&text));
                } else if let Ok(deg) = text.trim_end_matches('°').parse::<f32>() {
                    let sign = if angle < 0.0 { -1.0 } else { 1.0 };
                    self.finish_center_arc(Some(deg.to_radians() * sign));
                }
            }
            State::PushPull { face, n, dist, .. } => {
                if let Some(l) = value {
                    let sign = if dist.abs() > EPS {
                        dist.signum()
                    } else if n.z < -0.9 {
                        -1.0
                    } else {
                        1.0
                    };
                    self.finish_pushpull(face, l * sign);
                }
            }
            State::Move { delta, .. } => {
                if let Some(l) = value {
                    let dir = delta.normalize_or_zero();
                    if dir != Vec3::ZERO {
                        self.finish_move(Some(dir * l));
                    }
                }
            }
            State::RotateAngle { angle, .. } => {
                if let Ok(deg) = text.trim_end_matches('°').parse::<f32>() {
                    let sign = if angle < 0.0 { -1.0 } else { 1.0 };
                    self.finish_rotate(Some(deg.to_radians() * sign));
                }
            }
            State::ProtAngle { angle, .. } => {
                let sign = if angle < 0.0 { -1.0 } else { 1.0 };
                match parse_angle(&text) {
                    Some(a) => self.finish_protractor(Some(a * sign)),
                    None => self.status = "Type an angle (30) or a slope (1:2)".into(),
                }
            }
            State::Tape { start, dir, end } => {
                if let Some(l) = value {
                    let mut v = (end - start).normalize_or_zero();
                    if v == Vec3::ZERO {
                        if let Some(d) = dir {
                            v = drawing_axes(d).0; // any direction perpendicular to the edge
                        }
                    }
                    if v != Vec3::ZERO {
                        self.finish_tape(start, dir, start + v * l);
                    }
                }
            }
            State::Scale { .. } => {
                let f: Vec<f32> = parts.iter().filter_map(|p| p.parse::<f32>().ok()).collect();
                match f.len() {
                    1 => self.finish_scale(Some(Vec3::splat(f[0]))),
                    3 => self.finish_scale(Some(Vec3::new(f[0], f[1], f[2]))),
                    _ => self.status = "Type a factor (2) or x,y,z factors (1,2,1)".into(),
                }
            }
            State::Offset { face, dist } => {
                if let Some(l) = value {
                    self.finish_offset(face, if dist < 0.0 { -l } else { l });
                }
            }
            State::Idle => {
                let segmented = matches!(self.tool, Tool::Circle | Tool::Polygon | Tool::Arc | Tool::CenterArc | Tool::Pie);
                if segmented && text.ends_with('s') {
                    self.set_segments(parse_count(&text));
                } else if let Some(op) = self.last_op.clone() {
                    self.adjust_last_op(op, &text);
                }
            }
            _ => {}
        }
    }

    fn set_segments(&mut self, n: Option<usize>) {
        if let Some(n) = n.filter(|n| *n >= 3 && *n <= 999) {
            match self.tool {
                Tool::Polygon => self.polygon_sides = n,
                Tool::Arc | Tool::CenterArc | Tool::Pie => self.arc_segments = n,
                _ => self.circle_segments = n,
            }
            self.status = format!("Segments: {n}");
        }
    }

    /// Re-applies the last operation with a newly typed value.
    fn adjust_last_op(&mut self, op: LastOp, text: &str) {
        let t = text.trim();
        match op {
            LastOp::PushPull { before, face, sign } => {
                let Some(l) = self.units.parse(t) else { return };
                self.restore(before);
                self.model.push_pull(face, l * sign);
                self.sync();
                self.measure = self.units.format(l);
            }
            LastOp::Offset { before, face, sign } => {
                let Some(l) = self.units.parse(t) else { return };
                self.restore(before);
                self.model.offset_face(face, l * sign);
                self.sync();
                self.measure = self.units.format(l);
            }
            LastOp::Transform { before, t: targets, op, copy } => {
                let new_op = match &op {
                    XOp::Move(delta) => {
                        let array = copy && (t.ends_with('x') || t.starts_with('*') || t.starts_with('/'));
                        if array {
                            let n: usize = t.trim_matches(|c| c == 'x' || c == '*' || c == '/').trim().parse().unwrap_or(1);
                            let n = n.clamp(1, 500);
                            let step = if t.starts_with('/') { *delta / n as f32 } else { *delta };
                            self.restore(before.clone());
                            for k in 1..=n {
                                self.apply_transform(&targets, Mat4::from_translation(step * k as f32), true);
                            }
                            self.status = format!("Created {n} copies");
                            self.dirty = true;
                            return;
                        }
                        let Some(l) = self.units.parse(t) else { return };
                        self.measure = self.units.format(l);
                        XOp::Move(delta.normalize_or_zero() * l)
                    }
                    XOp::Rotate { center, n, angle } => {
                        let Ok(deg) = t.trim_end_matches('°').parse::<f32>() else { return };
                        self.measure = format!("{deg:.1}°");
                        XOp::Rotate { center: *center, n: *n, angle: deg.to_radians() * angle.signum() }
                    }
                };
                self.restore(before.clone());
                self.apply_transform(&targets, new_op.matrix(), copy);
                self.last_op = Some(LastOp::Transform { before, t: targets, op: new_op, copy });
            }
        }
        self.dirty = true;
    }

    // ----- operations -------------------------------------------------------

    fn line_to(&mut self, start: Vec3, end: Vec3) {
        if start.distance(end) < EPS {
            return;
        }
        let mut closed = false;
        self.commit(|m| {
            let a = m.vertex_on_geometry(start);
            let b = m.vertex_on_geometry(end);
            closed = m.insert_line(a, b);
            true
        });
        self.measure = self.units.format(start.distance(end));
        self.state = if closed { State::Idle } else { State::Line { start: end } };
    }

    fn finish_rect(&mut self, p0: Vec3, p1: Vec3, n: Vec3, u: Vec3, v: Vec3, on_face: bool) {
        let d = p1 - p0;
        let (a, b) = (d.dot(u), d.dot(v));
        if a.abs() < EPS || b.abs() < EPS {
            return;
        }
        let pts = [p0, p0 + u * a, p0 + u * a + v * b, p0 + v * b];
        let hint = on_face.then_some(n);
        self.commit(|m| m.add_polygon(&pts, hint).is_some());
        self.measure = format!("{}, {}", self.units.format(a.abs()), self.units.format(b.abs()));
        self.state = State::Idle;
    }

    fn finish_circle(&mut self, center: Vec3, edge: Vec3, n: Vec3, on_face: bool) {
        let rv = in_plane(edge - center, n);
        if rv.length() < EPS {
            return;
        }
        let segs = if self.tool == Tool::Polygon { self.polygon_sides } else { self.circle_segments };
        let pts = circle_points(center, n, rv, segs);
        let hint = on_face.then_some(n);
        self.commit(|m| {
            let ok = m.add_polygon(&pts, hint).is_some();
            m.mark_curve(&pts, true);
            ok
        });
        self.measure = self.units.format(rv.length());
        self.state = State::Idle;
    }

    /// Commits an open arc as one curve (it may close or split faces).
    fn finish_arc(&mut self, pts: Vec<Vec3>) {
        self.state = State::Idle;
        if pts.len() < 2 {
            return;
        }
        self.commit(|m| {
            let ids: Vec<Id> = pts.iter().map(|p| m.vertex_on_geometry(*p)).collect();
            m.insert_curve(&ids);
            true
        });
    }

    fn finish_center_arc(&mut self, angle_override: Option<f32>) {
        let State::ArcSweep { center, start, n, on_face, angle } = self.state.clone() else { return };
        let tau = std::f32::consts::TAU;
        let a = angle_override.unwrap_or(angle).clamp(-tau, tau);
        self.state = State::Idle;
        if a.abs() < 1e-4 {
            return;
        }
        let segs = self.arc_segments.max(2);
        let hint = on_face.then_some(n);
        self.measure = format!("{:.1}°", a.to_degrees().abs());
        if a.abs() > tau - 1e-3 {
            let pts = circle_points(center, n, start - center, segs);
            self.commit(|m| {
                let ok = m.add_polygon(&pts, hint).is_some();
                m.mark_curve(&pts, true);
                ok
            });
            return;
        }
        let arc = arc_points_center(center, start, n, a, segs);
        if self.tool == Tool::Pie {
            let mut poly = vec![center];
            poly.extend(arc.iter().copied());
            self.commit(|m| {
                let ok = m.add_polygon(&poly, hint).is_some();
                m.mark_curve(&arc, false);
                ok
            });
        } else {
            self.finish_arc(arc);
        }
    }

    /// Intersect Faces. With Model: the selected faces (or every face when
    /// nothing is selected) against all faces and groups in the context.
    /// With Selection: only the selected faces and groups against each other.
    fn intersect(&mut self, with_model: bool) {
        let sel_faces = self.sel.faces();
        let all: Vec<Id> = self.model.faces.keys().copied().collect();
        let targets = if sel_faces.is_empty() && with_model { all.clone() } else { sel_faces.clone() };
        if targets.is_empty() {
            self.status = "Select faces to intersect.".into();
            return;
        }
        let mut cutters = self.model.cut_faces(if with_model { &all } else { &sel_faces });
        let insts: Vec<Id> = if with_model { self.doc.defs[&self.ctx_def()].instances.keys().copied().collect() } else { self.sel.insts() };
        for i in insts {
            if let Some(mesh) = self.doc.instance_world_mesh(&self.path, i) {
                let ids: Vec<Id> = mesh.faces.keys().copied().collect();
                let mut c = mesh.cut_faces(&ids);
                c.iter_mut().for_each(|c| c.id = None); // ids belong to another mesh
                cutters.extend(c);
            }
        }
        let mut added = 0;
        self.commit(|m| {
            added = m.intersect_faces(&targets, &cutters);
            added > 0
        });
        self.status =
            if added > 0 { format!("Intersect Faces: added {added} edge(s)") } else { "Intersect Faces: no intersections found".into() };
    }

    /// Follow Me: sweeps the clicked profile face along the selected path.
    fn follow_me(&mut self, profile: Id) {
        let prof: HashSet<Id> = self.model.face_edges(profile).into_iter().collect();
        let mut path: Vec<Id> = self.sel.edges.iter().copied().filter(|e| !prof.contains(e)).collect();
        for f in self.sel.faces.iter().copied().filter(|f| *f != profile) {
            if let Some(face) = self.model.faces.get(&f) {
                let lp = &face.outer;
                for i in 0..lp.len() {
                    if let Some(e) = self.model.find_edge(lp[i], lp[(i + 1) % lp.len()]) {
                        if !prof.contains(&e) && !path.contains(&e) {
                            path.push(e);
                        }
                    }
                }
            }
        }
        if path.is_empty() {
            self.status = "Select the path first (edges, or a face to use its outline), then click the profile.".into();
            return;
        }
        let Some((verts, closed)) = self.model.chain_from_edges(&path) else {
            self.status = "The path must be one connected chain of edges (no branches).".into();
            return;
        };
        let pts: Vec<Vec3> = verts.iter().map(|v| self.model.pos(*v)).collect();
        if self.commit(|m| m.follow_me(profile, &pts, closed)).is_some() {
            self.sel.clear();
            self.status = format!("Follow Me: swept along {} segment(s)", path.len());
        } else {
            self.status = "Follow Me could not sweep that profile.".into();
        }
    }

    /// Which way along a face's normal counts as pulling it out (+1 or -1).
    /// A flat face lying on its own faces down, so pulling it up goes
    /// against its normal; everywhere else out is along the normal.
    fn pull_sign(&self, face: Id) -> f32 {
        let Some(f) = self.model.faces.get(&face) else { return 1.0 };
        let alone = self.model.face_edges(face).iter().all(|e| self.model.edge_faces(*e).len() <= 1);
        if alone && f.normal.z < -0.9 {
            -1.0
        } else {
            1.0
        }
    }

    fn finish_pushpull(&mut self, face: Id, d: f32) {
        let pull = d * self.pull_sign(face);
        if let Some(before) = self.commit(|m| m.push_pull(face, d)) {
            self.last_op = Some(LastOp::PushPull { before, face, sign: d.signum() });
            self.last_pp = Some(pull);
            self.measure = self.units.format(d.abs());
        }
        self.state = State::Idle;
    }

    /// Applies a world matrix to loose geometry and instances of the context.
    fn apply_transform(&mut self, t: &Targets, mat: Mat4, copy: bool) {
        let new_geo = apply_to_model(&mut self.model, t, mat, copy);
        let new_insts = self.doc.transform_instances(&self.path, &t.insts, mat, copy);
        let mut new_guides = Vec::new();
        for &i in &t.guides {
            let Some(g) = self.doc.guides.get(i).map(|g| g.transformed(mat)) else { continue };
            if copy {
                new_guides.push(self.doc.guides.len());
                self.doc.guides.push(g);
            } else {
                self.doc.guides[i] = g;
            }
        }
        let mut new_annots = Vec::new();
        for &i in &t.annots {
            let Some(a) = self.doc.annotations.get(i).map(|a| a.transformed(mat)) else { continue };
            if copy {
                new_annots.push(self.doc.annotations.len());
                self.doc.annotations.push(a);
            } else {
                self.doc.annotations[i] = a;
            }
        }
        self.sync();
        if copy {
            self.sel.clear();
            if let Some((f, e)) = new_geo {
                self.sel.faces.extend(f);
                self.sel.edges.extend(e);
            }
            self.sel.insts.extend(new_insts);
            self.sel.guides.extend(new_guides);
            self.sel.annots.extend(new_annots);
        }
    }

    fn finish_transform(&mut self, op: XOp, t: Targets, copy: bool) {
        self.state = State::Idle;
        let before = self.checkpoint();
        self.apply_transform(&t, op.matrix(), copy);
        self.last_op = Some(LastOp::Transform { before, t, op, copy });
    }

    fn finish_move(&mut self, delta_override: Option<Vec3>) {
        let State::Move { t, delta, copy, .. } = self.state.clone() else { return };
        let delta = delta_override.unwrap_or(delta);
        self.state = State::Idle;
        if delta.length() > EPS {
            self.measure = self.units.format(delta.length());
            self.finish_transform(XOp::Move(delta), t, copy);
        }
        self.end_auto_sel();
    }

    fn finish_rotate(&mut self, angle_override: Option<f32>) {
        let State::RotateAngle { t, center, n, angle, .. } = self.state.clone() else { return };
        let a = angle_override.unwrap_or(angle);
        self.state = State::Idle;
        if a.abs() > 1e-6 {
            self.measure = format!("{:.1}°", a.to_degrees().abs());
            self.finish_transform(XOp::Rotate { center, n, angle: a }, t, false);
        }
        self.end_auto_sel();
    }

    fn finish_scale(&mut self, factor_override: Option<Vec3>) {
        let State::Scale { t, center, factor, .. } = self.state.clone() else { return };
        let f = factor_override.unwrap_or(factor);
        self.state = State::Idle;
        if f.x.abs() < 1e-4 || f.y.abs() < 1e-4 || f.z.abs() < 1e-4 {
            self.status = "Scale factor cannot be zero".into();
            self.end_auto_sel();
            return;
        }
        self.checkpoint();
        self.apply_transform(&t, about(center, Mat4::from_scale(f)), false);
        self.measure = format!("{:.3}, {:.3}, {:.3}", f.x, f.y, f.z);
        self.end_auto_sel();
    }

    fn finish_offset(&mut self, face: Id, d: f32) {
        if let Some(before) = self.commit(|m| m.offset_face(face, d)) {
            self.last_op = Some(LastOp::Offset { before, face, sign: d.signum() });
            self.measure = self.units.format(d.abs());
        }
        self.state = State::Idle;
    }

    /// Geometry an edit tool acts on: the selection, or the hovered entity.
    fn take_targets(&mut self, s: &Snap) -> Option<Targets> {
        if self.sel.is_empty() {
            // Nothing is selected: act on what is under the cursor, for this
            // one operation. Move grabs a single corner point to stretch.
            if let (Tool::Move, SnapKind::Endpoint, Some(v)) = (self.tool, s.kind, s.vertex) {
                let verts = [v].into_iter().collect();
                return Some(Targets { verts, faces: vec![], edges: vec![], insts: vec![], guides: vec![], annots: vec![] });
            }
            if let Some(e) = self.hover_edge {
                self.sel.edges.insert(e);
            } else if let Some(a) = self.hover_annot {
                self.sel.annots.insert(a);
            } else if let Some(g) = self.hover_guide {
                self.sel.guides.insert(g);
            } else if let Some(f) = s.face {
                self.sel.faces.insert(f);
                self.sel.edges.extend(self.model.face_edges(f));
            } else {
                self.sel.insts.insert(s.instance?);
            }
            self.auto_sel = true;
        }
        let (faces, edges, insts) = (self.sel.faces(), self.sel.edges(), self.sel.insts());
        let (guides, annots) = (self.sel.guides(), self.sel.annots());
        Some(Targets { verts: self.model.vertices_of(&faces, &edges), faces, edges, insts, guides, annots })
    }

    /// Plane normal that glued instances slide along when moved on their own.
    fn glue_normal(&self, t: &Targets) -> Option<Vec3> {
        if !t.faces.is_empty() || !t.edges.is_empty() || t.insts.is_empty() {
            return None;
        }
        let ctx = self.doc.defs.get(&self.ctx_def())?;
        let mut n: Option<Vec3> = None;
        for i in &t.insts {
            let f = ctx.instances.get(i)?.glued?;
            let fnorm = self.model.faces.get(&f)?.normal;
            match n {
                None => n = Some(fnorm),
                Some(m) if m.dot(fnorm).abs() > 0.9999 => {}
                _ => return None,
            }
        }
        n
    }

    /// Geometry that is being dragged along right now (not while copying:
    /// the original then stays where it is and can be snapped to).
    fn moving(&self) -> Option<&Targets> {
        match &self.state {
            State::Move { t, copy: false, .. } | State::RotateAngle { t, .. } | State::Scale { t, .. } => Some(t),
            _ => None,
        }
    }

    /// Bulges that make a 2-point arc leave `p0`, or arrive at `p1`, tangent
    /// to an edge that ends there.
    fn arc_tangent_bulges(&self, p0: Vec3, p1: Vec3, n: Vec3) -> Vec<f32> {
        tangent_bulges(&self.model, p0, p1, n)
    }

    fn targets_bounds(&self, t: &Targets) -> Option<(Vec3, Vec3)> {
        let mut bb = self.model.bounds_of(&t.verts);
        for i in &t.insts {
            if let Some((_, corners)) = self.doc.instance_world_bounds(&self.path, *i) {
                for c in corners {
                    bb = grow(bb, c);
                }
            }
        }
        bb
    }

    fn plane_for(&self, s: &Snap) -> (Vec3, bool) {
        if let Some(i) = self.lock_axis {
            return (self.axes()[i], false);
        }
        match (s.normal, s.kind) {
            (Some(n), SnapKind::OnFace | SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::OnEdge) => (n, true),
            _ => (Vec3::Z, false),
        }
    }

    fn tool_click(&mut self, s: &Snap, mods: Modifiers) {
        match (self.tool, self.state.clone()) {
            (Tool::Line, State::Idle) => self.state = State::Line { start: s.pos },
            (Tool::Line, State::Line { start }) => self.line_to(start, s.pos),

            (Tool::Rectangle, State::Idle) => {
                let (n, on_face) = self.plane_for(s);
                let (u, v) = drawing_axes(n);
                self.state = State::Rect { p0: s.pos, p1: s.pos, n, u, v, on_face };
            }
            (Tool::Rectangle, State::Rect { p0, p1, n, u, v, on_face }) => self.finish_rect(p0, p1, n, u, v, on_face),

            (Tool::Circle | Tool::Polygon, State::Idle) => {
                let (n, on_face) = self.plane_for(s);
                self.state = State::Circle { center: s.pos, edge: s.pos, n, on_face };
            }
            (Tool::Circle | Tool::Polygon, State::Circle { center, edge, n, on_face }) => self.finish_circle(center, edge, n, on_face),

            (Tool::Arc, State::Idle) => {
                let (n, _) = self.plane_for(s);
                self.state = State::ArcEnd { p0: s.pos, p1: s.pos, n };
            }
            (Tool::Arc, State::ArcEnd { p0, p1, n }) => {
                if p0.distance(p1) > EPS {
                    self.state = State::ArcBulge { p0, p1, n, bulge: 0.0 };
                }
            }
            (Tool::Arc, State::ArcBulge { p0, p1, n, bulge }) => {
                let pts = arc_points_2pt(p0, p1, n, bulge, self.arc_segments.max(2));
                self.finish_arc(pts);
            }
            (Tool::CenterArc | Tool::Pie, State::Idle) => {
                let (n, on_face) = self.plane_for(s);
                self.state = State::ArcStart { center: s.pos, edge: s.pos, n, on_face };
            }
            (Tool::CenterArc | Tool::Pie, State::ArcStart { center, edge, n, on_face }) => {
                if center.distance(edge) > EPS {
                    self.state = State::ArcSweep { center, start: edge, n, on_face, angle: 0.0 };
                }
            }
            (Tool::CenterArc | Tool::Pie, State::ArcSweep { .. }) => self.finish_center_arc(None),

            (Tool::FollowMe, State::Idle) => match s.face {
                Some(f) => self.follow_me(f),
                None => self.status = "Click a face (the profile) to sweep along the selected path.".into(),
            },

            (Tool::PushPull, State::Idle) => {
                if let Some(f) = s.face {
                    let n = self.model.faces[&f].normal;
                    self.state = State::PushPull { face: f, start: s.pos, n, dist: 0.0 };
                } else if s.instance.is_some() {
                    self.status = "Double-click the group/component (Select tool) to edit its faces.".into();
                }
            }
            (Tool::PushPull, State::PushPull { face, dist, .. }) => self.finish_pushpull(face, dist),

            (Tool::Move, State::Idle) => match self.take_targets(s) {
                Some(t) => self.state = State::Move { t, base: s.pos, delta: Vec3::ZERO, copy: mods.ctrl },
                None => self.status = "Hover or select geometry to move.".into(),
            },
            (Tool::Move, State::Move { .. }) => self.finish_move(None),

            (Tool::Rotate, State::Idle) => match self.take_targets(s) {
                Some(t) => {
                    let n = match self.lock_axis {
                        Some(i) => self.axes()[i],
                        None => s.normal.filter(|_| s.kind == SnapKind::OnFace).unwrap_or(self.axes()[2]),
                    };
                    self.state = State::RotateCenter { t, center: s.pos, n };
                }
                None => self.status = "Select geometry to rotate first.".into(),
            },
            (Tool::Rotate, State::RotateCenter { t, center, n }) => {
                let d = in_plane(s.pos - center, n);
                if d.length() > EPS {
                    self.state = State::RotateAngle { t, center, n, ref_dir: d.normalize(), angle: 0.0 };
                }
            }
            (Tool::Rotate, State::RotateAngle { .. }) => self.finish_rotate(None),

            (Tool::Scale, State::Idle) => match self.take_targets(s) {
                Some(t) => {
                    if let Some((lo, hi)) = self.targets_bounds(&t) {
                        let center = Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, lo.z);
                        let ref_dist = (s.pos - center).length();
                        if ref_dist > EPS {
                            self.state = State::Scale { t, center, ref_dist, factor: Vec3::ONE };
                        }
                    }
                }
                None => self.status = "Select geometry to scale first.".into(),
            },
            (Tool::Scale, State::Scale { .. }) => self.finish_scale(None),

            (Tool::Offset, State::Idle) => {
                if let Some(f) = s.face {
                    self.state = State::Offset { face: f, dist: 0.0 };
                }
            }
            (Tool::Offset, State::Offset { face, dist }) => self.finish_offset(face, dist),

            (Tool::Tape, State::Idle) => {
                self.tape_result = None;
                // Starting on an edge or guide line (not at a point) makes a parallel guide.
                let dir = match s.kind {
                    SnapKind::OnEdge | SnapKind::OnGuide => s.line_dir.filter(|d| d.length() > 0.5),
                    _ => None,
                };
                self.state = State::Tape { start: s.pos, dir, end: s.pos };
            }
            (Tool::Tape, State::Tape { start, dir, end }) => self.finish_tape(start, dir, end),

            (Tool::Protractor, State::Idle) => {
                let n = match self.lock_axis {
                    Some(i) => self.axes()[i],
                    None => s.normal.filter(|_| s.kind == SnapKind::OnFace).unwrap_or(self.axes()[2]),
                };
                self.state = State::ProtBase { center: s.pos, n };
            }
            (Tool::Protractor, State::ProtBase { center, n }) => {
                let d = in_plane(s.pos - center, n);
                if d.length() > EPS {
                    self.state = State::ProtAngle { center, n, ref_dir: d.normalize(), angle: 0.0 };
                }
            }
            (Tool::Protractor, State::ProtAngle { .. }) => self.finish_protractor(None),
            (Tool::Dimension, State::Idle) => {
                let edge = s.edge.filter(|_| s.kind == SnapKind::OnEdge);
                if let Some((center, radius, normal, diameter)) = edge.and_then(|e| self.curve_circle(e)) {
                    self.state = State::DimRadial { center, radius, normal, diameter, at: s.pos };
                } else if let Some(e) = edge.and_then(|e| self.model.edges.get(&e)) {
                    let (a, b) = (self.model.pos(e.a), self.model.pos(e.b));
                    self.state = State::DimOffset { a, b, offset: Vec3::ZERO };
                } else {
                    self.state = State::DimSecond { a: s.pos };
                }
            }
            (Tool::Dimension, State::DimSecond { a }) => {
                if s.pos.distance(a) > EPS {
                    self.state = State::DimOffset { a, b: s.pos, offset: Vec3::ZERO };
                }
            }
            (Tool::Dimension, State::DimOffset { a, b, offset }) => self.add_annotation(Annotation::Dimension { a, b, offset, text: None }),
            (Tool::Dimension, State::DimRadial { center, radius, normal, diameter, at }) => {
                self.add_annotation(Annotation::Radial { center, radius, normal, diameter, at, text: None })
            }
            (Tool::Text, State::Idle) => {
                let on_something = s.hit
                    || matches!(
                        s.kind,
                        SnapKind::Endpoint
                            | SnapKind::Midpoint
                            | SnapKind::OnEdge
                            | SnapKind::GuidePoint
                            | SnapKind::Intersection
                            | SnapKind::OnGuide
                    );
                if on_something {
                    let text = self.default_text(s);
                    self.state = State::TextPlace { anchor: s.pos, text, at: s.pos };
                } else if let Some(pos) = self.cursor_frac {
                    self.add_annotation(Annotation::ScreenText { pos, text: "Text".into() });
                }
            }
            (Tool::Text, State::TextPlace { anchor, text, mut at }) => {
                if at.distance(anchor) < self.cam.distance * 0.01 {
                    at = anchor + self.cam.up() * self.cam.distance * 0.06;
                }
                self.add_annotation(Annotation::Text { anchor, at, text });
            }
            (t, State::Idle) if t.is_solid_tool() => match s.instance {
                Some(i) if self.solid_mesh(i).is_some() => {
                    self.sel.clear();
                    self.sel.insts.insert(i);
                    self.state = State::SolidFirst { first: i };
                    self.status = format!("{}: now click the second solid", t.label());
                }
                Some(_) => self.status = NOT_SOLID.into(),
                None => self.status = "Click a solid group or component.".into(),
            },
            (t, State::SolidFirst { first }) if t.is_solid_tool() => match s.instance {
                Some(i) if i == first => {}
                Some(i) if self.solid_mesh(i).is_some() => self.run_solid(t, first, i),
                Some(_) => self.status = NOT_SOLID.into(),
                None => {}
            },
            _ => {}
        }
    }

    // ----- solid tools -------------------------------------------------------

    /// World-space mesh of an instance, if it is a solid.
    fn solid_mesh(&self, iid: Id) -> Option<Model> {
        let m = self.doc.instance_world_mesh(&self.path, iid)?;
        solid::is_solid(&m).then_some(m)
    }

    /// Replaces instances by the solids that result from an operation.
    fn replace_with(&mut self, remove: &[Id], results: Vec<Model>, what: &str) {
        self.checkpoint();
        self.doc.remove_instances(&self.path, remove);
        self.sel.clear();
        let mut all_solid = true;
        for m in results.into_iter().filter(|m| !m.faces.is_empty()) {
            all_solid &= solid::is_solid(&m);
            if let Some(i) = self.doc.add_world_mesh(&self.path, &m, DefKind::Group) {
                self.sel.insts.insert(i);
            }
        }
        self.state = State::Idle;
        self.status = match (self.sel.insts.len(), all_solid) {
            (0, _) => format!("{what}: nothing is left"),
            (_, true) => what.to_string(),
            (_, false) => format!("{what} (warning: the result is not a closed solid)"),
        };
    }

    /// Runs a solid tool on `a` (picked first) and `b` (picked second). As in
    /// SketchUp, Subtract and Trim cut the first solid out of the second.
    fn run_solid(&mut self, tool: Tool, a: Id, b: Id) {
        let (Some(ma), Some(mb)) = (self.solid_mesh(a), self.solid_mesh(b)) else {
            self.status = NOT_SOLID.into();
            return;
        };
        let (results, remove, what): (Vec<Model>, Vec<Id>, &str) = match tool {
            Tool::OuterShell | Tool::Union => (vec![solid::boolean(&ma, &mb, BoolOp::Union)], vec![a, b], "Union"),
            Tool::SolidIntersect => (vec![solid::boolean(&ma, &mb, BoolOp::Intersect)], vec![a, b], "Intersect"),
            Tool::Subtract => (vec![solid::boolean(&mb, &ma, BoolOp::Subtract)], vec![a, b], "Subtract"),
            Tool::Trim => (vec![solid::boolean(&mb, &ma, BoolOp::Subtract)], vec![b], "Trim"),
            Tool::Split => (
                vec![
                    solid::boolean(&ma, &mb, BoolOp::Subtract),
                    solid::boolean(&mb, &ma, BoolOp::Subtract),
                    solid::boolean(&ma, &mb, BoolOp::Intersect),
                ],
                vec![a, b],
                "Split",
            ),
            _ => return,
        };
        if tool == Tool::SolidIntersect && results[0].faces.is_empty() {
            self.state = State::Idle;
            self.status = "The solids don't overlap.".into();
            return;
        }
        let keep_first = tool == Tool::Trim;
        self.replace_with(&remove, results, what);
        if keep_first {
            self.sel.insts.insert(a);
        }
    }

    /// Outer Shell / Union / Intersect applied to every selected solid at once.
    fn solid_selection(&mut self, tool: Tool) {
        let insts = self.sel.insts();
        let meshes: Option<Vec<Model>> = insts.iter().map(|i| self.solid_mesh(*i)).collect();
        let Some(meshes) = meshes else {
            self.status = NOT_SOLID.into();
            return;
        };
        let op = if tool == Tool::SolidIntersect { BoolOp::Intersect } else { BoolOp::Union };
        let mut it = meshes.into_iter();
        let first = it.next().unwrap();
        let result = it.fold(first, |acc, m| solid::boolean(&acc, &m, op));
        if result.faces.is_empty() {
            self.status = "The solids don't overlap.".into();
            return;
        }
        self.replace_with(&insts, vec![result], tool.label());
    }

    /// Updates live values (distance, angle, ...) of the active tool from the cursor.
    fn update_state(&mut self, s: &Snap, o: Vec3, d: Vec3) {
        let plane_point = |p0: Vec3, n: Vec3| match s.kind {
            SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::OnEdge => s.pos - n * (s.pos - p0).dot(n),
            _ => ray_plane(o, d, p0, n).unwrap_or(s.pos - n * (s.pos - p0).dot(n)),
        };
        let glue_n = match &self.state {
            State::Move { t, .. } => self.glue_normal(t),
            _ => None,
        };
        let mut st = self.state.clone();
        match &mut st {
            State::Line { start } => self.measure = self.units.format(s.pos.distance(*start)),
            State::Rect { p0, p1, n, u, v, .. } => {
                *p1 = plane_point(*p0, *n);
                let dd = *p1 - *p0;
                self.measure = format!("{}, {}", self.units.format(dd.dot(*u).abs()), self.units.format(dd.dot(*v).abs()));
            }
            State::Circle { center, edge, n, .. } => {
                *edge = plane_point(*center, *n);
                self.measure = self.units.format(center.distance(*edge));
            }
            State::ArcEnd { p0, p1, n } => {
                *p1 = plane_point(*p0, *n);
                self.measure = self.units.format(p0.distance(*p1));
            }
            State::ArcBulge { p0, p1, n, bulge } => {
                let q = plane_point(*p0, *n);
                let chord = *p1 - *p0;
                let l = chord.length().max(EPS);
                let m = n.cross(chord / l);
                let mut b = (q - (*p0 + *p1) * 0.5).dot(m);
                if (b.abs() - l / 2.0).abs() < l * 0.04 {
                    b = l / 2.0 * b.signum(); // snap to a half circle
                }
                // Snap to an arc that continues an edge smoothly.
                self.arc_tangent = false;
                for t in self.arc_tangent_bulges(*p0, *p1, *n) {
                    if (b - t).abs() < l.max(t.abs()) * 0.05 {
                        b = t;
                        self.arc_tangent = true;
                    }
                }
                *bulge = b;
                self.measure = self.units.format(b.abs());
            }
            State::ArcStart { center, edge, n, .. } => {
                *edge = plane_point(*center, *n);
                self.measure = self.units.format(center.distance(*edge));
            }
            State::ArcSweep { center, start, n, angle, .. } => {
                let p = plane_point(*center, *n);
                let (r0, cur) = (*start - *center, in_plane(p - *center, *n));
                if cur.length() > EPS {
                    let raw = r0.cross(cur).dot(*n).atan2(r0.dot(cur));
                    // Unwrap so the sweep can grow past 180°.
                    let (pi, tau) = (std::f32::consts::PI, std::f32::consts::TAU);
                    let mut a = raw;
                    while a - *angle > pi {
                        a -= tau;
                    }
                    while a - *angle < -pi {
                        a += tau;
                    }
                    a = a.clamp(-tau, tau);
                    let step = 15f32.to_radians();
                    let snapped = (a / step).round() * step;
                    if (a - snapped).abs() < 2.5f32.to_radians() {
                        a = snapped;
                    }
                    *angle = a;
                    self.measure = format!("{:.1}°", a.to_degrees());
                }
            }
            State::PushPull { face, start, n, dist } => {
                // The distance follows the cursor along the face's normal, or
                // stops level with other geometry the cursor is on.
                let own = self.model.faces.get(face).map(|f| f.vertices().collect::<HashSet<_>>()).unwrap_or_default();
                let free = (closest_on_line_to_ray(*start, *n, o, d) - *start).dot(*n);
                let level = (s.pos - *start).dot(*n);
                let own_edge = s.edge.and_then(|e| self.model.edges.get(&e)).is_some_and(|e| own.contains(&e.a) && own.contains(&e.b));
                let inferred = match s.kind {
                    SnapKind::Endpoint => s.vertex.is_none_or(|v| !own.contains(&v)),
                    SnapKind::GuidePoint | SnapKind::Intersection => true,
                    SnapKind::Midpoint | SnapKind::OnEdge => !own_edge,
                    // Level with another face that is parallel to this one, when pulling its way.
                    SnapKind::OnFace => s.face != Some(*face) && s.normal.is_some_and(|m| m.dot(*n).abs() > 0.9999) && level * free > 0.0,
                    _ => false,
                };
                *dist = if inferred && level.abs() > EPS { level } else { free };
                self.measure = self.units.format(dist.abs());
            }
            State::Move { base, delta, .. } => {
                *delta = match glue_n {
                    // Glued components slide within their face's plane.
                    Some(n) => plane_point(*base, n) - *base,
                    None => s.pos - *base,
                };
                self.measure = self.units.format(delta.length());
            }
            State::RotateAngle { center, n, ref_dir, angle, .. } | State::ProtAngle { center, n, ref_dir, angle } => {
                let p = ray_plane(o, d, *center, *n).unwrap_or(s.pos);
                let cur = in_plane(p - *center, *n);
                if cur.length() > EPS {
                    let mut a = ref_dir.cross(cur).dot(*n).atan2(ref_dir.dot(cur));
                    let step = 15f32.to_radians();
                    let snapped = (a / step).round() * step;
                    if (a - snapped).abs() < 2.5f32.to_radians() {
                        a = snapped;
                    }
                    *angle = a;
                    self.measure = format!("{:.1}°", a.to_degrees());
                }
            }
            State::Scale { center, ref_dist, factor, .. } => {
                let mut f = (s.pos - *center).length() / *ref_dist;
                if ((f * 20.0).round() / 20.0 - f).abs() < 0.01 {
                    f = (f * 20.0).round() / 20.0;
                }
                *factor = Vec3::splat(f.max(0.01));
                self.measure = format!("{f:.2}");
            }
            State::Offset { face, dist } => {
                if let Some(f) = self.model.faces.get(face) {
                    let p0 = self.model.pos(f.outer[0]);
                    let p = ray_plane(o, d, p0, f.normal).unwrap_or(s.pos);
                    let pts = self.model.loop_points(&f.outer);
                    let dmin = (0..pts.len()).map(|i| point_seg(p, pts[i], pts[(i + 1) % pts.len()]).0).fold(f32::MAX, f32::min);
                    *dist = if self.model.point_in_face(*face, p) { dmin } else { -dmin };
                    self.measure = self.units.format(dist.abs());
                }
            }
            State::DimOffset { a, b, offset } => {
                *offset = self.dim_offset(*a, *b, o, d);
                self.measure = self.units.format(a.distance(*b));
            }
            State::DimSecond { a } => self.measure = self.units.format(a.distance(s.pos)),
            State::DimRadial { center, normal, at, .. } => {
                *at = ray_plane(o, d, *center, *normal).unwrap_or(s.pos);
            }
            State::TextPlace { anchor, at, .. } => {
                *at = ray_plane(o, d, *anchor, -self.cam.forward()).unwrap_or(s.pos);
            }
            State::Tape { start, dir, end } => {
                *end = match dir {
                    Some(dir) => {
                        let v = s.pos - *start;
                        *start + v - *dir * v.dot(*dir)
                    }
                    None => s.pos,
                };
                self.measure = self.units.format(start.distance(*end));
            }
            _ => {}
        }
        self.state = st;
    }

    // ----- selection ---------------------------------------------------------

    fn click_select(&mut self, mods: Modifiers) {
        let toggle = mods.shift;
        let add = mods.ctrl;
        if !toggle && !add {
            self.sel.clear();
        }
        let flip = |set: &mut BTreeSet<Id>, id: Id| {
            if toggle && set.contains(&id) {
                set.remove(&id);
            } else {
                set.insert(id);
            }
        };
        if let Some(a) = self.hover_annot {
            if toggle && self.sel.annots.contains(&a) {
                self.sel.annots.remove(&a);
            } else {
                self.sel.annots.insert(a);
            }
            return;
        }
        if let Some(e) = self.hover_edge {
            // Arcs and circles select as a whole curve.
            let curve = self.model.curve_edges(e);
            if toggle && self.sel.edges.contains(&e) {
                curve.iter().for_each(|x| {
                    self.sel.edges.remove(x);
                });
            } else {
                self.sel.edges.extend(curve);
            }
        } else if let Some(g) = self.hover_guide {
            if toggle && self.sel.guides.contains(&g) {
                self.sel.guides.remove(&g);
            } else {
                self.sel.guides.insert(g);
            }
        } else if let Some(f) = self.hover_face {
            flip(&mut self.sel.faces, f);
        } else if let Some(i) = self.hover_inst {
            flip(&mut self.sel.insts, i);
        } else if !toggle && !add && !self.path.is_empty() {
            // Clicking outside the open group closes it, like SketchUp.
            self.exit_context();
            self.sel.clear();
        }
    }

    fn select_expand(&mut self) {
        if let Some(e) = self.hover_edge {
            self.sel.edges.insert(e);
            self.sel.faces.extend(self.model.edge_faces(e));
        } else if let Some(f) = self.hover_face {
            self.sel.faces.insert(f);
            self.sel.edges.extend(self.model.face_edges(f));
        }
    }

    fn select_connected(&mut self) {
        let seeds: Vec<Id> = if let Some(e) = self.hover_edge {
            self.model.edges.get(&e).map(|e| vec![e.a]).unwrap_or_default()
        } else if let Some(f) = self.hover_face {
            self.model.faces.get(&f).map(|f| vec![f.outer[0]]).unwrap_or_default()
        } else {
            vec![]
        };
        if seeds.is_empty() {
            return;
        }
        let (f, e) = self.model.connected(&seeds);
        self.sel.faces.extend(f);
        self.sel.edges.extend(e);
    }

    fn box_select(&mut self, rect: Rect, a: Pos2, b: Pos2, mods: Modifiers) {
        let r = Rect::from_two_pos(a, b);
        let window = b.x >= a.x;
        if !mods.shift && !mods.ctrl {
            self.sel.clear();
        }
        let proj = |p: Vec3| self.cam.project(rect, p);
        let mut edges = Vec::new();
        for (&id, e) in &self.model.edges {
            let (Some(pa), Some(pb)) = (proj(self.model.pos(e.a)), proj(self.model.pos(e.b))) else { continue };
            let hit = if window { r.contains(pa) && r.contains(pb) } else { seg_hits_rect(pa, pb, r) };
            if hit {
                edges.push(id);
            }
        }
        let mut faces = Vec::new();
        for (&id, f) in &self.model.faces {
            let pts: Vec<Option<Pos2>> = f.outer.iter().map(|v| proj(self.model.pos(*v))).collect();
            let hit = if window {
                pts.iter().all(|p| p.is_some_and(|p| r.contains(p)))
            } else {
                (0..pts.len()).any(|i| match (pts[i], pts[(i + 1) % pts.len()]) {
                    (Some(a), Some(b)) => seg_hits_rect(a, b, r),
                    _ => false,
                })
            };
            if hit {
                faces.push(id);
            }
        }
        let mut insts = Vec::new();
        for &iid in self.doc.defs[&self.ctx_def()].instances.keys() {
            let Some((_, corners)) = self.doc.instance_world_bounds(&self.path, iid) else { continue };
            let pts: Vec<Pos2> = corners.iter().filter_map(|c| proj(*c)).collect();
            if pts.len() < 8 {
                continue;
            }
            let hit = if window { pts.iter().all(|p| r.contains(*p)) } else { r.intersects(Rect::from_points(&pts)) };
            if hit {
                insts.push(iid);
            }
        }
        self.sel.edges.extend(edges);
        self.sel.faces.extend(faces);
        self.sel.insts.extend(insts);
    }

    fn paint_click(&mut self, s: &Snap, mods: Modifiers) {
        if let Some(f) = s.face {
            if mods.alt {
                self.material = self.model.faces[&f].material;
                self.status = "Sampled material".into();
                return;
            }
            let targets: Vec<Id> = if self.sel.faces.contains(&f) { self.sel.faces() } else { vec![f] };
            let mat = self.material;
            self.commit(|m| {
                for t in &targets {
                    if let Some(face) = m.faces.get_mut(t) {
                        face.material = mat;
                    }
                }
                true
            });
        } else if let Some(i) = s.instance {
            if mods.alt {
                self.material = self.doc.instance(&self.path, i).and_then(|x| x.material);
                return;
            }
            let targets = if self.sel.insts.contains(&i) { self.sel.insts() } else { vec![i] };
            self.set_instance_material(&targets, self.material);
        }
    }

    // ----- viewport -----------------------------------------------------------

    /// The point axis inference measures from (the tool's previous click).
    fn inference_origin(&self) -> Option<Vec3> {
        match &self.state {
            State::Line { start } | State::Tape { start, .. } => Some(*start),
            State::DimSecond { a } => Some(*a),
            State::ArcEnd { p0, .. } => Some(*p0),
            State::Move { base, .. } => Some(*base),
            _ => None,
        }
    }

    fn root_node(&self) -> Node {
        Node {
            def: ROOT,
            xf: Mat4::IDENTITY,
            level: Some(0),
            owner: None,
            mat: None,
            faded: !self.path.is_empty(),
            selected: false,
            hovered: false,
            tag: None,
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        paint_background(&painter, rect);

        if self.ai.pending_capture().is_some() {
            // AI Render wants a clean frame: no selection, hover highlights or overlays.
            let sel = std::mem::take(&mut self.sel);
            let hover = (self.hover_face.take(), self.hover_edge.take(), self.hover_inst.take(), self.hover_guide.take());
            let snap = self.snap.take();
            self.paint_scene(&painter, rect);
            self.sel = sel;
            (self.hover_face, self.hover_edge, self.hover_inst, self.hover_guide) = hover;
            self.snap = snap;
            return;
        }

        let mut scene = SceneOut::default();
        let opts = WalkOpts { active: &self.model, op: None, copy: false, render: false };
        self.walk(&opts, &mut scene, self.root_node());
        if let Some(t) = self.moving() {
            // What is being moved must not attract the cursor at its old place.
            let model = &self.model;
            scene.pick.retain(|(owner, _)| match owner {
                Owner::Face(f) => !model.faces.get(f).is_some_and(|face| face.vertices().any(|v| t.verts.contains(&v))),
                Owner::Instance(i) => !t.insts.contains(i),
                Owner::Outside => true,
            });
        }
        if self.show_guides && !self.doc.guides.is_empty() {
            let mut segs: Vec<(Vec3, Vec3)> =
                self.model.edges.values().filter(|e| !e.soft).map(|e| (self.model.pos(e.a), self.model.pos(e.b))).collect();
            segs.extend_from_slice(&scene.segs);
            scene.guide_pts = guide::intersections(&self.doc.guides, &segs);
        }

        let hover = ui.input(|i| i.pointer.hover_pos()).filter(|p| rect.contains(*p));
        let mods = ui.input(|i| i.modifiers);

        // Camera navigation (available in every tool).
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                let anchor = hover.and_then(|p| {
                    let picker = self.picker(rect, &scene);
                    let (o, d) = picker.ray(p);
                    match picker.pick_face(o, d) {
                        Some(h) => Some(o + d * h.t),
                        None => ray_plane(o, d, Vec3::ZERO, Vec3::Z),
                    }
                });
                self.cam_anim = None;
                self.cam.zoom((-scroll * 0.0018).exp(), anchor);
            }
        }
        let primary_nav = resp.dragged_by(PointerButton::Primary);
        let orbit = (resp.dragged_by(PointerButton::Middle) && !mods.shift) || (self.tool == Tool::Orbit && primary_nav);
        let pan = (resp.dragged_by(PointerButton::Middle) && mods.shift)
            || resp.dragged_by(PointerButton::Secondary)
            || (self.tool == Tool::Pan && primary_nav);
        if orbit || pan {
            self.cam_anim = None;
        }
        if orbit {
            self.cam.orbit(resp.drag_delta());
        }
        if pan {
            self.cam.pan(rect, resp.drag_delta());
        }

        // Inference.
        let from = self.inference_origin();
        let lock = if from.is_some() { self.lock_axis.or(self.shift_lock) } else { None };
        let picked = hover.map(|p| {
            let picker = self.picker(rect, &scene);
            let snap = picker.snap(p, from, lock);
            let (o, d) = picker.ray(p);
            let edge = picker.pick_model_edge(p, 6.0);
            let guide = if edge.is_none() { picker.pick_guide(p, 6.0) } else { None };
            (snap, edge, o, d, guide)
        });
        self.snap = picked.as_ref().map(|x| x.0.clone());
        if let Some(s) = &self.snap {
            // Remember what the cursor passed over, for "From Point" and
            // parallel / perpendicular inferences.
            if matches!(s.kind, SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::GuidePoint | SnapKind::Intersection)
                && !self.recent_pts.iter().any(|p| p.distance(s.pos) < 1e-4)
            {
                self.recent_pts.push(s.pos);
                if self.recent_pts.len() > 4 {
                    self.recent_pts.remove(0);
                }
            }
            if let (SnapKind::OnEdge, Some(d)) = (s.kind, s.line_dir) {
                if d.length() > 0.5 && !self.recent_dirs.iter().any(|e| e.dot(d).abs() > 0.9999) {
                    self.recent_dirs.push(d);
                    if self.recent_dirs.len() > 2 {
                        self.recent_dirs.remove(0);
                    }
                }
            }
        }
        self.hover_face = self.snap.as_ref().and_then(|s| s.face);
        self.hover_inst = self.snap.as_ref().and_then(|s| s.instance);
        self.hover_edge = picked.as_ref().and_then(|x| x.1);
        self.hover_guide = picked.as_ref().and_then(|x| x.4);
        self.hover_annot = hover.and_then(|p| self.pick_annotation(rect, p));
        self.cursor_frac = hover.map(|p| [(p.x - rect.left()) / rect.width(), (p.y - rect.top()) / rect.height()]);

        if let (Some((snap, _, o, d, _)), Some(pos)) = (picked, hover) {
            self.update_state(&snap, o, d);
            self.handle_pointer(ui, &resp, &snap, pos, mods, rect);
        }

        self.paint_scene(&painter, rect);
        self.paint_overlay(&painter, rect, hover);

        if resp.hovered() {
            let icon = match (self.tool, &self.state) {
                (_, State::Place { .. }) => egui::CursorIcon::Copy,
                (Tool::Select, _) => egui::CursorIcon::Default,
                (Tool::Orbit, _) => egui::CursorIcon::AllScroll,
                (Tool::Pan, _) => egui::CursorIcon::Grab,
                (Tool::Paint | Tool::Eraser, _) => egui::CursorIcon::PointingHand,
                _ => egui::CursorIcon::Crosshair,
            };
            ui.ctx().set_cursor_icon(icon);
        }
    }

    fn picker<'a>(&'a self, rect: Rect, scene: &'a SceneOut) -> Picker<'a> {
        Picker {
            model: &self.model,
            cam: &self.cam,
            rect,
            tris: &scene.pick,
            points: &scene.points,
            segs: &scene.segs,
            axes: self.axes(),
            guides: if self.show_guides { &self.doc.guides } else { &[] },
            guide_pts: &scene.guide_pts,
            skip: self.moving().map(|t| &t.verts),
            recent: &self.recent_pts,
            edge_dirs: &self.recent_dirs,
        }
    }

    fn handle_pointer(&mut self, ui: &egui::Ui, resp: &egui::Response, s: &Snap, pos: Pos2, mods: Modifiers, rect: Rect) {
        let clicked = resp.clicked();
        let drag_start = resp.drag_started_by(PointerButton::Primary);
        let dragging = resp.dragged_by(PointerButton::Primary);
        let drag_stop = resp.drag_stopped_by(PointerButton::Primary);
        if !ui.input(|i| i.pointer.primary_down() || i.pointer.primary_released()) {
            self.press = None;
        }

        if let State::Place { def } = self.state {
            if clicked {
                let xf = self.place_xf(def, s);
                self.place_at(def, xf);
            }
            return;
        }

        match self.tool {
            Tool::Orbit | Tool::Pan => {}
            Tool::Select => {
                if drag_start {
                    let start = ui.input(|i| i.pointer.press_origin()).unwrap_or(pos);
                    self.state = State::BoxSelect { start };
                }
                if drag_stop {
                    if let State::BoxSelect { start } = self.state.clone() {
                        self.box_select(rect, start, pos, mods);
                        self.state = State::Idle;
                    }
                }
                if resp.triple_clicked() {
                    self.select_connected();
                } else if resp.double_clicked() {
                    match (self.hover_edge, self.hover_face, self.hover_inst) {
                        (None, None, Some(i)) => self.enter_context(i),
                        _ => self.select_expand(),
                    }
                } else if clicked {
                    self.click_select(mods);
                }
            }
            Tool::Eraser => {
                if drag_start {
                    self.checkpoint();
                    self.state = State::Erase;
                }
                if dragging && matches!(self.state, State::Erase) {
                    if let Some(e) = self.hover_edge {
                        self.model.delete_edge(e);
                        self.sync();
                    } else if let Some(a) = self.hover_annot {
                        self.remove_annotations(&[a]);
                    } else if let Some(g) = self.hover_guide {
                        self.remove_guides(&[g]);
                    } else if let (None, Some(i)) = (s.face, s.instance) {
                        self.doc.remove_instances(&self.path, &[i]);
                    }
                }
                if drag_stop {
                    self.state = State::Idle;
                }
                if clicked {
                    if let Some(a) = self.hover_annot {
                        self.checkpoint();
                        self.remove_annotations(&[a]);
                    } else if let Some(e) = self.hover_edge {
                        self.commit(|m| {
                            m.delete_edge(e);
                            true
                        });
                    } else if let Some(g) = self.hover_guide {
                        self.checkpoint();
                        self.remove_guides(&[g]);
                    } else if let Some(f) = s.face {
                        self.commit(|m| {
                            m.delete_face(f);
                            true
                        });
                    } else if let Some(i) = s.instance {
                        self.checkpoint();
                        self.doc.remove_instances(&self.path, &[i]);
                    }
                }
            }
            Tool::Paint => {
                if clicked {
                    self.paint_click(s, mods);
                }
            }
            _ => {
                // Tools act when the button goes down, as in SketchUp: click,
                // move, click. Releasing the button far from where it went
                // down also counts (press, drag, release). A wobble of the
                // mouse during a click is neither.
                let (pressed, released) = ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_released()));
                if pressed && resp.is_pointer_button_down_on() {
                    self.press = Some(pos);
                    let now = ui.input(|i| i.time);
                    let busy = !matches!(self.state, State::Idle);
                    let same_spot = busy && self.last_click.is_some_and(|(p, _)| p.distance(pos) < 4.0);
                    let double = same_spot && self.last_click.is_some_and(|(_, t)| now - t < 0.45);
                    let repeat_is_noise = matches!(
                        self.tool,
                        Tool::Line
                            | Tool::Arc
                            | Tool::CenterArc
                            | Tool::Pie
                            | Tool::Rectangle
                            | Tool::Circle
                            | Tool::Polygon
                            | Tool::PushPull
                            | Tool::Offset
                            | Tool::Move
                            | Tool::Rotate
                            | Tool::Scale
                            | Tool::Protractor
                    );
                    if same_spot && repeat_is_noise {
                        // A second click on the same spot would only make
                        // zero-size geometry. Push/Pull and Offset let go.
                        if matches!(self.tool, Tool::PushPull | Tool::Offset) {
                            self.state = State::Idle;
                        }
                        // Double-click a face: push/pull it by the last distance.
                        if self.tool == Tool::PushPull && double {
                            match (self.last_pp, s.face) {
                                (Some(d), Some(f)) => {
                                    let d = d * self.pull_sign(f);
                                    self.finish_pushpull(f, d);
                                    self.status = format!("Push/Pull repeated: {}", self.measure);
                                }
                                (None, Some(_)) => self.status = "Double-click repeats the last Push/Pull: there is none yet.".into(),
                                _ => {}
                            }
                        }
                    } else {
                        self.tool_click(s, mods);
                    }
                    self.last_click = Some((pos, now));
                } else if released {
                    if let Some(down) = self.press.take() {
                        if down.distance(pos) > 10.0 && !matches!(self.state, State::Idle) {
                            self.tool_click(s, mods);
                            self.last_click = Some((pos, ui.input(|i| i.time)));
                        }
                    }
                }
            }
        }
    }

    fn state_targets(&self) -> Option<&Targets> {
        match &self.state {
            State::Move { t, .. } | State::RotateAngle { t, .. } | State::Scale { t, .. } => Some(t),
            _ => None,
        }
    }

    /// World matrix previewed by Move/Rotate/Scale, plus the copy flag.
    fn op_matrix(&self) -> (Option<Mat4>, bool) {
        match &self.state {
            State::Move { delta, copy, .. } => (Some(Mat4::from_translation(*delta)), *copy),
            State::RotateAngle { center, n, angle, .. } => (Some(about(*center, Mat4::from_axis_angle(*n, *angle))), false),
            State::Scale { center, factor, .. } => (Some(about(*center, Mat4::from_scale(*factor))), false),
            _ => (None, false),
        }
    }

    fn preview_model(&self) -> Option<Model> {
        let mut m = self.model.clone();
        if let ((Some(op), copy), Some(t)) = (self.op_matrix(), self.state_targets()) {
            apply_to_model(&mut m, t, op, copy);
            return Some(m);
        }
        match &self.state {
            State::PushPull { face, dist, .. } if dist.abs() > EPS => {
                m.push_pull(*face, *dist);
            }
            State::Offset { face, dist } if dist.abs() > EPS => {
                m.offset_face(*face, *dist);
            }
            _ => return None,
        }
        Some(m)
    }

    /// Recursively emits the document for rendering and/or picking.
    fn walk(&self, o: &WalkOpts, out: &mut SceneOut, n: Node) {
        let active_here = n.level == Some(self.path.len());
        let cuts = self.child_cuts(o, &n, active_here);
        if active_here {
            self.emit_active(o, out, &cuts, n.tag);
        } else if let Some(d) = self.doc.defs.get(&n.def) {
            self.emit_mesh(&d.mesh, &n, o.render, out, &cuts);
        }
        let Some(d) = self.doc.defs.get(&n.def) else { return };
        // Move / Rotate / Scale show what they would pick up when nothing is selected.
        let picks = matches!(self.tool, Tool::Move | Tool::Rotate | Tool::Scale) && self.sel.is_empty();
        let idle_hover = (matches!(self.state, State::Idle) && (picks || matches!(self.tool, Tool::Select | Tool::Paint | Tool::Eraser)))
            || self.tool.is_solid_tool();
        for (&iid, inst) in &d.instances {
            // Instances being moved are not snapped to at their old place.
            if active_here && !o.render && self.moving().is_some_and(|t| t.insts.contains(&iid)) {
                continue;
            }
            // Hidden tags hide instances, except the ones the user is editing inside.
            let editing = n.level.is_some_and(|k| self.path.get(k) == Some(&iid));
            if !editing && !self.doc.tag_visible(inst.tag) {
                continue;
            }
            let tag = inst.tag.or(n.tag);
            let mat = inst.material.or(n.mat);
            let base = n.xf * inst.xf;
            if active_here {
                let selected = self.sel.insts.contains(&iid);
                let child = Node {
                    def: inst.def,
                    xf: base,
                    level: None,
                    owner: Some(iid),
                    mat,
                    faded: false,
                    selected,
                    hovered: idle_hover && self.hover_inst == Some(iid),
                    tag,
                };
                let xf = match (selected, o.op) {
                    (true, Some(op)) => {
                        if o.copy {
                            self.walk(o, out, Node { selected: false, ..child });
                        }
                        op * base
                    }
                    _ => base,
                };
                self.walk(o, out, Node { xf, ..child });
                if selected && o.render {
                    if let Some((lo, hi)) = self.doc.local_bounds(inst.def) {
                        let c = box_corners(lo, hi).map(|p| xf.transform_point3(p));
                        for (a, b) in BOX_EDGES {
                            push_line(&mut out.lines, c[a], c[b], [0.1, 0.35, 1.0, 1.0]);
                        }
                    }
                }
            } else if let Some(k) = n.level {
                let on_path = self.path.get(k) == Some(&iid);
                let level = if on_path { Some(k + 1) } else { None };
                self.walk(
                    o,
                    out,
                    Node { def: inst.def, xf: base, level, owner: None, mat, faded: true, selected: false, hovered: false, tag },
                );
            } else {
                self.walk(o, out, Node { def: inst.def, xf: base, mat, tag, ..n });
            }
        }
    }

    /// Openings cut by glued children of a node, per host face, in the
    /// coordinates of the mesh being emitted (world space for the active
    /// context, where Move previews drag the opening along).
    fn child_cuts(&self, o: &WalkOpts, n: &Node, active_here: bool) -> HashMap<Id, Vec<Vec<Vec3>>> {
        let mut out: HashMap<Id, Vec<Vec<Vec3>>> = HashMap::new();
        let Some(d) = self.doc.defs.get(&n.def) else { return out };
        let mut cache: HashMap<DefId, Vec<Vec<Vec3>>> = HashMap::new();
        for (iid, inst) in &d.instances {
            let Some(host) = inst.glued.filter(|_| self.doc.tag_visible(inst.tag)) else { continue };
            let loops = cache.entry(inst.def).or_insert_with(|| self.doc.openings(inst.def));
            if loops.is_empty() {
                continue;
            }
            let mut xfs = Vec::new();
            if active_here {
                let base = n.xf * inst.xf;
                match (self.sel.insts.contains(iid), o.op) {
                    (true, Some(op)) => {
                        if o.copy {
                            xfs.push(base);
                        }
                        xfs.push(op * base);
                    }
                    _ => xfs.push(base),
                }
            } else {
                xfs.push(inst.xf);
            }
            for xf in xfs {
                for l in loops.iter() {
                    out.entry(host).or_default().push(l.iter().map(|p| xf.transform_point3(*p)).collect());
                }
            }
        }
        out
    }

    /// `Model::cut_triangles`, cached: openings are re-cut only when the face
    /// or an opening actually changes, not on every frame.
    fn cut_tris(&self, mesh: &Model, f: Id, loops: &[Vec<Vec3>]) -> Rc<Vec<[Vec3; 3]>> {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let mut put = |p: Vec3| p.to_array().iter().for_each(|x| x.to_bits().hash(&mut h));
        if let Some(face) = mesh.faces.get(&f) {
            put(face.normal);
            for lp in face.loops() {
                lp.iter().for_each(|v| put(mesh.pos(*v)));
                put(Vec3::NAN);
            }
        }
        for l in loops {
            l.iter().for_each(|p| put(*p));
            put(Vec3::INFINITY);
        }
        let key = h.finish();
        if let Some(t) = self.cut_cache.borrow().get(&key) {
            return t.clone();
        }
        let t = Rc::new(mesh.cut_triangles(f, loops));
        let mut cache = self.cut_cache.borrow_mut();
        if cache.len() > 512 {
            cache.clear();
        }
        cache.insert(key, t.clone());
        t
    }

    fn emit_active(&self, o: &WalkOpts, out: &mut SceneOut, cuts: &HashMap<Id, Vec<Vec<Vec3>>>, tag: Option<u32>) {
        let m = o.active;
        // Move / Rotate / Scale show what they would pick up when nothing is
        // selected: a corner point (the snap marker shows it), else an edge, else a face.
        let picks =
            matches!(self.tool, Tool::Move | Tool::Rotate | Tool::Scale) && matches!(self.state, State::Idle) && self.sel.is_empty();
        let grabs_point = self.tool == Tool::Move && self.snap.as_ref().is_some_and(|s| s.kind == SnapKind::Endpoint && s.vertex.is_some());
        let hl_face = match (self.tool, &self.state) {
            (Tool::PushPull | Tool::Paint | Tool::Offset | Tool::Select | Tool::Eraser | Tool::FollowMe, State::Idle) => self.hover_face,
            _ if picks && !grabs_point && self.hover_edge.is_none() => self.hover_face,
            _ => None,
        };
        let hl_edge = match self.tool {
            Tool::Select | Tool::Eraser => self.hover_edge,
            _ if picks && !grabs_point => self.hover_edge,
            _ => None,
        };
        let alpha = if self.xray { 0.45 } else { 1.0 };
        let shade = if o.render { m.shading_normals() } else { HashMap::new() };
        // Texture coordinates come from the context's own coordinates, so a
        // texture stays put on a group when you open it for editing.
        let to_local = self.ctx_xf.inverse();
        let n_local = Mat3::from_mat4(self.ctx_xf).transpose();
        for (&fid, face) in &m.faces {
            let tex = self.face_texture(face.material, false);
            let mut c = if tex.is_some() { [1.0; 3] } else { self.doc.material_color(face.material) };
            if self.color_by_tag {
                c = self.tag_color(tag);
            }
            let painted = face.material.is_some() || self.color_by_tag;
            let mut b = if painted { c } else { BACK_COLOR };
            if self.sel.faces.contains(&fid) {
                c = mix(c, [0.30, 0.45, 1.0], 0.5);
                b = mix(b, [0.30, 0.45, 1.0], 0.5);
            } else if Some(fid) == hl_face {
                c = mix(c, [0.55, 0.65, 1.0], 0.25);
                b = mix(b, [0.55, 0.65, 1.0], 0.25);
            }
            let ln = (n_local * face.normal).normalize_or_zero();
            let uv = |p: Vec3| tex.map(|(_, sz)| texture::uv(to_local.transform_point3(p), ln, sz)).unwrap_or([0.0; 2]);
            if let Some(cl) = cuts.get(&fid) {
                for &p in self.cut_tris(m, fid, cl).iter() {
                    if o.render {
                        for q in p {
                            push_face_vertex(tri_buf(out, tex), q, face.normal, [c[0], c[1], c[2], alpha], b, uv(q));
                        }
                    }
                    out.pick.push((Owner::Face(fid), p));
                }
                continue;
            }
            for t in m.triangulate_ids(fid) {
                let p = t.map(|v| m.pos(v));
                if o.render {
                    for k in 0..3 {
                        let nn = shade.get(&(fid, t[k])).copied().unwrap_or(face.normal);
                        push_face_vertex(tri_buf(out, tex), p[k], nn, [c[0], c[1], c[2], alpha], b, uv(p[k]));
                    }
                }
                out.pick.push((Owner::Face(fid), p));
            }
        }
        if !o.render {
            return;
        }
        for (&eid, e) in &m.edges {
            if e.soft && !self.show_hidden && !self.sel.edges.contains(&eid) {
                continue;
            }
            let col = if self.sel.edges.contains(&eid) {
                [0.10, 0.35, 1.0, 1.0]
            } else if Some(eid) == hl_edge {
                if self.tool == Tool::Eraser {
                    [1.0, 0.45, 0.1, 1.0]
                } else {
                    [0.3, 0.5, 1.0, 1.0]
                }
            } else if e.soft {
                [0.55, 0.57, 0.62, 1.0]
            } else {
                [0.07, 0.07, 0.09, 1.0]
            };
            push_line(&mut out.lines, m.pos(e.a), m.pos(e.b), col);
        }
    }

    fn emit_mesh(&self, mesh: &Model, n: &Node, render: bool, out: &mut SceneOut, cuts: &HashMap<Id, Vec<Vec<Vec3>>>) {
        let flip = n.xf.determinant() < 0.0;
        let owner = n.owner.map(Owner::Instance).unwrap_or(Owner::Outside);
        let alpha = if self.xray && !n.faded { 0.45 } else { 1.0 };
        let shade = if render { mesh.shading_normals() } else { HashMap::new() };
        let normal_mat = Mat3::from_mat4(n.xf).inverse().transpose();
        for (&fid, face) in &mesh.faces {
            let tex = self.face_texture(face.material.or(n.mat), n.faded);
            let uv = |p: Vec3| tex.map(|(_, sz)| texture::uv(p, face.normal, sz)).unwrap_or([0.0; 2]);
            let mut c = if tex.is_some() { [1.0; 3] } else { self.doc.material_color(face.material.or(n.mat)) };
            if self.color_by_tag {
                c = self.tag_color(n.tag);
            }
            let painted = face.material.or(n.mat).is_some() || self.color_by_tag;
            let mut b = if painted { c } else { BACK_COLOR };
            for (target, k, on) in [([0.30, 0.45, 1.0], 0.35, n.selected), ([0.55, 0.65, 1.0], 0.2, !n.selected && n.hovered)] {
                if on {
                    c = mix(c, target, k);
                    b = mix(b, target, k);
                }
            }
            if n.faded {
                c = mix(c, [0.93, 0.94, 0.96], 0.7);
                b = mix(b, [0.93, 0.94, 0.96], 0.7);
            }
            if let Some(cl) = cuts.get(&fid) {
                let nn = (normal_mat * face.normal).normalize_or_zero();
                for &(mut t) in self.cut_tris(mesh, fid, cl).iter() {
                    if flip {
                        t.swap(1, 2);
                    }
                    let p = t.map(|v| n.xf.transform_point3(v));
                    if render {
                        for k in 0..3 {
                            push_face_vertex(tri_buf(out, tex), p[k], nn, [c[0], c[1], c[2], alpha], b, uv(t[k]));
                        }
                    }
                    out.pick.push((owner, p));
                }
                continue;
            }
            for mut t in mesh.triangulate_ids(fid) {
                if flip {
                    t.swap(1, 2);
                }
                let p = t.map(|v| n.xf.transform_point3(mesh.pos(v)));
                if render {
                    for k in 0..3 {
                        let local = shade.get(&(fid, t[k])).copied().unwrap_or(face.normal);
                        let nn = (normal_mat * local).normalize_or_zero();
                        push_face_vertex(tri_buf(out, tex), p[k], nn, [c[0], c[1], c[2], alpha], b, uv(mesh.pos(t[k])));
                    }
                }
                out.pick.push((owner, p));
            }
        }
        let ecol = if n.faded {
            [0.66, 0.68, 0.72, 1.0]
        } else if n.selected {
            [0.10, 0.35, 1.0, 1.0]
        } else {
            [0.07, 0.07, 0.09, 1.0]
        };
        for e in mesh.edges.values() {
            if e.soft && !(render && self.show_hidden) {
                continue;
            }
            let (a, b) = (n.xf.transform_point3(mesh.pos(e.a)), n.xf.transform_point3(mesh.pos(e.b)));
            if render {
                push_line(&mut out.lines, a, b, if e.soft { [0.55, 0.57, 0.62, 1.0] } else { ecol });
            } else {
                out.segs.push((a, b));
            }
        }
        if !render {
            out.points.extend(mesh.vertices.values().map(|p| n.xf.transform_point3(*p)));
        }
    }

    fn paint_scene(&self, painter: &egui::Painter, rect: Rect) {
        let preview = self.preview_model();
        let (op, copy) = self.op_matrix();
        let opts = WalkOpts { active: preview.as_ref().unwrap_or(&self.model), op, copy, render: true };
        let mut out = SceneOut::default();
        self.walk(&opts, &mut out, self.root_node());

        // AI Render captures the model only: no guides, axes or edit-context box.
        let capture = self.ai.pending_capture();
        let clean = capture.is_some();
        if clean && !self.ai.settings.edges {
            out.lines.clear();
        }
        if !clean {
            self.emit_guides(&mut out.lines);
        }

        // Component being placed follows the cursor.
        if let (State::Place { def }, Some(s)) = (&self.state, &self.snap) {
            let xf = self.place_xf(*def, s);
            let node = Node { def: *def, xf, level: None, owner: None, mat: None, faded: false, selected: true, hovered: false, tag: None };
            self.walk(&opts, &mut out, node);
        }

        if self.show_axes && !clean {
            let len = (self.cam.distance * 50.0).max(1000.0);
            for (i, ax) in AXES.iter().enumerate() {
                let c = axis_rgb(i);
                push_line(&mut out.lines, Vec3::ZERO, *ax * len, [c[0], c[1], c[2], 0.9]);
                push_line(&mut out.lines, Vec3::ZERO, -*ax * len, [c[0], c[1], c[2], 0.3]);
            }
        }
        if !self.path.is_empty() && !clean {
            // Local axes and dashed-looking bounds of the open group/component.
            let origin = self.ctx_xf.transform_point3(Vec3::ZERO);
            let len = self.cam.distance * 0.25;
            for (i, ax) in self.axes().iter().enumerate() {
                let c = axis_rgb(i);
                push_line(&mut out.lines, origin, origin + *ax * len, [c[0], c[1], c[2], 1.0]);
            }
            if let Some((lo, hi)) = self.doc.local_bounds(self.ctx_def()) {
                let c = box_corners(lo, hi).map(|p| self.ctx_xf.transform_point3(p));
                for (a, b) in BOX_EDGES {
                    for k in 0..10 {
                        if k % 2 == 0 {
                            let (s, e) = (k as f32 / 10.0, (k + 1) as f32 / 10.0);
                            push_line(&mut out.lines, c[a].lerp(c[b], s), c[a].lerp(c[b], e), [0.35, 0.35, 0.4, 1.0]);
                        }
                    }
                }
            }
        }

        let mut tris = out.tris;
        let mut textured = Vec::new();
        for (mat, mut buf) in std::mem::take(&mut out.tex) {
            match self.texture_pixels(mat) {
                Some((key, pixels)) => textured.push(TexBatch { key, pixels, tris: buf }),
                None => {
                    // The image could not be decoded: fall back to the plain color.
                    let c = self.doc.material_color(Some(mat));
                    for v in buf.as_chunks_mut::<FLOATS>().0 {
                        for k in 0..3 {
                            v[6 + k] *= c[k];
                            v[12 + k] *= c[k];
                        }
                    }
                    tris.extend(buf);
                }
            }
        }
        let sun = self.doc.sun;
        let (light, shadow) = if sun.enabled && sun.above_horizon() {
            let dir = sun.direction();
            let shadow = self.doc.world_bounds().map(|(lo, hi)| {
                let (light_vp, glo, ghi) = sun::light_matrix(dir, lo, hi, 0.0);
                let pad = Vec3::new(1.0, 1.0, 0.0) * ((ghi - glo).length() * 0.05 + 0.5);
                let (a, b) = (glo - pad, ghi + pad);
                let q = [a, Vec3::new(b.x, a.y, 0.0), b, Vec3::new(a.x, b.y, 0.0)];
                let mut ground = Vec::new();
                for k in [0, 1, 2, 0, 2, 3] {
                    push_vertex(&mut ground, q[k], Vec3::Z, [0.0, 0.0, 0.0, 1.0]);
                }
                ShadowData { light_vp, ground }
            });
            (dir, shadow)
        } else {
            let headlight = (self.cam.eye() - self.cam.target).normalize() + Vec3::Z * 0.9 + self.cam.right() * 0.4;
            (headlight.normalize_or(Vec3::Z), None)
        };
        let data =
            DrawData { tris, textured, lines: out.lines, view_proj: self.cam.view_proj(rect), light, xray: self.xray && !clean, shadow };
        // Screenshot self-test: capture once the scene has settled.
        let shot = self.shot.as_ref().filter(|s| s.1 == 8).map(|s| (s.0.clone(), s.2.clone()));
        if let Some(r) = &self.renderer {
            let r = r.clone();
            let cb = egui::PaintCallback {
                rect,
                callback: Arc::new(eframe::egui_glow::CallbackFn::new(move |info, painter| {
                    let v = info.viewport_in_pixels();
                    let vp = [v.left_px, v.from_bottom_px, v.width_px, v.height_px];
                    r.lock().unwrap().paint(painter.gl(), &data, vp, painter.intermediate_fbo());
                    if let Some((path, done)) = &shot {
                        let _ = read_viewport(painter.gl(), vp).save(path);
                        done.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    if let Some(capture) = &capture {
                        *capture.lock().unwrap() = Some(read_viewport(painter.gl(), vp));
                    }
                })),
            };
            painter.add(cb);
        }
    }

    /// Guides as dashed lines (selected ones follow Move/Rotate/Scale previews),
    /// plus the guide a Tape Measure or Protractor would create.
    fn emit_guides(&self, lines: &mut Vec<f32>) {
        let (dash, half) = (self.cam.distance * 0.012, self.cam.distance * 12.0);
        let focus = self.cam.target;
        let mut draw = |g: &Guide, col: [f32; 4]| match *g {
            Guide::Line { p, d } => {
                for (a, b) in guide::dashes(p, d, focus, half, dash) {
                    push_line(lines, a, b, col);
                }
            }
            Guide::Point { p } => {
                let r = self.cam.distance * 0.008;
                for ax in [Vec3::X, Vec3::Y, Vec3::Z] {
                    push_line(lines, p - ax * r, p + ax * r, col);
                }
            }
        };
        if self.show_guides {
            let (op, copy) = self.op_matrix();
            let moving = self.state_targets().map(|t| t.guides.clone()).unwrap_or_default();
            let hover_ok = matches!(self.tool, Tool::Select | Tool::Eraser) && matches!(self.state, State::Idle | State::Erase);
            for (i, g) in self.doc.guides.iter().enumerate() {
                let col = if self.sel.guides.contains(&i) {
                    [0.10, 0.35, 1.0, 1.0]
                } else if hover_ok && self.hover_guide == Some(i) {
                    if self.tool == Tool::Eraser {
                        [1.0, 0.45, 0.1, 1.0]
                    } else {
                        [0.3, 0.5, 1.0, 1.0]
                    }
                } else {
                    [0.25, 0.25, 0.30, 0.9]
                };
                match op.filter(|_| moving.contains(&i)) {
                    Some(m) => {
                        if copy {
                            draw(g, [0.25, 0.25, 0.30, 0.9]);
                        }
                        draw(&g.transformed(m), col);
                    }
                    None => draw(g, col),
                }
            }
        }
        let preview = [0.75, 0.0, 0.45, 1.0];
        match &self.state {
            State::Tape { dir: Some(d), end, .. } if self.tape_guides => draw(&Guide::Line { p: *end, d: *d }, preview),
            State::Tape { dir: None, end, start } if self.tape_guides && start.distance(*end) > EPS => {
                draw(&Guide::Point { p: *end }, preview)
            }
            State::ProtAngle { center, n, ref_dir, angle } => {
                if let Some(g) = Guide::line(*center, glam::Quat::from_axis_angle(*n, *angle) * *ref_dir) {
                    draw(&g, preview);
                }
            }
            _ => {}
        }
    }

    fn paint_overlay(&self, painter: &egui::Painter, rect: Rect, hover: Option<Pos2>) {
        let proj = |p: Vec3| self.cam.project(rect, p);
        let ink = Color32::from_rgb(25, 25, 30);
        let snap_pos = self.snap.as_ref().map(|s| s.pos);
        let axes = self.axes();

        let polyline = |pts: &[Vec3], closed: bool, stroke: Stroke| {
            let sp: Vec<Pos2> = pts.iter().filter_map(|p| proj(*p)).collect();
            if sp.len() == pts.len() && sp.len() >= 2 {
                if closed {
                    painter.add(Shape::closed_line(sp, stroke));
                } else {
                    painter.add(Shape::line(sp, stroke));
                }
            }
        };
        let dashed = |a: Vec3, b: Vec3, color: Color32| {
            if let (Some(a), Some(b)) = (proj(a), proj(b)) {
                painter.extend(Shape::dashed_line(&[a, b], stroke(1.5, color), 6.0, 4.0));
            }
        };

        match &self.state {
            State::Line { start } => {
                if let Some(p) = snap_pos {
                    let color = match self.snap.as_ref().map(|s| s.kind) {
                        Some(SnapKind::Axis(i)) => axis_color(i),
                        Some(SnapKind::Parallel | SnapKind::Perpendicular) => MAGENTA,
                        _ => parallel_axis(p - *start, &axes).map(axis_color).unwrap_or(ink),
                    };
                    polyline(&[*start, p], false, stroke(2.0, color));
                }
            }
            State::Rect { p0, p1, u, v, .. } => {
                let d = *p1 - *p0;
                let (a, b) = (d.dot(*u), d.dot(*v));
                let pts = [*p0, *p0 + *u * a, *p0 + *u * a + *v * b, *p0 + *v * b];
                let square = (a.abs() - b.abs()).abs() < 1e-3 * a.abs().max(1e-3) && a.abs() > EPS;
                polyline(&pts, true, stroke(2.0, if square { Color32::from_rgb(200, 0, 160) } else { ink }));
            }
            State::Circle { center, edge, n, .. } => {
                let rv = in_plane(*edge - *center, *n);
                if rv.length() > EPS {
                    let segs = if self.tool == Tool::Polygon { self.polygon_sides } else { self.circle_segments };
                    polyline(&circle_points(*center, *n, rv, segs), true, stroke(2.0, ink));
                    dashed(*center, *center + rv, ink);
                }
            }
            State::ArcEnd { p0, p1, .. } => {
                let color = parallel_axis(*p1 - *p0, &axes).map(axis_color).unwrap_or(ink);
                polyline(&[*p0, *p1], false, stroke(2.0, color));
            }
            State::ArcBulge { p0, p1, n, bulge } => {
                let half = (bulge.abs() - p0.distance(*p1) / 2.0).abs() < 1e-5;
                let (col, note) = if self.arc_tangent {
                    (Color32::from_rgb(0, 170, 200), "Tangent at Vertex")
                } else if half {
                    (MAGENTA, "Half Circle")
                } else {
                    (ink, "")
                };
                polyline(&arc_points_2pt(*p0, *p1, *n, *bulge, self.arc_segments.max(2)), false, stroke(2.0, col));
                dashed(*p0, *p1, Color32::GRAY);
                if let (false, Some(h)) = (note.is_empty(), hover) {
                    label(painter, h + Vec2::new(16.0, -30.0), note, col);
                }
            }
            State::ArcStart { center, edge, n, .. } | State::ArcSweep { center, start: edge, n, .. } => {
                let col = parallel_axis(*n, &axes).map(axis_color).unwrap_or(Color32::from_rgb(150, 60, 200));
                let r = self.cam.distance * 0.07;
                let (u, _) = drawing_axes(*n);
                polyline(&circle_points(*center, *n, u * r, 64), true, stroke(1.0, col));
                dashed(*center, *edge, col);
                if let State::ArcSweep { start, angle, .. } = &self.state {
                    let pts = arc_points_center(*center, *start, *n, *angle, self.arc_segments.max(2));
                    polyline(&pts, false, stroke(2.0, ink));
                    if let Some(end) = pts.last() {
                        if self.tool == Tool::Pie {
                            polyline(&[*start, *center, *end], false, stroke(2.0, ink));
                        } else {
                            dashed(*center, *end, col);
                        }
                    }
                }
            }
            State::RotateCenter { center, n, .. }
            | State::RotateAngle { center, n, .. }
            | State::ProtBase { center, n }
            | State::ProtAngle { center, n, .. } => {
                let r = self.cam.distance * 0.07;
                let (u, _) = drawing_axes(*n);
                let col = parallel_axis(*n, &axes).map(axis_color).unwrap_or(Color32::from_rgb(150, 60, 200));
                polyline(&circle_points(*center, *n, u * r, 64), true, stroke(2.0, col));
                for (k, dir) in circle_points(Vec3::ZERO, *n, u, 24).into_iter().enumerate() {
                    let inner = if k % 6 == 0 { 0.8 } else { 0.9 };
                    polyline(&[*center + dir * r * inner, *center + dir * r], false, stroke(1.0, col));
                }
                if let State::RotateAngle { ref_dir, angle, .. } | State::ProtAngle { ref_dir, angle, .. } = &self.state {
                    let q = glam::Quat::from_axis_angle(*n, *angle);
                    polyline(&[*center, *center + *ref_dir * r * 1.2], false, stroke(1.5, col));
                    polyline(&[*center, *center + (q * *ref_dir) * r * 1.2], false, stroke(2.0, ink));
                } else if let Some(p) = snap_pos {
                    dashed(*center, p, col);
                }
            }
            State::Move { base, delta, copy, .. } => {
                let col = match self.snap.as_ref().map(|s| s.kind) {
                    Some(SnapKind::Axis(i)) => axis_color(i),
                    Some(SnapKind::Parallel | SnapKind::Perpendicular) => MAGENTA,
                    _ => parallel_axis(*delta, &axes).map(axis_color).unwrap_or(ink),
                };
                dashed(*base, *base + *delta, col);
                if *copy {
                    if let Some(h) = hover {
                        painter.text(h + Vec2::new(14.0, -18.0), Align2::LEFT_BOTTOM, "+ copy", FontId::proportional(13.0), ink);
                    }
                }
            }
            State::Offset { face, dist } => {
                if let Some(f) = self.model.faces.get(face) {
                    let pts = self.model.loop_points(&f.outer);
                    polyline(&offset_loop(&pts, f.normal, *dist), true, stroke(2.0, ink));
                }
            }
            State::Tape { start, end, .. } => {
                dashed(*start, *end, ink);
                if let (Some(h), false) = (hover, self.tape_guides) {
                    label(painter, h + Vec2::new(16.0, -30.0), "measure only", ink);
                }
            }
            State::BoxSelect { start } => {
                if let Some(h) = hover {
                    let r = Rect::from_two_pos(*start, h);
                    let window = h.x >= start.x;
                    let col = if window { Color32::from_rgb(30, 90, 230) } else { Color32::from_rgb(20, 150, 70) };
                    painter.rect_filled(r, 0.0, col.gamma_multiply(0.08));
                    if window {
                        painter.rect_stroke(r, 0.0, stroke(1.0, col), egui::StrokeKind::Inside);
                    } else {
                        let c = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
                        painter.extend(Shape::dashed_line(&c, stroke(1.0, col), 5.0, 4.0));
                    }
                }
            }
            _ => {}
        }

        if let Some((a, b)) = self.tape_result {
            dashed(a, b, Color32::from_rgb(200, 0, 120));
            if let Some(m) = proj((a + b) * 0.5) {
                label(painter, m, &self.units.format(a.distance(b)), Color32::from_rgb(200, 0, 120));
            }
        }

        // Dimensions & text, then the one being placed.
        if self.show_annotations {
            let hover_ok = matches!(self.tool, Tool::Select | Tool::Eraser) && matches!(self.state, State::Idle | State::Erase);
            for (i, a) in self.doc.annotations.iter().enumerate() {
                let col = if self.sel.annots.contains(&i) {
                    Color32::from_rgb(25, 90, 240)
                } else if hover_ok && self.hover_annot == Some(i) {
                    if self.tool == Tool::Eraser {
                        Color32::from_rgb(240, 110, 20)
                    } else {
                        Color32::from_rgb(90, 130, 240)
                    }
                } else {
                    Color32::from_rgb(30, 40, 90)
                };
                let moved = match (self.op_matrix(), self.state_targets()) {
                    ((Some(m), copy), Some(t)) if t.annots.contains(&i) => {
                        if copy {
                            self.draw_annotation(painter, rect, a, Color32::from_rgb(30, 40, 90));
                        }
                        Some(a.transformed(m))
                    }
                    _ => None,
                };
                self.draw_annotation(painter, rect, moved.as_ref().unwrap_or(a), col);
            }
        }
        let preview_col = Color32::from_rgb(200, 0, 120);
        match &self.state {
            State::DimSecond { a } => {
                if let Some(p) = snap_pos {
                    dashed(*a, p, preview_col);
                }
            }
            State::DimOffset { a, b, offset } => {
                self.draw_annotation(painter, rect, &Annotation::Dimension { a: *a, b: *b, offset: *offset, text: None }, preview_col)
            }
            State::DimRadial { center, radius, normal, diameter, at } => {
                let r = Annotation::Radial { center: *center, radius: *radius, normal: *normal, diameter: *diameter, at: *at, text: None };
                self.draw_annotation(painter, rect, &r, preview_col);
            }
            State::TextPlace { anchor, text, at } => {
                self.draw_annotation(painter, rect, &Annotation::Text { anchor: *anchor, at: *at, text: text.clone() }, preview_col)
            }
            _ => {}
        }

        // Inference marker + tooltip.
        let shows_inference = matches!(self.state, State::Place { .. })
            || !matches!(self.tool, Tool::Select | Tool::Orbit | Tool::Pan | Tool::Paint | Tool::Eraser);
        if let (true, Some(s), Some(h)) = (shows_inference, &self.snap, hover) {
            if let Some(p) = proj(s.pos) {
                let (col, txt) = match s.kind {
                    SnapKind::Endpoint => (Color32::from_rgb(0, 170, 0), true),
                    SnapKind::Midpoint => (Color32::from_rgb(0, 170, 210), true),
                    SnapKind::OnEdge => (Color32::from_rgb(210, 0, 0), true),
                    SnapKind::OnFace => (Color32::from_rgb(20, 40, 220), true),
                    SnapKind::Axis(i) => (axis_color(i), true),
                    SnapKind::GuidePoint | SnapKind::Intersection => (Color32::from_rgb(0, 150, 0), true),
                    SnapKind::OnGuide => (Color32::from_rgb(110, 110, 120), true),
                    SnapKind::Parallel | SnapKind::Perpendicular => (MAGENTA, true),
                    _ => (ink, false),
                };
                // "From Point": a dotted line to the point this position lines up with.
                if let Some(r) = s.from_point {
                    let c = parallel_axis(s.pos - r, &axes).map(axis_color).unwrap_or(Color32::from_rgb(90, 90, 100));
                    if let Some(rp) = proj(r) {
                        painter.extend(Shape::dotted_line(&[rp, p], c, 6.0, 1.5));
                        painter.circle_filled(rp, 3.5, c);
                    }
                }
                match s.kind {
                    SnapKind::Endpoint | SnapKind::Midpoint => {
                        painter.circle_filled(p, 5.0, col);
                        painter.circle_stroke(p, 5.0, stroke(1.0, Color32::BLACK));
                    }
                    SnapKind::OnEdge => {
                        painter.rect_filled(Rect::from_center_size(p, Vec2::splat(8.0)), 0.0, col);
                    }
                    SnapKind::OnFace => {
                        let d = 5.0;
                        let pts = vec![p + Vec2::new(0.0, -d), p + Vec2::new(d, 0.0), p + Vec2::new(0.0, d), p + Vec2::new(-d, 0.0)];
                        painter.add(Shape::convex_polygon(pts, col, stroke(1.0, Color32::BLACK)));
                    }
                    SnapKind::Axis(_) => {
                        painter.circle_stroke(p, 4.0, stroke(2.0, col));
                    }
                    _ => {}
                }
                if txt {
                    let text = match (s.kind, s.instance) {
                        (SnapKind::Endpoint | SnapKind::Midpoint | SnapKind::OnEdge | SnapKind::OnFace, Some(i)) => {
                            let name = self.doc.instance(&self.path, i).map(|x| self.def_name(x.def)).unwrap_or_default();
                            format!("{} in {}", s.kind.label(), name)
                        }
                        _ if s.from_point.is_some() => "From Point".to_string(),
                        _ => s.kind.label().to_string(),
                    };
                    label(painter, h + Vec2::new(16.0, 16.0), &text, col);
                }
            }
        }

        // Breadcrumb of the open edit context.
        let mut y = 10.0;
        if !self.path.is_empty() {
            let mut names = vec!["Model".to_string()];
            let mut cur = ROOT;
            for i in &self.path {
                if let Some(inst) = self.doc.defs.get(&cur).and_then(|d| d.instances.get(i)) {
                    names.push(self.def_name(inst.def));
                    cur = inst.def;
                }
            }
            let text = format!("Editing: {}   (Esc to close)", names.join(" › "));
            label(painter, rect.left_top() + Vec2::new(12.0, y), &text, Color32::from_rgb(40, 40, 60));
            y += 26.0;
        }
        if let Some(i) = self.lock_axis {
            painter.text(
                rect.left_top() + Vec2::new(12.0, y),
                Align2::LEFT_TOP,
                format!("Locked to {} axis", ["red", "green", "blue"][i]),
                FontId::proportional(13.0),
                axis_color(i),
            );
        }

        // Axis gizmo (bottom-left).
        let origin = rect.left_bottom() + Vec2::new(46.0, -46.0);
        if let Some(t) = proj(self.cam.target) {
            for i in 0..3 {
                if let Some(e) = proj(self.cam.target + AXES[i] * self.cam.distance * 0.05) {
                    let dir = (e - t).normalized();
                    let tip = origin + dir * 30.0;
                    painter.line_segment([origin, tip], stroke(2.5, axis_color(i)));
                    painter.text(tip + dir * 8.0, Align2::CENTER_CENTER, ["X", "Y", "Z"][i], FontId::proportional(12.0), axis_color(i));
                }
            }
        }
    }

    // ----- panels ---------------------------------------------------------------

    fn hint(&self) -> &'static str {
        match (self.tool, &self.state) {
            (_, State::Place { .. }) => "Click to place the component (gluing components snap onto faces). Esc cancels.",
            (Tool::Select, _) => {
                "Click to select (Shift toggles, Ctrl adds). Drag L→R = window, R→L = crossing. Double-click a group to edit it."
            }
            (Tool::Line, State::Line { .. }) => "Click end point or type a length + Enter. Arrow keys lock axes, Esc finishes.",
            (Tool::Line, _) => "Click to start a line. Closed coplanar loops become faces; lines across faces split them.",
            (Tool::Arc, State::ArcEnd { .. }) => "Click the arc's end point or type the chord length.",
            (Tool::Arc, State::ArcBulge { .. }) => {
                "Move to set the bulge (snaps to a half circle and to tangents), click — or type a bulge, 5r for a radius, 12s for segments."
            }
            (Tool::Arc, _) => "Click the arc's start point. Arcs connecting to edges close faces; arcs across a face split it.",
            (Tool::CenterArc | Tool::Pie, State::ArcStart { .. }) => "Click the start point (sets the radius) or type a radius.",
            (Tool::CenterArc | Tool::Pie, State::ArcSweep { .. }) => "Move to sweep (snaps to 15°), click — or type degrees.",
            (Tool::CenterArc, _) => "Click the arc's center. Type e.g. 16s to change segments.",
            (Tool::Pie, _) => "Click the pie's center (makes a closed wedge face).",
            (Tool::FollowMe, _) if self.sel.is_empty() => {
                "Select a path first (edges, or a face to use its outline), then click the profile face."
            }
            (Tool::FollowMe, _) => "Click the profile face to sweep it along the selected path.",
            (Tool::Rectangle, State::Rect { .. }) => "Click opposite corner or type width,height (e.g. 4,3).",
            (Tool::Rectangle, _) => "Click the first corner (on the ground or on a face). Arrow keys pick a vertical plane.",
            (Tool::Circle | Tool::Polygon, State::Circle { .. }) => "Click to set the radius or type a value.",
            (Tool::Circle | Tool::Polygon, _) => "Click the center point. Type e.g. 32s to change the number of segments.",
            (Tool::PushPull, State::PushPull { .. }) => {
                "Move to set the distance, then click — or type a distance. Point at other geometry to stop level with it."
            }
            (Tool::PushPull, _) => "Click a face to push or pull it. Double-click a face to repeat the last distance.",
            (Tool::Move, State::Move { .. }) => {
                "Click destination or type a distance. Ctrl toggles copy; after a copy type 3x or /3 for arrays."
            }
            (Tool::Move, _) => "Click what to move: a corner point, an edge, a face or a group. Or select first, then click a base point.",
            (Tool::Rotate, State::RotateCenter { .. }) => "Click a reference point to start the angle.",
            (Tool::Rotate, State::RotateAngle { .. }) => "Click to set the angle (snaps to 15°) or type degrees.",
            (Tool::Rotate, _) => "Select geometry, then click the rotation center (aligns to the hovered face).",
            (Tool::Scale, State::Scale { .. }) => "Move to scale, click to finish, or type a factor (2) / x,y,z factors (1,2,1).",
            (Tool::Scale, _) => "Select geometry, then click a point to start scaling about its base center.",
            (Tool::Offset, State::Offset { .. }) => "Move inside/outside the face to set the offset, click or type a value.",
            (Tool::Offset, _) => "Click a face to offset its outline.",
            (Tool::Eraser, _) => "Click or drag over edges or groups to erase them. Erasing a seam between coplanar faces merges them.",
            (Tool::Paint, _) => "Click faces or groups to paint with the current material. Alt+click samples a material.",
            (Tool::Tape, State::Tape { dir: Some(_), .. }) => {
                "Move off the edge and click (or type a distance) for a parallel guide; click again in place for a guide on the edge."
            }
            (Tool::Tape, State::Tape { .. }) => {
                "Click the second point (or type a distance) to measure and add a guide point. Ctrl: measure only."
            }
            (Tool::Tape, _) => "Click a point to measure from, or an edge to make a parallel guide line. Ctrl toggles measure-only.",
            (Tool::Dimension, State::DimSecond { .. }) => "Click the second point.",
            (Tool::Dimension, State::DimOffset { .. }) => "Move to offset the dimension line, then click to place it.",
            (Tool::Dimension, State::DimRadial { .. }) => "Move to place the radius/diameter label, then click.",
            (Tool::Dimension, _) => "Click two points, or an edge, to dimension it. Click an arc or circle for its radius/diameter.",
            (Tool::Text, State::TextPlace { .. }) => "Move to place the text and click. Edit it afterwards in Entity Info.",
            (Tool::Text, _) => "Click a face, edge or point for leader text (area / length / coordinates), or empty space for screen text.",
            (Tool::Protractor, State::ProtBase { .. }) => "Click a point to set the baseline the angle is measured from.",
            (Tool::Protractor, State::ProtAngle { .. }) => "Click to place the guide (snaps to 15°) or type an angle (30) or slope (1:2).",
            (Tool::Protractor, _) => "Click the protractor's center (aligns to the hovered face; arrow keys lock a plane).",
            (Tool::Orbit, _) => "Drag to orbit. Middle-drag orbits in any tool; scroll zooms.",
            (Tool::Pan, _) => "Drag to pan. Shift+middle-drag or right-drag pans in any tool.",
            (Tool::Subtract, State::SolidFirst { .. }) => "Click the solid to subtract it from. The first solid is removed.",
            (Tool::Trim, State::SolidFirst { .. }) => "Click the solid to trim. The first solid is kept.",
            (_, State::SolidFirst { .. }) => "Click the second solid.",
            (Tool::OuterShell, _) => "Click two solids (or select several first) to merge them into one shell.",
            (Tool::SolidIntersect, _) => "Click two solids (or select several first) to keep only what they share.",
            (Tool::Union, _) => "Click two solids (or select several first) to combine them.",
            (Tool::Subtract, _) => "Click the solid to subtract, then the solid to cut it from.",
            (Tool::Trim, _) => "Click the cutting solid, then the solid to trim (the cutter stays).",
            (Tool::Split, _) => "Click two solids to split them into their differences and their overlap.",
        }
    }

    fn vcb_label(&self) -> &'static str {
        match (self.tool, &self.state) {
            (Tool::Line | Tool::Tape, _) => "Length",
            (Tool::Arc, State::ArcEnd { .. }) => "Length",
            (Tool::Arc, State::ArcBulge { .. }) => "Bulge",
            (Tool::CenterArc | Tool::Pie, State::ArcStart { .. }) => "Radius",
            (Tool::CenterArc | Tool::Pie, State::ArcSweep { .. }) => "Angle",
            (Tool::Arc | Tool::CenterArc | Tool::Pie, _) => "Segments",
            (Tool::Rectangle, _) => "Dimensions",
            (Tool::Circle, State::Idle) => "Segments",
            (Tool::Polygon, State::Idle) => "Sides",
            (Tool::Circle | Tool::Polygon, _) => "Radius",
            (Tool::PushPull | Tool::Move | Tool::Offset, _) => "Distance",
            (Tool::Rotate | Tool::Protractor, _) => "Angle",
            (Tool::Scale, _) => "Scale",
            _ => "Measurements",
        }
    }

    fn menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New                Ctrl+N").clicked() {
                        self.new_model();
                        ui.close_menu();
                    }
                    if ui.button("Open…             Ctrl+O").clicked() {
                        ui.close_menu();
                        self.open();
                    }
                    if ui.button("Save               Ctrl+S").clicked() {
                        ui.close_menu();
                        self.save();
                    }
                    if ui.button("Save As…   Ctrl+Shift+S").clicked() {
                        ui.close_menu();
                        self.save_as();
                    }
                    ui.separator();
                    if ui.button("Import OBJ…").clicked() {
                        ui.close_menu();
                        self.import_obj();
                    }
                    if ui.button("Export OBJ…").clicked() {
                        ui.close_menu();
                        self.export(false);
                    }
                    if ui.button("Export STL…").clicked() {
                        ui.close_menu();
                        self.export(true);
                    }
                    ui.separator();
                    if ui.button("AI Render…").clicked() {
                        ui.close_menu();
                        self.ai.open = true;
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo        Ctrl+Z")).clicked() {
                        self.undo();
                        ui.close_menu();
                    }
                    if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("Redo        Ctrl+Y")).clicked() {
                        self.redo();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Select All   Ctrl+A").clicked() {
                        self.select_all();
                        ui.close_menu();
                    }
                    if ui.button("Select None   Esc").clicked() {
                        self.sel.clear();
                        ui.close_menu();
                    }
                    if ui.button("Delete        Del").clicked() {
                        self.delete_selection();
                        ui.close_menu();
                    }
                    ui.separator();
                    let has_sel = !self.sel.is_empty();
                    let has_inst = !self.sel.insts.is_empty();
                    if ui.add_enabled(has_sel, egui::Button::new("Make Group        Ctrl+G")).clicked() {
                        self.make_group(DefKind::Group);
                        ui.close_menu();
                    }
                    if ui.add_enabled(has_sel, egui::Button::new("Make Component   G")).clicked() {
                        self.make_group(DefKind::Component);
                        ui.close_menu();
                    }
                    if ui.add_enabled(has_inst, egui::Button::new("Explode")).clicked() {
                        self.explode();
                        ui.close_menu();
                    }
                    if ui.add_enabled(has_inst, egui::Button::new("Make Unique")).clicked() {
                        self.make_unique();
                        ui.close_menu();
                    }
                    if ui.add_enabled(self.sel.insts.len() == 1, egui::Button::new("Edit Group/Component")).clicked() {
                        let i = self.sel.insts()[0];
                        self.enter_context(i);
                        ui.close_menu();
                    }
                    if ui.add_enabled(!self.path.is_empty(), egui::Button::new("Close Group/Component")).clicked() {
                        self.exit_context();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add_enabled(!self.doc.guides.is_empty(), egui::Button::new("Delete Guides")).clicked() {
                        self.delete_all_guides();
                        ui.close_menu();
                    }
                    ui.menu_button("Intersect Faces", |ui| {
                        if ui.button("With Model").clicked() {
                            self.intersect(true);
                            ui.close_menu();
                        }
                        if ui.add_enabled(!self.sel.is_empty(), egui::Button::new("With Selection")).clicked() {
                            self.intersect(false);
                            ui.close_menu();
                        }
                    });
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Zoom Extents   Shift+Z").clicked() {
                        self.zoom_extents();
                        ui.close_menu();
                    }
                    ui.separator();
                    let views: [(&str, f32, f32); 6] = [
                        ("Iso", -60.0, 30.0),
                        ("Top", -90.0, 89.9),
                        ("Front", -90.0, 0.0),
                        ("Right", 0.0, 0.0),
                        ("Back", 90.0, 0.0),
                        ("Left", 180.0, 0.0),
                    ];
                    for (name, yaw, pitch) in views {
                        if ui.button(name).clicked() {
                            self.cam.set_view(yaw, pitch);
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    ui.checkbox(&mut self.cam.perspective, "Perspective");
                    ui.checkbox(&mut self.xray, "X-Ray");
                    ui.checkbox(&mut self.show_axes, "Axes");
                    ui.checkbox(&mut self.show_hidden, "Hidden Geometry (soft edges)");
                    ui.checkbox(&mut self.show_guides, "Guides");
                    ui.checkbox(&mut self.color_by_tag, "Color by Tag");
                    ui.checkbox(&mut self.doc.sun.enabled, "Shadows");
                    ui.checkbox(&mut self.show_annotations, "Dimensions & Text");
                    ui.separator();
                    if ui.button("Add Scene").clicked() {
                        self.add_scene();
                        ui.close_menu();
                    }
                    if ui.add_enabled(!self.doc.scenes.is_empty(), egui::Button::new("Next Scene   PgDn")).clicked() {
                        self.step_scene(true);
                        ui.close_menu();
                    }
                    if ui.add_enabled(!self.doc.scenes.is_empty(), egui::Button::new("Previous Scene   PgUp")).clicked() {
                        self.step_scene(false);
                        ui.close_menu();
                    }
                });
                ui.menu_button("Tools", |ui| {
                    for (_, tools) in TOOL_GROUPS {
                        for &t in *tools {
                            if ui.button(format!("{} {}  ({})", t.icon(), t.label(), t.shortcut())).clicked() {
                                self.set_tool(t);
                                ui.close_menu();
                            }
                        }
                        ui.separator();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard shortcuts").clicked() {
                        self.show_help = true;
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn toolbar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("tools").resizable(false).exact_width(140.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.toolbar_contents(ui));
        });
    }

    fn toolbar_contents(&mut self, ui: &mut egui::Ui) {
        {
            ui.add_space(4.0);
            for (group, tools) in TOOL_GROUPS {
                ui.label(RichText::new(*group).small().weak());
                for &t in *tools {
                    let text = RichText::new(format!("{}  {}", t.icon(), t.label())).size(14.0);
                    let r = ui.add_sized([128.0, 26.0], egui::SelectableLabel::new(self.tool == t, text)).on_hover_text(format!(
                        "{} — shortcut: {}",
                        t.label(),
                        t.shortcut()
                    ));
                    if r.clicked() {
                        self.set_tool(t);
                    }
                }
                ui.add_space(6.0);
            }
            ui.label(RichText::new("GROUPS").small().weak());
            let has_sel = !self.sel.is_empty();
            if ui
                .add_enabled(has_sel, egui::Button::new("📦  Make Group").min_size(Vec2::new(128.0, 24.0)))
                .on_hover_text("Ctrl+G")
                .clicked()
            {
                self.make_group(DefKind::Group);
            }
            if ui
                .add_enabled(has_sel, egui::Button::new("🔷  Make Component").min_size(Vec2::new(128.0, 24.0)))
                .on_hover_text("G")
                .clicked()
            {
                self.make_group(DefKind::Component);
            }
            ui.separator();
            if ui.add_sized([128.0, 24.0], egui::Button::new("⛶  Zoom Extents")).clicked() {
                self.zoom_extents();
            }
            ui.horizontal(|ui| {
                if ui.small_button("Top").clicked() {
                    self.cam.set_view(-90.0, 89.9);
                }
                if ui.small_button("Front").clicked() {
                    self.cam.set_view(-90.0, 0.0);
                }
                if ui.small_button("Iso").clicked() {
                    self.cam.set_view(-60.0, 30.0);
                }
            });
        }
    }

    fn instance_info(&mut self, ui: &mut egui::Ui, iid: Id) {
        let Some(inst) = self.doc.instance(&self.path, iid).cloned() else { return };
        let def = inst.def;
        let kind = self.doc.defs[&def].kind;
        let count = self.doc.instances_of(def);
        ui.label(RichText::new(kind.label()).strong());
        ui.horizontal(|ui| {
            ui.label("Name");
            if let Some(d) = self.doc.defs.get_mut(&def) {
                ui.text_edit_singleline(&mut d.name);
            }
        });
        if kind == DefKind::Component {
            ui.label(format!("Placed {count} time(s) — edits apply to all"));
            let (mut glue, mut cut) = (self.doc.defs[&def].glue, self.doc.defs[&def].cut_opening);
            let mut changed = ui.checkbox(&mut glue, "Glue to faces").on_hover_text("Places on faces and slides along them").changed();
            changed |= ui
                .add_enabled(glue, egui::Checkbox::new(&mut cut, "Cut opening"))
                .on_hover_text("Cuts the component's outline (its red/green plane) out of the face it is glued to")
                .changed();
            if changed {
                self.checkpoint();
                if let Some(d) = self.doc.defs.get_mut(&def) {
                    d.glue = glue;
                    d.cut_opening = cut;
                }
                self.doc.refresh_glue();
            }
            if glue {
                let glued = self.doc.instance(&self.path, iid).and_then(|i| i.glued).is_some();
                ui.label(RichText::new(if glued { "Glued to a face" } else { "Not on a face" }).weak());
            }
        }
        let o = (self.ctx_xf * inst.xf).transform_point3(Vec3::ZERO);
        ui.label(format!("Origin: {}, {}, {}", self.units.format(o.x), self.units.format(o.y), self.units.format(o.z)));
        let d = &self.doc.defs[&def];
        let solid_vol = if d.instances.is_empty() {
            let det = (self.ctx_xf * inst.xf).determinant().abs();
            solid::is_solid(&d.mesh).then(|| solid::volume(&d.mesh).abs() * det)
        } else {
            self.solid_mesh(iid).map(|m| solid::volume(&m).abs())
        };
        match solid_vol {
            Some(v) => ui.label(format!("Solid — volume {}", self.units.format_volume(v))),
            None => ui.label(RichText::new("Not a solid (solid tools need closed shells)").weak()),
        };
        ui.horizontal_wrapped(|ui| {
            if ui.button("Edit").clicked() {
                self.enter_context(iid);
            }
            if ui.button("Explode").clicked() {
                self.explode();
            }
            if kind == DefKind::Component && count > 1 && ui.button("Make Unique").clicked() {
                self.make_unique();
            }
            if kind == DefKind::Group && ui.button("Make Component").clicked() {
                self.convert_to_component(def);
            }
        });
        self.tag_picker(ui, &[iid]);
    }

    fn outliner(&mut self, ui: &mut egui::Ui, def: DefId, path: &mut Vec<Id>, depth: usize) {
        let items: Vec<(Id, DefId, String, DefKind)> = self
            .doc
            .defs
            .get(&def)
            .map(|d| {
                d.instances
                    .iter()
                    .map(|(i, inst)| {
                        let cd = self.doc.defs.get(&inst.def);
                        (*i, inst.def, cd.map(|x| x.name.clone()).unwrap_or_default(), cd.map(|x| x.kind).unwrap_or(DefKind::Group))
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (iid, cdef, name, kind) in items {
            let parent = path.clone();
            let selected = self.path == parent && self.sel.insts.contains(&iid);
            path.push(iid);
            let editing = self.path == *path;
            let icon = if kind == DefKind::Component { "🔷" } else { "📦" };
            let mut text = RichText::new(format!("{icon} {name}"));
            if editing {
                text = text.strong().italics();
            }
            let mut clicked = false;
            ui.horizontal(|ui| {
                ui.add_space(depth as f32 * 14.0);
                clicked = ui.selectable_label(selected, text).clicked();
            });
            if clicked {
                self.navigate(parent, iid);
            }
            if depth < 32 {
                self.outliner(ui, cdef, path, depth + 1);
            }
            path.pop();
        }
    }

    fn inspector(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("inspector").default_width(260.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Entity Info");
                let (nf, ne, ni) = (self.sel.faces.len(), self.sel.edges.len(), self.sel.insts.len());
                let ng = self.sel.guides.len();
                if ng > 0 {
                    let lines = self.sel.guides.iter().filter(|i| matches!(self.doc.guides.get(**i), Some(Guide::Line { .. }))).count();
                    ui.label(format!("{lines} guide line(s), {} guide point(s)", ng - lines));
                }
                if !self.sel.annots.is_empty() {
                    self.annotation_info(ui);
                }
                if nf + ne + ni == 0 {
                    if ng == 0 && self.sel.annots.is_empty() {
                        ui.label(RichText::new("No selection").weak());
                    }
                } else {
                    if ni == 1 && nf + ne == 0 {
                        let iid = *self.sel.insts.iter().next().unwrap();
                        self.instance_info(ui, iid);
                    } else {
                        ui.label(format!("{nf} face(s), {ne} edge(s), {ni} group(s)/component(s)"));
                        if ni > 0 {
                            let insts = self.sel.insts();
                            self.tag_picker(ui, &insts);
                        }
                    }
                    if nf == 1 {
                        let f = *self.sel.faces.iter().next().unwrap();
                        ui.label(format!("Area: {}", self.units.format_area(self.model.face_area(f))));
                        let n = self.model.faces[&f].normal;
                        ui.label(format!("Normal: ({:.2}, {:.2}, {:.2})", n.x, n.y, n.z));
                    } else if nf > 1 {
                        let total: f32 = self.sel.faces.iter().map(|f| self.model.face_area(*f)).sum();
                        ui.label(format!("Total area: {}", self.units.format_area(total)));
                    }
                    if ne == 1 && nf == 0 {
                        let e = *self.sel.edges.iter().next().unwrap();
                        ui.label(format!("Length: {}", self.units.format(self.model.edge_length(e))));
                    } else if ne > 1 && nf == 0 {
                        let curve_of = |e: &Id| self.model.edges.get(e).and_then(|x| x.curve);
                        let first = self.sel.edges.iter().next().and_then(curve_of);
                        if first.is_some() && self.sel.edges.iter().all(|e| curve_of(e) == first) {
                            let len: f32 = self.sel.edges.iter().map(|e| self.model.edge_length(*e)).sum();
                            ui.label(format!("Curve: {ne} segments, length {}", self.units.format(len)));
                        }
                    }
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Make Group").clicked() {
                            self.make_group(DefKind::Group);
                        }
                        if ui.button("Make Component").clicked() {
                            self.make_group(DefKind::Component);
                        }
                        if ui.button("Intersect with Model").on_hover_text("Edit ▸ Intersect Faces ▸ With Model").clicked() {
                            self.intersect(true);
                        }
                        if nf > 0 && ui.button("Reverse faces").clicked() {
                            let faces = self.sel.faces();
                            self.commit(|m| {
                                for f in &faces {
                                    if let Some(face) = m.faces.get_mut(f) {
                                        face.outer.reverse();
                                        face.holes.iter_mut().for_each(|h| h.reverse());
                                    }
                                }
                                m.recompute_normals();
                                true
                            });
                        }
                    });
                }

                ui.separator();
                ui.heading("Materials");
                ui.horizontal_wrapped(|ui| {
                    let swatch = |ui: &mut egui::Ui, color: Color32, selected: bool, tip: &str, textured: bool| {
                        let s = if selected { stroke(3.0, Color32::from_rgb(20, 90, 255)) } else { stroke(1.0, Color32::GRAY) };
                        let label = RichText::new(if textured { "▦" } else { "" }).color(Color32::from_white_alpha(200));
                        ui.add(egui::Button::new(label).fill(color).stroke(s).min_size(Vec2::splat(30.0))).on_hover_text(tip).clicked()
                    };
                    if swatch(ui, to_color32(self.doc.material_color(None)), self.material.is_none(), "Default", false) {
                        self.material = None;
                    }
                    for i in 0..self.doc.materials.len() {
                        let m = &self.doc.materials[i];
                        let (c, name, textured) = (to_color32(m.color), m.name.clone(), m.texture.is_some());
                        if swatch(ui, c, self.material == Some(i), &name, textured) {
                            self.material = Some(i);
                        }
                    }
                });
                if let Some(i) = self.material.filter(|i| *i < self.doc.materials.len()) {
                    ui.horizontal(|ui| {
                        ui.color_edit_button_rgb(&mut self.doc.materials[i].color);
                        ui.text_edit_singleline(&mut self.doc.materials[i].name);
                    });
                    self.texture_controls(ui, i);
                } else {
                    ui.label(RichText::new("Default material").weak());
                }
                ui.horizontal(|ui| {
                    if ui.button("+ New").clicked() {
                        let n = self.doc.materials.len() + 1;
                        self.doc.materials.push(Material { name: format!("Material {n}"), color: [0.85, 0.55, 0.35], texture: None });
                        self.material = Some(self.doc.materials.len() - 1);
                    }
                    if ui.add_enabled(!self.sel.is_empty(), egui::Button::new("Paint selection")).clicked() {
                        let (faces, insts, mat) = (self.sel.faces(), self.sel.insts(), self.material);
                        if !insts.is_empty() {
                            self.set_instance_material(&insts, mat);
                        }
                        self.commit(|m| {
                            for f in &faces {
                                if let Some(face) = m.faces.get_mut(f) {
                                    face.material = mat;
                                }
                            }
                            !faces.is_empty()
                        });
                    }
                });

                ui.separator();
                self.tags_panel(ui);
                ui.separator();
                self.scenes_panel(ui);
                ui.separator();
                self.shadows_panel(ui);

                ui.separator();
                ui.heading("Components");
                let comps: Vec<(DefId, String, usize)> = self
                    .doc
                    .defs
                    .iter()
                    .filter(|(_, d)| d.kind == DefKind::Component)
                    .map(|(id, d)| (*id, d.name.clone(), self.doc.instances_of(*id)))
                    .collect();
                if comps.is_empty() {
                    ui.label(RichText::new("Select geometry and press G to make a component.").weak());
                }
                for (id, name, count) in comps {
                    ui.horizontal(|ui| {
                        ui.label(format!("🔷 {name}"));
                        ui.label(RichText::new(format!("×{count}")).weak());
                        if ui.small_button("Place").clicked() {
                            self.start_place(id);
                        }
                    });
                }
                if ui.small_button("Purge unused").clicked() {
                    self.checkpoint();
                    let n = self.doc.purge_components();
                    self.status = format!("Purged {n} unused definition(s)");
                }

                ui.separator();
                ui.heading("Outliner");
                let mut root_label = RichText::new("🏠 Model");
                if self.path.is_empty() {
                    root_label = root_label.strong();
                }
                if ui.selectable_label(false, root_label).clicked() && !self.path.is_empty() {
                    self.path.clear();
                    self.load_context();
                }
                self.outliner(ui, ROOT, &mut Vec::new(), 1);

                ui.separator();
                ui.heading("Model Info");
                egui::Grid::new("info").num_columns(2).show(ui, |ui| {
                    ui.label("Faces (this context)");
                    ui.label(self.model.faces.len().to_string());
                    ui.end_row();
                    ui.label("Edges (this context)");
                    ui.label(self.model.edges.len().to_string());
                    ui.end_row();
                    ui.label("Definitions");
                    ui.label((self.doc.defs.len() - 1).to_string());
                    ui.end_row();
                    ui.label("Units");
                    egui::ComboBox::from_id_salt("units").selected_text(self.units.label()).show_ui(ui, |ui| {
                        for u in Units::ALL {
                            ui.selectable_value(&mut self.units, u, u.label());
                        }
                    });
                    ui.end_row();
                    ui.label("Circle segments");
                    ui.add(egui::DragValue::new(&mut self.circle_segments).range(3..=360));
                    ui.end_row();
                    ui.label("Polygon sides");
                    ui.add(egui::DragValue::new(&mut self.polygon_sides).range(3..=100));
                    ui.end_row();
                    ui.label("Arc segments");
                    ui.add(egui::DragValue::new(&mut self.arc_segments).range(2..=360));
                    ui.end_row();
                });
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(self.hint()).size(13.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let typing = !self.vcb.is_empty();
                    let value = if typing { self.vcb.clone() } else { self.measure.clone() };
                    egui::Frame::group(ui.style())
                        .fill(if typing { Color32::from_rgb(255, 250, 220) } else { ui.visuals().extreme_bg_color })
                        .show(ui, |ui| {
                            ui.set_min_width(170.0);
                            ui.label(RichText::new(value).monospace().size(14.0));
                        });
                    ui.label(RichText::new(self.vcb_label()).strong());
                    ui.separator();
                    ui.label(RichText::new(&self.status).weak());
                });
            });
        });
    }

    // ----- AI Render ---------------------------------------------------------

    /// Names of the materials used in the model, to tell the AI what things are made of.
    fn used_materials(&self) -> Vec<String> {
        let mut used: BTreeSet<usize> = self.model.faces.values().filter_map(|f| f.material).collect();
        for d in self.doc.defs.values() {
            used.extend(d.mesh.faces.values().filter_map(|f| f.material));
            used.extend(d.instances.values().filter_map(|i| i.material));
        }
        used.into_iter().filter_map(|i| self.doc.materials.get(i)).map(|m| m.name.clone()).take(12).collect()
    }

    /// The service to use, with the key from `OPENSKETCH_AI_KEY` if none was typed.
    fn ai_profile(&self) -> ai::Profile {
        let mut p = self.ai.settings.profile();
        if p.api_key.trim().is_empty() {
            p.api_key = std::env::var("OPENSKETCH_AI_KEY").unwrap_or_default();
        }
        p
    }

    /// Render was clicked: ask the next frame for a clean picture of the view.
    fn ai_begin(&mut self, ctx: &egui::Context) {
        self.ai.error = None;
        self.ai.note = None;
        if self.ai.settings.provider.needs_key() && self.ai_profile().api_key.trim().is_empty() {
            self.ai.error = Some("Enter your API key first.".into());
            return;
        }
        self.ai.save_settings();
        self.ai.grab = Some(Arc::new(Mutex::new(None)));
        ctx.request_repaint();
    }

    /// The picture of the view arrived: send it off on a background thread.
    fn ai_send(&mut self, ctx: &egui::Context, capture: image::RgbaImage) {
        let s = &self.ai.settings;
        let (kind, profile) = (s.provider, self.ai_profile());
        let img = ai::prepare(&capture, s.max_side, kind);
        let png = match ai::png_bytes(&img) {
            Ok(png) => png,
            Err(e) => {
                self.ai.error = Some(format!("Could not prepare the picture: {e}"));
                return;
            }
        };
        let prompt = ai::compose_prompt(kind, ai::style(&s.style), &self.ai.description, &self.used_materials(), s.fidelity, &s.avoid);
        let job = ai::Job { png, width: img.width(), height: img.height(), prompt, negative: s.avoid.clone(), fidelity: s.fidelity };
        self.ai.source = Some(ctx.load_texture("ai-source", color_image(&img), egui::TextureOptions::LINEAR));
        let (tx, rx) = std::sync::mpsc::channel();
        let wake = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(ai::render(kind, &profile, &job));
            wake.request_repaint();
        });
        self.ai.job = Some((rx, std::time::Instant::now()));
    }

    fn ai_poll(&mut self, ctx: &egui::Context) {
        let captured = self.ai.grab.as_ref().and_then(|g| g.lock().unwrap().take());
        if let Some(img) = captured {
            self.ai.grab = None;
            self.ai_send(ctx, img);
        } else if self.ai.grab.is_some() {
            ctx.request_repaint();
        }
        let Some((rx, _)) = &self.ai.job else { return };
        let answer = match rx.try_recv() {
            Ok(answer) => answer,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("The render stopped unexpectedly.".to_string()),
        };
        self.ai.job = None;
        match answer.and_then(|bytes| {
            image::load_from_memory(&bytes)
                .map_err(|_| "The service sent an image OpenSketch can't open (only PNG and JPEG work).".to_string())
        }) {
            Ok(img) => {
                let img = img.to_rgba8();
                let tex = ctx.load_texture("ai-result", color_image(&img), egui::TextureOptions::LINEAR);
                self.ai.result = Some((tex, img));
                self.ai.show_source = false;
            }
            Err(e) => self.ai.error = Some(e),
        }
    }

    /// Scripted input self-test: the steps that need the app itself.
    fn run_script(&mut self, ctx: &egui::Context) {
        let Some(script) = &mut self.script else { return };
        ctx.request_repaint();
        if script.wait > 0 {
            return;
        }
        match script.pending.take() {
            Some(Step::Shot(path)) => {
                script.shot = Some(path);
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            Some(Step::Dump(path)) => {
                let m = &self.model;
                let state = match &self.state {
                    State::Idle => "idle",
                    _ => "busy",
                };
                let mut text = format!(
                    "vertices {}\nedges {}\nfaces {}\ninstances {}\nselected {}\nstate {}\nmeasure {}\n",
                    m.vertices.len(),
                    m.edges.len(),
                    m.faces.len(),
                    self.doc.defs[&self.ctx_def()].instances.len(),
                    self.sel.faces.len() + self.sel.edges.len() + self.sel.insts.len(),
                    state,
                    self.measure
                );
                text += &format!("status {}\n", self.status);
                if let Some((lo, hi)) = m.bounds() {
                    text += &format!("bounds {:.3} {:.3} {:.3} .. {:.3} {:.3} {:.3}\n", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
                }
                let _ = std::fs::write(path, text);
            }
            Some(Step::Quit) => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            _ => {}
        }
    }

    /// AI Render self-test: clicks Render once the window has settled, waits
    /// for the answer, saves a picture of the whole window and exits.
    fn ai_self_test(&mut self, ctx: &egui::Context) {
        let Some((path, stage)) = self.ai_test.clone() else { return };
        ctx.request_repaint();
        let busy = self.ai.job.is_some() || self.ai.grab.is_some();
        let next = match stage {
            10 => {
                self.ai_begin(ctx);
                11
            }
            11 if busy => 11,
            20 => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                21
            }
            21.. => {
                let shot = ctx.input(|i| {
                    i.events.iter().find_map(|e| match e {
                        egui::Event::Screenshot { image, .. } => Some(image.clone()),
                        _ => None,
                    })
                });
                if let Some(shot) = shot {
                    if let Some(img) = image::RgbaImage::from_raw(shot.width() as u32, shot.height() as u32, shot.as_raw().to_vec()) {
                        let _ = img.save(&path);
                    }
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                stage
            }
            _ => stage + 1,
        };
        self.ai_test = Some((path, next));
    }

    fn ai_window(&mut self, ctx: &egui::Context) {
        let mut open = self.ai.open;
        egui::Window::new("AI Render").open(&mut open).default_size([940.0, 600.0]).min_width(680.0).min_height(420.0).show(ctx, |ui| {
            // Controls on the left, the picture fills the rest; both take the
            // window's full height so resizing the window resizes the picture.
            let height = ui.available_height();
            let column = egui::Layout::top_down(egui::Align::Min);
            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(Vec2::new(300.0, height), column, |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.ai_controls(ui));
                });
                ui.separator();
                ui.allocate_ui_with_layout(Vec2::new(ui.available_width(), height), column, |ui| self.ai_preview(ui));
            });
        });
        if !open {
            self.ai.save_settings();
        }
        self.ai.open = open;
    }

    fn ai_controls(&mut self, ui: &mut egui::Ui) {
        let busy = self.ai.job.is_some() || self.ai.grab.is_some();
        let (mut go, mut cancel) = (false, false);
        let AiUi { settings, description, show_key, job, error, note, .. } = &mut self.ai;

        ui.label(RichText::new("Service").strong());
        egui::ComboBox::from_id_salt("ai_service").width(280.0).selected_text(settings.provider.label()).show_ui(ui, |ui| {
            for k in ai::Kind::ALL {
                ui.selectable_value(&mut settings.provider, k, k.label());
            }
        });
        let kind = settings.provider;
        let p = settings.profile_mut();
        ui.add_space(4.0);
        ui.label(if kind.needs_key() { "API key" } else { "Password (only if you set one)" });
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut p.api_key).password(!*show_key).desired_width(215.0));
            ui.checkbox(show_key, "Show");
        });
        let (label, url) = kind.key_page();
        ui.hyperlink_to(RichText::new(label).small(), url);
        ui.collapsing("Connection (advanced)", |ui| {
            ui.label("Address");
            ui.add(egui::TextEdit::singleline(&mut p.base_url).desired_width(f32::INFINITY));
            ui.label("Model");
            ui.add(egui::TextEdit::singleline(&mut p.model).hint_text(kind.model_hint()).desired_width(f32::INFINITY));
            ui.label("Extra options (name=value, one per line)");
            ui.add(egui::TextEdit::multiline(&mut p.extra).desired_rows(3).desired_width(f32::INFINITY).font(egui::TextStyle::Monospace));
            if ui.button("Reset to defaults").clicked() {
                let key = std::mem::take(&mut p.api_key);
                *p = ai::Profile { api_key: key, ..kind.default_profile() };
            }
        });
        let sent_to = ai::host(&p.base_url);

        ui.separator();
        ui.label(RichText::new("Describe the scene").strong());
        ui.add(
            egui::TextEdit::multiline(description)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text("e.g. a small brick house with a wooden roof in a garden"),
        );
        ui.add_space(4.0);
        ui.label(RichText::new("Style").strong());
        egui::ComboBox::from_id_salt("ai_style").width(280.0).selected_text(ai::style(&settings.style).name).show_ui(ui, |ui| {
            for s in ai::STYLES {
                ui.selectable_value(&mut settings.style, s.name.to_string(), s.name);
            }
        });
        ui.add_space(4.0);
        ui.label(RichText::new("Keep my model's shape").strong());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Loosely").weak());
            ui.add(egui::Slider::new(&mut settings.fidelity, 0.0..=1.0).show_value(false));
            ui.label(RichText::new("Exactly").weak());
        });
        ui.collapsing("More options", |ui| {
            ui.label("Avoid");
            ui.add(egui::TextEdit::multiline(&mut settings.avoid).desired_rows(3).desired_width(f32::INFINITY));
            if ui.small_button("Reset the list").clicked() {
                settings.avoid = ai::DEFAULT_AVOID.to_string();
            }
            ui.horizontal(|ui| {
                ui.label("Picture size");
                egui::ComboBox::from_id_salt("ai_size").selected_text(format!("{} px", settings.max_side)).show_ui(ui, |ui| {
                    for side in [1024, 1536, 2048] {
                        ui.selectable_value(&mut settings.max_side, side, format!("{side} px"));
                    }
                });
            });
            ui.checkbox(&mut settings.edges, "Send the model's edge lines");
            ui.checkbox(&mut settings.remember_keys, "Remember API keys on this computer");
            if let Some(path) = ai::settings_path() {
                ui.label(RichText::new(format!("Kept as plain text in {}", path.display())).weak().small());
            }
        });

        ui.separator();
        ui.label(RichText::new(format!("A picture of your current view and your description are sent to {sent_to}.")).weak().small());
        ui.add_space(4.0);
        if ui.add_enabled(!busy, egui::Button::new(RichText::new("Render").strong()).min_size(Vec2::new(284.0, 32.0))).clicked() {
            go = true;
        }
        if let Some((_, started)) = job {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("Rendering… {} s", started.elapsed().as_secs()));
                cancel = ui.small_button("Cancel").clicked();
            });
            ui.label(RichText::new("This can take a minute or two.").weak().small());
        }
        if let Some(e) = error {
            ui.colored_label(Color32::from_rgb(190, 40, 30), e.as_str());
        }
        if let Some(n) = note {
            ui.label(RichText::new(n.as_str()).weak());
        }
        if cancel {
            // The answer, if it still comes, is ignored.
            self.ai.job = None;
        }
        if go {
            self.ai_begin(ui.ctx());
        }
    }

    fn ai_preview(&mut self, ui: &mut egui::Ui) {
        let mut save = false;
        if self.ai.result.is_some() {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.ai.show_source, false, "AI render");
                ui.selectable_value(&mut self.ai.show_source, true, "Your view");
                save = ui.button("Save image…").clicked();
            });
            ui.add_space(4.0);
        }
        let shown = match (&self.ai.result, &self.ai.source) {
            (Some((tex, _)), _) if !self.ai.show_source => Some(tex),
            (_, source) => source.as_ref(),
        };
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        match shown {
            Some(tex) => {
                let size = tex.size_vec2();
                let scale = (rect.width() / size.x).min(rect.height() / size.y).max(0.01);
                egui::Image::new((tex.id(), size)).paint_at(ui, Rect::from_center_size(rect.center(), size * scale));
            }
            None => {
                let hint = "Set up the view in the main window, then click Render.\nThe picture appears here.";
                ui.painter().text(rect.center(), Align2::CENTER_CENTER, hint, FontId::proportional(14.0), ui.visuals().weak_text_color());
            }
        }
        if save {
            self.ai_save_image();
        }
    }

    fn ai_save_image(&mut self) {
        let Some((_, img)) = &self.ai.result else { return };
        let Some(path) = rfd::FileDialog::new().add_filter("PNG image", &["png"]).set_file_name("render.png").save_file() else { return };
        let path = if path.extension().is_some() { path } else { path.with_extension("png") };
        match img.save(&path) {
            Ok(()) => {
                self.ai.error = None;
                self.ai.note = Some(format!("Saved {}", path.display()));
            }
            Err(e) => self.ai.error = Some(format!("Could not save the image: {e}")),
        }
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_help;
        egui::Window::new("Keyboard shortcuts").open(&mut open).resizable(false).show(ctx, |ui| {
            egui::ScrollArea::vertical().max_height(700.0).show(ui, |ui| {
                egui::Grid::new("keys").striped(true).show(ui, |ui| {
                    let rows = [
                        ("Space", "Select"),
                        ("L / A / R / C / N", "Line / 2-Point Arc / Rectangle / Circle / Polygon"),
                        ("Follow Me", "Select a path (or a face), then click the profile face"),
                        ("Intersect Faces", "Edit menu: cut edges where faces/groups cross"),
                        ("P / F", "Push/Pull / Offset"),
                        ("Double-click (Push/Pull)", "Repeat the last distance on that face"),
                        ("Click or drag", "Tools work click-move-click, or press-drag-release"),
                        ("M / Q / S", "Move / Rotate / Scale"),
                        ("E / B / T", "Eraser / Paint Bucket / Tape Measure"),
                        ("Protractor", "Click center → baseline → angle (type 30 or a slope 1:2)"),
                        ("Dimension", "Two points or an edge; click an arc/circle for R / Ø"),
                        ("Text", "Click geometry for leader text, empty space for screen text"),
                        ("Solid Tools", "Click the first solid, then the second"),
                        ("O / H", "Orbit / Pan"),
                        ("G / Ctrl+G", "Make Component / Make Group"),
                        ("Double-click group", "Edit it (Esc or click outside closes)"),
                        ("Shift+Z", "Zoom extents"),
                        ("Middle-drag", "Orbit (Shift = pan)"),
                        ("Right-drag", "Pan"),
                        ("Scroll", "Zoom to cursor"),
                        ("Arrow →/←/↑", "Lock red / green / blue axis (↓ unlocks)"),
                        ("Hold Shift", "Lock the current axis inference"),
                        ("Type + Enter", "Exact value (2.5, 300mm, 4,3, 24s, 3x, /4)"),
                        ("Ctrl (Move)", "Toggle copy"),
                        ("Ctrl (Tape Measure)", "Toggle measure-only (no guide)"),
                        ("Delete", "Erase selection"),
                        ("PgDn / PgUp", "Next / previous scene"),
                        ("Ctrl+Z / Ctrl+Y", "Undo / Redo"),
                        ("Ctrl+S / Ctrl+O", "Save / Open"),
                        ("Esc", "Cancel / clear selection / close group"),
                    ];
                    for (k, v) in rows {
                        ui.label(RichText::new(k).monospace().strong());
                        ui.label(v);
                        ui.end_row();
                    }
                });
            });
        });
        self.show_help = open;
    }
}

/// Reads the 3D viewport back as an image (screenshot self-test, AI Render).
fn read_viewport(gl: &glow::Context, vp: [i32; 4]) -> image::RgbaImage {
    use eframe::glow::HasContext;
    let (w, h) = (vp[2].max(1), vp[3].max(1));
    let mut buf = vec![0u8; (w * h * 4) as usize];
    unsafe {
        gl.read_pixels(vp[0], vp[1], w, h, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut buf)));
    }
    let row = (w * 4) as usize;
    let flipped: Vec<u8> = buf.chunks_exact(row).rev().flatten().copied().collect();
    image::RgbaImage::from_raw(w as u32, h as u32, flipped).unwrap_or_else(|| image::RgbaImage::new(1, 1))
}

/// A small textured, sunlit scene for the screenshot self-test.
fn demo_document(ground: bool) -> Document {
    let mut doc = Document::new();
    let mat = |doc: &Document, name: &str| doc.materials.iter().position(|m| m.name == name);
    let (grass, brick, wood, stone, tiles) =
        (mat(&doc, "Grass"), mat(&doc, "Brick"), mat(&doc, "Wood"), mat(&doc, "Stone"), mat(&doc, "Floor Tiles"));
    let mut m = Model::new();
    let rect = |m: &mut Model, x0: f32, y0: f32, x1: f32, y1: f32| {
        m.add_polygon(&[Vec3::new(x0, y0, 0.0), Vec3::new(x1, y0, 0.0), Vec3::new(x1, y1, 0.0), Vec3::new(x0, y1, 0.0)], None)
    };
    if ground {
        if let Some(g) = rect(&mut m, -7.0, -6.0, 7.0, 7.0) {
            m.faces.get_mut(&g).unwrap().material = grass;
        }
    }
    let paint_box = |m: &mut Model, f: Id, h: f32, side: Option<usize>, top: Option<usize>| {
        let before: HashSet<Id> = m.faces.keys().copied().collect();
        m.push_pull(f, -h);
        for (id, face) in m.faces.iter_mut() {
            if !before.contains(id) || *id == f {
                face.material = if face.normal.z.abs() > 0.9 { top } else { side };
            }
        }
    };
    if let Some(h) = rect(&mut m, 0.0, 0.0, 4.0, 3.0) {
        paint_box(&mut m, h, 2.6, brick, wood);
    }
    // A recessed window in the front wall.
    let win = [Vec3::new(1.2, 0.0, 1.0), Vec3::new(2.8, 0.0, 1.0), Vec3::new(2.8, 0.0, 2.0), Vec3::new(1.2, 0.0, 2.0)];
    if let Some(w) = m.add_polygon(&win, Some(-Vec3::Y)) {
        m.push_pull(w, -0.25);
    }
    if let Some(b) = rect(&mut m, -4.5, -1.0, -2.5, 1.0) {
        paint_box(&mut m, b, 1.2, stone, stone);
    }
    if let Some(t) = rect(&mut m, -4.5, 2.5, -1.0, 5.0) {
        m.faces.get_mut(&t).unwrap().material = tiles;
    }
    doc.defs.get_mut(&ROOT).unwrap().mesh = m;
    doc.sun = Sun { enabled: true, day: 172, hour: 15.5, ..Sun::default() };
    doc
}

fn label(painter: &egui::Painter, pos: Pos2, text: &str, color: Color32) {
    let galley = painter.layout_no_wrap(text.to_string(), FontId::proportional(12.5), color);
    let r = Rect::from_min_size(pos, galley.size()).expand(3.0);
    painter.rect_filled(r, 3.0, Color32::from_white_alpha(230));
    painter.rect_stroke(r, 3.0, stroke(1.0, color.gamma_multiply(0.6)), egui::StrokeKind::Inside);
    painter.galley(pos, galley, color);
}

fn paint_background(painter: &egui::Painter, rect: Rect) {
    let mut mesh = egui::Mesh::default();
    let top = Color32::from_rgb(190, 211, 232);
    let bottom = Color32::from_rgb(236, 239, 242);
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(Shape::mesh(mesh));
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let ctx_def = self.ctx_def();
        let model = &self.model;
        let insts = &self.doc.defs[&ctx_def].instances;
        self.sel.faces.retain(|f| model.faces.contains_key(f));
        self.sel.edges.retain(|e| model.edges.contains_key(e));
        let doc = &self.doc;
        self.sel.insts.retain(|i| insts.get(i).is_some_and(|x| doc.tag_visible(x.tag)));

        self.handle_keyboard(ctx);
        self.menu_bar(ctx);
        self.status_bar(ctx);
        self.toolbar(ctx);
        self.inspector(ctx);
        self.scene_tabs(ctx);
        self.animate_camera(ctx);
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| self.viewport(ui));
        if self.show_help {
            self.help_window(ctx);
        }
        self.ai_poll(ctx);
        if self.ai.open {
            self.ai_window(ctx);
        }
        self.ai_self_test(ctx);
        self.run_script(ctx);
        if let Some(shot) = &mut self.shot {
            shot.1 += 1;
            if shot.2.load(std::sync::atomic::Ordering::SeqCst) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ctx.request_repaint();
        }
        let name =
            self.file.as_ref().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "Untitled".into());
        let title = format!("{}{} — OpenSketch", name, if self.dirty { " *" } else { "" });
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        if let Some(script) = &mut self.script {
            script.feed(raw_input);
        }
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let (Some(gl), Some(r)) = (gl, &self.renderer) {
            r.lock().unwrap().destroy(gl);
        }
    }
}

/// Angle typed into the Measurements box: degrees ("30", "30°") or a slope
/// rise:run ("1:2"). Returns radians.
fn parse_angle(text: &str) -> Option<f32> {
    let t = text.trim().trim_end_matches('°');
    if let Some((a, b)) = t.split_once(':') {
        let (rise, run) = (a.trim().parse::<f32>().ok()?, b.trim().parse::<f32>().ok()?);
        return (rise != 0.0 || run != 0.0).then(|| rise.atan2(run));
    }
    t.parse::<f32>().ok().map(f32::to_radians)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arcs_find_the_bulge_that_continues_an_edge() {
        // An edge arriving at the origin along +X, then an arc on to (1, 1).
        let mut m = Model::new();
        let a = m.add_vertex(Vec3::new(-1.0, 0.0, 0.0));
        let o = m.add_vertex(Vec3::ZERO);
        m.add_edge(a, o);
        let (p0, p1) = (Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0));
        let bulges = tangent_bulges(&m, p0, p1, Vec3::Z);
        assert_eq!(bulges.len(), 1);
        // A quarter circle of radius 1 around (0, 1): sagitta 1 - cos 45°.
        assert!((bulges[0].abs() - (1.0 - 0.5f32.sqrt())).abs() < 1e-4, "{bulges:?}");
        let pts = arc_points_2pt(p0, p1, Vec3::Z, bulges[0], 64);
        assert!(pts.iter().all(|p| (p.distance(Vec3::Y) - 1.0).abs() < 1e-3));
        assert!((pts[1] - pts[0]).normalize().dot(Vec3::X) > 0.999, "leaves along the edge");
        // The same at the far end: an edge leaving (1, 1) along +Y.
        let mut m = Model::new();
        let e = m.add_vertex(p1);
        let f = m.add_vertex(Vec3::new(1.0, 2.0, 0.0));
        m.add_edge(e, f);
        let bulges = tangent_bulges(&m, p0, p1, Vec3::Z);
        assert_eq!(bulges.len(), 1);
        let pts = arc_points_2pt(p0, p1, Vec3::Z, bulges[0], 64);
        assert!((pts[64] - pts[63]).normalize().dot(Vec3::Y) > 0.999, "arrives along the edge");
        // An edge that continues the chord in a straight line gives no arc; nor does none.
        let mut m = Model::new();
        let a = m.add_vertex(Vec3::new(-1.0, -1.0, 0.0));
        let o = m.add_vertex(Vec3::ZERO);
        m.add_edge(a, o);
        assert!(tangent_bulges(&m, p0, p1, Vec3::Z).is_empty());
        assert!(tangent_bulges(&Model::new(), p0, p1, Vec3::Z).is_empty());
    }

    #[test]
    fn two_point_arc_geometry() {
        let (p0, p1) = (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Half circle: every point is on the unit circle, apex at +Y.
        let pts = arc_points_2pt(p0, p1, Vec3::Z, 1.0, 12);
        assert_eq!(pts.len(), 13);
        assert!(pts.iter().all(|p| (p.length() - 1.0).abs() < 1e-4));
        assert!((pts[6] - Vec3::Y).length() < 1e-4);
        assert_eq!(pts[12], p1);
        // Shallow arc: apex at the bulge height, on the other side for a negative bulge.
        let pts = arc_points_2pt(p0, p1, Vec3::Z, -0.25, 8);
        assert!((pts[4] - Vec3::new(0.0, -0.25, 0.0)).length() < 1e-4);
        // Bulge larger than half the chord: major arc.
        let pts = arc_points_2pt(p0, p1, Vec3::Z, 1.5, 24);
        assert!((pts[12].y - 1.5).abs() < 1e-3);
    }

    #[test]
    fn center_arc_quarter() {
        let pts = arc_points_center(Vec3::ZERO, Vec3::X, Vec3::Z, std::f32::consts::FRAC_PI_2, 6);
        assert!((pts[6] - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn protractor_angles_and_slopes() {
        let deg = |t: &str| parse_angle(t).map(f32::to_degrees);
        assert!((deg("30").unwrap() - 30.0).abs() < 1e-4);
        assert!((deg("45°").unwrap() - 45.0).abs() < 1e-4);
        assert!((deg("1:1").unwrap() - 45.0).abs() < 1e-4);
        assert!((deg("1:2").unwrap() - 26.565).abs() < 1e-2);
        assert!(parse_angle("abc").is_none());
    }
}
