//! Document: definitions (geometry containers) and their instances.
//!
//! As in SketchUp, the model itself is the root definition. A group is a
//! definition with a single instance; a component definition can be placed many
//! times, and editing it updates every instance.
//!
//! While the user edits inside a group/component ("edit context"), the app works
//! on a copy of that definition's mesh transformed into world space; `load` and
//! `sync` convert between the two.

use crate::annotation::Annotation;
use crate::camera::Camera;
use crate::guide::Guide;
use crate::model::{default_materials, Id, Material, Model, EPS};
use crate::sun::Sun;
use glam::{Mat3, Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub type DefId = u32;
pub const ROOT: DefId = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefKind {
    Root,
    Group,
    Component,
}

impl DefKind {
    pub fn label(self) -> &'static str {
        match self {
            DefKind::Root => "Model",
            DefKind::Group => "Group",
            DefKind::Component => "Component",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub def: DefId,
    /// Placement relative to the parent definition.
    pub xf: Mat4,
    /// Material applied to faces that have none (SketchUp's "paint the group").
    #[serde(default)]
    pub material: Option<usize>,
    /// Face of the parent definition this instance is glued to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glued: Option<Id>,
    /// Tag (layer); `None` is "Untagged".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<u32>,
}

/// A tag (SketchUp's layers): toggles visibility of the instances using it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tag {
    pub name: String,
    pub visible: bool,
    /// Used by View ▸ Color by Tag.
    pub color: [f32; 3],
}

/// Display settings a scene remembers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneStyle {
    pub xray: bool,
    pub hidden_geometry: bool,
    pub guides: bool,
    pub axes: bool,
    pub color_by_tag: bool,
    pub hide_annotations: bool,
    /// Shadow settings (None in scenes saved before shadows existed).
    pub sun: Option<Sun>,
}

/// A saved view: camera, which tags are hidden, and display settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub name: String,
    pub camera: Camera,
    #[serde(default)]
    pub hidden_tags: Vec<u32>,
    #[serde(default)]
    pub style: SceneStyle,
}

const TAG_COLORS: [[f32; 3]; 8] = [
    [0.93, 0.45, 0.40],
    [0.40, 0.68, 0.93],
    [0.55, 0.80, 0.45],
    [0.95, 0.75, 0.30],
    [0.70, 0.52, 0.88],
    [0.35, 0.80, 0.78],
    [0.92, 0.55, 0.75],
    [0.65, 0.62, 0.55],
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Definition {
    pub name: String,
    pub kind: DefKind,
    pub mesh: Model,
    #[serde(default)]
    pub instances: BTreeMap<Id, Instance>,
    /// Instances stick to faces: placed with blue along the face normal and
    /// moved only within the face plane.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub glue: bool,
    /// Glued instances cut their z = 0 outline out of the face (windows, doors).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cut_opening: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub defs: BTreeMap<DefId, Definition>,
    pub materials: Vec<Material>,
    pub next_id: u32,
    /// Construction guides, in world space.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guides: Vec<Guide>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<u32, Tag>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scenes: Vec<Scene>,
    /// Dimensions and text, in world space.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<Annotation>,
    /// Shadows: on/off, time, date and location.
    #[serde(default, skip_serializing_if = "Sun::is_default")]
    pub sun: Sun,
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

pub fn box_corners(lo: Vec3, hi: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(lo.x, lo.y, lo.z),
        Vec3::new(hi.x, lo.y, lo.z),
        Vec3::new(hi.x, hi.y, lo.z),
        Vec3::new(lo.x, hi.y, lo.z),
        Vec3::new(lo.x, lo.y, hi.z),
        Vec3::new(hi.x, lo.y, hi.z),
        Vec3::new(hi.x, hi.y, hi.z),
        Vec3::new(lo.x, hi.y, hi.z),
    ]
}

/// Placement frame of a component glued to a face with normal `n` at `p`:
/// blue along the normal, red level (perpendicular to `up`), green up the face.
/// Horizontal faces use `x_hint` for red.
pub fn glue_frame(n: Vec3, p: Vec3, up: Vec3, x_hint: Vec3) -> Mat4 {
    let z = n.normalize();
    let mut x = up.cross(z);
    if x.length() < 1e-3 {
        x = x_hint - z * x_hint.dot(z);
    }
    let x = x.normalize();
    let y = z.cross(x);
    Mat4::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), p.extend(1.0))
}

fn grow(bb: Option<(Vec3, Vec3)>, p: Vec3) -> Option<(Vec3, Vec3)> {
    Some(match bb {
        Some((lo, hi)) => (lo.min(p), hi.max(p)),
        None => (p, p),
    })
}

impl Document {
    pub fn new() -> Self {
        let root = Definition {
            name: "Model".into(),
            kind: DefKind::Root,
            mesh: Model::new(),
            instances: BTreeMap::new(),
            glue: false,
            cut_opening: false,
        };
        Self {
            defs: BTreeMap::from([(ROOT, root)]),
            materials: default_materials(),
            next_id: 1,
            guides: Vec::new(),
            tags: BTreeMap::new(),
            scenes: Vec::new(),
            annotations: Vec::new(),
            sun: Sun::default(),
        }
    }

    /// Wraps a mesh saved by an older version (no groups) into a document.
    pub fn from_model(mut m: Model) -> Self {
        let mut d = Self::new();
        if !m.materials.is_empty() {
            d.materials = std::mem::take(&mut m.materials);
        }
        d.defs.get_mut(&ROOT).unwrap().mesh = m;
        d
    }

    fn nid(&mut self) -> u32 {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    pub fn material_color(&self, m: Option<usize>) -> [f32; 3] {
        m.and_then(|i| self.materials.get(i)).map(|m| m.color).unwrap_or([0.96, 0.96, 0.94])
    }

    // ----- edit context ------------------------------------------------------

    /// Definition being edited for a path of nested instance ids.
    pub fn ctx_def(&self, path: &[Id]) -> DefId {
        let mut cur = ROOT;
        for i in path {
            match self.defs.get(&cur).and_then(|d| d.instances.get(i)) {
                Some(inst) => cur = inst.def,
                None => break,
            }
        }
        cur
    }

    /// World transform of the edit context.
    pub fn ctx_xf(&self, path: &[Id]) -> Mat4 {
        let mut cur = ROOT;
        let mut m = Mat4::IDENTITY;
        for i in path {
            match self.defs.get(&cur).and_then(|d| d.instances.get(i)) {
                Some(inst) => {
                    m *= inst.xf;
                    cur = inst.def;
                }
                None => break,
            }
        }
        m
    }

    /// The context's mesh in world coordinates.
    pub fn load(&self, path: &[Id]) -> Model {
        let mesh = &self.defs[&self.ctx_def(path)].mesh;
        if path.is_empty() {
            mesh.clone()
        } else {
            mesh.transformed(self.ctx_xf(path))
        }
    }

    /// Stores the world-space working mesh back into the context's definition.
    pub fn sync(&mut self, path: &[Id], active: &Model) {
        let d = self.ctx_def(path);
        let mesh = if path.is_empty() { active.clone() } else { active.transformed(self.ctx_xf(path).inverse()) };
        self.defs.get_mut(&d).unwrap().mesh = mesh;
        self.refresh_glue();
    }

    // ----- tags ---------------------------------------------------------------

    /// Instances with no tag, or a missing tag, are always shown.
    pub fn tag_visible(&self, tag: Option<u32>) -> bool {
        tag.and_then(|t| self.tags.get(&t)).is_none_or(|t| t.visible)
    }

    pub fn add_tag(&mut self) -> u32 {
        let names: HashSet<&str> = self.tags.values().map(|t| t.name.as_str()).collect();
        let name = (1..).map(|n| format!("Tag {n}")).find(|n| !names.contains(n.as_str())).unwrap();
        let color = TAG_COLORS[self.tags.len() % TAG_COLORS.len()];
        let id = self.nid();
        self.tags.insert(id, Tag { name, visible: true, color });
        id
    }

    /// Deletes a tag; its instances become untagged.
    pub fn delete_tag(&mut self, id: u32) {
        self.tags.remove(&id);
        for d in self.defs.values_mut() {
            for i in d.instances.values_mut() {
                if i.tag == Some(id) {
                    i.tag = None;
                }
            }
        }
        for sc in &mut self.scenes {
            sc.hidden_tags.retain(|t| *t != id);
        }
    }

    /// Number of instances (anywhere in the model) using a tag.
    pub fn tag_usage(&self, id: u32) -> usize {
        self.defs.values().map(|d| d.instances.values().filter(|i| i.tag == Some(id)).count()).sum()
    }

    pub fn hidden_tags(&self) -> Vec<u32> {
        self.tags.iter().filter(|(_, t)| !t.visible).map(|(i, _)| *i).collect()
    }

    pub fn next_scene_name(&self) -> String {
        let names: HashSet<&str> = self.scenes.iter().map(|s| s.name.as_str()).collect();
        (1..).map(|n| format!("Scene {n}")).find(|n| !names.contains(n.as_str())).unwrap()
    }

    // ----- gluing & openings ---------------------------------------------------

    /// Opening outlines of a definition in its own coordinates (empty unless
    /// it glues and cuts).
    pub fn openings(&self, def: DefId) -> Vec<Vec<Vec3>> {
        match self.defs.get(&def) {
            Some(d) if d.glue && d.cut_opening => d.mesh.opening_loops(),
            _ => vec![],
        }
    }

    /// Point (local, on z = 0) that must lie on the host face: the middle of
    /// the openings, else of the bounding box.
    fn glue_ref(&self, def: DefId) -> Vec3 {
        let loops = self.defs.get(&def).map(|d| d.mesh.opening_loops()).unwrap_or_default();
        let mut bb = None;
        for p in loops.iter().flatten() {
            bb = grow(bb, *p);
        }
        let (lo, hi) = bb.or_else(|| self.local_bounds(def)).unwrap_or((Vec3::ZERO, Vec3::ZERO));
        let c = (lo + hi) * 0.5;
        Vec3::new(c.x, c.y, 0.0)
    }

    /// Face of `mesh` that an instance's z = 0 plane sits on, preferring the
    /// face it was glued to before.
    fn find_host(&self, mesh: &Model, inst: &Instance) -> Option<Id> {
        let o = inst.xf.transform_point3(Vec3::ZERO);
        let z = (Mat3::from_mat4(inst.xf).inverse().transpose() * Vec3::Z).normalize_or_zero();
        let r = inst.xf.transform_point3(self.glue_ref(inst.def));
        let ok = |f: Id| {
            mesh.faces.get(&f).is_some_and(|face| {
                face.normal.dot(z).abs() > 0.9999
                    && (o - mesh.pos(face.outer[0])).dot(face.normal).abs() < EPS * 10.0
                    && mesh.point_in_loop(&face.outer, r, face.normal)
            })
        };
        inst.glued.filter(|&f| ok(f)).or_else(|| mesh.faces.keys().copied().find(|&f| ok(f)))
    }

    /// Re-checks which face every gluing instance sits on (after edits, moves
    /// or setting changes). Instances no longer on a face come unglued.
    pub fn refresh_glue(&mut self) {
        let mut updates = Vec::new();
        for (&did, d) in &self.defs {
            for (&iid, inst) in &d.instances {
                let host = match self.defs.get(&inst.def) {
                    Some(cd) if cd.glue => self.find_host(&d.mesh, inst),
                    _ => None,
                };
                if host != inst.glued {
                    updates.push((did, iid, host));
                }
            }
        }
        for (did, iid, host) in updates {
            if let Some(i) = self.defs.get_mut(&did).and_then(|d| d.instances.get_mut(&iid)) {
                i.glued = host;
            }
        }
    }

    /// For a component made from geometry sitting on a face (drawn on it,
    /// pushed into it, or sticking out of a wall): that face, and the normal
    /// the component's blue axis should follow.
    fn glue_host(active: &Model, faces: &[Id], edges: &[Id]) -> Option<(Id, Vec3)> {
        let verts = active.vertices_of(faces, edges);
        if verts.len() < 3 {
            return None;
        }
        let sel: HashSet<Id> = faces.iter().copied().collect();
        let tol = EPS * 10.0;
        let mut best: Option<(usize, Id, Vec3)> = None;
        for (&fid, f) in &active.faces {
            if sel.contains(&fid) || !f.vertices().any(|v| verts.contains(&v)) {
                continue;
            }
            let p0 = active.pos(f.outer[0]);
            // Ground faces face down by convention; components sit on top of them.
            let ground = f.normal.z < -0.999 && f.outer.iter().all(|&v| active.pos(v).z.abs() < tol);
            let n = if ground { Vec3::Z } else { f.normal };
            let ds: Vec<f32> = verts.iter().map(|&v| (active.pos(v) - p0).dot(n)).collect();
            let on = ds.iter().filter(|d| d.abs() < tol).count();
            let flat = on == ds.len();
            let behind = ds.iter().all(|&d| d < tol);
            let front = ds.iter().all(|&d| d > -tol) && n.z.abs() < 0.1;
            if on >= 3 && (flat || behind || front) && best.is_none_or(|b| on > b.0) {
                best = Some((on, fid, n));
            }
        }
        best.map(|b| (b.1, b.2))
    }

    /// Openings cut by the glued children of `def`, per host face, transformed by `xf`.
    pub fn child_openings(&self, def: DefId, xf: Mat4) -> HashMap<Id, Vec<Vec<Vec3>>> {
        let mut out: HashMap<Id, Vec<Vec<Vec3>>> = HashMap::new();
        let Some(d) = self.defs.get(&def) else { return out };
        for c in d.instances.values() {
            let Some(host) = c.glued else { continue };
            let m = xf * c.xf;
            for l in self.openings(c.def) {
                out.entry(host).or_default().push(l.iter().map(|p| m.transform_point3(*p)).collect());
            }
        }
        out
    }

    // ----- queries -------------------------------------------------------------

    pub fn instance(&self, path: &[Id], iid: Id) -> Option<&Instance> {
        self.defs.get(&self.ctx_def(path))?.instances.get(&iid)
    }

    pub fn instances_of(&self, def: DefId) -> usize {
        self.defs.values().map(|d| d.instances.values().filter(|i| i.def == def).count()).sum()
    }

    /// True if `outer` is `target` or (transitively) contains an instance of it.
    pub fn contains_def(&self, outer: DefId, target: DefId) -> bool {
        outer == target || self.defs.get(&outer).is_some_and(|d| d.instances.values().any(|i| self.contains_def(i.def, target)))
    }

    /// Bounding box of a definition in its own coordinates, including children.
    pub fn local_bounds(&self, def: DefId) -> Option<(Vec3, Vec3)> {
        let d = self.defs.get(&def)?;
        let mut bb = d.mesh.bounds();
        for inst in d.instances.values() {
            if let Some((lo, hi)) = self.local_bounds(inst.def) {
                for c in box_corners(lo, hi) {
                    bb = grow(bb, inst.xf.transform_point3(c));
                }
            }
        }
        bb
    }

    pub fn world_bounds(&self) -> Option<(Vec3, Vec3)> {
        self.local_bounds(ROOT)
    }

    /// World-space bounds of a child instance of the context.
    pub fn instance_world_bounds(&self, path: &[Id], iid: Id) -> Option<(Vec3, [Vec3; 8])> {
        let inst = self.instance(path, iid)?;
        let (lo, hi) = self.local_bounds(inst.def)?;
        let xf = self.ctx_xf(path) * inst.xf;
        let corners = box_corners(lo, hi).map(|c| xf.transform_point3(c));
        Some((xf.transform_point3((lo + hi) * 0.5), corners))
    }

    pub fn next_name(&self, kind: DefKind) -> String {
        let names: HashSet<&str> = self.defs.values().map(|d| d.name.as_str()).collect();
        (1..).map(|n| format!("{}#{n}", kind.label())).find(|n| !names.contains(n.as_str())).unwrap()
    }

    // ----- editing -------------------------------------------------------------

    pub fn add_def(&mut self, name: String, kind: DefKind, mesh: Model) -> DefId {
        let id = self.nid();
        self.defs.insert(id, Definition { name, kind, mesh, instances: BTreeMap::new(), glue: false, cut_opening: false });
        id
    }

    pub fn add_instance(&mut self, parent: DefId, inst: Instance) -> Id {
        let id = self.nid();
        self.defs.get_mut(&parent).unwrap().instances.insert(id, inst);
        id
    }

    /// Copies a definition; nested groups are copied too (groups stay unique),
    /// nested components keep pointing at their shared definition.
    pub fn deep_copy_def(&mut self, def: DefId) -> DefId {
        let mut d = self.defs[&def].clone();
        d.name = self.next_name(d.kind);
        let children = std::mem::take(&mut d.instances);
        let id = self.nid();
        self.defs.insert(id, d);
        for (_, mut inst) in children {
            if self.defs.get(&inst.def).map(|x| x.kind) == Some(DefKind::Group) {
                inst.def = self.deep_copy_def(inst.def);
            }
            self.add_instance(id, inst);
        }
        id
    }

    /// Removes group definitions that are no longer placed anywhere.
    pub fn gc(&mut self) {
        loop {
            let dead: Vec<DefId> =
                self.defs.iter().filter(|(id, d)| d.kind == DefKind::Group && self.instances_of(**id) == 0).map(|(id, _)| *id).collect();
            if dead.is_empty() {
                break;
            }
            for d in dead {
                self.defs.remove(&d);
            }
        }
    }

    /// Removes component definitions with no instances ("Purge Unused").
    pub fn purge_components(&mut self) -> usize {
        let mut removed = 0;
        loop {
            let dead: Vec<DefId> =
                self.defs.iter().filter(|(id, d)| d.kind != DefKind::Root && self.instances_of(**id) == 0).map(|(id, _)| *id).collect();
            if dead.is_empty() {
                break;
            }
            removed += dead.len();
            for d in dead {
                self.defs.remove(&d);
            }
        }
        removed
    }

    /// Turns selected geometry/instances of the context into a new group or
    /// component. `active` is the world-space working mesh of the context;
    /// call `sync` afterwards. Returns the new instance id.
    #[allow(clippy::too_many_arguments)]
    pub fn make_group(&mut self, path: &[Id], active: &mut Model, faces: &[Id], edges: &[Id], insts: &[Id], kind: DefKind) -> Option<Id> {
        let ctx = self.ctx_def(path);
        let inv = self.ctx_xf(path).inverse();
        let mut all_edges: Vec<Id> = edges.to_vec();
        for f in faces {
            all_edges.extend(active.face_edges(*f));
        }
        all_edges.sort_unstable();
        all_edges.dedup();
        if faces.is_empty() && all_edges.is_empty() && insts.is_empty() {
            return None;
        }

        let cxf = self.ctx_xf(path);
        let glue = if kind == DefKind::Component { Self::glue_host(active, faces, &all_edges) } else { None };
        let part_world = active.extract(faces, &all_edges);
        let part = part_world.transformed(inv);
        let mut bb = part.bounds();
        let mut moved = Vec::new();
        for i in insts {
            if let Some(inst) = self.defs.get_mut(&ctx).and_then(|d| d.instances.remove(i)) {
                moved.push((*i, inst));
            }
        }
        for (_, inst) in &moved {
            if let Some((lo, hi)) = self.local_bounds(inst.def) {
                for c in box_corners(lo, hi) {
                    bb = grow(bb, inst.xf.transform_point3(c));
                }
            }
        }
        // Like SketchUp, the new definition's origin is its bounding-box corner.
        // Components on a face get axes aligned to it, origin on its plane.
        let frame = match glue {
            Some((host, n)) => {
                let (up, xh) = (cxf.transform_vector3(Vec3::Z), cxf.transform_vector3(Vec3::X));
                let rot = glue_frame(n, Vec3::ZERO, up, xh);
                let to = rot.inverse();
                let mut lo: Option<(Vec3, Vec3)> = None;
                for p in part_world.vertices.values() {
                    lo = grow(lo, to.transform_point3(*p));
                }
                for (_, inst) in &moved {
                    if let Some((a, b)) = self.local_bounds(inst.def) {
                        for c in box_corners(a, b) {
                            lo = grow(lo, to.transform_point3((cxf * inst.xf).transform_point3(c)));
                        }
                    }
                }
                let lo = lo.map(|b| b.0).unwrap_or(Vec3::ZERO);
                let plane_z = to.transform_point3(active.pos(active.faces[&host].outer[0])).z;
                let origin = rot.transform_point3(Vec3::new(lo.x, lo.y, plane_z));
                inv * glue_frame(n, origin, up, xh)
            }
            None => Mat4::from_translation(bb.map(|b| b.0).unwrap_or(Vec3::ZERO)),
        };
        let to_local = frame.inverse();
        active.remove_geometry(faces, &all_edges);

        let name = self.next_name(kind);
        let def = self.add_def(name, kind, part.transformed(to_local));
        for (iid, mut inst) in moved {
            inst.xf = to_local * inst.xf;
            self.defs.get_mut(&def).unwrap().instances.insert(iid, inst);
        }
        if let Some((host, n)) = glue {
            let d = self.defs.get_mut(&def).unwrap();
            d.glue = true;
            d.cut_opening = true;
            // The opening is cut dynamically from now on, so heal the host face.
            let world = cxf * frame;
            let material = active.faces.get(&host).and_then(|f| f.material);
            let face_n = active.faces.get(&host).map(|f| f.normal).unwrap_or(n);
            for lp in self.defs[&def].mesh.opening_loops() {
                let ids: Option<Vec<Id>> = lp.iter().map(|p| active.find_vertex(world.transform_point3(*p))).collect();
                if let Some(ids) = ids {
                    active.fill_and_merge(&ids, face_n, material);
                }
            }
        }
        Some(self.add_instance(ctx, Instance { def, xf: frame, material: None, glued: None, tag: None }))
    }

    /// Wraps a world-space mesh (e.g. a solid tool result) into a new group or
    /// component of the context, origin at its bounding-box corner.
    pub fn add_world_mesh(&mut self, path: &[Id], world: &Model, kind: DefKind) -> Option<Id> {
        let ctx = self.ctx_def(path);
        let local = world.transformed(self.ctx_xf(path).inverse());
        let (lo, _) = local.bounds()?;
        let name = self.next_name(kind);
        let def = self.add_def(name, kind, local.transformed(Mat4::from_translation(-lo)));
        Some(self.add_instance(ctx, Instance { def, xf: Mat4::from_translation(lo), material: None, glued: None, tag: None }))
    }

    /// Explodes an instance back into loose geometry of the context.
    /// Returns (new faces, new edges, released child instances). Call `sync` afterwards.
    pub fn explode(&mut self, path: &[Id], active: &mut Model, iid: Id) -> Option<(Vec<Id>, Vec<Id>, Vec<Id>)> {
        let ctx = self.ctx_def(path);
        let cxf = self.ctx_xf(path);
        let inst = self.defs.get_mut(&ctx)?.instances.remove(&iid)?;
        let def = self.defs.get(&inst.def)?.clone();
        if let Some(host) = inst.glued.filter(|_| def.glue && def.cut_opening) {
            // The opening becomes real geometry.
            if let Some(face) = active.faces.get(&host) {
                let (n, p0) = (face.normal, active.pos(face.outer[0]));
                let m = cxf * inst.xf;
                let loops: Vec<Vec<Vec3>> =
                    def.mesh.opening_loops().iter().map(|l| l.iter().map(|p| m.transform_point3(*p)).collect()).collect();
                active.subtract_loops(n, p0, &loops);
            }
        }
        let mut geo = def.mesh.transformed(cxf * inst.xf);
        if inst.material.is_some() {
            for f in geo.faces.values_mut() {
                f.material = f.material.or(inst.material);
            }
        }
        let (faces, edges) = active.merge_from(&geo);
        let still_used = self.instances_of(inst.def) > 0;
        let mut released = Vec::new();
        for c in def.instances.values() {
            let mut cdef = c.def;
            if still_used && self.defs.get(&cdef).map(|d| d.kind) == Some(DefKind::Group) {
                cdef = self.deep_copy_def(cdef);
            }
            let ni = Instance { def: cdef, xf: inst.xf * c.xf, material: c.material.or(inst.material), glued: None, tag: c.tag };
            released.push(self.add_instance(ctx, ni));
        }
        self.gc();
        Some((faces, edges, released))
    }

    /// Applies a world-space transform to child instances of the context.
    /// With `copy`, new instances are created instead (groups get their own copy).
    pub fn transform_instances(&mut self, path: &[Id], insts: &[Id], world: Mat4, copy: bool) -> Vec<Id> {
        let ctx = self.ctx_def(path);
        let cxf = self.ctx_xf(path);
        let local = cxf.inverse() * world * cxf;
        let mut created = Vec::new();
        for iid in insts {
            let Some(inst) = self.defs.get(&ctx).and_then(|d| d.instances.get(iid)).cloned() else { continue };
            let xf = local * inst.xf;
            if copy {
                let def = if self.defs[&inst.def].kind == DefKind::Group { self.deep_copy_def(inst.def) } else { inst.def };
                created.push(self.add_instance(ctx, Instance { def, xf, material: inst.material, glued: inst.glued, tag: inst.tag }));
            } else if let Some(i) = self.defs.get_mut(&ctx).and_then(|d| d.instances.get_mut(iid)) {
                i.xf = xf;
            }
        }
        created
    }

    pub fn remove_instances(&mut self, path: &[Id], insts: &[Id]) {
        let ctx = self.ctx_def(path);
        if let Some(d) = self.defs.get_mut(&ctx) {
            for i in insts {
                d.instances.remove(i);
            }
        }
        self.gc();
    }

    /// Gives an instance its own copy of its definition.
    pub fn make_unique(&mut self, path: &[Id], iid: Id) -> bool {
        let ctx = self.ctx_def(path);
        let Some(def) = self.instance(path, iid).map(|i| i.def) else { return false };
        let new_def = self.deep_copy_def(def);
        self.defs.get_mut(&ctx).unwrap().instances.get_mut(&iid).unwrap().def = new_def;
        true
    }

    /// Places a new instance of `def` inside the context with a world placement.
    pub fn place(&mut self, path: &[Id], def: DefId, world: Mat4) -> Option<Id> {
        let ctx = self.ctx_def(path);
        if self.contains_def(def, ctx) {
            return None;
        }
        let xf = self.ctx_xf(path).inverse() * world;
        let id = self.add_instance(ctx, Instance { def, xf, material: None, glued: None, tag: None });
        self.refresh_glue();
        Some(id)
    }

    /// Everything merged into a single world-space mesh (for export).
    pub fn flatten(&self) -> Model {
        let mut out = Model::new();
        self.flatten_into(ROOT, Mat4::IDENTITY, None, &mut out);
        out
    }

    /// A child instance of the context flattened into one world-space mesh.
    pub fn instance_world_mesh(&self, path: &[Id], iid: Id) -> Option<Model> {
        let inst = self.instance(path, iid)?;
        let mut out = Model::new();
        self.flatten_into(inst.def, self.ctx_xf(path) * inst.xf, inst.material, &mut out);
        Some(out)
    }

    fn flatten_into(&self, def: DefId, xf: Mat4, mat: Option<usize>, out: &mut Model) {
        let Some(d) = self.defs.get(&def) else { return };
        let mut g = d.mesh.transformed(xf);
        let cuts = self.child_openings(def, xf);
        let planes: Vec<(Vec3, Vec3, Vec<Vec<Vec3>>)> =
            cuts.into_iter().filter_map(|(f, loops)| g.faces.get(&f).map(|face| (face.normal, g.pos(face.outer[0]), loops))).collect();
        for (n, p0, loops) in planes {
            g.subtract_loops(n, p0, &loops);
        }
        if mat.is_some() {
            for f in g.faces.values_mut() {
                f.material = f.material.or(mat);
            }
        }
        out.merge_from(&g);
        for c in d.instances.values() {
            self.flatten_into(c.def, xf * c.xf, c.material.or(mat), out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1x1x1 box at (2, 0, 0) in the root context, returned as a working mesh.
    fn box_model() -> Model {
        let mut m = Model::new();
        let f = m
            .add_polygon(&[Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0)], None)
            .unwrap();
        m.push_pull(f, -1.0);
        m
    }

    fn all(m: &Model) -> (Vec<Id>, Vec<Id>) {
        (m.faces.keys().copied().collect(), m.edges.keys().copied().collect())
    }

    #[test]
    fn group_and_explode_roundtrip() {
        let mut doc = Document::new();
        let mut active = box_model();
        let (f, e) = all(&active);
        let inst = doc.make_group(&[], &mut active, &f, &e, &[], DefKind::Group).unwrap();
        doc.sync(&[], &active);
        assert!(active.faces.is_empty() && active.vertices.is_empty());
        let def = doc.instance(&[], inst).unwrap().def;
        assert_eq!(doc.defs[&def].mesh.faces.len(), 6);
        // Origin at the bounding-box corner.
        let (lo, _) = doc.local_bounds(def).unwrap();
        assert!(lo.length() < 1e-5);

        let (faces, _, _) = doc.explode(&[], &mut active, inst).unwrap();
        assert_eq!(faces.len(), 6);
        assert_eq!(doc.defs.len(), 1, "exploded group definition is garbage-collected");
        let (lo, hi) = active.bounds().unwrap();
        assert!((lo - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-5 && (hi - Vec3::new(3.0, 1.0, 1.0)).length() < 1e-5);
    }

    #[test]
    fn component_edits_are_shared() {
        let mut doc = Document::new();
        let mut active = box_model();
        let (f, e) = all(&active);
        let a = doc.make_group(&[], &mut active, &f, &e, &[], DefKind::Component).unwrap();
        doc.sync(&[], &active);
        let b = doc.transform_instances(&[], &[a], Mat4::from_translation(Vec3::X * 5.0), true)[0];
        let def = doc.instance(&[], a).unwrap().def;
        assert_eq!(doc.instance(&[], b).unwrap().def, def);
        assert_eq!(doc.instances_of(def), 2);

        // Edit inside instance b: pull its top face up.
        let path = [b];
        let mut ctx = doc.load(&path);
        let top = *ctx.faces.iter().find(|(_, f)| f.normal.z > 0.99).unwrap().0;
        assert!(ctx.bounds().unwrap().0.x > 6.9, "context mesh is in world space");
        ctx.push_pull(top, 1.0);
        doc.sync(&path, &ctx);
        let (_, hi) = doc.local_bounds(def).unwrap();
        assert!((hi.z - 2.0).abs() < 1e-4, "shared definition got taller");

        // Make unique detaches b.
        assert!(doc.make_unique(&[], b));
        assert_ne!(doc.instance(&[], b).unwrap().def, def);
        assert_eq!(doc.instances_of(def), 1);
    }

    #[test]
    fn context_load_sync_roundtrip() {
        let mut doc = Document::new();
        let mut active = box_model();
        let (f, e) = all(&active);
        let g = doc.make_group(&[], &mut active, &f, &e, &[], DefKind::Group).unwrap();
        doc.sync(&[], &active);
        doc.transform_instances(&[], &[g], Mat4::from_rotation_z(0.7) * Mat4::from_translation(Vec3::Y), false);
        let world = doc.load(&[g]);
        let before = doc.defs[&doc.ctx_def(&[g])].mesh.clone();
        doc.sync(&[g], &world);
        let after = &doc.defs[&doc.ctx_def(&[g])].mesh;
        for (id, p) in &before.vertices {
            assert!(p.distance(after.vertices[id]) < 1e-5);
        }
    }

    #[test]
    fn cannot_place_component_inside_itself() {
        let mut doc = Document::new();
        let mut active = box_model();
        let (f, e) = all(&active);
        let a = doc.make_group(&[], &mut active, &f, &e, &[], DefKind::Component).unwrap();
        let def = doc.instance(&[], a).unwrap().def;
        assert!(doc.place(&[a], def, Mat4::IDENTITY).is_none());
        assert!(doc.place(&[], def, Mat4::from_translation(Vec3::ONE)).is_some());
    }

    #[test]
    fn flatten_includes_instances() {
        let mut doc = Document::new();
        let mut active = box_model();
        let (f, e) = all(&active);
        let a = doc.make_group(&[], &mut active, &f, &e, &[], DefKind::Component).unwrap();
        doc.sync(&[], &active);
        doc.transform_instances(&[], &[a], Mat4::from_translation(Vec3::Y * 3.0), true);
        assert_eq!(doc.flatten().faces.len(), 12);
    }

    // ----- gluing & cutting openings ---------------------------------------------

    /// A 4 x 3 wall in the y = 0 plane facing -Y.
    fn wall() -> (Model, Id) {
        let mut m = Model::new();
        let w = m
            .add_polygon(&[Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 3.0)], Some(-Vec3::Y))
            .unwrap();
        (m, w)
    }

    /// Draws an x0..x1 / z0..z1 rectangle on the wall and returns its face.
    fn on_wall(m: &mut Model, x0: f32, x1: f32, z0: f32, z1: f32) -> Id {
        m.add_polygon(&[Vec3::new(x0, 0.0, z0), Vec3::new(x1, 0.0, z0), Vec3::new(x1, 0.0, z1), Vec3::new(x0, 0.0, z1)], Some(-Vec3::Y))
            .unwrap()
    }

    fn wall_faces(m: &Model) -> Vec<Id> {
        m.faces.iter().filter(|(_, f)| f.normal.y < -0.99 && m.pos(f.outer[0]).y.abs() < 1e-4).map(|(i, _)| *i).collect()
    }

    #[test]
    fn window_on_wall_becomes_cutting_component() {
        let mut doc = Document::new();
        let (mut active, _) = wall();
        let win = on_wall(&mut active, 1.0, 2.0, 1.0, 2.0);
        assert_eq!(active.faces.len(), 2);
        let inst = doc.make_group(&[], &mut active, &[win], &[], &[], DefKind::Component).unwrap();
        doc.sync(&[], &active);

        // The wall is healed; the opening is now cut dynamically.
        assert_eq!(active.faces.len(), 1);
        let w = *active.faces.keys().next().unwrap();
        assert!(active.faces[&w].holes.is_empty());
        assert!((active.face_area(w) - 12.0).abs() < 1e-3);
        let i = doc.instance(&[], inst).unwrap().clone();
        let def = &doc.defs[&i.def];
        assert!(def.glue && def.cut_opening);
        assert_eq!(i.glued, Some(w));
        assert!(i.xf.z_axis.truncate().dot(-Vec3::Y) > 0.999, "blue axis points out of the wall");
        assert!(i.xf.x_axis.truncate().dot(Vec3::X) > 0.999, "red axis runs along the wall");
        assert_eq!(doc.openings(i.def).len(), 1);

        // Rendering cuts the wall without touching it.
        let cut: f32 = doc
            .child_openings(ROOT, Mat4::IDENTITY)
            .get(&w)
            .map(|l| active.cut_triangles(w, l).iter().map(|t| (t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5).sum())
            .unwrap();
        assert!((cut - 11.0).abs() < 1e-3);
        // Export: the wall has a real hole and the window face fills it.
        let flat = doc.flatten();
        assert!(flat.faces.keys().any(|f| flat.faces[f].holes.len() == 1 && (flat.face_area(*f) - 11.0).abs() < 1e-3));

        // Sliding along the wall keeps it glued; pulling it off unglues it.
        doc.transform_instances(&[], &[inst], Mat4::from_translation(Vec3::X * 0.5), false);
        doc.refresh_glue();
        assert_eq!(doc.instance(&[], inst).unwrap().glued, Some(w));
        doc.transform_instances(&[], &[inst], Mat4::from_translation(-Vec3::Y), false);
        doc.refresh_glue();
        assert_eq!(doc.instance(&[], inst).unwrap().glued, None);
        doc.transform_instances(&[], &[inst], Mat4::from_translation(Vec3::Y), false);
        doc.refresh_glue();
        assert_eq!(doc.instance(&[], inst).unwrap().glued, Some(w));

        // Exploding makes the opening real.
        let (faces, _, _) = doc.explode(&[], &mut active, inst).unwrap();
        assert_eq!(faces.len(), 1);
        let walls = wall_faces(&active);
        assert_eq!(walls.len(), 2, "wall with a hole + the window face");
        assert!(walls.iter().any(|f| active.faces[f].holes.len() == 1 && (active.face_area(*f) - 11.0).abs() < 1e-3));
    }

    #[test]
    fn door_on_wall_edge_heals_and_cuts() {
        let mut doc = Document::new();
        let (mut active, _) = wall();
        let door = on_wall(&mut active, 1.5, 2.5, 0.0, 2.2);
        assert_eq!(active.faces.len(), 2, "U-shaped wall + door");
        let inst = doc.make_group(&[], &mut active, &[door], &[], &[], DefKind::Component).unwrap();
        doc.sync(&[], &active);
        assert_eq!(active.faces.len(), 1);
        let w = *active.faces.keys().next().unwrap();
        assert!((active.face_area(w) - 12.0).abs() < 1e-3);
        assert_eq!(active.faces[&w].outer.len(), 4, "split points on the bottom edge are gone");
        assert_eq!(doc.instance(&[], inst).unwrap().glued, Some(w));

        let flat = doc.flatten();
        let walls = wall_faces(&flat);
        assert!(walls.iter().any(|f| (flat.face_area(*f) - (12.0 - 2.2)).abs() < 1e-3), "wall has a door-shaped notch");
        assert!(flat.vertices.values().all(|p| p.z > -1e-4));
    }

    #[test]
    fn recessed_window_glues_with_outward_axis() {
        let mut doc = Document::new();
        let (mut active, wall_id) = wall();
        let win = on_wall(&mut active, 1.0, 2.0, 1.0, 2.0);
        active.push_pull(win, -0.2); // into the wall
        let recess: Vec<Id> = active.faces.keys().copied().filter(|f| *f != wall_id).collect();
        assert_eq!(recess.len(), 5, "back + 4 sides");
        let inst = doc.make_group(&[], &mut active, &recess, &[], &[], DefKind::Component).unwrap();
        doc.sync(&[], &active);
        assert_eq!(active.faces.len(), 1);
        assert!(active.faces.values().all(|f| f.holes.is_empty()));
        let i = doc.instance(&[], inst).unwrap();
        assert!(i.glued.is_some());
        assert!(i.xf.z_axis.truncate().dot(-Vec3::Y) > 0.999);
        let (lo, hi) = doc.local_bounds(i.def).unwrap();
        assert!(lo.z < -0.19 && hi.z.abs() < 1e-4, "geometry goes into the wall (negative blue)");
        let flat = doc.flatten();
        assert!(flat.faces.keys().any(|f| flat.faces[f].holes.len() == 1 && (flat.face_area(*f) - 11.0).abs() < 1e-3));
    }

    #[test]
    fn box_on_ground_is_not_glued() {
        let mut doc = Document::new();
        let mut active = Model::new();
        active.add_polygon(&[Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), Vec3::new(5.0, 5.0, 0.0), Vec3::new(0.0, 5.0, 0.0)], None);
        let f = active
            .add_polygon(&[Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0), Vec3::new(2.0, 2.0, 0.0), Vec3::new(1.0, 2.0, 0.0)], None)
            .unwrap();
        active.push_pull(f, -1.0); // ground faces point down: negative pulls up
        let ground = *active.faces.iter().max_by(|a, b| active.face_area(*a.0).total_cmp(&active.face_area(*b.0))).unwrap().0;
        let boxf: Vec<Id> = active.faces.keys().copied().filter(|x| *x != ground).collect();
        let inst = doc.make_group(&[], &mut active, &boxf, &[], &[], DefKind::Component).unwrap();
        doc.sync(&[], &active);
        assert!(!doc.defs[&doc.instance(&[], inst).unwrap().def].glue);
    }

    #[test]
    fn placing_a_gluing_component_on_a_wall() {
        let mut doc = Document::new();
        let (mut active, _) = wall();
        let win = on_wall(&mut active, 1.0, 2.0, 1.0, 2.0);
        let a = doc.make_group(&[], &mut active, &[win], &[], &[], DefKind::Component).unwrap();
        doc.sync(&[], &active);
        let def = doc.instance(&[], a).unwrap().def;
        let w = *active.faces.keys().next().unwrap();
        let xf = glue_frame(-Vec3::Y, Vec3::new(2.5, 0.0, 0.5), Vec3::Z, Vec3::X);
        let b = doc.place(&[], def, xf).unwrap();
        assert_eq!(doc.instance(&[], b).unwrap().glued, Some(w));
        assert_eq!(doc.child_openings(ROOT, Mat4::IDENTITY)[&w].len(), 2);
    }

    // ----- tags & scenes ---------------------------------------------------------

    #[test]
    fn tags_hide_and_delete() {
        let mut doc = Document::new();
        let mut active = box_model();
        let (f, e) = all(&active);
        let a = doc.make_group(&[], &mut active, &f, &e, &[], DefKind::Component).unwrap();
        let b = doc.transform_instances(&[], &[a], Mat4::from_translation(Vec3::X * 3.0), true)[0];
        let t = doc.add_tag();
        assert_eq!(doc.tags[&t].name, "Tag 1");
        let t2 = doc.add_tag();
        assert_eq!(doc.tags[&t2].name, "Tag 2");
        doc.defs.get_mut(&ROOT).unwrap().instances.get_mut(&a).unwrap().tag = Some(t);
        assert_eq!(doc.tag_usage(t), 1);
        assert!(doc.tag_visible(Some(t)) && doc.tag_visible(None));
        doc.tags.get_mut(&t).unwrap().visible = false;
        assert!(!doc.tag_visible(doc.instance(&[], a).unwrap().tag));
        assert!(doc.tag_visible(doc.instance(&[], b).unwrap().tag), "untagged stays visible");
        // A copy keeps its tag.
        let c = doc.transform_instances(&[], &[a], Mat4::from_translation(Vec3::Y * 3.0), true)[0];
        assert_eq!(doc.instance(&[], c).unwrap().tag, Some(t));

        doc.scenes.push(Scene {
            name: doc.next_scene_name(),
            camera: Camera::default(),
            hidden_tags: doc.hidden_tags(),
            style: SceneStyle::default(),
        });
        assert_eq!(doc.scenes[0].hidden_tags, vec![t]);
        doc.delete_tag(t);
        assert_eq!(doc.instance(&[], a).unwrap().tag, None);
        assert!(doc.scenes[0].hidden_tags.is_empty());
        assert_eq!(doc.next_scene_name(), "Scene 2");
    }

    #[test]
    fn tags_and_scenes_are_saved() {
        let mut doc = Document::new();
        let t = doc.add_tag();
        doc.tags.get_mut(&t).unwrap().visible = false;
        let cam = Camera { distance: 7.5, yaw: 1.0, ..Camera::default() };
        let style = SceneStyle { xray: true, axes: true, ..SceneStyle::default() };
        doc.scenes.push(Scene { name: "Front".into(), camera: cam, hidden_tags: vec![t], style });
        let back: Document = serde_json::from_str(&serde_json::to_string(&doc).unwrap()).unwrap();
        assert_eq!(back.tags, doc.tags);
        assert_eq!(back.scenes, doc.scenes);
        let plain = serde_json::to_string(&Document::new()).unwrap();
        assert!(!plain.contains("tags") && !plain.contains("scenes"));
    }
}
