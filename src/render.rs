//! OpenGL renderer drawn inside the egui viewport through a paint callback.
//!
//! Faces are drawn untextured in one batch, plus one batch per textured
//! material. With shadows on, a depth pass renders the scene from the sun into
//! a shadow map first; the main pass then lights faces with the sun and darkens
//! what the map says is in shadow. A transparent ground plane catches shadows
//! where there is no geometry, like SketchUp's ground shadows.

use crate::texture::Pixels;
use eframe::glow::{self, HasContext};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Floats per vertex: position (3) + normal (3) + rgba (4) + uv (2) + back-face rgb (3).
pub const FLOATS: usize = 15;
const SHADOW_SIZE: i32 = 2048;

pub struct TexBatch {
    /// Identifies the image (see `Texture::key`).
    pub key: u64,
    pub pixels: Arc<Pixels>,
    pub tris: Vec<f32>,
}

pub struct ShadowData {
    pub light_vp: Mat4,
    /// Ground shadow-catcher triangles.
    pub ground: Vec<f32>,
}

pub struct DrawData {
    pub tris: Vec<f32>,
    pub textured: Vec<TexBatch>,
    pub lines: Vec<f32>,
    pub view_proj: Mat4,
    /// Direction towards the light (the sun when shadows are on).
    pub light: Vec3,
    pub xray: bool,
    pub shadow: Option<ShadowData>,
}

pub struct Renderer {
    program: glow::Program,
    depth_program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    textures: HashMap<u64, glow::Texture>,
    shadow_map: Option<(glow::Framebuffer, glow::Texture)>,
}

const VS: &str = r#"
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec2 a_uv;
layout(location = 4) in vec3 a_back;
uniform mat4 u_vp;
uniform mat4 u_light_vp;
out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;
out vec4 v_light;
out vec3 v_back;
void main() {
    gl_Position = u_vp * vec4(a_pos, 1.0);
    v_back = a_back;
    v_normal = a_normal;
    v_color = a_color;
    v_uv = a_uv;
    v_light = u_light_vp * vec4(a_pos, 1.0);
}
"#;

const FS: &str = r#"
precision highp float;
in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;
in vec4 v_light;
in vec3 v_back;
uniform int u_mode;       // 0 = flat (lines), 1 = lit faces, 2 = ground shadow catcher
uniform vec3 u_light;
uniform int u_textured;
uniform sampler2D u_tex;
uniform int u_shadows;
uniform sampler2D u_shadow_map;
out vec4 out_color;

float lit_amount(vec3 n) {
    if (u_shadows == 0) return 1.0;
    vec3 p = v_light.xyz / v_light.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float ndl = abs(dot(n, normalize(u_light)));
    float bias = 0.0008 + 0.003 * (1.0 - ndl);
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float sum = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float d = texture(u_shadow_map, p.xy + vec2(float(x), float(y)) * texel).r;
            sum += (p.z - bias > d) ? 0.0 : 1.0;
        }
    }
    return sum / 9.0;
}

void main() {
    if (u_mode == 0) { out_color = v_color; return; }
    if (u_mode == 2) {
        float l = lit_amount(vec3(0.0, 0.0, 1.0));
        out_color = vec4(0.05, 0.06, 0.12, (1.0 - l) * 0.42);
        return;
    }
    vec3 n = normalize(v_normal);
    // Painted faces show their material on both sides; unpainted ones get
    // SketchUp's blue-gray back color (chosen per vertex by the app).
    vec3 col = gl_FrontFacing ? v_color.rgb : v_back;
    if (!gl_FrontFacing) n = -n;
    if (u_textured == 1) col *= texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
    float diff = max(dot(n, normalize(u_light)), 0.0);
    if (u_shadows == 1) {
        float l = diff > 0.0 ? lit_amount(n) : 0.0;
        out_color = vec4(col * (0.52 + 0.55 * diff * l), v_color.a);
    } else {
        out_color = vec4(col * (0.58 + 0.42 * diff), v_color.a);
    }
}
"#;

const DEPTH_VS: &str = r#"
layout(location = 0) in vec3 a_pos;
uniform mat4 u_light_vp;
void main() { gl_Position = u_light_vp * vec4(a_pos, 1.0); }
"#;

const DEPTH_FS: &str = r#"
precision highp float;
void main() {}
"#;

fn as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

unsafe fn compile(gl: &glow::Context, vs: &str, fs: &str) -> glow::Program {
    let header = if cfg!(target_arch = "wasm32") { "#version 300 es\n" } else { "#version 330\n" };
    let program = gl.create_program().expect("cannot create program");
    let mut shaders = Vec::new();
    for (ty, src) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
        let s = gl.create_shader(ty).expect("cannot create shader");
        gl.shader_source(s, &format!("{header}{src}"));
        gl.compile_shader(s);
        assert!(gl.get_shader_compile_status(s), "shader error: {}", gl.get_shader_info_log(s));
        gl.attach_shader(program, s);
        shaders.push(s);
    }
    gl.link_program(program);
    assert!(gl.get_program_link_status(program), "link error: {}", gl.get_program_info_log(program));
    for s in shaders {
        gl.detach_shader(program, s);
        gl.delete_shader(s);
    }
    program
}

impl Renderer {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile(gl, VS, FS);
            let depth_program = compile(gl, DEPTH_VS, DEPTH_FS);
            let vao = gl.create_vertex_array().expect("cannot create VAO");
            let vbo = gl.create_buffer().expect("cannot create VBO");
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let stride = (FLOATS * 4) as i32;
            for (i, (n, off)) in [(3, 0), (3, 12), (4, 24), (2, 40), (3, 48)].into_iter().enumerate() {
                gl.enable_vertex_attrib_array(i as u32);
                gl.vertex_attrib_pointer_f32(i as u32, n, glow::FLOAT, false, stride, off);
            }
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            Self { program, depth_program, vao, vbo, textures: HashMap::new(), shadow_map: None }
        }
    }

    unsafe fn texture(&mut self, gl: &glow::Context, key: u64, px: &Pixels) -> glow::Texture {
        if let Some(t) = self.textures.get(&key) {
            return *t;
        }
        let t = gl.create_texture().expect("cannot create texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(t));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            px.w as i32,
            px.h as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(&px.rgba)),
        );
        gl.generate_mipmap(glow::TEXTURE_2D);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
        self.textures.insert(key, t);
        t
    }

    unsafe fn shadow_target(&mut self, gl: &glow::Context) -> (glow::Framebuffer, glow::Texture) {
        if let Some(s) = self.shadow_map {
            return s;
        }
        let tex = gl.create_texture().expect("cannot create shadow texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::DEPTH_COMPONENT24 as i32,
            SHADOW_SIZE,
            SHADOW_SIZE,
            0,
            glow::DEPTH_COMPONENT,
            glow::UNSIGNED_INT,
            glow::PixelUnpackData::Slice(None),
        );
        for (p, v) in [
            (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
            (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, p, v as i32);
        }
        let fbo = gl.create_framebuffer().expect("cannot create framebuffer");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::TEXTURE_2D, Some(tex), 0);
        gl.draw_buffer(glow::NONE);
        gl.read_buffer(glow::NONE);
        self.shadow_map = Some((fbo, tex));
        (fbo, tex)
    }

    unsafe fn draw(gl: &glow::Context, mode: u32, data: &[f32]) {
        if !data.is_empty() {
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, as_bytes(data), glow::DYNAMIC_DRAW);
            gl.draw_arrays(mode, 0, (data.len() / FLOATS) as i32);
        }
    }

    /// `viewport` is the callback's viewport (x, y from bottom, w, h) and
    /// `target` the framebuffer egui expects us to draw into.
    pub fn paint(&mut self, gl: &glow::Context, d: &DrawData, viewport: [i32; 4], target: Option<glow::Framebuffer>) {
        unsafe {
            gl.bind_vertex_array(Some(self.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.disable(glow::CULL_FACE);

            // --- Shadow map: the scene's depth seen from the sun. ---
            let shadow_tex = if let Some(sh) = &d.shadow {
                let (fbo, tex) = self.shadow_target(gl);
                let scissor = gl.is_enabled(glow::SCISSOR_TEST);
                gl.disable(glow::SCISSOR_TEST);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
                gl.enable(glow::DEPTH_TEST);
                gl.depth_func(glow::LEQUAL);
                gl.depth_mask(true);
                gl.clear_depth_f32(1.0);
                gl.clear(glow::DEPTH_BUFFER_BIT);
                gl.disable(glow::BLEND);
                gl.enable(glow::POLYGON_OFFSET_FILL);
                gl.polygon_offset(1.5, 3.0);
                gl.use_program(Some(self.depth_program));
                let loc = gl.get_uniform_location(self.depth_program, "u_light_vp");
                gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, &sh.light_vp.to_cols_array());
                Self::draw(gl, glow::TRIANGLES, &d.tris);
                for b in &d.textured {
                    Self::draw(gl, glow::TRIANGLES, &b.tris);
                }
                gl.disable(glow::POLYGON_OFFSET_FILL);
                gl.bind_framebuffer(glow::FRAMEBUFFER, target);
                gl.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                if scissor {
                    gl.enable(glow::SCISSOR_TEST);
                }
                Some(tex)
            } else {
                None
            };

            // --- Main pass. ---
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            let p = self.program;
            gl.use_program(Some(p));
            let loc = |n: &str| gl.get_uniform_location(p, n);
            gl.uniform_matrix_4_f32_slice(loc("u_vp").as_ref(), false, &d.view_proj.to_cols_array());
            gl.uniform_3_f32(loc("u_light").as_ref(), d.light.x, d.light.y, d.light.z);
            gl.uniform_1_i32(loc("u_tex").as_ref(), 0);
            gl.uniform_1_i32(loc("u_shadow_map").as_ref(), 1);
            gl.uniform_1_i32(loc("u_textured").as_ref(), 0);
            match (&d.shadow, shadow_tex) {
                (Some(sh), Some(tex)) => {
                    gl.uniform_1_i32(loc("u_shadows").as_ref(), 1);
                    gl.uniform_matrix_4_f32_slice(loc("u_light_vp").as_ref(), false, &sh.light_vp.to_cols_array());
                    gl.active_texture(glow::TEXTURE1);
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                    gl.active_texture(glow::TEXTURE0);
                    // Ground shadows: blended onto the background, no depth write.
                    gl.uniform_1_i32(loc("u_mode").as_ref(), 2);
                    gl.depth_mask(false);
                    Self::draw(gl, glow::TRIANGLES, &sh.ground);
                    gl.depth_mask(true);
                }
                _ => {
                    gl.uniform_1_i32(loc("u_shadows").as_ref(), 0);
                    gl.uniform_matrix_4_f32_slice(loc("u_light_vp").as_ref(), false, &Mat4::IDENTITY.to_cols_array());
                }
            }

            // Faces (pushed back slightly so edges draw on top).
            gl.uniform_1_i32(loc("u_mode").as_ref(), 1);
            gl.enable(glow::POLYGON_OFFSET_FILL);
            gl.polygon_offset(1.0, 1.0);
            if d.xray {
                gl.depth_mask(false);
            }
            Self::draw(gl, glow::TRIANGLES, &d.tris);
            if !d.textured.is_empty() {
                gl.uniform_1_i32(loc("u_textured").as_ref(), 1);
                for b in &d.textured {
                    let t = self.texture(gl, b.key, &b.pixels);
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(t));
                    Self::draw(gl, glow::TRIANGLES, &b.tris);
                }
                gl.uniform_1_i32(loc("u_textured").as_ref(), 0);
            }
            gl.disable(glow::POLYGON_OFFSET_FILL);
            gl.depth_mask(true);

            // Edges and axes.
            gl.uniform_1_i32(loc("u_mode").as_ref(), 0);
            Self::draw(gl, glow::LINES, &d.lines);

            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
            gl.disable(glow::DEPTH_TEST);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_program(self.depth_program);
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.vbo);
            for t in self.textures.values() {
                gl.delete_texture(*t);
            }
            if let Some((fbo, tex)) = self.shadow_map {
                gl.delete_framebuffer(fbo);
                gl.delete_texture(tex);
            }
        }
    }
}

/// A face vertex: front color `c`, back-face color `back`, texture coords `uv`.
pub fn push_face_vertex(buf: &mut Vec<f32>, p: Vec3, n: Vec3, c: [f32; 4], back: [f32; 3], uv: [f32; 2]) {
    buf.extend_from_slice(&[p.x, p.y, p.z, n.x, n.y, n.z, c[0], c[1], c[2], c[3], uv[0], uv[1], back[0], back[1], back[2]]);
}

pub fn push_vertex(buf: &mut Vec<f32>, p: Vec3, n: Vec3, c: [f32; 4]) {
    push_face_vertex(buf, p, n, c, [c[0], c[1], c[2]], [0.0, 0.0]);
}

pub fn push_line(buf: &mut Vec<f32>, a: Vec3, b: Vec3, c: [f32; 4]) {
    push_vertex(buf, a, Vec3::Z, c);
    push_vertex(buf, b, Vec3::Z, c);
}
