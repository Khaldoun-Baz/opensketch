//! AI Render: sends a picture of the current view plus a description to an
//! image model and gets a photorealistic image back.
//!
//! OpenSketch ships no AI account. The user picks a service, pastes their own
//! API key (or points at a Stable Diffusion running on their own computer) and
//! can change the address, the model name and any extra option, so new models
//! and compatible services work without a new release.
//!
//! The module is split so that everything except the network call is a pure
//! function with unit tests: [`prepare`] fits the capture to what the services
//! accept, [`compose_prompt`] writes the prompt, [`build`] makes the HTTP
//! request and [`parse`] reads the answer. [`render`] strings them together.

use crate::texture::{base64_decode, base64_encode};
use image::RgbaImage;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Services
// ---------------------------------------------------------------------------

/// The request/response dialect to speak. Each one can be pointed at any
/// address that speaks the same dialect.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub enum Kind {
    /// `POST {base}/images/edits` (multipart), as served by OpenAI and compatible services.
    OpenAi,
    /// `POST {base}/interactions` (JSON), Google's Gemini image models.
    Gemini,
    /// `POST {base}/v2beta/stable-image/{model}` (multipart), Stability AI.
    Stability,
    /// `POST {base}/sdapi/v1/img2img` (JSON): AUTOMATIC1111 / Forge on your own computer.
    Automatic1111,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::OpenAi, Kind::Gemini, Kind::Stability, Kind::Automatic1111];

    pub fn label(self) -> &'static str {
        match self {
            Kind::OpenAi => "OpenAI (or compatible)",
            Kind::Gemini => "Google Gemini",
            Kind::Stability => "Stability AI",
            Kind::Automatic1111 => "Stable Diffusion on this PC",
        }
    }

    /// Online services need a key; a local Stable Diffusion usually doesn't.
    pub fn needs_key(self) -> bool {
        self != Kind::Automatic1111
    }

    /// Where to get a key (or the software), shown as a link.
    pub fn key_page(self) -> (&'static str, &'static str) {
        match self {
            Kind::OpenAi => ("Get an OpenAI API key", "https://platform.openai.com/api-keys"),
            Kind::Gemini => ("Get a Gemini API key", "https://aistudio.google.com/apikey"),
            Kind::Stability => ("Get a Stability AI key", "https://platform.stability.ai/account/keys"),
            Kind::Automatic1111 => {
                ("Stable Diffusion web UI (start it with --api)", "https://github.com/AUTOMATIC1111/stable-diffusion-webui")
            }
        }
    }

    /// What the "Model" box means for this service.
    pub fn model_hint(self) -> &'static str {
        match self {
            Kind::OpenAi => "Model name, e.g. gpt-image-2",
            Kind::Gemini => "Model name, e.g. gemini-3-pro-image",
            Kind::Stability => "Endpoint: control/structure or control/sketch",
            Kind::Automatic1111 => "Checkpoint name (empty = the one loaded)",
        }
    }

    /// Instruction-following models get told what to do with the picture;
    /// diffusion models get a description of the picture they should make.
    fn instructed(self) -> bool {
        matches!(self, Kind::OpenAi | Kind::Gemini)
    }

    /// Widest (and tallest) picture the service accepts, as long side / short side.
    fn max_aspect(self) -> f32 {
        match self {
            Kind::OpenAi => 2.9,
            Kind::Stability => 2.4,
            Kind::Gemini | Kind::Automatic1111 => 4.0,
        }
    }

    pub fn default_profile(self) -> Profile {
        let (base_url, model, extra) = match self {
            Kind::OpenAi => ("https://api.openai.com/v1", "gpt-image-2", "quality=high"),
            Kind::Gemini => ("https://generativelanguage.googleapis.com/v1beta", "gemini-3-pro-image", ""),
            Kind::Stability => ("https://api.stability.ai", "control/structure", "style_preset=photographic"),
            Kind::Automatic1111 => ("http://127.0.0.1:7860", "", "steps=30\ncfg_scale=7"),
        };
        Profile { base_url: base_url.into(), model: model.into(), api_key: String::new(), extra: extra.into() }
    }
}

/// How to reach one service. Everything is editable in the AI Render window.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Profile {
    pub base_url: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub api_key: String,
    /// Extra request options, one `name=value` per line. They are added to the
    /// request and replace OpenSketch's own value of the same name.
    #[serde(default)]
    pub extra: String,
}

// ---------------------------------------------------------------------------
// Styles and the prompt
// ---------------------------------------------------------------------------

pub struct Style {
    pub name: &'static str,
    /// Completes "Turn this picture into …" for instruction-following models.
    instruction: &'static str,
    /// Comma-separated look for diffusion models.
    keywords: &'static str,
    /// Realistic styles ask for real surroundings and get the default "avoid" list.
    pub realistic: bool,
}

const CAMERA: &str = "photorealistic, ultra-detailed real-world materials with fine surface texture and natural imperfections, \
physically accurate light and soft shadows, global illumination, subtle reflections, high dynamic range, sharp focus, \
professional architectural photography, full-frame camera, 35mm lens, 8k";

pub const STYLES: &[Style] = &[
    Style {
        name: "Hyper-realistic photo",
        instruction: "a hyper-realistic photograph that is indistinguishable from a real photo taken in clear natural daylight",
        keywords: "hyper-realistic photograph, clear natural daylight",
        realistic: true,
    },
    Style {
        name: "Golden hour",
        instruction: "a hyper-realistic photograph taken at golden hour, with warm low sunlight and long soft shadows",
        keywords: "hyper-realistic photograph, golden hour, warm low sunlight, long soft shadows",
        realistic: true,
    },
    Style {
        name: "Overcast day",
        instruction: "a hyper-realistic photograph taken on an overcast day, with soft even light and gentle shadows",
        keywords: "hyper-realistic photograph, overcast sky, soft diffuse light",
        realistic: true,
    },
    Style {
        name: "Dusk with lights on",
        instruction: "a hyper-realistic photograph taken at dusk, with a deep blue sky and warm light glowing from the windows and lamps",
        keywords: "hyper-realistic photograph, blue hour dusk, warm interior lights glowing through windows",
        realistic: true,
    },
    Style {
        name: "Interior photo",
        instruction:
            "a hyper-realistic interior photograph, with daylight coming through the windows and realistic furniture-grade finishes",
        keywords: "hyper-realistic interior photograph, daylight through windows, interior design magazine",
        realistic: true,
    },
    Style {
        name: "Aerial photo",
        instruction: "a hyper-realistic aerial drone photograph in clear daylight",
        keywords: "hyper-realistic aerial drone photograph, clear daylight",
        realistic: true,
    },
    Style {
        name: "Watercolor",
        instruction: "a loose architectural watercolor painting on textured paper",
        keywords: "architectural watercolor painting, loose washes, textured paper",
        realistic: false,
    },
    Style {
        name: "Pencil sketch",
        instruction: "a hand-drawn architectural pencil sketch with light shading",
        keywords: "architectural pencil sketch, hand drawn, light hatching",
        realistic: false,
    },
    Style {
        name: "White clay model",
        instruction: "a photograph of a matte white architectural scale model under soft studio light",
        keywords: "matte white architectural scale model, soft studio light, photograph",
        realistic: false,
    },
];

/// Things realistic renders should stay away from.
pub const DEFAULT_AVOID: &str = "cartoon, illustration, painting, 3D render, CGI, video game graphics, plastic look, flat shading, \
black outlines, sketch lines, blurry, low quality, distorted geometry, extra buildings, people, text, watermark";

pub fn style(name: &str) -> &'static Style {
    STYLES.iter().find(|s| s.name == name).unwrap_or(&STYLES[0])
}

/// Writes the prompt. `fidelity` (0–1) is how strictly the model's shape must
/// be kept; `materials` are the names of the materials used in the model.
pub fn compose_prompt(kind: Kind, style: &Style, description: &str, materials: &[String], fidelity: f32, avoid: &str) -> String {
    let description = description.trim().trim_end_matches('.');
    let mats = materials.join(", ");
    if !kind.instructed() {
        // Diffusion models: describe the picture. The "avoid" list travels separately.
        let mut parts: Vec<String> = Vec::new();
        if !description.is_empty() {
            parts.push(description.to_string());
        }
        if !mats.is_empty() {
            parts.push(format!("materials: {mats}"));
        }
        parts.push(style.keywords.to_string());
        if style.realistic {
            parts.push(CAMERA.to_string());
        }
        return parts.join(", ");
    }
    let mut p = format!("The attached picture is a screenshot of a 3D model in a modeling program. Turn it into {}.", style.instruction);
    if !description.is_empty() {
        p += &format!(" The scene: {description}.");
    }
    if !mats.is_empty() {
        p += &format!(" Materials used in the model: {mats}; render each one as its real-world counterpart.");
    }
    p += if fidelity >= 0.66 {
        " Keep exactly the same geometry, proportions, openings, layout and camera angle as the picture: do not add, remove, move or reshape any part of the model."
    } else if fidelity >= 0.33 {
        " Keep the same overall shapes, layout and camera angle as the picture; small realistic details may be added."
    } else {
        " Use the picture as a loose guide for the layout and camera angle; realistic detail and surroundings may be added freely."
    };
    if style.realistic {
        p += " Replace the flat colors, repeating textures and drawn edge lines with real materials, and replace the plain background with a real sky and fitting, believable surroundings.";
        p += &format!(" Look: {CAMERA}.");
    }
    let avoid = avoid.trim().trim_end_matches('.');
    if !avoid.is_empty() {
        p += &format!(" Avoid: {avoid}.");
    }
    p
}

// ---------------------------------------------------------------------------
// The picture that is sent
// ---------------------------------------------------------------------------

/// Fits a viewport capture to what the services accept: the aspect ratio is
/// limited by cropping around the center, the long side is scaled down to
/// `max_side`, both sides become multiples of 16 and the picture is opaque.
pub fn prepare(img: &RgbaImage, max_side: u32, kind: Kind) -> RgbaImage {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let lim = kind.max_aspect();
    let (mut cw, mut ch) = (w, h);
    if cw as f32 > ch as f32 * lim {
        cw = (ch as f32 * lim) as u32;
    }
    if ch as f32 > cw as f32 * lim {
        ch = (cw as f32 * lim) as u32;
    }
    let scale = (max_side.max(64) as f32 / cw.max(ch) as f32).min(1.0);
    let round = |v: u32| ((v as f32 * scale) as u32 / 16 * 16).max(64);
    let (mut tw, mut th) = (round(cw), round(ch));
    // Rounding must not push the shape back over the limit.
    if tw as f32 > th as f32 * lim {
        tw = ((th as f32 * lim) as u32 / 16 * 16).max(64);
    }
    if th as f32 > tw as f32 * lim {
        th = ((tw as f32 * lim) as u32 / 16 * 16).max(64);
    }
    // Crop to the target's aspect ratio before scaling, so nothing is stretched.
    let aspect = tw as f32 / th as f32;
    if cw as f32 > ch as f32 * aspect {
        cw = ((ch as f32 * aspect) as u32).clamp(1, w);
    } else {
        ch = ((cw as f32 / aspect) as u32).clamp(1, h);
    }
    let crop = image::imageops::crop_imm(img, (w - cw) / 2, (h - ch) / 2, cw, ch).to_image();
    let mut out = if (cw, ch) == (tw, th) { crop } else { image::imageops::resize(&crop, tw, th, image::imageops::FilterType::Triangle) };
    for px in out.pixels_mut() {
        px[3] = 255;
    }
    out
}

pub fn png_bytes(img: &RgbaImage) -> Result<Vec<u8>, String> {
    let rgb = image::DynamicImage::ImageRgba8(img.clone()).to_rgb8();
    let mut out = std::io::Cursor::new(Vec::new());
    rgb.write_to(&mut out, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(out.into_inner())
}

// ---------------------------------------------------------------------------
// Building the request
// ---------------------------------------------------------------------------

/// What gets rendered: the prepared picture and the words that go with it.
pub struct Job {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub prompt: String,
    /// The "avoid" list, for services that take it separately.
    pub negative: String,
    pub fidelity: f32,
}

#[derive(Debug)]
pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// `name=value` lines; blank lines and lines starting with `#` are skipped.
pub fn parse_extra(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .filter(|(k, _)| !k.is_empty())
        .collect()
}

fn set_field(fields: &mut Vec<(String, String)>, name: &str, value: String) {
    match fields.iter_mut().find(|f| f.0 == name) {
        Some(f) => f.1 = value,
        None => fields.push((name.to_string(), value)),
    }
}

/// Extra options for JSON requests: values that read as JSON (numbers, true,
/// objects…) are sent as such, everything else as text.
fn merge_extra(body: &mut Value, extra: &str) {
    if let Some(obj) = body.as_object_mut() {
        for (k, v) in parse_extra(extra) {
            obj.insert(k, serde_json::from_str(&v).unwrap_or(Value::String(v)));
        }
    }
}

fn multipart(fields: &[(String, String)], file_field: &str, png: &[u8]) -> (String, Vec<u8>) {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let boundary = format!("----OpenSketch{nanos:x}");
    let mut body = Vec::with_capacity(png.len() + 1024);
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
    }
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"view.png\"\r\nContent-Type: image/png\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(png);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// True for addresses on this computer or the local network.
fn is_local(url: &str) -> bool {
    let rest = url.split_once("://").map(|x| x.1).unwrap_or(url);
    let host = rest.split(['/', '?']).next().unwrap_or("");
    let host =
        if let Some(v6) = host.strip_prefix('[') { v6.split(']').next().unwrap_or("") } else { host.split(':').next().unwrap_or("") };
    host == "localhost"
        || host == "::1"
        || host.ends_with(".local")
        || host.ends_with(".localhost")
        || host.starts_with("127.")
        || host.starts_with("10.")
        || host.starts_with("192.168.")
}

/// The host a request goes to, for the privacy note in the window.
pub fn host(url: &str) -> String {
    let rest = url.trim().split_once("://").map(|x| x.1).unwrap_or(url.trim());
    rest.split(['/', '?']).next().unwrap_or("").to_string()
}

pub fn build(kind: Kind, profile: &Profile, job: &Job) -> Result<HttpRequest, String> {
    let base = profile.base_url.trim().trim_end_matches('/');
    let key = profile.api_key.trim();
    let model = profile.model.trim();
    if !(base.starts_with("https://") || base.starts_with("http://")) {
        return Err("The address must start with https:// (or http:// for a service on your own computer).".into());
    }
    if kind.needs_key() && key.is_empty() {
        return Err("Enter your API key first.".into());
    }
    if !key.is_empty() && base.starts_with("http://") && !is_local(base) {
        return Err("Use an https:// address, so your API key isn't sent unprotected.".into());
    }
    let mut headers: Vec<(String, String)> = Vec::new();
    let (url, content_type, body) = match kind {
        Kind::OpenAi => {
            headers.push(("Authorization".into(), format!("Bearer {key}")));
            let mut fields = vec![("model".to_string(), model.to_string()), ("prompt".to_string(), job.prompt.clone())];
            // The gpt-image-2 family accepts any size, so the result can match the view.
            if model.starts_with("gpt-image-2") {
                fields.push(("size".into(), format!("{}x{}", job.width, job.height)));
            }
            for (k, v) in parse_extra(&profile.extra) {
                set_field(&mut fields, &k, v);
            }
            let (ct, body) = multipart(&fields, "image[]", &job.png);
            (format!("{base}/images/edits"), ct, body)
        }
        Kind::Gemini => {
            headers.push(("x-goog-api-key".into(), key.to_string()));
            headers.push(("Api-Revision".into(), "2026-05-20".into()));
            let mut body = json!({
                "model": model,
                "input": [
                    { "type": "text", "text": job.prompt },
                    { "type": "image", "mime_type": "image/png", "data": base64_encode(&job.png) },
                ],
            });
            merge_extra(&mut body, &profile.extra);
            (format!("{base}/interactions"), "application/json".to_string(), body.to_string().into_bytes())
        }
        Kind::Stability => {
            headers.push(("Authorization".into(), format!("Bearer {key}")));
            headers.push(("Accept".into(), "image/*".into()));
            let mut fields = vec![
                ("prompt".to_string(), job.prompt.clone()),
                ("control_strength".to_string(), format!("{:.2}", job.fidelity.clamp(0.0, 1.0))),
                ("output_format".to_string(), "png".to_string()),
            ];
            if !job.negative.trim().is_empty() {
                fields.push(("negative_prompt".into(), job.negative.trim().to_string()));
            }
            for (k, v) in parse_extra(&profile.extra) {
                set_field(&mut fields, &k, v);
            }
            let (ct, body) = multipart(&fields, "image", &job.png);
            let endpoint = if model.is_empty() { "control/structure" } else { model.trim_matches('/') };
            (format!("{base}/v2beta/stable-image/{endpoint}"), ct, body)
        }
        Kind::Automatic1111 => {
            if !key.is_empty() {
                // `--api-auth user:password` uses Basic authentication.
                let auth = if key.contains(':') { format!("Basic {}", base64_encode(key.as_bytes())) } else { format!("Bearer {key}") };
                headers.push(("Authorization".into(), auth));
            }
            let mut body = json!({
                "init_images": [base64_encode(&job.png)],
                "prompt": job.prompt,
                "negative_prompt": job.negative.trim(),
                // Strictly keeping the shape means changing the picture less.
                "denoising_strength": 0.9 - 0.5 * job.fidelity.clamp(0.0, 1.0),
                "width": job.width / 8 * 8,
                "height": job.height / 8 * 8,
            });
            if !model.is_empty() {
                body["override_settings"] = json!({ "sd_model_checkpoint": model });
            }
            merge_extra(&mut body, &profile.extra);
            (format!("{base}/sdapi/v1/img2img"), "application/json".to_string(), body.to_string().into_bytes())
        }
    };
    headers.push(("Content-Type".into(), content_type));
    Ok(HttpRequest { url, headers, body })
}

// ---------------------------------------------------------------------------
// Reading the answer
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub enum Parsed {
    /// The image file (PNG or JPEG).
    Bytes(Vec<u8>),
    /// The image has to be downloaded from here.
    Url(String),
}

fn decode_b64(data: &str) -> Option<Vec<u8>> {
    // Some servers answer with a data URL.
    let data = match data.split_once("base64,") {
        Some((head, rest)) if head.starts_with("data:") => rest,
        _ => data,
    };
    base64_decode(data).filter(|b| !b.is_empty())
}

/// Finds the first image anywhere in a JSON answer, whatever the service
/// calls it: `b64_json`, `inlineData`/`inline_data`, an `image` block…
fn find_image(v: &Value) -> Option<Vec<u8>> {
    match v {
        Value::Object(o) => {
            if let Some(b) = o.get("b64_json").and_then(Value::as_str).and_then(decode_b64) {
                return Some(b);
            }
            let mime = o.get("mime_type").or_else(|| o.get("mimeType")).and_then(Value::as_str).unwrap_or("");
            let is_image = mime.starts_with("image/") || o.get("type").and_then(Value::as_str) == Some("image");
            if is_image {
                if let Some(b) = o.get("data").and_then(Value::as_str).and_then(decode_b64) {
                    return Some(b);
                }
            }
            o.values().find_map(find_image)
        }
        Value::Array(a) => a.iter().find_map(find_image),
        _ => None,
    }
}

/// The first piece of text in a JSON answer (a model explaining why it made no image).
fn find_text(v: &Value) -> Option<String> {
    match v {
        Value::Object(o) => match o.get("text").and_then(Value::as_str) {
            Some(t) if !t.trim().is_empty() => Some(t.trim().to_string()),
            _ => o.values().find_map(find_text),
        },
        Value::Array(a) => a.iter().find_map(find_text),
        _ => None,
    }
}

fn shorten(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() > 300 {
        format!("{}…", text.chars().take(300).collect::<String>())
    } else {
        text
    }
}

pub fn parse(kind: Kind, content_type: &str, body: &[u8]) -> Result<Parsed, String> {
    const NO_IMAGE: &str = "The service answered, but without an image.";
    if content_type.starts_with("image/") {
        return Ok(Parsed::Bytes(body.to_vec()));
    }
    let v: Value = serde_json::from_slice(body).map_err(|_| format!("{NO_IMAGE} ({})", shorten(&String::from_utf8_lossy(body))))?;
    if v.get("finish_reason").and_then(Value::as_str) == Some("CONTENT_FILTERED") {
        return Err("The service's content filter blocked this image.".into());
    }
    match kind {
        Kind::OpenAi => {
            if let Some(url) = v.pointer("/data/0/url").and_then(Value::as_str) {
                return Ok(Parsed::Url(url.to_string()));
            }
        }
        Kind::Stability => {
            if let Some(b) = v.get("image").and_then(Value::as_str).and_then(decode_b64) {
                return Ok(Parsed::Bytes(b));
            }
        }
        Kind::Automatic1111 => {
            if let Some(b) = v.pointer("/images/0").and_then(Value::as_str).and_then(decode_b64) {
                return Ok(Parsed::Bytes(b));
            }
        }
        Kind::Gemini => {}
    }
    if let Some(b) = find_image(&v) {
        return Ok(Parsed::Bytes(b));
    }
    Err(match find_text(&v) {
        Some(t) => format!("{NO_IMAGE} It said: {}", shorten(&t)),
        None => NO_IMAGE.to_string(),
    })
}

/// Turns an error answer into something a person can act on.
pub fn explain(kind: Kind, status: u16, body: &str) -> String {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| v.get("errors").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" ")))
        .or_else(|| ["message", "detail", "error"].iter().find_map(|k| v.get(*k).and_then(Value::as_str)).map(str::to_string))
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| shorten(body));
    let hint = match status {
        401 => " Check the API key.",
        402 | 429 => " The account may be out of credit, or sending too many requests.",
        404 => " Check the address and the model name.",
        413 => " The picture is too large: choose a smaller image size.",
        _ => "",
    };
    let mut message = shorten(&message);
    if !message.is_empty() && !message.ends_with(['.', '!', '?']) {
        message.push('.');
    }
    if message.is_empty() {
        format!("{} answered with error {status}.{hint}", kind.label())
    } else {
        format!("{} answered with error {status}: {message}{hint}", kind.label())
    }
}

// ---------------------------------------------------------------------------
// The network call
// ---------------------------------------------------------------------------

const MAX_ANSWER: u64 = 96 * 1024 * 1024;

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(20)).timeout(Duration::from_secs(600)).build()
}

fn read_answer(kind: Kind, url: &str, result: Result<ureq::Response, ureq::Error>) -> Result<(String, Vec<u8>), String> {
    match result {
        Ok(resp) => {
            let content_type = resp.content_type().to_string();
            let mut body = Vec::new();
            resp.into_reader().take(MAX_ANSWER).read_to_end(&mut body).map_err(|e| format!("The answer was cut off: {e}"))?;
            Ok((content_type, body))
        }
        Err(ureq::Error::Status(status, resp)) => {
            let mut text = String::new();
            let _ = resp.into_reader().take(1024 * 1024).read_to_string(&mut text);
            Err(explain(kind, status, &text))
        }
        Err(ureq::Error::Transport(t)) => {
            let hint =
                if kind == Kind::Automatic1111 { " Is Stable Diffusion running, and was it started with the --api option?" } else { "" };
            Err(format!("Could not reach {}: {t}.{hint}", host(url)))
        }
    }
}

/// Sends the job and returns the image file the service made. Blocks until
/// the answer arrives, so call it from a background thread.
pub fn render(kind: Kind, profile: &Profile, job: &Job) -> Result<Vec<u8>, String> {
    let http = build(kind, profile, job)?;
    let agent = agent();
    let mut req = agent.post(&http.url);
    for (k, v) in &http.headers {
        req = req.set(k, v);
    }
    let (content_type, body) = read_answer(kind, &http.url, req.send_bytes(&http.body))?;
    match parse(kind, &content_type, &body)? {
        Parsed::Bytes(b) => Ok(b),
        Parsed::Url(url) => {
            if !(url.starts_with("https://") || (url.starts_with("http://") && is_local(&url))) {
                return Err("The service pointed at an image address OpenSketch won't open.".into());
            }
            read_answer(kind, &url, agent.get(&url).call()).map(|x| x.1)
        }
    }
}

// ---------------------------------------------------------------------------
// Settings (kept in the user's settings folder, never in the model file)
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub provider: Kind,
    pub profiles: BTreeMap<Kind, Profile>,
    pub style: String,
    /// How strictly the model's shape is kept, 0–1.
    pub fidelity: f32,
    /// Long side of the picture that is sent, in pixels.
    pub max_side: u32,
    /// Send the model's edge lines along (off = shaded faces only).
    pub edges: bool,
    pub avoid: String,
    /// Write API keys to the settings file (plain text).
    pub remember_keys: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            provider: Kind::OpenAi,
            profiles: BTreeMap::new(),
            style: STYLES[0].name.to_string(),
            fidelity: 0.8,
            max_side: 1536,
            edges: true,
            avoid: DEFAULT_AVOID.to_string(),
            remember_keys: true,
        }
    }
}

impl Settings {
    pub fn profile(&self) -> Profile {
        self.profiles.get(&self.provider).cloned().unwrap_or_else(|| self.provider.default_profile())
    }

    pub fn profile_mut(&mut self) -> &mut Profile {
        let kind = self.provider;
        self.profiles.entry(kind).or_insert_with(|| kind.default_profile())
    }

    pub fn load_from(path: &Path) -> Settings {
        std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        let mut copy = self.clone();
        if !copy.remember_keys {
            for p in copy.profiles.values_mut() {
                p.api_key.clear();
            }
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let text = serde_json::to_string_pretty(&copy).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            // The file can hold API keys: readable by the owner only.
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }
}

/// `ai.json` in the user's settings folder (`OPENSKETCH_CONFIG_DIR` overrides the folder).
pub fn settings_path() -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let dir = if let Some(dir) = var("OPENSKETCH_CONFIG_DIR") {
        dir
    } else if cfg!(windows) {
        var("APPDATA")?.join("OpenSketch")
    } else if cfg!(target_os = "macos") {
        var("HOME")?.join("Library/Application Support/OpenSketch")
    } else {
        var("XDG_CONFIG_HOME").or_else(|| var("HOME").map(|h| h.join(".config")))?.join("opensketch")
    };
    Some(dir.join("ai.json"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    fn job() -> Job {
        let img = RgbaImage::from_pixel(64, 64, image::Rgba([200, 120, 60, 255]));
        Job {
            png: png_bytes(&img).unwrap(),
            width: 64,
            height: 64,
            prompt: "a brick house".into(),
            negative: "cartoon".into(),
            fidelity: 0.8,
        }
    }

    fn profile(kind: Kind, base: &str) -> Profile {
        Profile { base_url: base.into(), api_key: "test-key".into(), ..kind.default_profile() }
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    fn header<'a>(req: &'a HttpRequest, name: &str) -> &'a str {
        req.headers.iter().find(|h| h.0.eq_ignore_ascii_case(name)).map(|h| h.1.as_str()).unwrap_or("")
    }

    #[test]
    fn prepare_limits_size_aspect_and_alpha() {
        // A very wide viewport is cropped to what Stability accepts and scaled down.
        let wide = RgbaImage::from_pixel(3000, 600, image::Rgba([10, 20, 30, 0]));
        let out = prepare(&wide, 1536, Kind::Stability);
        assert!(out.width() <= 1536 && out.width().is_multiple_of(16) && out.height().is_multiple_of(16));
        assert!(out.width() as f32 / out.height() as f32 <= 2.4);
        assert!(out.pixels().all(|p| p[3] == 255));
        // An ordinary viewport keeps its shape (to within the rounding to 16 px).
        let img = RgbaImage::new(1036, 844);
        let out = prepare(&img, 1536, Kind::OpenAi);
        assert_eq!((out.width(), out.height()), (1024, 832));
        // Small pictures are never blown up, tall ones are limited too.
        let tall = RgbaImage::new(200, 2000);
        let out = prepare(&tall, 1024, Kind::Stability);
        assert!(out.height() <= 1024 && out.height() as f32 / out.width() as f32 <= 2.4);
        let odd = prepare(&RgbaImage::new(207, 2000), 1024, Kind::Stability);
        assert!(odd.height() as f32 / odd.width() as f32 <= 2.4, "{:?}", odd.dimensions());
        assert_eq!(prepare(&RgbaImage::new(320, 240), 2048, Kind::Gemini).dimensions(), (320, 240));
    }

    #[test]
    fn prompt_follows_the_service_and_the_fidelity() {
        let mats = vec!["Brick".to_string(), "Wood".to_string()];
        let p = compose_prompt(Kind::OpenAi, style("Hyper-realistic photo"), "a small house.", &mats, 0.9, "people");
        assert!(p.contains("hyper-realistic photograph") && p.contains("The scene: a small house."));
        assert!(p.contains("Brick, Wood") && p.contains("Keep exactly the same geometry") && p.contains("Avoid: people."));
        let loose = compose_prompt(Kind::Gemini, style("Watercolor"), "", &[], 0.1, "");
        assert!(loose.contains("watercolor") && loose.contains("loose guide") && !loose.contains("Avoid") && !loose.contains("The scene"));
        // Diffusion services get a description, not instructions.
        let d = compose_prompt(Kind::Stability, style("Golden hour"), "a small house", &mats, 0.9, "people");
        assert!(d.starts_with("a small house, materials: Brick, Wood, hyper-realistic photograph, golden hour"));
        assert!(!d.contains("Keep exactly") && !d.contains("people"));
        // An unknown style name falls back to the default.
        assert_eq!(style("nope").name, "Hyper-realistic photo");
    }

    #[test]
    fn extra_options_are_parsed_and_override() {
        let extra = "# comment\n quality = high \n\nsize=1024x1024\nbroken line\n=novalue";
        assert_eq!(parse_extra(extra), vec![("quality".to_string(), "high".to_string()), ("size".to_string(), "1024x1024".to_string())]);
        let mut p = profile(Kind::OpenAi, "https://api.openai.com/v1");
        p.extra = "size=1024x1024\ninput_fidelity=high".into();
        let req = build(Kind::OpenAi, &p, &job()).unwrap();
        assert!(contains(&req.body, b"name=\"size\"\r\n\r\n1024x1024\r\n"));
        assert!(!contains(&req.body, b"64x64"));
        assert!(contains(&req.body, b"name=\"input_fidelity\"\r\n\r\nhigh\r\n"));
    }

    #[test]
    fn openai_request() {
        let req = build(Kind::OpenAi, &profile(Kind::OpenAi, "https://api.openai.com/v1/"), &job()).unwrap();
        assert_eq!(req.url, "https://api.openai.com/v1/images/edits");
        assert_eq!(header(&req, "authorization"), "Bearer test-key");
        let boundary = header(&req, "content-type").strip_prefix("multipart/form-data; boundary=").unwrap().to_string();
        assert!(req.body.ends_with(format!("\r\n--{boundary}--\r\n").as_bytes()));
        assert!(contains(&req.body, b"name=\"model\"\r\n\r\ngpt-image-2\r\n"));
        assert!(contains(&req.body, b"name=\"prompt\"\r\n\r\na brick house\r\n"));
        assert!(contains(&req.body, b"name=\"size\"\r\n\r\n64x64\r\n"));
        assert!(contains(&req.body, b"name=\"quality\"\r\n\r\nhigh\r\n"));
        assert!(contains(&req.body, b"name=\"image[]\"; filename=\"view.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG"));
        // Models outside the gpt-image-2 family only take fixed sizes: none is sent.
        let mut p = profile(Kind::OpenAi, "https://api.openai.com/v1");
        p.model = "gpt-image-1.5".into();
        assert!(!contains(&build(Kind::OpenAi, &p, &job()).unwrap().body, b"name=\"size\""));
    }

    #[test]
    fn gemini_request() {
        let mut p = profile(Kind::Gemini, "https://generativelanguage.googleapis.com/v1beta");
        p.extra = "store=false".into();
        let req = build(Kind::Gemini, &p, &job()).unwrap();
        assert_eq!(req.url, "https://generativelanguage.googleapis.com/v1beta/interactions");
        assert_eq!(header(&req, "x-goog-api-key"), "test-key");
        assert_eq!(header(&req, "content-type"), "application/json");
        let v: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(v["model"], "gemini-3-pro-image");
        assert_eq!(v["input"][0]["text"], "a brick house");
        assert_eq!(v["input"][1]["type"], "image");
        assert_eq!(v["input"][1]["mime_type"], "image/png");
        assert_eq!(base64_decode(v["input"][1]["data"].as_str().unwrap()).unwrap(), job().png);
        assert_eq!(v["store"], false);
    }

    #[test]
    fn stability_request() {
        let req = build(Kind::Stability, &profile(Kind::Stability, "https://api.stability.ai"), &job()).unwrap();
        assert_eq!(req.url, "https://api.stability.ai/v2beta/stable-image/control/structure");
        assert_eq!(header(&req, "accept"), "image/*");
        assert!(contains(&req.body, b"name=\"control_strength\"\r\n\r\n0.80\r\n"));
        assert!(contains(&req.body, b"name=\"negative_prompt\"\r\n\r\ncartoon\r\n"));
        assert!(contains(&req.body, b"name=\"style_preset\"\r\n\r\nphotographic\r\n"));
        assert!(contains(&req.body, b"name=\"image\"; filename=\"view.png\""));
    }

    #[test]
    fn automatic1111_request() {
        let mut p = Kind::Automatic1111.default_profile();
        let req = build(Kind::Automatic1111, &p, &job()).unwrap();
        assert_eq!(req.url, "http://127.0.0.1:7860/sdapi/v1/img2img");
        assert_eq!(header(&req, "authorization"), "");
        let v: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(v["prompt"], "a brick house");
        assert_eq!(v["negative_prompt"], "cartoon");
        assert!((v["denoising_strength"].as_f64().unwrap() - 0.5).abs() < 1e-6);
        assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(64), Some(64)));
        assert_eq!(v["steps"], 30);
        assert_eq!(v["cfg_scale"], 7);
        assert!(v.get("override_settings").is_none());
        // A checkpoint name and --api-auth credentials.
        p.model = "realisticVision".into();
        p.api_key = "user:pass".into();
        let req = build(Kind::Automatic1111, &p, &job()).unwrap();
        assert_eq!(header(&req, "authorization"), format!("Basic {}", base64_encode(b"user:pass")));
        let v: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(v["override_settings"]["sd_model_checkpoint"], "realisticVision");
    }

    #[test]
    fn keys_are_required_and_never_sent_unprotected() {
        let mut p = Kind::OpenAi.default_profile();
        assert_eq!(build(Kind::OpenAi, &p, &job()).unwrap_err(), "Enter your API key first.");
        p.api_key = "k".into();
        p.base_url = "http://example.com/v1".into();
        assert!(build(Kind::OpenAi, &p, &job()).unwrap_err().contains("https://"));
        p.base_url = "example.com".into();
        assert!(build(Kind::OpenAi, &p, &job()).unwrap_err().contains("must start with"));
        // A compatible server on this computer or the local network is fine over http.
        for base in ["http://localhost:8080/v1", "http://127.0.0.1:1234/v1", "http://192.168.1.20/v1", "http://[::1]:8080/v1"] {
            p.base_url = base.into();
            assert!(build(Kind::OpenAi, &p, &job()).is_ok(), "{base}");
        }
        assert_eq!(host("https://api.openai.com/v1"), "api.openai.com");
        assert_eq!(host("http://127.0.0.1:7860"), "127.0.0.1:7860");
    }

    #[test]
    fn answers_are_understood() {
        let png = job().png;
        let b64 = base64_encode(&png);
        let bytes = Parsed::Bytes(png.clone());
        // OpenAI: base64 or a link.
        let body = json!({ "created": 1, "data": [{ "b64_json": b64 }] }).to_string();
        assert_eq!(parse(Kind::OpenAi, "application/json", body.as_bytes()).unwrap(), bytes);
        let body = json!({ "data": [{ "url": "https://example.com/a.png" }] }).to_string();
        assert_eq!(parse(Kind::OpenAi, "application/json", body.as_bytes()).unwrap(), Parsed::Url("https://example.com/a.png".into()));
        // Gemini: wherever the image block is, in either spelling.
        let body = json!({ "steps": [{ "type": "model_output", "content": [{ "type": "text", "text": "Here" }, { "type": "image", "mime_type": "image/png", "data": b64 }] }] });
        assert_eq!(parse(Kind::Gemini, "application/json", body.to_string().as_bytes()).unwrap(), bytes);
        let body = json!({ "candidates": [{ "content": { "parts": [{ "inlineData": { "mimeType": "image/png", "data": b64 } }] } }] });
        assert_eq!(parse(Kind::Gemini, "application/json", body.to_string().as_bytes()).unwrap(), bytes);
        // Stability: the raw image, or JSON.
        assert_eq!(parse(Kind::Stability, "image/png", &png).unwrap(), bytes);
        let body = json!({ "image": b64, "finish_reason": "SUCCESS", "seed": 1 }).to_string();
        assert_eq!(parse(Kind::Stability, "application/json; type=image/png", body.as_bytes()).unwrap(), bytes);
        let body = json!({ "image": b64, "finish_reason": "CONTENT_FILTERED" }).to_string();
        assert!(parse(Kind::Stability, "application/json", body.as_bytes()).unwrap_err().contains("content filter"));
        // AUTOMATIC1111: plain base64 or a data URL.
        let body = json!({ "images": [format!("data:image/png;base64,{b64}")], "info": "{}" }).to_string();
        assert_eq!(parse(Kind::Automatic1111, "application/json", body.as_bytes()).unwrap(), bytes);
        // No image: the model's own words are passed on.
        let body = json!({ "steps": [{ "content": [{ "type": "text", "text": "I can't help with that." }] }] }).to_string();
        let err = parse(Kind::Gemini, "application/json", body.as_bytes()).unwrap_err();
        assert!(err.contains("without an image") && err.contains("I can't help with that."));
        assert!(parse(Kind::OpenAi, "text/html", b"<html>gateway</html>").unwrap_err().contains("without an image"));
    }

    #[test]
    fn errors_are_explained() {
        let e = explain(Kind::OpenAi, 401, r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#);
        assert_eq!(e, "OpenAI (or compatible) answered with error 401: Incorrect API key provided. Check the API key.");
        let e = explain(
            Kind::Stability,
            400,
            r#"{"id":"x","name":"bad_request","errors":["image: aspect ratio must be between 1:2.5 and 2.5:1"]}"#,
        );
        assert!(e.contains("aspect ratio must be between"));
        assert!(explain(Kind::Automatic1111, 404, r#"{"detail":"Not Found"}"#).contains("Not Found. Check the address"));
        assert_eq!(explain(Kind::Gemini, 500, ""), "Google Gemini answered with error 500.");
    }

    #[test]
    fn settings_roundtrip_and_keys_can_stay_out_of_the_file() {
        let dir = std::env::temp_dir().join(format!("opensketch-ai-test-{}", std::process::id()));
        let path = dir.join("ai.json");
        let mut s = Settings { provider: Kind::Stability, fidelity: 0.4, ..Settings::default() };
        s.profile_mut().api_key = "secret".into();
        s.profile_mut().model = "control/sketch".into();
        s.save_to(&path).unwrap();
        assert_eq!(Settings::load_from(&path), s);
        s.remember_keys = false;
        s.save_to(&path).unwrap();
        assert!(!std::fs::read_to_string(&path).unwrap().contains("secret"));
        let loaded = Settings::load_from(&path);
        assert_eq!(loaded.profile().api_key, "");
        assert_eq!(loaded.profile().model, "control/sketch");
        // A missing or broken file gives the defaults; unknown services fall back too.
        std::fs::write(&path, "{ not json").unwrap();
        assert_eq!(Settings::load_from(&path), Settings::default());
        assert_eq!(Settings::default().profile(), Kind::OpenAi.default_profile());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A one-request HTTP server on this computer. Returns its address and a
    /// handle that yields the request head and body it received.
    fn serve_once(status: &str, content_type: &str, answer: Vec<u8>) -> (String, std::thread::JoinHandle<(String, Vec<u8>)>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (status, content_type) = (status.to_string(), content_type.to_string());
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let (mut head, mut length) = (String::new(), 0usize);
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap();
                }
                head += &line;
            }
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).unwrap();
            let mut stream = reader.into_inner();
            let head_out = format!(
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                answer.len()
            );
            stream.write_all(head_out.as_bytes()).unwrap();
            stream.write_all(&answer).unwrap();
            stream.flush().unwrap();
            (head, body)
        });
        (base, handle)
    }

    #[test]
    fn renders_through_every_service_end_to_end() {
        let result = png_bytes(&RgbaImage::from_pixel(32, 32, image::Rgba([1, 2, 3, 255]))).unwrap();
        let b64 = base64_encode(&result);
        let cases = [
            (Kind::OpenAi, "application/json", json!({ "data": [{ "b64_json": b64 }] }).to_string().into_bytes(), "POST /images/edits "),
            (
                Kind::Gemini,
                "application/json",
                json!({ "steps": [{ "type": "model_output", "content": [{ "type": "image", "mime_type": "image/png", "data": b64 }] }] })
                    .to_string()
                    .into_bytes(),
                "POST /interactions ",
            ),
            (Kind::Stability, "image/png", result.clone(), "POST /v2beta/stable-image/control/structure "),
            (Kind::Automatic1111, "application/json", json!({ "images": [b64] }).to_string().into_bytes(), "POST /sdapi/v1/img2img "),
        ];
        for (kind, content_type, answer, request_line) in cases {
            let (base, server) = serve_once("200 OK", content_type, answer);
            let out = render(kind, &profile(kind, &base), &job()).unwrap();
            assert_eq!(out, result, "{kind:?}");
            assert!(image::load_from_memory(&out).is_ok());
            let (head, body) = server.join().unwrap();
            assert!(head.starts_with(request_line), "{kind:?}: {head}");
            assert!(head.to_ascii_lowercase().contains("test-key") || kind == Kind::Automatic1111, "{kind:?}: {head}");
            assert!(contains(&body, b"a brick house"), "{kind:?}");
        }
    }

    #[test]
    fn service_errors_and_unreachable_services_are_reported() {
        let (base, server) =
            serve_once("401 Unauthorized", "application/json", br#"{"error":{"message":"Incorrect API key provided"}}"#.to_vec());
        let err = render(Kind::OpenAi, &profile(Kind::OpenAi, &base), &job()).unwrap_err();
        assert!(err.contains("error 401: Incorrect API key provided"), "{err}");
        server.join().unwrap();
        // Nothing listens on this port any more.
        let err = render(Kind::Automatic1111, &profile(Kind::Automatic1111, &base), &job()).unwrap_err();
        assert!(err.starts_with("Could not reach 127.0.0.1:") && err.contains("--api"), "{err}");
    }
}
