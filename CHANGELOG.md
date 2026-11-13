# Changelog

All notable changes to OpenSketch are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **AI Render** (File ▸ AI Render…): turn the current view into a
  hyper-realistic picture with an AI image model, using your own account with
  OpenAI (or a compatible service), Google Gemini or Stability AI, or a Stable
  Diffusion running on your own computer. Styles, a "keep my model's shape"
  control, editable address / model / extra options, before-and-after view and
  PNG export.
- Push/Pull stops level with a corner, edge or parallel face you point at, and
  double-clicking a face repeats the last distance.
- Move picks up what is under the cursor when nothing is selected, including a
  single corner point, and highlights it first. Faces that stop being flat
  after a move fold along new edges (auto-fold).
- 2-point arcs snap to tangents ("Tangent at Vertex").
- Inference: "From Point" alignment along an axis, and parallel /
  perpendicular to an edge.
- `OPENSKETCH_SCRIPT` plays scripted mouse and keyboard input for testing.
- File ▸ Samples ▸ Stone house with arches: a complete sample house built
  with the kernel (walls pulled up and windows pushed in), in named groups,
  with tags for the roof and the tree and a saved "Street view" scene.
- Limestone blocks and clay roof tiles textures.
- `opensketch file.osk` opens a model; `opensketch --sample stone-house`
  opens a sample.

### Changed
- Tools act when the mouse button goes down (click-move-click), and also
  finish on release after a real drag. A small wobble during a click no longer
  starts and finishes a tiny shape.
- Geometry being moved, rotated or scaled no longer snaps to its own old
  position.
- A Move, Rotate or Scale that picked its own target leaves nothing selected
  afterwards, so the next one picks what is under the cursor again.

## [0.1.1] - 2026-11-10

### Changed
- Help ▸ Keyboard shortcuts now lists Protractor, Dimension, Text, Solid
  Tools and Ctrl with the Tape Measure, and scrolls so it fits on small
  screens ([#11](https://github.com/Khaldoun-Baz/opensketch/pull/11), by
  [agenticsymphony-tech](https://github.com/agenticsymphony-tech): the
  project's first outside contribution).

### Project
- New website: <https://khaldoun-baz.github.io/opensketch/>
- GitHub Discussions is open for questions and ideas.

## [0.1.0] - 2026-11-07

First public release.

### Modeling
- Line, 2-point arc, center arc, pie, rectangle, circle and polygon tools,
  on the ground, on faces or on locked planes. Lines split faces, closed
  loops become faces, overlapping shapes divide into regions.
- Push/Pull following SketchUp's rules: faces move and stretch their sides,
  shapes inside faces become pockets (or holes once they reach the far side),
  flush pushes merge back. Offset, Follow Me, Move/Rotate/Scale with copies
  and arrays, Eraser, Paint Bucket.
- Inference snapping: endpoints, midpoints, edges, faces, axes, guides and
  intersections. Typed values in the Measurements box, re-typing adjusts the
  last operation.
- Groups and components with in-place editing, Outliner and Components panel.
  Components glue to faces and cut openings (windows, doors).
- Intersect Faces and Solid Tools: Outer Shell, Intersect, Union, Subtract,
  Trim, Split.

### Construction and documentation
- Tape Measure guides, Protractor, Dimensions (linear, radius, diameter) and
  Text (leader and screen text).
- Tags with visibility and Color by Tag; Scenes with animated transitions.

### Rendering
- Built-in textures (brick, wood, shingles, grass, cobblestone, tiles) and
  image textures embedded in the model file.
- Shadows from a sun positioned by time, date and latitude.

### Files
- Native `.osk` format, OBJ (with textures) and STL export, OBJ import.

### Known limitations
- Dimensions don't follow geometry when it moves.
- Window openings cut only the face they're glued to, not through thick walls.
- No copy/paste yet, and no tool to reposition a texture on a single face.

[Unreleased]: https://github.com/Khaldoun-Baz/opensketch/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Khaldoun-Baz/opensketch/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Khaldoun-Baz/opensketch/releases/tag/v0.1.0
