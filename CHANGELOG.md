# Changelog

All notable changes to OpenSketch are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
