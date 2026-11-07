# OpenSketch

A SketchUp-style push/pull 3D modeler written in Rust (egui + OpenGL).

```bash
cargo run --release
```

## What works today

| Area | Features |
|---|---|
| Drawing | Line (L), 2-Point Arc (A), Arc (center), Pie, Rectangle (R), Circle (C), Polygon (N) — on the ground, on faces, or on locked vertical planes. Arcs/circles are curves (select as one), arcs close and split faces; type `5r` radius, `12s` segments |
| Follow Me | Select a path (edges or a face outline), click the profile face: mitred sweeps along open paths, rings and lathe/revolve along closed paths |
| Intersections | Coplanar edges split where they cross; overlapping shapes, lines across faces and shapes touching a face's edge divide faces into proper regions (no overlapping faces). Edit ▸ Intersect Faces ▸ With Model / With Selection cuts edges where faces and groups/components cross |
| Smoothing | Shallow-angle seams from push/pull and Follow Me become soft edges (hidden) with smooth shading; View ▸ Hidden Geometry shows them |
| Groups & Components | Make Group (Ctrl+G), Make Component (G), double-click to edit in place (rest of the model fades; Esc or click outside closes), Explode, Make Unique, convert group → component, Components panel with Place, Outliner tree, nested groups, paint/move/rotate/scale/copy/erase whole instances |
| Editing | Push/Pull (P), Move (M, Ctrl = copy, `3x` / `/3` arrays), Rotate (Q, 15° snaps), Scale (S, uniform or `x,y,z`), Offset (F), Eraser (E), Paint Bucket (B, Alt = sample) |
| Sticky geometry | Shared vertices/edges, auto-faces from closed coplanar loops, lines split faces, shapes drawn inside faces become holes, coplanar faces heal after push/pull or seam erase |
| Inference | Endpoint, Midpoint, On Edge, On Face, red/green/blue axis; arrow keys lock axes, Shift holds the current inference |
| Measurements box | Type while drawing: `2.5`, `300mm`, `4,3`, `24s`, `45` (degrees). Retype right after Push/Pull, Move, Rotate or Offset to adjust it |
| Camera | Orbit (O / middle-drag), Pan (H / Shift+middle / right-drag), zoom to cursor, Zoom Extents (Shift+Z), standard views, perspective/parallel, X-ray |
| Selection | Click, Shift toggle, Ctrl add, double-click (face + edges), triple-click (connected), window (L→R) and crossing (R→L) boxes |
| Component openings | Make Component (G) on geometry drawn on a face (a window or door rectangle, or a recess pushed into a wall) glues it to the face and cuts an opening: blue axis out of the face, wall healed, hole cut live. Glued components place onto faces, slide only within the face when moved, and the opening follows; Entity Info toggles Glue to faces / Cut opening. Explode and OBJ/STL export make the opening real geometry |
| Solid tools | Outer Shell, Intersect, Union, Subtract, Trim, Split on solid groups/components (closed shells: every edge bounds two faces). Click the first solid, then the second; Outer Shell / Union / Intersect also work on several selected solids. Results are new groups with coplanar seams cleaned up; Entity Info shows whether an instance is a solid and its volume |
| Guides | Tape Measure (T): click a point and measure (adds a guide point), or start on an edge / guide line and move off it for a parallel guide line (click twice in place for a guide on the edge); type a distance; Ctrl toggles measure-only. Protractor: center (aligns to the hovered face, arrow keys lock a plane), baseline, then angle with 15° snaps, or type `30` or a slope `1:2`. Guides are snapped to (On Guide, Guide Point, Intersection with edges and other guides), can be selected, moved/copied, erased, hidden (View ▸ Guides) and cleared (Edit ▸ Delete Guides); saved in the file, not exported |
| Tags & Scenes | Tags panel: add, rename, recolor, hide/show and delete tags; assign a tag to selected groups/components in Entity Info (➕ makes a new one). Hidden tags hide their instances (not while you edit inside one); View ▸ Color by Tag. Scenes panel / View menu: Add Scene saves the camera, hidden tags and display settings (X-ray, hidden geometry, guides, axes, color by tag); scene tabs above the viewport and PgDn/PgUp switch scenes with an animated camera; ⟳ updates a scene. Saved in the file |
| Dimensions & Text | Dimension: click two points (or an edge), move out to offset the dimension line along a plane facing you, click; click an arc or circle for its radius (R) or diameter (Ø). Text: click a face / edge / point for leader text that starts as its area / length / coordinates, or empty space for screen text. Labels follow the display units; edit text in Entity Info (`<>` = measured value). Select, Move/copy, erase, hide (View ▸ Dimensions & Text); saved in the file and in scenes |
| Textures | Built-in procedural textures (brick, wood planks, roof shingles, grass, cobblestone, floor tiles) on the default materials, or Materials ▸ Load image… (PNG/JPEG, embedded in the `.osk` file). Real-world texture size per material; textures are projected onto each face's plane and stay put on groups. Painted faces show their material on both sides. OBJ export writes the texture PNGs (`map_Kd`) and UVs |
| Shadows | Shadows panel (or View ▸ Shadows): time of day, date, latitude and north angle drive the sun; shadow-mapped shadows on faces and on the ground (even where there is no ground face), sun-based shading. Saved in the file and in scenes |
| Files | Native `.osk` (JSON), OBJ import, OBJ (+MTL) and STL export, undo/redo (Ctrl+Z / Ctrl+Y) |

## Code map

- `src/document.rs` — definitions/instances (groups & components), edit contexts, explode, flatten for export
- `src/model.rs` — geometry kernel: vertices/edges/faces with holes, face splitting, loop detection, push/pull, offset, healing, triangulation (unit tests at the bottom)
- `src/solid.rs` — solid check, volume and booleans (Union / Intersect / Subtract) for the Solid Tools
- `src/guide.rs` — guide lines/points, their intersections and dashed drawing
- `src/texture.rs` — built-in procedural textures, image textures (embedded as base64), UV projection
- `src/sun.rs` — sun position from time/date/latitude, shadow-map light frustum
- `src/inference.rs` — snapping and picking
- `src/app.rs` — UI, tools, Measurements box, selection
- `src/render.rs` — OpenGL renderer: plain and textured face batches, shadow-map pass, ground shadows, edges
- `src/camera.rs`, `src/units.rs`, `src/io.rs`

## Roadmap (next SketchUp features)

1. Component axes tool, Edit ▸ Paste in Place, openings cutting through thick walls
2. Position Texture (move/rotate/scale a texture on a face), styles (sketchy edges, watermarks)
3. Dimensions that stay attached to geometry when it moves

## Rendering self-test

`OPENSKETCH_SHOT=shot.png cargo run` opens a small textured, sunlit demo scene,
saves the 3D view to `shot.png` and exits (add `OPENSKETCH_SHOT_NO_GROUND=1` to
check ground shadows without a ground face). Handy for checking rendering
changes without clicking through the UI.
