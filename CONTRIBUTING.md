# Contributing to OpenSketch

Thanks for helping build an open, SketchUp-style 3D modeler in Rust! Bug
reports, feature ideas, docs and code are all welcome.

## Getting started

1. Install Rust (stable, 1.88 or newer) from <https://rustup.rs>.
2. On Linux, install the windowing libraries:
   ```bash
   sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
   ```
3. Clone and run:
   ```bash
   git clone https://github.com/Khaldoun-Baz/opensketch.git
   cd opensketch
   cargo run --release
   ```

## Before you open a pull request

CI runs these on every pull request; please run them locally first:

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

If you touch rendering, also run the screenshot self-test and look at the image:

```bash
OPENSKETCH_SHOT=shot.png cargo run
```

## Where things live

The README has a code map. In short:

- `src/model.rs`: the geometry kernel (faces, edges, push/pull, intersections). Most modeling bugs live here.
- `src/document.rs`: groups, components, tags, scenes.
- `src/app.rs`: tools, UI and drawing.
- `src/render.rs`: OpenGL (textures, shadows).

## Guidelines

- **Fix bugs with a test.** For a modeling bug, add a unit test that builds the
  geometry, performs the operation and checks the result (`crate::solid::is_solid`
  and `volume` are handy). It should fail before your fix and pass after.
- **SketchUp is the behavioral reference.** When unsure how a tool should
  behave, describe what SketchUp does in the issue or PR. Do not copy SketchUp's
  code, icons, textures, trademarks or file formats you don't have rights to;
  OpenSketch is an independent project.
- **Keep files compatible.** `.osk` files are JSON. New fields need
  `#[serde(default)]` so older files still open.
- **Keep PRs focused.** One feature or fix per PR is easier to review.
  Describe what changed and how you tested it.

## Good first contributions

- Items from the README roadmap (Position Texture, component axes tool,
  Paste in Place, styles).
- Issues labeled **good first issue** or **help wanted**.
- Docs: a getting-started guide with screenshots, keyboard reference.

## Reporting bugs

Open an issue with the steps to reproduce, what you expected, what happened,
and your OS. Attaching the `.osk` file helps a lot.

By contributing, you agree that your contributions are licensed under the MIT
License (see `LICENSE`), and to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
