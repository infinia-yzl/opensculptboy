# Design: a revamp of the web viewer's interface

Status: proposal, 2026-10-28. The owner decides the questions at the end before any work starts.

## Why

A new visitor opens `viewer/dist/anny_viewer.html`, sees Anny's default body and uses it. The page gives no clear way to bring in a character of their own, so they never see what a fitted character card looks like in the page: its baked skin, its own hair, brows, eyes, lashes, teeth and mouth poses, and its outfits. Since the first viewer, the page has gained wearables, a card's look, expressions and the character sheet, and each of them joined the panel where it fitted at the time. This review looks at the page as a new visitor meets it and proposes a structure for the tasks people come for: make a character, dress it, give it expressions, pose it, make a sheet of it and export it.

## How the review was made

The page of `main` (commit 4c9bc16) ran in Chromium through Playwright with a fresh browser profile, at 1920 by 1080 and at a phone size of 390 by 844. A script opened every section of both segments, the help, the Performance card and the Randomise menu, pressed every panel key with the figure focused, and opened four files a visitor would try: a character card of sliders only, a minimal GLB, the same file named `.vrm`, through both the Wearables section and the Characters section. A fitted reference card was also opened through the Wearables section. The review pictures are annotated screenshots and wireframes of the proposal.

## What a new visitor finds

### Bringing in their own character
- The only way to open a character card is the **Wearables** section, the sixth tab of the left rail. Its name says clothes, and its lead text speaks of "outfits and wearable packages".
- The **Characters** section has its own **Open file**, which reads a different format: the page's "preset" (a look with skin tone, hair, eyes, sliders and face). It takes a character card without a message, keeps the card's name and sliders, and drops the card's face shapes and facial actions, because the card names them `face_shapes` and the preset `face`.
- A GLB or VRM file gives "The files hold no character card and no wearable package." The message names no next step. OpenSculptBoy's own GLB and VRM files carry their card in the scene extras, and the page does not read it from there.
- A card without outfits opens with "No outfit" as the only choice, the toast says "0 of 0 pieces shown", and the section shows triangle counts and dressing times. The body and the face change, but the section reads as if the open failed.
- A fitted card (with its detail layer, baked textures, eyes and groom) is made only by `opensculptboy fit` on the command line. The page links to none of this.

### Why the default looks plainer than a fitted card
- The default figure stands undressed in an A-pose, with the page's own skin shading, eyes and strand hair. A fitted card shows baked base colour, normal and roughness maps, the reference's hair and brows, its eyes and blinks, lashes, teeth, inner mouth and mouth poses. Nothing on the page tells a visitor that this difference exists or how to reach it.
- The four presets ("Asian woman", "Asian man", "Eurasian woman", "Eurasian man") are the first choices on the page. They name the models by ethnicity and gender, which presents them as humans, against the project's rule that the models are humanoid 3D characters. None of them is dressed or styled beyond a hair style.
- The page ships no dressed character and no sample card, and the repository ships no garments of its own yet.

### Where the features sit
- The page has two segments named by their side of the screen: **Character** on the left (Characters, Body, Face, Hair, Skin & eyes, Wearables) and **Stage** on the right (Pose, Expression, Rig, Scene, Sheet). The segment names show only in the code and in screen-reader labels.
- Tasks cut across this split. Wearables holds the card loader and, for a card with reference eyes, the blink controls. Expression and Sheet sit beside Rig and Scene, which are tools for looking at the model. Opening Sheet closes the Character inspector and moves the figure to the left edge, and the layout view and the inspector both carry the title "Sheet" with Build and Download in each.
- With a fitted card open, the Hair section still offers the page's 25 styles and sliders, and Skin & eyes still offers tone and undertone, although the card's groom and baked skin take their place in the picture. The page does not say which controls the card overrides.
- On a phone the eleven tabs share one bottom bar of 390 pixels: the labels run into each other ("PoseExpression") and the last tab is cut off.

### The export story
- The page writes a PNG of the view (**Save image**), a pose card (body sliders, face shapes and the pose, without hair, colours, facial actions or outfit), a silhouette SVG, a preset JSON and the sheet PNG.
- It writes no GLB, no VRM and no character card with its files. Edits made on a loaded card cannot be saved. The README sends visitors from the page to the command line for every export.
- The page's look (hair style, hair colour, skin tone and undertone, eye colour) has no field in the character card, so a card cannot carry what the page shows.

### Naming
- The page is "OpenSculptBoy Viewer" in the top bar and the title, and the help sheet is titled "Anny Viewer". The Body section's lead says "anny's phenotype sliders", and the canvas label says "anny's body model".
- The default character is called "Baseline". One thing has three names: "preset" (Characters), "look" (code and colour chips) and "character card" (Wearables, docs and command line).
- The Body section heads the race sliders "Ethnicity" and explains "Eurasian is Asian and Caucasian at equal values".
- Messages in the Wearables section speak to developers: bone names ("the page moves its bones jaw with head"), mesh group names, the count of pushed-out vertices and their distance in millimetres.

### Keyboard and mouse
- With the figure focused, the panel keys C, B, F, H, E, P, X, R, L and S open their sections, `?` opens the help and the backquote key opens the Performance card. **W does nothing**: the help lists it for Wearables, the key handler has no lower-case `w`, and Shift+W shows the skin weights.
- The keys show only in tooltips and in the help. The first-run hint names drag, right-drag, scroll, double-click and `?`, and the same hint shows on a phone, where right-drag, scroll and double-click do not exist; there it covers the figure's legs.
- In the automated run, the focused rail tab kept its tooltip over the heading of the next section after a panel closed. This needs a check by hand.

### Performance cues
- On a desktop GPU the page is ready in about 9.5 seconds, behind a loader with a progress bar. The file is 30 MB.
- A fitted card with two outfits took 16 seconds to open, with only "Reading the files…" as feedback.
- A Draft sheet of the default character built in 18 seconds, with panels filling in and a progress bar. The refinement ring ("Refined", or a percentage) is clear once learnt, and its meaning shows only in a tooltip.

### Accessibility
- The rails are tab lists with arrow-key navigation, dialogs trap focus, toasts are live regions, and coarse pointers get 44-pixel targets. These are good foundations.
- The faint text colour (`#6f6962`) on the panel colour (`#16171b`) has a contrast of 3.3 to 1, under the 4.5 to 1 that WCAG AA asks for text of this size. Rail labels and sub-headings are 10.5 pixels.
- The page lists its keys in a dialog only; a visitor with a screen reader meets eleven tabs and no summary of what the character holds.

## The proposal

### Task modes
The page keeps its single canvas, its dark studio look and its section builders. The two side rails give way to six **modes** in the top bar, one per task, and each mode opens one inspector on the left with its own sub-tabs:

| Mode | Holds | Comes from today's |
|---|---|---|
| **Create** | Body, Face, Hair, Skin & eyes as sub-tabs; Randomise in each | Body, Face, Hair, Skin & eyes, the Randomise menu |
| **Dress** | the outfits and pieces of the character, Add pieces (a drop zone for packages and outfit files), Hide covered skin, and the fit details behind a fold | Wearables, without the card loader |
| **Express** | the expression library (today the 14 presets and 10 visemes), intensity, the card's own expressions; later the planned library of named expressions, the editor and From a camera | Expression |
| **Pose** | poses, clips, From a picture, head correction, soft tissue | Pose |
| **Sheet** | the layout view in the centre and the panel settings on the left | Sheet |
| **Export** | the character card (download), GLB and VRM (the command to copy until the page writes them), the sheet, a picture of the view, the pose card and the silhouette | Save image, Pose card, Silhouette, Preset Download |

Rig, Scene (lights, quality, frame rate) and Performance move into a **View** menu beside the view bar, since they serve every mode. The view bar keeps the framings and angles. The modes take the keys 1 to 6 only if the framing keys move; otherwise they keep letters (C for Create, D Dress, X Express, P Pose, S Sheet, E Export), shown on the mode buttons.

A **Character** panel on the right summarises what the character holds: shape, skin (page skin, or baked textures), hair (page style, or reference groom), eyes, mouth, outfit and expressions, each with a link to its mode. For a page character it also links to [Fit to a reference](../reference-fit.md) with one line on what a fit adds. With a card open, the controls that the card overrides say so ("The card's own hair is shown." with a switch for the page's hair) in Create's Hair and Skin tabs.

On a phone the six modes form the bottom bar, and the inspector is the bottom sheet that the page already has.

### The first run and the start screen
The first visit opens a **start screen** over the dimmed figure. Open in the top bar (Ctrl+O) brings it back.
- **Open your character**: a drop zone and Choose files or Choose a folder, with the formats it reads, and a link for other 3D models to the fit.
- **Starter characters**: four to six dressed and styled characters with names (see "The default character").
- **Base body**: Anny's default shape in simple basewear, for people who want to start from the base.
- **Random**: a random body, face, hair and colours, as Random character does now.
- **Continue**: the character of the last visit, kept in the browser.

A drop anywhere on the page keeps working in every mode, and the page routes each file by its content.

### What the page opens

| File | Today | Proposed |
|---|---|---|
| Character card `.json` with its files (picked, a folder, or dropped) | Wearables only | Open, the start screen, or a drop anywhere |
| A `.zip` of a card and its files | no | yes: one file to share, and what Export writes |
| GLB or VRM written by OpenSculptBoy | error | the card in its scene extras; a card of sliders opens whole, and a fitted card asks for its files or its `.zip` |
| Preset (look) JSON | Characters only | read anywhere and converted to a card, with a note |
| Outfit file and wearable packages | Wearables | Dress, Open or a drop; added to the open character |
| Pose card | no | Pose: applies the pose |
| Sheet settings (`opensculptboy/sheet-settings@1`) | no | Sheet |
| A picture | Pose and Sheet slots | Pose (From a picture), and later Express (From a picture) |
| Any other GLB, VRM or FBX | error with no next step | a message that explains the fit and links to it |

### One file for a character
The page's look (hair style and its sliders, hair colour, skin tone and undertone, eye colour) joins the character card as a `look` block, so one card carries everything the page shows and the preset format can retire (the page keeps reading old presets). The exporters ignore the block until they learn to use it. This is a card schema change and needs the owner's decision.

### The default character
- The page opens on a named starter character in simple basewear, at the three-quarter view, with a page hair style and colours chosen on review renders.
- The starters are character cards of sliders, face shapes, a page hair style, colours and basewear, so each costs a few kilobytes in the page. Their names are character names, and their descriptions speak of style ("Travel", "Work", "Robe").
- The repository's own procedural basewear (our own design, CC0) dresses them. The rules for public wearables apply ([Wearables](../wearables.md)): each garment records its design source.
- Fitted sample cards made from open data (a CC0 3D character fitted with `opensculptboy fit`) would show the baked look in the page. Each brings textures of several megabytes, so they live as separate files beside the page, loaded from the start screen.

### Naming
- One name for the page: "OpenSculptBoy". The help, the canvas label and the leads use it, and they call the body model "Anny" with a capital.
- One name for the file: "character card". "Preset" and "look" leave the interface.
- The default character gets a character name. "Baseline" stays as the name of the reset.
- Developer numbers (triangle counts, push-out, group and bone names) move behind a **Details** fold in Dress, in plain words above them ("2 pieces were moved out of the skin by at most 5 mm").
- The heading and the note of the race sliders need new words from the owner. The API keeps its names.

### Keyboard, mouse and help
- Every mode button and sub-tab shows its key, and the help lists the keys by mode with a search field.
- A command list (`/` or Ctrl+K) finds every action by name and shows its key.
- The first-run hint follows the pointer type: touch gestures on a phone, mouse gestures on a desktop.
- Every key that the help lists works, and a test presses each one in the built page.

### Performance cues
- A card's open shows its steps with a progress bar (reading, building the body, the textures, the groom, dressing), as the sheet build does.
- The refinement ring gets a short label in the View menu ("Picture refined" or "Refining, 40 %").

### Accessibility
- The faint text colour rises to at least 4.5 to 1 against the panels, and labels and sub-headings to at least 12 pixels.
- The Character panel is a landmark with a heading, so a screen-reader user hears what the character holds.

## Plan

| Phase | Work | Files in `viewer/src/ui/` (and others) | Done when |
|---|---|---|---|
| 1. Fixes within today's layout | Rename the presets and the help; route a card opened in Characters to the card loader; an **Open** button in the top bar; messages with a next step for GLB, VRM and unknown files; plain-word notes and a Details fold in Wearables; the `w` key; keys shown on the rail tabs; a touch hint on phones; contrast and sizes | `character_ui.ts`, `wearables_ui.ts`, `index.ts`, `shortcuts.ts`, `segments.ts`, `shell.html`, `main.ts` (the preset table) | a visitor opens a card from the top bar; every listed key works in a page test |
| 2. Task modes and the start screen | The mode bar and one inspector per mode, the Character panel, the View menu, the start screen, the phone bar of six modes | new `modes.ts`, `start.ts`, `character_panel.ts`, `view_menu.ts`; `index.ts` and `segments.ts` rebuilt on them; the section builders kept | the tasks of the owner's test list run without the old rails, at 1920 by 1080 and at 390 by 844 |
| 3. The default character and the file | Basewear garments and four to six starter cards; the `look` block in the card; Save card and the `.zip`; reading the card from OpenSculptBoy's GLB and VRM; the Export mode with the commands to copy | `start.ts`, `export_ui.ts`, `wearables/files.ts`, `card/data.ts`; `opensculptboy/character.py` for the `look` block | the first view shows a dressed starter; a card saved in the page opens again in the page and exports with the command |
| 4. Growth of the modes | Express: the planned expression library and editor, with From a camera; Dress: outfit drops, a check overlay that marks skin showing through, and the quality items of each piece; Sheet: the planned styles and the bundle | `expression_ui.ts`, `wearables_ui.ts`, `sheet_ui.ts` | each feature has its page test and review pictures |
| 5. Export from the page (to decide) | GLB and VRM written by the page, or by a local helper that runs the Python exporter on the visitor's computer | to decide | to decide |

Each phase gets page tests in the built page (local only, as `test_viewer_wearables_page` and `test_viewer_sheet` are): the start screen, the routing of each file type with synthetic files, the keys, and the layout at both sizes, with screenshots in a review page.

## Questions for the owner
1. Do the six task modes (Create, Dress, Express, Pose, Sheet, Export) and a View menu for Rig, Scene and Performance fit how you want people to use the page?
2. Should the start screen show on the first visit and on Open, or should the page open straight on a starter character with Open in the top bar?
3. May we design our own procedural basewear (CC0) and ship four to six starter cards in the page? Which names and styles do you want for them?
4. Do you want fitted sample cards made from open data, and do you know a CC0 or CC-BY source of humanoid 3D characters that suits?
5. Which words should replace the preset names and the heading "Ethnicity" of the race sliders?
6. May the character card gain a `look` block (schema 5) for the page's hair, colours and skin, so that the preset format can retire?
7. For GLB and VRM from the page: a browser exporter, a local helper that runs the Python exporter, or the command to copy for now?
8. Should the default character wear basewear by default, with a switch to show the body undressed?

Related: [Web viewer](../viewer.md), [Character card](../character-card.md), [Wearables](../wearables.md), [Fit to a reference](../reference-fit.md), [VRM avatars](../vrm.md).
