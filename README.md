<p align="center">
  <img src="docs/figures/opensculptboy_banner.svg" alt="OpenSculptBoy: open humanoid characters for free, unlimited creativity, built on the Anny body model" width="900" style="display:block;width:100%;max-width:900px;margin:auto"/>
</p>

# OpenSculptBoy

[![Tests](https://github.com/infinia-yzl/opensculptboy/actions/workflows/tests.yml/badge.svg)](https://github.com/infinia-yzl/opensculptboy/actions/workflows/tests.yml)

OpenSculptBoy is an open toolkit for humanoid 3D characters. You shape a character with sliders, pose it, and export it as a file that game engines, 3D tools and VTuber apps open.

It is built on [Anny](https://github.com/naver/anny), the parametric body model of NAVER LABS Europe.

> **Built and maintained by AI agents.** AI coding agents ([Claude Code](https://claude.com/claude-code)) write and maintain everything that this repository adds to Anny. One human author ([infinia-yzl](https://github.com/infinia-yzl)) sets the goals and publishes the repository. Agents make mistakes, so review and test the code before you rely on it. The project comes without any warranty (see [Disclaimer](#disclaimer-and-no-warranty)).

## From sliders to avatar

1. **Shape**: sculpt a body and a face with sliders, or roll a random face. Add one of 25 hairstyles. [Web viewer](docs/viewer.md) · [Body model](docs/body-model.md)
2. **Pose**: pick from 50 poses and 7 animation clips, or copy the pose of a photo or a drawing. [Pose from a picture](docs/pose-from-picture.md)
3. **Bring your own**: fit the body to a 3D character that you already have, and keep its fine shape, skin, eyes, teeth and mouth movements. [Fit to a reference](docs/reference-fit.md)
4. **Take it anywhere**: export a `.glb` file for Blender, Godot, Unity, Unreal Engine or three.js, or a `.vrm` avatar for VTuber apps, at the level of detail that you need. [glTF export](docs/export.md) · [VRM avatars](docs/vrm.md) · [Budget tiers](docs/budget-tiers.md)

## How it works

Anny turns a few numbers into a body mesh with a skeleton: shape sliders, face shapes and a pose. OpenSculptBoy stores those numbers, with any fitted detail and textures, in a **character card** (a JSON file), and its exporters turn a card into a `.glb` or `.vrm` file. The [character card](docs/character-card.md) page lists every field.

## Install

```bash
pip uninstall anny   # both distributions install the `anny` package
pip install "opensculptboy @ git+https://github.com/infinia-yzl/opensculptboy.git"
```

Extras add optional parts: `[viewer]` for the viewer build and the fit, `[pose]` for the pose from a picture, `[examples]` for the tutorials, and `[smpl]` for the SMPL and SMPL-X wrappers (non-commercial data). In a clone of this repository, run `uv sync --extra examples`.

## Quick start

Open `viewer/dist/anny_viewer.html` in a browser to shape a character.

From the command line:

```bash
opensculptboy character > ada.json                                      # a character card; edit its values
opensculptboy export ada.glb --character ada.json --animation walk
opensculptboy export ada.vrm --character ada.json --author "Your name"
```

From Python:

```python
from opensculptboy import Character, export_glb

export_glb("ada.glb", Character(name="ada", phenotype={"muscle": 0.7}), animations=["walk"])
```

## Documentation

| Guide | For |
|---|---|
| [Web viewer](docs/viewer.md) | shaping and posing a character in the browser |
| [glTF export](docs/export.md) and [VRM avatars](docs/vrm.md) | getting a character into engines, 3D tools and VTuber apps |
| [Pose from a picture](docs/pose-from-picture.md) | posing a character from a photo or a drawing |
| [Fit to a reference](docs/reference-fit.md) | rebuilding an existing 3D character on Anny's body |
| [Budget tiers](docs/budget-tiers.md) | exporting one character at several levels of detail |
| [Body model](docs/body-model.md) | the Python API: shapes, poses, rigs, topologies, hair and caching |
| [Character card](docs/character-card.md) | the card format |
| [Wearables](docs/wearables.md) | dressing a character: wearable packages, outfits, and the clothing of a reference character |
| [Eyes](docs/eyes.md), [Teeth](docs/teeth.md), [Mouth](docs/mouth.md), [Sculpt topology](docs/sculpt-topology.md) | how each part is built |
| [CHANGELOG](CHANGELOG.md) | what changed, Anny's releases included |

## Credits

OpenSculptBoy builds on:
- **[Anny](https://github.com/naver/anny)** by Romain Brégier, Guénolé Fiche, Laura Bravo-Sánchez, Thomas Lucas, Matthieu Armando, Philippe Weinzaepfel, Grégory Rogez and Fabien Baradel at [NAVER LABS Europe](https://europe.naverlabs.com/) (Apache License 2.0, Copyright (c) 2025 NAVER Corp.). The body model, its parameters, topologies and rigs, and the `anny` package come from Anny. If you use OpenSculptBoy in research, please cite the Anny paper (see [Citation](#citation)).
- **[MakeHuman](https://static.makehumancommunity.org/)** and **[MPFB2](https://github.com/makehumancommunity/mpfb2/)**, whose CC0 assets (targets, rigs, poses and the face units of Mika Suominen) are the base of Anny.
- **[SOMA-X](https://github.com/NVlabs/SOMA-X)** (NVIDIA), for the `soma` rig and topology.
- **[ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit)**, MediaPipe, and the datasets listed in [LICENSE_THINGS](LICENSE_THINGS) and `src/anny/data/faces/SOURCES.md`, for the face shapes.

These authors have not reviewed or endorsed OpenSculptBoy.

## License

The code of Anny (Copyright (c) 2025 NAVER Corp.) and the changes in this repository use the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The bundled data keeps its own licences: CC0 for the MakeHuman, MPFB2 and Face Units assets, Apache 2.0 for SOMA-X and the MediaPipe landmarks, and MIT for ICT-FaceKit. [LICENSE_THINGS](LICENSE_THINGS) and `src/anny/data/faces/SOURCES.md` list each source. The SMPL and SMPL-X data downloads on first use and is for non-commercial use only ([its licence](http://download.europe.naverlabs.com/humans/Anny/noncommercial.zip)).

## Disclaimer and no warranty

- **No warranty.** The software, the data, the viewer and the documentation are provided "AS IS", without warranty of any kind, either express or implied, including any warranty of title, non-infringement, merchantability, fitness for a particular purpose, accuracy, security or uninterrupted operation (Apache License 2.0, section 7).
- **No liability.** To the fullest extent that the law allows, the author of this repository, the AI agents that helped to write it, and the authors of the upstream works named in [Credits](#credits) are not liable for any damage, loss or claim that arises from the use of, or the inability to use, this project. That covers direct, indirect, incidental, special and consequential damages, including lost data, lost profit and business interruption (Apache License 2.0, section 8).
- **AI-written work.** The author does not guarantee that the code, the data or the documentation is correct, original, secure or free of third-party rights. Review and test them for your own purpose.
- **Model numbers only.** Sizes, volumes and masses describe a 3D model. Do not rely on them for medical, safety-critical, legal or identity decisions.
- **Third-party assets.** Bundled and downloaded data keep their own licences (see [LICENSE_THINGS](LICENSE_THINGS)), and the SMPL and SMPL-X data is for non-commercial use only. Check that the licences fit your use.
- **Your content.** You are responsible for the characters, images and products that you create, and for following the laws that apply to you.
- **No affiliation.** OpenSculptBoy is not affiliated with, endorsed by or sponsored by NAVER, MakeHuman, NVIDIA, Anthropic or the other parties named here. Their names and marks belong to their owners.
- **Not legal advice.** This section does not change the [Apache License 2.0](LICENSE), and the license text applies where the two differ. Some rights cannot be excluded by contract in some countries.

## Citation

If you use the body model in research, please cite the Anny paper:

```bibtex
@inproceedings{anny,
      title={Human Mesh Modeling for {A}nny {B}ody},
      author={Romain Br{\'e}gier and Gu{\'e}nol{\'e} Fiche and Laura Bravo-S{\'a}nchez and Thomas Lucas and Matthieu Armando and Philippe Weinzaepfel and Gr{\'e}gory Rogez and Fabien Baradel},
      booktitle="Computer Vision -- ECCV 2026",
      year={2026},
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="170--188",
      isbn="978-3-032-37314-4"
}
```
