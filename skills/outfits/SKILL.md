---
name: outfits
description: Dress an OpenSculptBoy character from a request in words ("a hooded travel cloak over a linen shirt and brown trousers"). Use when an agent finds, imports, binds, assembles, checks, tunes or exports wearables and outfits with the opensculptboy wearables commands in a checkout of this repository.
---

# Outfits

Every step is a command that prints one JSON result on stdout. Read `ok`; on `false`, act on `error.code` and follow `error.hint`. Run the commands with `uv run opensculptboy ...` from the checkout. [docs/wearables.md](../../docs/wearables.md) holds the full reference.

## Workflow

1. **Find pieces.** `opensculptboy wearables catalogue wear/ --query "linen shirt" --slot torso` lists the packages of a folder: `name`, `label`, `slot`, `layer`, `tags`, `description`, `colours` (hex per material), `groups`, `triangles`, `licence`, `quality` and `thumbnail`. Match the request's words against `tags` and `description`, and the colours against `colours[].hex`. `--mhclo packs/` adds the MakeHuman assets not imported yet (`kind: "mhclo"`, with the `import` command). `--thumbnails` renders `NAME.thumb.png` next to each package; look at it before choosing.
2. **Make missing packages.** A MakeHuman asset: `opensculptboy wearables import-mhclo FILE.mhclo --out wear/NAME --tag WORDS --description TEXT`. A garment modelled on `base.glb`: `opensculptboy wearables bind FILE.glb --out wear/NAME --slot SLOT --layer N --tag WORDS --description TEXT`. Give every new package tags and a one-line description, so that later searches find it. Use `--dry-run` first when unsure.
3. **Assemble.** `opensculptboy wearables add card.json wear/shirt.wearable.json wear/trousers.wearable.json --outfit travel --label "Travel clothes" --activate`. An outfit file goes on a card alone: `opensculptboy wearables add card.json travel.outfit.json --activate`.
4. **Check.** `opensculptboy wearables check card.json --sheet card.check.png`. Exit status 0: every gate passes; 1: a gate fails; 2: an error. Read `gates[]` (`name`, `pass`, `value`, `limit`, `worst.body`, `worst.state`, `worst.frame`, `worst.region`, `worst.piece`), then open the sheet: red dots mark skin through a piece, orange dots a lower piece through an upper one. For quick rounds pass `--bodies card,default --clips walk --frames 2`; run the full check before an export.
5. **Tune**, then check again (see the table below).
6. **Export.** `opensculptboy export out.glb --character card.json` and `opensculptboy export out.vrm --character card.json --author NAME`. The exports carry the credit line of every piece.
7. **Share.** `opensculptboy wearables outfit export card.json --outfit travel --out travel.outfit.json` writes the outfit with paths relative to the file.

## Acting on the results

| Finding | Action |
|---|---|
| `skin_rest` fails, `worst.region` names where | the piece is too tight or its cover mask is too wide: rebind with more room, or bind with `--covers auto` |
| `layers_rest` fails, `worst.piece` is `LOWER through UPPER` | raise the upper piece's layer (`overrides.layer`), or add a hides rule that hides the lower piece's group under the upper one |
| `skin_clips` or `skin_tpose` fails at one region | a joint bends through a tight piece; report it with the sheet, or loosen the piece there |
| `quality` fails | `pieces[].quality.items` lists each item with its `hint`; fix the source file and import or bind again |
| `piece_exists`, `outfit_exists`, `file_exists` | nothing was changed; pass `--replace` only when the user wants the old one gone |
| `unknown_slot` | pass `--slot` with one of the slots of docs/wearables.md |
| `too_far_from_skin` | the garment was modelled on another body or in another frame; model it on `base.glb` |
| `private_in_repository` | write the private files into a folder outside the checkout |

Tune in the card's outfit pieces or in an outfit file: `overrides` (`layer`, `label`), `hides` (rules toward other pieces' groups) and `springs`. Each card command takes `--dry-run`; a second run with the same arguments reports `changed: false`.

## Gates

- `quality`: no item of error severity in a worn piece.
- `skin_rest`, `layers_rest`: zero vertices more than 2 mm outside the piece that covers them, on every test body at rest.
- `skin_tpose`, `layers_tpose`, `skin_clips`, `layers_clips`: at most 1 % of the covered vertices (`--pose-limit`).
- The test bodies are the card's own body, the default body and the lowest and highest weight, muscle and height (`--bodies`).

## Licences and provenance

- Licences are the user's call. Record what the source says (`--licence`, `--author`, `--source`, `--design-source`); never refuse or drop a piece because of its licence. The exports write a credit line for every piece with a recorded licence, author or source; keep those lines when sharing files.
- A package made from someone's own files is `provenance: private`. Private packages, the cards and outfit files that list them, their sheets and their exports stay outside every checkout of this repository; the commands refuse such writes with `private_in_repository`.
