# OpenSculptBoy
# Apache License, Version 2.0
"""
Registration of anny to a reference mesh of any topology, in three stages.

1. **Placement.** With landmarks, a rigid alignment of anny's landmarks onto the reference's
   (Umeyama 1991, :func:`anny.faces.authoring.nicp.umeyama`); without them, the centres of
   the bounding landmarks of :func:`~opensculptboy.fit.reference.auto_landmarks`. The height
   phenotype starts at the value whose stature matches the reference.
2. **Parameters.** Adam over the phenotypes, the local changes, the face shapes, the pose of
   every bone but the root and the eyes, and a rigid placement, in rounds: each round finds the
   closest points again and runs a few steps on

   - point-to-surface distances both ways (anny's vertices to the reference surface, point to
     plane; a sample of the reference's vertices to anny's nearest vertices), with pairs too far
     apart or facing away from each other left out;
   - the landmarks (:data:`~opensculptboy.fit.reference.LANDMARKS`, weighted);
   - small ridge penalties on the local changes, the face shapes and the bone turns.

   The first rounds free the placement, the phenotypes and the large joints alone.
3. **Non-rigid refinement.** Anny's whole mesh moves onto the reference by a non-rigid ICP in
   the spirit of optimal-step NICP (Amberg et al., CVPR 2007), as in
   :func:`anny.faces.authoring.nicp.refine`: closest points pull the vertices, and a Laplacian
   term keeps the displacement smooth with a stiffness that decreases over the rounds. Region
   masks leave out the eyeballs, the eye sockets and the inside of the mouth, which follow the
   skin around them. With an ear mask (``masks.ear`` of the handoff), :class:`EarSplit` fits
   anny's outer ears (:func:`pinna`) to the reference's ears alone and the rest to the rest,
   in both stages, with the root of the ears on the attachment line, and :func:`fair_region`
   and :func:`repair_folds` fair the ears on the reference. Each level of the detail layer
   then repeats the refinement on its subdivided mesh, starting from the level below.

The residuals (:mod:`opensculptboy.fit.residual`) of each level become the card's detail layer:
:func:`fit_reference` returns a :class:`FitResult`, and :meth:`FitResult.write` writes the card
and its offsets files.

The fit needs SciPy (the ``viewer`` or ``examples`` extra).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np
import torch

from opensculptboy.character import Character
from opensculptboy.fit import residual as R_
from opensculptboy.eyes.geometry import CORNEA_GAP_MM, LID_CLEARANCE_MM
from opensculptboy.fit.reference import (
    LANDMARKS,
    Reference,
    auto_landmark_vertices,
    fit_sphere,
    vertex_normals,
)

# Bones whose turns the parameter stage leaves at rest.
FIXED_BONES = ("root", "eye.L", "eye.R")
# Words of the bones that the first rounds free: the trunk and the large joints.
MAJOR_BONES = (
    "spine",
    "neck",
    "head",
    "clavicle",
    "shoulder",
    "upperarm",
    "lowerarm",
    "wrist",
    "upperleg",
    "lowerleg",
    "foot",
)
# Words of the finger bones, which :attr:`FitOptions.finger_rounds` keeps at rest at first.
FINGER_BONES = ("finger", "metacarpal", "thumb")
# Base-mesh face labels that the registration leaves out (with the eyeballs).
EXCLUDED_LABELS = ("eye_cavity.L", "eye_cavity.R", "mouth_cavity")
# The roles of the new vertices of the sculpt patches in the registration, by patch label (a
# label or the prefix of finer labels, see opensculptboy.topology.lookup.labelled; the last
# matching entry wins): "fit" (fitted as skin), "exclude" (left out and moved by follow_skin,
# as the eye sockets) or "anchors" (left out, and placed after each level's refinement at the
# blend of its anchors that defines it, see PatchAnchors). A new vertex without an entry
# takes the base-mesh label of its strongest anchor. The eye-corner patch's caruncle and fold
# of the conjunctiva lie inside the corner, where the reference shows the socket's tissue or
# nothing: fitted as skin they crumple, and moved by follow_skin each vertex takes the
# displacement of other skin and is pushed behind it, which tears the corner into flaps; at
# the blend of their anchors they keep their designed shape between the fitted lids (its new
# vertices all carry the label "eye_corner"; the finer labels name its faces).
PATCH_ROLES: dict[str, str] = {
    "eye_corner/caruncle": "anchors",
    "eye_corner/conjunctiva": "anchors",
    "eye_corner": "anchors",
}
PATCH_ROLE_NAMES = ("fit", "exclude", "anchors")
# the vertices of an "anchors" patch's region that lie this far from an eyeball's centre at
# rest (in eyeball radii) or nearer move with the eyeball (PatchAnchors)
ATTACH_RADIUS = 0.99
# the rings of fitted skin around an "anchors" patch's faces and anchors that also take the
# harmonic extension in place of their fitted offsets (PatchAnchors)
SKIN_RINGS = 0
# seat_lids: how far inside the reference's eyeball sphere (in its radii) the exported
# eyeball's own sphere may lie; outward-facing skin within it counts as touching the eyeball
SPHERE_TOLERANCE = 0.005
# The topologies that the fit takes (FitOptions.topology).
FIT_TOPOLOGIES = ("anny", "sculpt")


@dataclasses.dataclass
class FitOptions:
    """Settings of :func:`fit_reference`; the defaults suit a full figure in metres."""

    # the mesh of the card: "anny", or "sculpt" (anny's mesh with the patches of
    # opensculptboy.topology); the card and its offsets take it
    topology: str = "anny"
    rounds: int = 10
    steps: int = 30
    major_rounds: int = (
        3  # rounds that free only the placement, phenotypes and large joints
    )
    # rounds that keep the fingers at rest (at least the major rounds): until the distance
    # limit is short, so that a finger does not slide onto its neighbour
    finger_rounds: int = 7
    reverse_samples: int = 20000
    landmark_weight: float = 1.0
    auto_landmark_weight: float = 0.2
    # ridge penalties (per squared unit, summed over the parameters)
    local_change_ridge: float = 1e-7
    face_shape_ridge: float = 1e-7
    pose_ridge: float = 1e-6
    # correspondences farther apart than this (metres) are left out, first and last round
    distance_limits: tuple = (0.05, 0.01)
    normal_threshold: float = 0.2
    # NICP stiffness per stage, level 0 and the subdivided levels
    stiffness: tuple = (10.0, 3.0, 1.0, 0.3)
    detail_stiffness: tuple = (1.0, 0.3, 0.1, 0.03)
    nicp_iterations: int = 3
    nicp_max_distance: float = 0.02
    # the stiffness of the refinement on the outer ears, relative to the rest (ear_weights)
    ear_stiffness: float = 3.0
    # rounds of tangential relaxation of the outer ears after each refinement (0: none)
    ear_relax_iterations: int = 10
    # after each refinement, the vertices within this distance (metres) of the reference
    # whose normals agree to project_normal_threshold move onto it, but on the ears (0: none)
    project_distance: float = 0.002
    project_normal_threshold: float = 0.8
    # then the skin of the head (the regions of skin_fair_regions, but the outer ears and the
    # skin the ear fairing takes, the lips' inner margin and lining, and the patches placed by
    # their anchors) is faired on the reference (fair_skin): the fairness of each level (see
    # _per_level; 0: none), the rounds, and the distance (metres) within which a vertex takes
    # its closest point. A coarse reference looks smooth only through the normals it shades
    # with, and a fine level projected onto its flat triangles copies their facets as steps
    skin_fairness: tuple = (0.0, 4.0, 160.0)
    skin_fair_iterations: int = 2
    skin_fair_distance: float = 0.002
    skin_fair_regions: tuple = ("face", "jaw", "neck")
    # the fairing also leaves out the lids and eye corners that seat_lids places, and this
    # many rings of skin around them on level 0 (twice as many on each finer level)
    skin_fair_lid_rings: int = 1
    # ... and the skin within skin_fair_lid_blend_rings rings of them (on level 0, twice as
    # many on each finer level) is faired with a hold that falls from
    # this weight on the first ring to 0 past the last (lid_blend; 0: held fast), so that the
    # faired skin joins the lids without a step; one value, or one per level (on level 1 the
    # blend turned more faces of the synthetic target's eye corners over than anny's own)
    skin_fair_lid_blend: float | tuple = (0.0, 0.0, 50.0)
    skin_fair_lid_blend_rings: int = 1
    # the data weight of the lips' inner margin: the vertices within lip_margin_rings faces
    # of the inside of the mouth, which follow their neighbours when it is 0
    lip_margin_weight: float = 0.0
    lip_margin_rings: int = 1
    # the offsets of the inside of the mouth and of the lips' inner margin on each level:
    # "smooth" (the harmonic extension of the offsets of the skin around them, see
    # smooth_mouth) or "follow" (each vertex with the nearest fitted skin in space, as the
    # eye sockets; with closed lips the upper and the lower lip's linings take each other's
    # displacement and tear apart when the mouth opens)
    mouth_lining: str = "smooth"
    # rounds of repair_mouth on each level (0: none): the tears of the mouth that the offsets
    # add to the parametric mesh, smoothed away in the rest frame
    mouth_repair_rounds: int = 3
    # the vertices that repair_mouth may move: those within mouth_repair_reach faces of the
    # inside of the mouth (the lips' hidden roll; None: the whole mouth of metrics.mouth_zone,
    # which moved the visible lips off a real target by up to 1.2 mm at the 95th percentile),
    # each tear with mouth_repair_rings rings of its neighbours; one value, or one per level
    # (_per_level): on level 2 it reaches the skin around the mouth's corners, where anny's
    # lips fold into each other and the fit left a fan of creases (on level 1 the same reach
    # deepened the corners' pockets)
    mouth_repair_reach: int | None | tuple = (5, 5, 8)
    mouth_repair_rings: int = 1
    # the vertices of the lips (outside the inside of the mouth) that are each other's nearest
    # vertex of the other lip within this distance (metres) on the parametric mesh (lip_contact, the lips split by
    # jaw_share) meet at the mean of their fitted positions (stitch_lips), so that the closed
    # lips close, and repair_mouth holds them (0: no stitch, the default: on a real replica
    # 4 mm found 7 to 34 pairs per level, too few to close the slit, and on the synthetic
    # target it doubled the mouth's tears)
    lip_contact: float = 0.0
    # repair_mouth also holds the vertices of the mouth within this distance (metres) of the
    # reference whose normals agree with it: the visible lips (0: none held)
    mouth_repair_hold: float = 0.0
    # seal the closed lips (opensculptboy.fit.lips): the rows where anny's upper and lower lips
    # end move onto the line where the reference's lips meet (its mouth opening, or the bottom
    # of the crease between its lips), and the lips within lip_seal_rings faces (on level 0,
    # twice as many on each finer level) follow; repair_mouth then holds those rows
    lip_seal: bool = True
    lip_seal_rings: int = 3
    # the skin of the mouth that lies on the reference stays where the fit put it, but within
    # this distance (metres) of the sealed rows; one value, or one per level: on level 2 the
    # wider reach lets the seal's move spread over more skin around the corners
    lip_seal_reach: float | tuple = (0.0015, 0.0015, 0.003)
    # move the vertex nearest each corner landmark onto the line's end as well
    lip_seal_corners: bool = True
    # and the rows of both lips within this distance (metres) past the line's ends, onto the
    # line where the lips meet carried on past its end, so that the corners close
    lip_seal_beyond: float = 0.002
    # seal only lips that the reference closes: no seal when the reference's lips stand
    # further apart than this (metres, the median of its line), as a reference cut open at
    # its mouth does, and none on a level whose rows would move further than
    # lip_seal_max_shift (metres, their mean) to reach the line
    lip_seal_max_gap: float = 0.001
    lip_seal_max_shift: float = 0.004
    # after the seal and the repair, the skin within lip_corner_radius (metres) of each end of
    # the reference's lip line (the mouth's visible corners), outside the inside of the mouth,
    # is faired on the reference (fair_skin) with this fairness on each level (_per_level; 0:
    # none): the skin fairing leaves the lips' roll out, and the closed mouth showed short lines
    # beside its corners, the reference's facets and the folds of anny's lips meeting the
    # cheek. The closest points count only where the normals agree to
    # lip_corner_normal_threshold (at 0.5 the walls of the reference's corner pocket drew the
    # corner into a pit), and the sealed rows within the radius are held by lip_corner_hold
    # (0: free, which on a real replica left the fewest lines and kept the lips closed; a held
    # row ended in a fan of creases)
    lip_corner_fairness: float | tuple = (0.0, 40.0, 160.0)
    lip_corner_radius: float = 0.004
    lip_corner_hold: float = 0.0
    lip_corner_normal_threshold: float = 0.8
    # which family fits the MakeHuman targets that local changes and face shapes share:
    # "face_shapes" (the local changes on them stay at 0), "local_changes" or "both"
    shared_targets: str = "face_shapes"
    # experimental, off by default (see docs/reference-fit.md, "Measuring a fit"): the
    # surface of the refinement: the Phong tessellation of the reference's triangles
    # with this share (0: the flat triangles, whose facets a fine level would copy), curved
    # by the normals the source shades with when target_shading (else its geometric normals)
    target_smooth: float = 0.0
    target_shading: bool = False
    # "vertex" (the nearest fitted vertex of each reference vertex) or "surface" (its closest
    # point on the fitted triangles, spread over their corners); see nicp
    reverse: str = "vertex"
    # the normal agreement of the correspondences on the outer ears (ear_weights over 0.5),
    # where the front and the back of a thin rim face apart
    ear_normal_threshold: float = 0.2
    # the ears of a reference with an "ear" region (masks.ear of the handoff; see EarSplit):
    # anny's outer ears (pinna, those vertices more than pinna_height off the head) fit the
    # reference's ears alone and the rest the rest, the root of the ears takes the attachment
    # line with this share of the data weight, and the ears are faired on the reference
    # (fair_region) and their folds repaired (repair_folds) in place of ear_relax_iterations.
    # False: the ears fit as without the region
    ear_split: bool = True
    pinna_height: float = 0.002
    # the stiffness of the outer ears with the split, in place of ear_stiffness: the fairing
    # and the repair keep the folds, and a stiffer ear reaches a pointed tip less
    ear_split_stiffness: float = 1.0
    ear_attachment_weight: float = 1.0
    ear_attachment_reach: float = 0.0
    ear_fair_iterations: int = 5
    ear_fairness: float = 1.0
    # the fairing also takes this many rings of the head's skin around the ears, on level 0
    # (twice as many on each finer level), which then matches the reference's head alone
    ear_fair_rings: int = 2
    ear_repair_rounds: int = 3
    # on each detail level, the mask of anny's outer ears moves within this many rings of its
    # border (twice as many on each finer level) to where the mesh crosses the attachment
    # line (relabel_ears; 0: the pinna of level 0 throughout)
    ear_relabel_rings: int = 3
    # after each level, the lids and the inner corner of each eye onto what the reference
    # shows there: its skin outside its eyeball's sphere, then the sphere (seat_lids), so that
    # they meet the eyeball flush. The region: the fitted skin within lid_reach eyeball radii
    # of the centre and lid_front degrees of the gaze, and the left-out vertices (the socket's
    # lining, the eye-corner patch) within lid_lining_reach; the sphere target lies at
    # lid_sphere of the radius, and the reference's skin counts as outside the sphere beyond
    # 1 + lid_margin radii. Needs the reference's eyeball spheres (Reference.eyes)
    seat_lids: bool = True
    lid_reach: float = 1.35
    lid_lining_reach: float = 1.6
    lid_front: float = 100.0
    lid_sphere: float = 0.99
    lid_margin: float = 0.005
    lid_iterations: int = 10
    # the fairness of seat_lids on level 0, 16 times more on each finer level (the
    # bi-Laplacian of a finer mesh is smaller by the fourth power of its edges)
    lid_fairness: float = 20.0
    lid_weight: float = 10.0
    # the rounds in which seat_lids draws the skin out of the eyes' clearance band within its
    # faired solve, per level (0: moved radially after the solve alone). Level 2's vertices
    # lie closer together than the band is wide, and moved alone, neighbours along the lid's
    # edge stepped across it; on the coarser levels the step does not show, and on the
    # synthetic target the faired version turned more of the corner's faces over on level 1
    lid_clearance_rounds: tuple = (0, 0, 4)
    # per level: whether seat_lids follows the reference's skin where it goes under its
    # eyeball's sphere (lid_follow_inside), and whether the clearance band also moves the
    # skin that faces the eyeball (lid_clearance_inward). On level 2, skin facing the
    # eyeball that the band moved under it stood a ring away from the lid at 1.02 to 1.05
    # radii, a crease along the lower lid at the inner corner; the reference's own skin faces
    # the eyeball there and goes under it. On level 1 following it raised the corner's
    # reverse error (p95), and the band's move shows no crease
    lid_follow_inside: tuple = (False, False, True)
    lid_clearance_inward: tuple = (True, True, False)
    # the vertices of the fit model's level-0 mesh, as a (V,) bool mask, that take no data:
    # the parts that the reference lacks, such as a statue's lost hand. They keep anny's
    # shape and follow the fitted skin around them through the stiffness (None: all of the
    # skin takes data)
    no_data: np.ndarray | None = None
    # the pose the parameter stage starts from: (B, 3, 3) rotations of the fit model's bones
    # (a library pose for a reference that stands far from anny's rest; the fixed bones keep
    # theirs); None: the rest pose
    initial_pose: np.ndarray | None = None
    # a solid figure (a statue: carved eyes and closed lips): after each level, the linings
    # of the eye sockets and the mouth within this distance (metres) of the reference move
    # onto it and close the openings, and anny's eyeballs shrink by eyeball_shrink about
    # their centres behind them (0: openings kept)
    close_openings: float = 0.0
    eyeball_shrink: float = 0.6
    # the measures of metrics.measure, before and after the projection, in FitResult.measures
    measure: bool = True
    seed: int = 0
    verbose: bool = False


@dataclasses.dataclass
class FitResult:
    """What a fit gives: the card, the residual offsets of each level and the errors."""

    character: Character
    rotation: np.ndarray  # (3, 3) anny -> reference: x_ref = R x + t
    translation: np.ndarray  # (3,)
    parametric_rest: np.ndarray  # (V0, 3) rest mesh of the card's parameters
    offsets: dict  # level -> (N_level, 3) rest-frame offsets
    errors: dict  # "parametric" and each level -> surface_error() dict
    seconds: dict  # time of each stage
    reference: Reference | None = None
    # "parametric" and each level -> {"before_projection", "final"} -> Measures.summary()
    # (opensculptboy.fit.metrics: unmasked, signed, per region); empty without options.measure
    measures: dict = dataclasses.field(default_factory=dict)
    meshes: dict = dataclasses.field(
        default_factory=dict
    )  # level -> (vertices, triangles)
    # the share of the jaw on each vertex of level 0 and the reference's poses it comes from
    # (opensculptboy.mouth.weights.attach_jaw_share), or None
    jaw_share: np.ndarray | None = None
    jaw_share_poses: tuple = ()

    def write(self, card_path: str | pathlib.Path, textures: dict | None = None):
        """
        Write the card (JSON) and, next to it, ``<card>.detail<N>.safetensors`` for each level.
        ``textures`` maps texture slots to files already written next to the card.
        """
        from opensculptboy.detail import save_offsets

        card_path = pathlib.Path(card_path)
        stem = (
            card_path.name[: -len(card_path.suffix)]
            if card_path.suffix
            else card_path.name
        )
        levels = []
        for subdiv in sorted(self.offsets):
            name = f"{stem}.detail{subdiv}.safetensors"
            save_offsets(
                card_path.parent / name,
                self.offsets[subdiv],
                subdiv,
                self.character.topology,
                {"source": "opensculptboy fit"},
            )
            entry = {"subdiv": int(subdiv), "offsets": name}
            if subdiv in self.errors:
                entry["error_mm"] = round(self.errors[subdiv]["forward"]["mean"], 4)
            levels.append(entry)
        detail = {
            "levels": levels,
            "fit": {
                "placement": {
                    "rotation": np.round(self.rotation, 9).tolist(),
                    "translation": np.round(self.translation, 9).tolist(),
                    "frame": "x_reference = rotation @ x_anny + translation (Anny's axes)",
                },
                "errors_mm": {
                    str(k): {d: round(v[d]["mean"], 4) for d in ("forward", "reverse")}
                    for k, v in self.errors.items()
                },
                "seconds": {k: round(v, 1) for k, v in self.seconds.items()},
            },
        }
        if textures:
            detail["textures"] = dict(textures)
        if self.jaw_share is not None:
            from opensculptboy.mouth.weights import write_mouth_file

            name = f"{stem}.jaw_share.safetensors"
            write_mouth_file(
                card_path.parent / name, self.jaw_share, self.jaw_share_poses
            )
            detail["mouth"] = {
                "jaw_share": name,
                "poses": list(self.jaw_share_poses),
            }
        self.character.detail = detail
        card_path.write_text(
            json.dumps(self.character.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return card_path


def _log(options: FitOptions, message: str) -> None:
    if options.verbose:
        print(message, flush=True)


def fit_model(topology: str = "anny"):
    """The Anny model of the fit (float64, every local change and face shape, LBS) on
    ``topology``: ``"anny"`` or ``"sculpt"`` (:data:`FIT_TOPOLOGIES`)."""
    from opensculptboy.topology.sculpt import build_model

    if topology not in FIT_TOPOLOGIES:
        raise ValueError(
            f"The fit takes the topologies {', '.join(FIT_TOPOLOGIES)}; got {topology!r}."
        )
    return build_model(
        topology,
        local_changes="all",
        face_shapes="all",
        facial_actions="none",
        skinning_method="lbs",
    ).to(dtype=torch.float64)


def _fit_topology(model) -> str:
    from opensculptboy.topology.lookup import model_topology

    return model_topology(model)


def excluded_vertices(model) -> np.ndarray:
    """Mask (V,) of the vertices that the registration leaves out: the eye sockets, the inside
    of the mouth, and every closed part apart from the body (the eyeballs and the tongue).

    On ``sculpt``, a new vertex takes the base-mesh label of its strongest anchor, unless its
    patch label has a role in :data:`PATCH_ROLES` (``"exclude"`` and ``"anchors"`` leave it
    out), and anny's vertices that no face uses any
    more (inside the removed faces) are left out as closed parts of their own."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.export.vrm_tables import eyeball_vertices
    from opensculptboy.topology.lookup import base_mesh_rows

    mesh = base_mesh()
    base = base_mesh_rows(model)
    out_base = np.zeros(len(mesh.V), bool)
    for label in EXCLUDED_LABELS:
        out_base[np.unique(mesh.quads[mesh.face_labels == label])] = True
    excluded = out_base[base]
    roles = patch_roles(model)
    excluded[roles != ""] = np.isin(roles[roles != ""], ("exclude", "anchors"))
    try:
        for rows in eyeball_vertices(model).values():
            excluded[rows] = True
    except ValueError:
        pass
    faces = model.faces.cpu().numpy()
    n = len(excluded)
    edges = np.concatenate(
        [faces[:, [k, (k + 1) % faces.shape[1]]] for k in range(faces.shape[1])]
    )
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    excluded |= labels != np.argmax(counts)
    return excluded


def patch_roles(model) -> np.ndarray:
    """(V,) the role of each vertex in :data:`PATCH_ROLES` (``""`` for anny's vertices and
    for new vertices without an entry); the last matching entry wins."""
    from opensculptboy.topology.lookup import labelled

    roles = np.full(len(model.template_vertices), "", dtype=object)
    for label, role in PATCH_ROLES.items():
        if role not in PATCH_ROLE_NAMES:
            raise ValueError(f"Unknown patch role {role!r} of {label!r}.")
        roles[labelled(model, label)] = role
    return roles


class PatchAnchors:
    """
    The rule of the patch role ``"anchors"``: each new vertex of such a patch is its fixed
    convex blend of anny vertices (its anchors, :func:`opensculptboy.topology.lookup.vertex_anchors`),
    in the rest frame of the card: its detail offset is the same blend of its anchors' offsets,
    so that its rest position is the blend of their rest positions, as the model defines it.

    Most anchors of the eye-corner patch are vertices that the fit leaves out (anny's pocket,
    which no face uses on ``sculpt``, and the lining of the socket), and the patch spans from
    the lid margins to the eyeball. Their offsets come from the two things the patch lies
    between:

    - the vertices that lie inside the eyeball's sphere at rest (:data:`ATTACH_RADIUS`) among
      the left-out vertices joined to the anchors (``free``) and anny's vertices of the patch's
      faces (``attached``) move with the eyeball: the similarity (a scale about its centre and
      a translation) that takes its fitted sphere at rest from before to after the fit, so
      that the fold of the patch stays tucked under an eyeball that the fit moves or scales;
    - the other free vertices, and the fitted skin of the patch's faces and of its anchors
      with :data:`SKIN_RINGS` rings around them (``skin``), take a harmonic extension over
      anny's quad mesh of the offsets around them (the fitted skin further out and the
      attached vertices). A blend of anchors that move apart amplifies the steps between
      them, and the fitted skin at the corner has such steps where it meets what the fit
      leaves out; the harmonic field has none, so that the patch keeps its shape.

    These vertices take these offsets too, so that the patch and its anchors agree in the
    written offsets.

    On a detail level the corner starts as the subdivision of the level below (the offsets
    of that level carried up, ``lifted``), and the rule applies to what the level's
    refinement adds to them: on the level's vertex points of anny's vertices (the first rows
    of every level), whose result the subdivision carries to the level's vertices that come
    from the patch, the free and the attached vertices (``share`` over 0), in full to those
    the refinement leaves out and in proportion to their share to those it fits
    (``fitted``). So the patch of a level is the subdivision of the patched quads of the
    level below, moved as the skin and the eyeball around it moved on this level.
    On a detail level the rule places every vertex that it touches (the skin along the
    patch's border included) in full; :func:`fit_levels` still fits them first, so that the
    reference's points over the corner find them and do not drag the skin around them in.
    """

    def __init__(self, model, excluded: np.ndarray, rest: np.ndarray | None = None):
        from scipy.sparse import coo_matrix, csc_matrix, diags
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse.linalg import splu

        from opensculptboy.export.vrm_tables import eyeball_vertices
        from opensculptboy.topology.lookup import model_sculpt_data, vertex_anchors
        from opensculptboy.topology.patch import base_quad_mesh

        self.count = len(model.template_vertices)
        self.rows = np.flatnonzero(patch_roles(model) == "anchors")
        self.free = np.zeros(0, np.int64)
        self.unknown = np.zeros(0, np.int64)
        self.attached = np.zeros(0, np.int64)
        self.written = np.zeros(self.count, bool)
        if len(self.rows) == 0:
            return
        anchors, weights = vertex_anchors(model)
        self.anchors, self.weights = anchors[self.rows], weights[self.rows]
        data = model_sculpt_data(model)
        n = data.vertex_count
        if rest is None:
            rest = model.template_vertices.detach().cpu().double().numpy()
        rest = np.asarray(rest, dtype=np.float64)
        quads = base_quad_mesh().quads
        edges = np.concatenate([quads[:, [k, (k + 1) % 4]] for k in range(4)])
        edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], 1), axis=0)
        left_out = np.asarray(excluded[:n], bool)
        inner = edges[left_out[edges].all(1)]
        graph = coo_matrix(
            (np.ones(len(inner)), (inner[:, 0], inner[:, 1])), shape=(n, n)
        )
        _, component = connected_components(graph, directed=False)
        used = self.anchors[self.weights > 0]
        free = left_out & np.isin(component, component[used[left_out[used]]])
        # anny's vertices of the patch's faces
        faces = np.asarray(data.faces)
        border = np.zeros(n, bool)
        mine = np.isin(faces, self.rows).any(1)
        corners = np.unique(faces[mine])
        border[corners[corners < n]] = True
        # the vertices inside an eyeball's sphere move with it
        self.eyes = []
        try:
            eyes = list(eyeball_vertices(model).values())
        except ValueError:
            eyes = []
        attach = np.zeros(n, bool)
        which = np.full(n, -1)
        best = np.full(n, np.inf)
        for k, rows in enumerate(eyes):
            rows = np.asarray(rows)
            centre, radius, _ = fit_sphere(rest[rows])
            d = np.linalg.norm(rest[:n] - centre, axis=1) / radius
            closer = (free | border) & (d < ATTACH_RADIUS) & (d < best)
            which[closer], best[closer] = k, d[closer]
            attach |= closer
            self.eyes.append(rows)
        self.attached = np.flatnonzero(attach)
        self.attached_eye = which[self.attached]
        adjacency = coo_matrix(
            (np.ones(2 * len(edges)), (edges.ravel(), edges[:, ::-1].ravel())),
            shape=(n, n),
        ).tocsr()
        # the fitted skin of the patch's faces, of its anchors and SKIN_RINGS rings around
        # them: the blends of anchors that move apart amplify every step between them
        skin = border.copy()
        skin[used] = True
        for _ in range(SKIN_RINGS):
            skin |= adjacency @ skin.astype(np.float64) > 0
        skin &= ~left_out
        unknown = (free | skin) & ~attach
        index = np.flatnonzero(unknown)
        # a component of unknown vertices without a known neighbour keeps its offsets
        reached = np.zeros(n, bool)
        reached[index] = np.asarray(adjacency[index][:, ~unknown].sum(1)).ravel() > 0
        _, own = connected_components(adjacency[index][:, index], directed=False)
        index = index[np.isin(own, own[reached[index]])]
        if len(index):
            known = np.ones(n, bool)
            known[index] = False
            inside = adjacency[index][:, index]
            degree = np.asarray(adjacency[index].sum(1)).ravel()
            self._solve = splu(csc_matrix(diags(degree) - inside))
            self._fixed = np.flatnonzero(known)
            self._boundary = adjacency[index][:, self._fixed]
        self.unknown = index
        self.free = np.flatnonzero(free)
        self.written[self.rows] = True
        self.written[self.unknown] = True
        self.written[self.attached] = True

    @property
    def active(self) -> bool:
        return len(self.rows) > 0

    def level_zero(self, field: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """The rule on a field (V0, D) of offsets of anny's vertices and the patch's, added to
        the positions ``rest`` (V0, 3): the attached vertices with their eyeball, the other
        free vertices harmonic, the patch's vertices the blend of their anchors."""
        out = np.array(field, dtype=np.float64, copy=True)
        if not self.active:
            return out
        rest = np.asarray(rest, dtype=np.float64)
        for k, rows in enumerate(self.eyes):
            sel = self.attached[self.attached_eye == k]
            if len(sel) == 0:
                continue
            c0, r0, _ = fit_sphere(rest[rows])
            c1, r1, _ = fit_sphere(rest[rows] + out[rows])
            out[sel] = c1 + (rest[sel] - c0) * (r1 / r0) - rest[sel]
        if len(self.unknown):
            out[self.unknown] = self._solve.solve(self._boundary @ out[self._fixed])
        out[self.rows] = np.einsum("nk, nkd -> nd", self.weights, out[self.anchors])
        return out

    def share(self, topology) -> np.ndarray:
        """(N,) the share of each vertex of a level in the vertices the rule writes."""
        return np.asarray(topology.apply(self.written.astype(np.float64)))

    def apply(self, offsets, topology, parametric, fitted=None, lifted=None):
        """The offsets (N, 3) of a level (:class:`~opensculptboy.detail.DetailTopology`) with
        the rule applied: ``parametric`` (N, 3) is the level's parametric rest mesh,
        ``fitted`` (N,) marks the vertices that the level's refinement fits (default: none),
        and ``lifted`` (N, 3) holds the offsets of the level below carried to this level
        (default: zero)."""
        offsets = np.asarray(offsets, dtype=np.float64)
        if not self.active:
            return offsets
        if topology.subdiv == 0 or lifted is None:
            lifted = np.zeros_like(offsets)
        added = offsets - lifted
        base = (np.asarray(parametric) + lifted)[: self.count]
        field = self.level_zero(added[: self.count], base)
        if topology.subdiv == 0:
            out = offsets.copy()
            out[self.written] = field[self.written]
            return out
        share = self.share(topology)
        if fitted is not None:
            share = np.where(np.asarray(fitted, bool), share, share > 0)
        else:
            share = (share > 0).astype(np.float64)
        share = share[:, None]
        return lifted + added + share * (np.asarray(topology.apply(field)) - added)


SHARED_TARGETS = ("face_shapes", "local_changes", "both")


def shared_target_controls(model) -> tuple[np.ndarray, np.ndarray]:
    """
    Masks of the local changes (L,) and the face shapes (F,) that act on a MakeHuman target of
    the other family: each face shape sums the left and right targets of a local change (the
    face shape ``chin-prominent`` and the local change ``chin-prominent-incr`` move the same
    vertices), so fitted together they would split one movement between two controls.
    """
    from anny.models.face_shapes import face_shape_spec

    labels = list(model.blendshape_labels)
    local_targets = {}
    for i, pos in enumerate(model.local_change_labels):
        row = labels.index(f"local_change:{pos}")
        local_targets[i] = {pos, labels[row + 1].split(":", 1)[1]}
    used = set().union(*local_targets.values()) if local_targets else set()
    face_names = set(model.face_shape_labels)
    face_targets = {}
    for p in face_shape_spec():
        if p.name in face_names:
            face_targets[p.name] = {
                t.rsplit("/", 1)[-1] for t in p.positive + p.negative
            }
    on_faces = set().union(*face_targets.values()) if face_targets else set()
    local = np.array(
        [
            bool(local_targets[i] & on_faces)
            for i in range(len(model.local_change_labels))
        ],
        dtype=bool,
    )
    face = np.array(
        [bool(face_targets.get(n, set()) & used) for n in model.face_shape_labels],
        dtype=bool,
    )
    return local, face


def lip_margin(model, excluded: np.ndarray | None = None, rings: int = 1) -> np.ndarray:
    """Mask (V,) of the lips' inner margin: the fitted vertices within ``rings`` faces of the
    inside of the mouth (the base-mesh label ``mouth_cavity``)."""
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.topology.lookup import base_mesh_rows

    mesh = base_mesh()
    base = base_mesh_rows(model)
    cavity = np.zeros(len(mesh.V), bool)
    cavity[np.unique(mesh.quads[mesh.face_labels == "mouth_cavity"])] = True
    cavity = cavity[base]
    if excluded is None:
        excluded = excluded_vertices(model)
    faces = model.faces.cpu().numpy()
    out = cavity.copy()
    for _ in range(rings):
        out[np.unique(faces[out[faces].any(1)])] = True
    return out & ~excluded


def mouth_lining(model, rings: int = 1) -> np.ndarray:
    """Mask (V,) of the inside of the mouth (the base-mesh label ``mouth_cavity``) and the
    lips' inner margin within ``rings`` faces of it."""
    return lip_margin(model, np.zeros(len(model.template_vertices), bool), rings)


def smooth_mouth(
    offsets: np.ndarray, triangles: np.ndarray, lining: np.ndarray
) -> np.ndarray:
    """
    The offsets (N, 3) with the rows of ``lining`` replaced by the harmonic extension of the
    others: each lining vertex takes the mean of its neighbours' offsets, with the skin around
    the lining held. The inside of the mouth then moves with the lips as one smooth sheet in
    the rest frame, so that the mouth's own shapes (jawOpen and the others, which act on the
    rest mesh) open it without tears. A component of the lining that touches no other vertex
    keeps its offsets.
    """
    from scipy.sparse import diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    lining = np.asarray(lining, bool)
    if not lining.any():
        return offsets
    A = _adjacency(len(offsets), triangles)
    rows = np.nonzero(lining)[0]
    border = np.nonzero(~lining)[0]
    A_ll = A[rows][:, rows]
    A_lb = A[rows][:, border]
    touching = np.asarray(A_lb.sum(1)).ravel() > 0
    _, labels = connected_components(A_ll, directed=False)
    anchored = np.isin(labels, np.unique(labels[touching]))
    keep = rows[anchored]
    if len(keep) == 0:
        return np.array(offsets, dtype=np.float64, copy=True)
    A_ll = A[keep][:, keep]
    A_lb = A[keep][:, border]
    degree = np.asarray(A[keep].sum(1)).ravel()
    L = (diags(degree) - A_ll).tocsc()
    out = np.array(offsets, dtype=np.float64, copy=True)
    out[keep] = spsolve(L, A_lb @ out[border]).reshape(len(keep), -1)
    return out


def jaw_share(model) -> np.ndarray:
    """
    (V,) how far each vertex moves when the jaw opens, relative to the most it moves: the
    length of the displacement of anny's ``jawOpen`` face unit (MakeHuman's
    ``faceunits01``, on the base mesh), over its largest. About 1 on the lower lip and the
    chin, 0 on the upper lip; a model needs no facial actions for it.
    """
    from anny.paths import get_anny_root_dir
    from opensculptboy.topology.lookup import base_mesh_values

    path = (
        pathlib.Path(get_anny_root_dir())
        / "data/faceunits01/targets/faceunits/jawOpen.target"
    )
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    index = np.array([int(r[0]) for r in rows])
    length = np.linalg.norm(
        np.array([[float(x) for x in r[1:4]] for r in rows]), axis=1
    )
    from anny.faces.authoring.base_mesh import base_mesh

    table = np.zeros(len(base_mesh().V))
    table[index] = length
    out = np.asarray(base_mesh_values(model, table))
    return out / max(float(out.max()), 1e-12)


def lip_contact(
    parametric: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    distance: float = 0.0015,
) -> np.ndarray:
    """
    Pairs (P, 2) of a vertex of ``upper`` and a vertex of ``lower`` that are each other's
    nearest vertex of the other lip and lie within ``distance`` of each other on the
    parametric mesh (``parametric``, the card without its offsets): the rows where anny's
    closed lips meet, one partner per vertex.
    """
    from scipy.spatial import cKDTree

    up, low = np.nonzero(upper)[0], np.nonzero(lower)[0]
    if len(up) == 0 or len(low) == 0:
        return np.zeros((0, 2), np.int64)
    d_up, k_up = cKDTree(parametric[low]).query(parametric[up])
    _, k_low = cKDTree(parametric[up]).query(parametric[low])
    mutual = k_low[k_up] == np.arange(len(up))
    near = mutual & (d_up < distance)
    return np.stack([up[near], low[k_up[near]]], 1)


def stitch_lips(
    offsets: np.ndarray, parametric: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    """
    The offsets (N, 3) with the two vertices of each pair of :func:`lip_contact` moved to the
    mean of their fitted rest positions (``parametric`` plus the offsets), so that the closed
    lips meet there. Fitted to a closed mouth, the visible lips follow the reference down to
    its crease and the roll behind it takes no data, so that the lips part by a slit through
    which the inside of the mouth shows. The mouth's own shapes part the pair again when it
    opens.
    """
    out = np.asarray(offsets, dtype=np.float64).copy()
    if len(pairs) == 0:
        return out
    a, b = pairs[:, 0], pairs[:, 1]
    mean = 0.5 * (parametric[a] + out[a] + parametric[b] + out[b])
    out[a] = mean - parametric[a]
    out[b] = mean - parametric[b]
    return out


def repair_mouth(
    offsets: np.ndarray,
    parametric: np.ndarray,
    triangles: np.ndarray,
    zone: np.ndarray,
    rounds: int = 3,
    rings: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Repair the tears of the mouth in the rest frame: the vertices of ``zone`` at an edge that
    folds by more than ``metrics.TEAR_ANGLE`` degrees and by ``metrics.TEAR_EXCESS`` more than
    on the parametric mesh (``parametric``, the card without its offsets, whose mouth folds
    only where the lips meet), and ``rings`` rings of their neighbours, take the harmonic
    extension of the offsets around them (:func:`smooth_mouth`); up to ``rounds`` times while
    tears remain. The reference cannot judge these folds: its closed lips meet in a crease of
    their own, which hides a crumple of anny's lining that the mouth's shapes later open.
    Returns the offsets and the number of tearing vertices found first.
    """
    from opensculptboy.export.mesh_checks import adjacent_triangles, bends
    from opensculptboy.fit.metrics import TEAR_ANGLE, TEAR_EXCESS

    T = np.asarray(triangles)
    zone = np.asarray(zone, bool)
    pairs = adjacent_triangles(T)
    pairs = pairs[zone[T[pairs[:, 0]]].all(1) & zone[T[pairs[:, 1]]].all(1)]
    if len(pairs) == 0 or rounds <= 0:
        return np.asarray(offsets, dtype=np.float64), 0
    plain = bends(parametric, T, pairs)
    A = _adjacency(len(offsets), T)
    out = np.asarray(offsets, dtype=np.float64).copy()
    first = None
    for _ in range(rounds):
        bend = bends(parametric + out, T, pairs)
        torn = (bend > TEAR_ANGLE) & (bend - plain > TEAR_EXCESS)
        bad = np.zeros(len(out), bool)
        bad[np.unique(T[pairs[torn]].ravel())] = True
        if first is None:
            first = int(bad.sum())
        if not bad.any():
            break
        for _ in range(rings):
            bad |= A @ bad.astype(np.float64) > 0
        out = smooth_mouth(out, T, bad & zone)
    return out, first or 0


def ear_weights(model) -> np.ndarray:
    """
    A weight (V,) of 1 on the outer ears that falls to 0 over the skin around them: the
    displacement of MakeHuman's ear-translation targets (``l-ear-trans-up`` and
    ``r-ear-trans-up``, which move the outer ear rigidly and fade out around it), relative to
    their largest. Zero everywhere on a model without those local changes.
    """
    labels = list(model.blendshape_labels)
    out = np.zeros(len(model.template_vertices))
    for side in "lr":
        label = f"local_change:{side}-ear-trans-up"
        if label not in labels:
            continue
        d = model.blendshapes[labels.index(label)].detach().double().norm(dim=-1)
        out = np.maximum(out, (d / d.max().clamp_min(1e-12)).cpu().numpy())
    return out


def _adjacency(n: int, triangles: np.ndarray):
    """The vertex adjacency (n, n) of a triangle mesh, as a CSR matrix of ones."""
    from scipy.sparse import coo_matrix

    t = np.asarray(triangles)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def pinna(
    model,
    rest_vertices: np.ndarray,
    ears: np.ndarray | None = None,
    height=0.002,
    root: np.ndarray | None | bool = True,
):
    """
    Mask (V,) of the outer ears proper on anny's level-0 rest mesh: of the vertices that the
    ear-translation targets move in full (:func:`ear_weights` over 0.99, which also hold a
    strip of the head's skin around the root, in front of the ear most), those that stand
    more than ``height`` off the head, with the holes that they enclose (the concha). The
    head's surface around each ear is a quadratic fitted to the skin around it (weights
    between 0.2 and 0.95), in the frame of its mean normal.

    With a root line (``root``: a vertex mask, or True for the line that the ear-root patch of
    the ``sculpt`` topology marks, :func:`opensculptboy.topology.lookup.ear_root_line`), the
    outer ear is instead the part that the line cuts off the head, the line included, so that
    the border of the mask (where the ear split attaches the ear) runs along the line. The
    vertices that no face uses never belong to it.
    """
    from scipy.sparse.csgraph import connected_components

    from opensculptboy.topology.lookup import ear_root_line, referenced_vertices

    V = np.asarray(rest_vertices, dtype=np.float64)
    T = model.faces.cpu().numpy()
    w = ear_weights(model) if ears is None else np.asarray(ears)
    used = referenced_vertices(model)
    core = (w > 0.99) & used
    out = np.zeros(len(V), bool)
    if not core.any():
        return out
    if root is True:
        root = ear_root_line(model)
    elif root is False:
        root = None
    if root is not None:
        return _cut_off_ears(V, T, np.asarray(root, bool) & used, core, used)
    N = vertex_normals(V, T)
    high = np.zeros(len(V), bool)
    for s in (1.0, -1.0):
        side = np.sign(V[:, 0]) == s
        ring = side & (w > 0.2) & (w < 0.95)
        rows = np.nonzero(side & core)[0]
        if ring.sum() < 6 or len(rows) == 0:
            continue
        centre = V[ring].mean(0)
        normal = N[ring].mean(0)
        normal /= np.linalg.norm(normal)
        a = np.cross(normal, [0.0, 0.0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)

        def local(X):
            d = X - centre
            u, v = d @ a, d @ b
            return np.c_[np.ones_like(u), u, v, u * u, u * v, v * v], d @ normal

        M, z = local(V[ring])
        coef = np.linalg.lstsq(M, z, rcond=None)[0]
        M, z = local(V[rows])
        high[rows] = z - M @ coef > height
    A = _adjacency(len(V), T)
    low = np.nonzero(~high)[0]
    _, labels = connected_components(A[low][:, low], directed=False)
    open_ = np.zeros(labels.max() + 1, bool)
    open_[labels[~core[low]]] = True
    out = high.copy()
    out[low[~open_[labels]]] = True
    return out & used


def _cut_off_ears(V, T, root, core, used) -> np.ndarray:
    """The outer ears that a root line cuts off the head: the parts of the mesh without the
    line that hold vertices of the ears' core and none of the head's far skin (the part
    connected to the most vertices is the head and the body), with the line."""
    from scipy.sparse.csgraph import connected_components

    A = _adjacency(len(V), T)
    rest = np.nonzero(used & ~root)[0]
    _, labels = connected_components(A[rest][:, rest], directed=False)
    sizes = np.bincount(labels)
    head = np.argmax(sizes)
    has_core = np.zeros(len(sizes), bool)
    has_core[labels[core[rest]]] = True
    ear = has_core & (np.arange(len(sizes)) != head)
    out = root.copy()
    out[rest[ear[labels]]] = True
    return out


def mask_border(triangles: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The vertices of a vertex mask that share an edge with a vertex outside it."""
    T = np.asarray(triangles)
    mask = np.asarray(mask, bool)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    cross = mask[e[:, 0]] != mask[e[:, 1]]
    out = np.zeros(len(mask), bool)
    out[e[cross].ravel()] = True
    return out & mask


def _border_edges(triangles: np.ndarray) -> np.ndarray:
    """The edges (E, 2), sorted, that only one triangle has."""
    T = np.asarray(triangles)
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    unique, counts = np.unique(e, axis=0, return_counts=True)
    return unique[counts == 1]


def welded_ids(vertices: np.ndarray, tolerance: float = 1e-7) -> np.ndarray:
    """
    One id (V,) per vertex, shared by the vertices that lie within ``tolerance`` of each other:
    the vertices of one point of the surface that a file splits at its UV seams, or at the line
    where two pieces of a mesh are stitched (a game's ear and the head under it). The edges of
    a mesh whose corners are written by these ids join the pieces again.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    V = np.asarray(vertices, dtype=np.float64)
    pairs = cKDTree(V).query_pairs(tolerance, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(len(V))
    A = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(V), len(V))
    )
    return connected_components(A, directed=False)[1]


def _rows_in(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Mask of the rows (n, 2) of integers that ``table`` (m, 2) holds."""
    if len(rows) == 0 or len(table) == 0:
        return np.zeros(len(rows), bool)
    key = int(max(rows.max(), table.max())) + 1
    return np.isin(rows[:, 0] * key + rows[:, 1], table[:, 0] * key + table[:, 1])


class EarSplit:
    """
    A reference cut at its ears: the triangles of its ``"ear"`` region (``masks.ear`` of the
    handoff) and the others, the head, which keep the triangles across the region's border.
    The two meet only along the attachment line, the ``"ear_attachment"`` region of the
    reference (``masks.ear[].attachment``), or the border of the ear region when the source
    gives none. The fit matches anny's outer ears (:func:`pinna`) to the ear part alone and
    the rest of anny to the head alone, so that neither reaches across the gap behind a
    standing ear, and pulls the root of anny's ear onto the attachment line. Both parts hold
    the reference's triangles within ``reach`` of the attachment line, where a mask and anny's
    own outer ear may part by a few millimetres: there the skin of either may lie.
    """

    def __init__(self, reference: Reference, reach: float = 0.0):
        from scipy.spatial import cKDTree

        ear = np.asarray(reference.regions["ear"], bool)
        T = reference.triangles
        attachment = reference.regions.get("ear_attachment")
        if attachment is None or not np.asarray(attachment).any():
            attachment = mask_border(T, ear)
        attachment = np.asarray(attachment, bool)
        on_ear = ear[T].all(1)
        shared = np.zeros(len(T), bool)
        if reach > 0:
            # both parts hold the strip of the reference along the attachment line
            gap, _ = cKDTree(reference.vertices[attachment]).query(reference.vertices)
            shared = (gap[T] < reach).any(1)
        self.ear = reference.subset(on_ear)
        self.head = reference.subset(~on_ear | shared)
        e = np.unique(
            np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1),
            axis=0,
        )
        e = e[attachment[e].all(1)]
        lone = np.setdiff1d(np.nonzero(attachment)[0], e.ravel())
        e = np.concatenate([e, np.c_[lone, lone]])  # lone vertices as point segments
        self.segments = reference.vertices[e]  # (S, 2, 3)
        self.attachment = attachment
        self._tree = None
        # the vertices of each part along the cut: on a border of the part that is not a
        # border of the reference. The borders join the vertices that the source splits at one
        # point (welded_ids): an ear stitched to the head by coincident vertices, as a game
        # writes it, has the attachment line as a border of the reference itself, which would
        # leave the cut empty
        ids = welded_ids(reference.vertices)
        whole = _border_edges(ids[T])
        for part in (self.ear, self.head):
            joined = ids[part.source_vertices]
            edges = _border_edges(part.triangles)
            # a border of the part where the source splits its vertices (a UV seam) is none
            own = _rows_in(
                np.sort(joined[edges], 1), _border_edges(joined[part.triangles])
            )
            cut = own & ~_rows_in(np.sort(joined[edges], 1), whole)
            part.cut = np.zeros(len(part.vertices), bool)
            part.cut[edges[cut].ravel()] = True
        # the vertices along the cut pull no vertex of their part in the reverse term: the
        # nearest of the head's is the skin beside the root of the ear, which they drew onto
        # the attachment line next to the root (that the attachment term holds there), and the
        # two creased into a crack
        for part in (self.ear, self.head):
            part.skip_reverse = part.cut

    def attach(self, points: np.ndarray):
        """The closest points (n, 3) of the attachment line and their distances."""
        from scipy.spatial import cKDTree

        S = self.segments
        if self._tree is None:
            self._tree = cKDTree(S.mean(1))
        points = np.asarray(points, dtype=np.float64)
        k = min(8, len(S))
        _, near = self._tree.query(points, k=k)
        near = near.reshape(len(points), k)
        a, b = S[near, 0], S[near, 1]
        ab = b - a
        t = ((points[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-20)
        c = a + np.clip(t, 0.0, 1.0)[..., None] * ab
        d = np.linalg.norm(c - points[:, None], axis=-1)
        best = np.argmin(d, 1)
        rows = np.arange(len(points))
        return c[rows, best], d[rows, best]


class Target:
    """
    The surface that the refinement fits: a :class:`Reference`, as its flat triangles or as
    their Phong tessellation (``smooth``, curved by the normals the source shades with when
    ``shading``; see :meth:`Reference.closest`). With an :class:`EarSplit` and the mask
    ``anny_ear`` of the fitted mesh's outer ears, the closest points of anny's ears lie on the
    reference's ears and those of the rest on the rest.
    """

    def __init__(
        self,
        reference: Reference,
        smooth: float = 0.0,
        shading: bool = False,
        split: EarSplit | None = None,
        anny_ear: np.ndarray | None = None,
        border: np.ndarray | None = None,
    ):
        self.reference = reference
        self.smooth = smooth
        self.shading = shading
        self.vertices = reference.vertices
        self.normals = reference.shading_normals if shading else reference.normals
        self.split = split
        self.anny_ear = None if anny_ear is None else np.asarray(anny_ear, bool)
        # the reference's vertices along its open borders (a statue's cut supports), whose
        # pairs drop out in both terms; None: every pair counts
        self.border = None if border is None else np.asarray(border, bool)
        self.skip_reverse = self.border

    @property
    def is_split(self) -> bool:
        return self.split is not None and self.anny_ear is not None

    def parts(self, rows=None):
        """(mask of the points, the reference part they match) for the points of the fitted
        mesh's vertices ``rows`` (all of them by default)."""
        if not self.is_split:
            return [(None, self.reference)]
        ear = self.anny_ear if rows is None else self.anny_ear[rows]
        return [(ear, self.split.ear), (~ear, self.split.head)]

    def closest(self, points: np.ndarray, rows=None, edge: bool = False):
        """
        Closest points, normals and distances of ``points``, the points of the fitted mesh's
        vertices ``rows`` (all of them by default). With ``edge``, also the mask of the points
        that lie past the cut of their part of a split reference: their closest point lies on
        the cut (the attachment line), and the offset to it runs along the surface more than
        across it and is over 0.2 mm. Those are the points of anny's head past the line, or of
        its ears, which would all clamp onto it and crease there; their pairs drop out, as
        Amberg et al. drop the pairs on the borders of a target, and the attachment term holds
        the root of the ear on the line. All False without a split. With a ``border``, the
        points whose closest point lies on it in the same way are marked as well.
        """
        points = np.asarray(points, dtype=np.float64)
        C = np.zeros_like(points)
        N = np.zeros_like(points)
        d = np.zeros(len(points))
        off = np.zeros(len(points), bool)
        for mask, part in self.parts(rows):
            take = slice(None) if mask is None else np.nonzero(mask)[0]
            if mask is not None and len(take) == 0:
                continue
            c, n, dd, tri, bary = part.closest(points[take], self.smooth, self.shading)
            C[take], N[take], d[take] = c, n, dd
            if edge and hasattr(part, "cut"):
                corners = part.cut[part.triangles[tri]]
                on_cut = (corners | (bary < 1e-6)).all(1)
                offset = points[take] - c
                across = np.abs((offset * n).sum(1))
                along = np.sqrt(np.maximum(dd**2 - across**2, 0.0))
                off[take] = on_cut & (along > 0.5 * dd) & (dd > 2e-4)
            if edge and mask is None and self.border is not None:
                on_border = (self.border[part.triangles[tri]] | (bary < 1e-6)).all(1)
                offset = points[take] - c
                across = np.abs((offset * n).sum(1))
                along = np.sqrt(np.maximum(dd**2 - across**2, 0.0))
                off[take] |= on_border & (along > 0.5 * dd) & (dd > 2e-4)
        return (C, N, d, off) if edge else (C, N, d)


def nicp(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference: Reference,
    weights: np.ndarray,
    stiffness=(10.0, 3.0, 1.0, 0.3),
    iterations: int = 3,
    max_distance: float = 0.02,
    normal_threshold: float = 0.2,
    anchor: float = 1e-4,
    reverse_weight: float | np.ndarray = 1.0,
    reverse_normal_threshold: float = 0.5,
    stiffness_scale: np.ndarray | None = None,
    reverse: str = "vertex",
    attachment_weight: float = 0.0,
) -> np.ndarray:
    """
    Non-rigid ICP of a mesh onto the reference: in each round, solve

        min_X sum_i w_i |x_i - c_i|^2 + r_i |x_i - p_i|^2 + alpha |S^1/2 L (X - X_0)|^2
              + anchor |X - X_0|^2

    with c_i the closest point of the reference, L the uniform Laplacian of the mesh and X_0
    the mesh at the start of the stiffness ``alpha``. Pairs farther than ``max_distance`` or
    whose normals disagree (dot product under ``normal_threshold``) drop out; ``weights`` of 0
    leave a vertex to the smoothness term. The small ``anchor`` keeps closed parts without
    data (the eyeballs) where they are, which :func:`follow_skin` then corrects.
    ``stiffness_scale`` (S, per vertex; 1 by default) makes a region stiffer than the rest.

    The reverse term (``reverse="vertex"``) pulls each vertex that is the nearest of some
    reference vertices to their mean, with weight ``reverse_weight`` (a number, or one per
    vertex) times its data weight.
    ``reverse="surface"`` (experimental) instead pulls the fitted surface onto the reference's
    vertices: each reference
    vertex p_j within ``max_distance`` of the surface, whose normal agrees with the surface's
    there to ``reverse_normal_threshold``, adds ``reverse_weight`` m_j |sum_k b_jk x_k - p_j|^2
    with b_j the barycentric coordinates of its closest point on the fitted triangle (of
    vertices that all take data) and m_j the smallest data weight of its corners. Closest
    points alone never pull a surface past its border, so without it a part of the reference
    that reaches beyond anny's (the tip of a pointed ear, eyelids that cover more of the eye)
    stays uncovered. Spread over the triangle, the pull does not drag single vertices: a level
    finer than the reference has several vertices per reference vertex, and pulling the
    nearest of them alone left pinched points along the rim of an ear.

    A split ``reference`` (a :class:`Target` with an :class:`EarSplit`) matches anny's outer
    ears to the reference's ears alone and the rest to the rest, in both terms, and with
    ``attachment_weight`` pulls the border of anny's ears (their root) toward the closest
    points of the attachment line, with that share of the data weight.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    from anny.faces.authoring.nicp import uniform_laplacian

    target = reference if isinstance(reference, Target) else Target(reference)
    X = np.asarray(vertices, dtype=np.float64).copy()
    n = len(X)
    tri = np.asarray(triangles)
    L = uniform_laplacian(n, tri)
    if stiffness_scale is not None:
        LtL = (L.T @ scipy.sparse.diags(np.asarray(stiffness_scale, float)) @ L).tocsc()
    else:
        LtL = (L.T @ L).tocsc()
    base = np.asarray(weights, dtype=np.float64)
    kept = np.nonzero(base > 0)[0]
    fitted = tri[(base[tri] > 0).all(1)]
    border = None
    if target.is_split and attachment_weight > 0:
        border = np.nonzero(mask_border(tri, target.anny_ear) & (base > 0))[0]
    for alpha in stiffness:
        start = X.copy()
        smooth = alpha * (LtL @ start)
        for _ in range(iterations):
            C, N, d, off = target.closest(X, edge=True)
            normals = vertex_normals(X, triangles)
            ok = (d < max_distance) & ((normals * N).sum(1) > normal_threshold) & ~off
            w = base * ok
            if border is not None and len(border):
                C, w = _attach(
                    X, border, target.split, C, w, base, attachment_weight, max_distance
                )
            if reverse == "vertex":
                r, P = _reverse_pairs(
                    X, normals, kept, target, max_distance, reverse_normal_threshold
                )
                r = reverse_weight * base * r
                rhs = w[:, None] * C + r[:, None] * P + anchor * start + smooth
                A = (scipy.sparse.diags(w + r + anchor) + alpha * LtL).tocsc()
                solve = scipy.sparse.linalg.factorized(A)
                X = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
                continue
            B, P = _reverse_rows(
                X,
                normals,
                fitted,
                base,
                target,
                max_distance,
                reverse_normal_threshold,
                reverse_weight,
            )
            rhs = w[:, None] * C + anchor * start + smooth + B.T @ P
            A = (scipy.sparse.diags(w + anchor) + B.T @ B + alpha * LtL).tocsc()
            solve = scipy.sparse.linalg.factorized(A)
            X = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    return X


def _reverse_rows(
    X, normals, fitted, base, target, max_distance, normal_threshold, weight
):
    """The rows of the reverse term of :func:`nicp`: a sparse (M, N) matrix of the weighted
    barycentric coordinates of the closest points of M reference vertices on the fitted
    triangles, and the weighted reference vertices (M, 3)."""
    import scipy.sparse

    from anny.hair import closest_triangles

    n = len(X)
    weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), (n,))
    if len(fitted) == 0 or not (weight > 0).any():
        return scipy.sparse.csr_matrix((0, n)), np.zeros((0, 3))
    points = target.vertices
    t, bary, d = closest_triangles(points, X, fitted)
    corners = fitted[t]
    normal = np.einsum("nk, nkd -> nd", bary, normals[corners])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    ok = (d < max_distance) & ((normal * target.normals).sum(1) > normal_threshold)
    rows = np.nonzero(ok)[0]
    m = np.sqrt((weight * base)[corners[rows]].min(1))
    values = (m[:, None] * bary[rows]).ravel()
    B = scipy.sparse.csr_matrix(
        (values, (np.repeat(np.arange(len(rows)), 3), corners[rows].ravel())),
        shape=(len(rows), n),
    )
    return B, m[:, None] * points[rows]


def project_onto(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference: Reference,
    keep: np.ndarray,
    distance: float,
    normal_threshold: float = 0.8,
) -> np.ndarray:
    """The vertices of ``keep`` within ``distance`` of the reference whose normals agree with
    the reference's to ``normal_threshold`` moved onto their closest points."""
    target = reference if isinstance(reference, Target) else Target(reference)
    X = np.asarray(vertices, dtype=np.float64).copy()
    C, N, d, off = target.closest(X, edge=True)
    normals = vertex_normals(X, triangles)
    move = keep & (d < distance) & ((normals * N).sum(1) > normal_threshold) & ~off
    X[move] = C[move]
    return X


def relax_along(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference: Reference,
    weights: np.ndarray,
    iterations: int = 10,
    max_distance: float = 0.02,
    step: float = 0.5,
    normal_threshold: float = 0.5,
) -> np.ndarray:
    """
    Spread the vertices of a region evenly over the reference: each round moves each vertex
    by ``step`` times ``weights`` of its uniform umbrella vector (the mean of its neighbours
    minus itself) within its tangent plane, then onto its closest point of the reference
    (when that lies within ``max_distance`` and faces the same way). Where the reverse term
    pulled the border of a part far out (the tip of a pointed ear), the triangles next to it
    stretch far more than the rest; spread out, the stretch, and with it the stretch of the
    texture, is shared by the whole part.
    """
    import scipy.sparse

    target = reference if isinstance(reference, Target) else Target(reference)
    X = np.asarray(vertices, dtype=np.float64).copy()
    w = np.asarray(weights, dtype=np.float64)
    rows = np.nonzero(w > 1e-3)[0]
    if len(rows) == 0 or iterations <= 0:
        return X
    t = np.asarray(triangles)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    n = len(X)
    A = scipy.sparse.coo_matrix(
        (np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)
    ).tocsr()
    A.data[:] = 1.0
    degree = np.asarray(A.sum(1)).ravel()
    for _ in range(iterations):
        normals = vertex_normals(X, t)[rows]
        umbrella = (A[rows] @ X) / np.maximum(degree[rows], 1)[:, None] - X[rows]
        umbrella -= (umbrella * normals).sum(1, keepdims=True) * normals
        moved = X[rows] + step * w[rows, None] * umbrella
        C, N, d = target.closest(moved, rows)
        ok = (d < max_distance) & ((normals * N).sum(1) > normal_threshold)
        X[rows] = np.where(ok[:, None], C, moved)
    return X


def _attach(X, border, split, C, w, base, weight, max_distance):
    """
    The attachment term of :func:`nicp`, folded into its data term (targets C, weights w):
    each vertex of the root of anny's ear (``border``) toward its closest point of the
    attachment line, and each vertex of the line pulls the nearest root vertex, so that the
    root covers the whole line, its ends included; both with ``weight`` times the data
    weight.
    """
    from scipy.spatial import cKDTree

    a, da = split.attach(X[border])
    target = np.where((da < max_distance)[:, None], a, 0.0)
    count = (da < max_distance).astype(np.float64)
    line = split.segments.reshape(-1, 3)
    d, nearest = cKDTree(X[border]).query(line)
    ok = d < max_distance
    np.add.at(target, nearest[ok], line[ok])
    np.add.at(count, nearest[ok], 1.0)
    has = count > 0
    rows = border[has]
    point = target[has] / count[has, None]
    share = weight * base[rows]
    C = C.copy()
    w = w.copy()
    C[rows] = (w[rows, None] * C[rows] + share[:, None] * point) / np.maximum(
        w[rows] + share, 1e-12
    )[:, None]
    w[rows] += share
    return C, w


def _mean_laplacian(n: int, triangles: np.ndarray):
    """The mean-value Laplacian I - D^-1 A (n, n) of a triangle mesh's edge graph."""
    import scipy.sparse

    A = _adjacency(n, triangles)
    degree = np.maximum(np.asarray(A.sum(1)).ravel(), 1.0)
    return (scipy.sparse.identity(n) - scipy.sparse.diags(1.0 / degree) @ A).tocsr()


def fair_region(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference,
    region: np.ndarray,
    iterations: int = 5,
    fairness: float = 1.0,
    max_distance: float = 0.02,
    normal_threshold: float = 0.0,
    reverse_weight: float = 1.0,
    reverse_normal_threshold: float = 0.5,
    pool: np.ndarray | None = None,
    anchor: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fair the vertices of ``region`` on the reference, the others held: each round solves

        min sum_i w_i |x_i - c_i|^2 + r_i |x_i - p_i|^2 + fairness |B X|^2

    over the region's vertices, with c_i the closest point of the reference (within
    ``max_distance``, its normal agreeing to ``normal_threshold``), p_i the mean of the
    reference vertices whose nearest vertex of the region x_i is (the reverse term of
    :func:`nicp`, which keeps a pointed tip reached), and B the bi-Laplacian (the square of the
    mean-value Laplacian) of the positions. The bi-Laplacian of a smooth surface is small, and
    that of a crumple, a fold or a spike large, while the closest points let the vertices slide
    within the reference: the folds of the reference stay, those of the fit go, and the
    vertices spread evenly. Each round finds the closest points again. ``pool`` (a mask that
    holds the region; the region by default) gives the vertices among which the reverse term
    finds the nearest of each reference vertex, so that a small region takes only the
    reference vertices nearest to it. ``anchor`` (a weight per vertex) adds
    ``anchor_i |x_i - x_i^0|^2`` to the sum, which holds the vertices of a large weight near
    where they were: a region joins the held vertices around it smoothly through a band whose
    weights fall from large to 0.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    target = reference if isinstance(reference, Target) else Target(reference)
    X = np.asarray(vertices, dtype=np.float64).copy()
    region = np.asarray(region, bool)
    rows = np.nonzero(region)[0]
    if len(rows) == 0 or iterations <= 0:
        return X
    others = np.nonzero(~region)[0]
    pool_rows = rows if pool is None else np.nonzero(np.asarray(pool, bool))[0]
    T = np.asarray(triangles)
    L = _mean_laplacian(len(X), T)
    B = (L @ L).tocsr()
    Q = (B.T @ B).tocsr()
    Qrr = Q[rows][:, rows]
    Qro = Q[rows][:, others]
    X0 = X[rows].copy()
    h = np.zeros(len(rows)) if anchor is None else np.asarray(anchor, np.float64)[rows]
    for _ in range(iterations):
        C, N, d, off = target.closest(X[rows], rows, edge=True)
        normals = vertex_normals(X, T)
        ok = (d < max_distance) & ((normals[rows] * N).sum(1) > normal_threshold)
        ok &= ~off
        w = ok.astype(np.float64)
        r, P = _reverse_pairs(
            X, normals, pool_rows, target, max_distance, reverse_normal_threshold
        )
        r = (np.broadcast_to(reverse_weight, (len(X),)) * r)[rows]
        A = (scipy.sparse.diags(w + r + h + 1e-6) + fairness * Qrr).tocsc()
        rhs = (
            w[:, None] * C
            + r[:, None] * P[rows]
            + h[:, None] * X0
            + 1e-6 * X[rows]
            - fairness * (Qro @ X[others])
        )
        solve = scipy.sparse.linalg.factorized(A)
        X[rows] = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    return X


def fair_skin(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference,
    region: np.ndarray,
    fairness: float,
    iterations: int = 2,
    max_distance: float = 0.002,
    normal_threshold: float = 0.5,
    anchor: np.ndarray | None = None,
) -> np.ndarray:
    """
    The skin of ``region`` faired on the reference: :func:`fair_region` without its reverse
    term, so that each vertex keeps to its closest point of the reference while the
    bi-Laplacian evens out what is finer than the fairness lets through. A coarse reference
    looks smooth only through the normals it shades with; its flat triangles are facets a few
    millimetres across, and a fine level projected onto them (:func:`project_onto`) copies
    them as steps, which shading by the fitted mesh's own normals shows as layers. The
    fairing keeps the reference's shape at the scale of its triangles and above and drops
    the steps between them; the closest points hold the surface on the reference, so the
    vertices of the reference's triangles end up slightly off a convex surface they span.
    """
    return fair_region(
        vertices,
        triangles,
        reference,
        region,
        iterations,
        fairness,
        max_distance,
        normal_threshold,
        reverse_weight=0.0,
        pool=region,
        anchor=anchor,
    )


def _fair_corners(
    offsets,
    parametric,
    triangles,
    target,
    corners,
    cavity,
    held_rows,
    pose,
    unpose,
    fairness,
    radius,
    hold,
    iterations,
    max_distance,
    normal_threshold: float = 0.5,
):
    """Fair the skin around the mouth's corners on the reference (:func:`fair_skin`): the
    vertices within ``radius`` of either corner (``corners``, posed and placed), but those of
    the inside of the mouth (``cavity``), with the rows of ``held_rows`` (the sealed edges of
    both lips) held by ``hold``. Returns the rest-frame offsets and the count of vertices
    faired. The rest of the mesh stays."""
    n = len(parametric)
    every = np.arange(n)
    posed = pose(parametric + offsets, every)
    near = np.min(
        np.linalg.norm(posed[:, None, :] - np.asarray(corners)[None, :, :], axis=2),
        axis=1,
    )
    region = (near < radius) & ~np.asarray(cavity, bool)
    if not region.any():
        return offsets, 0
    anchor = np.zeros(n)
    for rows in held_rows:
        anchor[np.asarray(rows, np.int64)] = hold
    faired = fair_skin(
        posed,
        triangles,
        target,
        region,
        fairness,
        iterations,
        max_distance,
        normal_threshold,
        anchor=anchor,
    )
    rows = np.nonzero(region)[0]
    out = np.asarray(offsets, dtype=np.float64).copy()
    out[rows] = unpose(faired[rows], rows) - parametric[rows]
    return out, int(region.sum())


def lid_blend(
    triangles: np.ndarray, held: np.ndarray, rings: int, weight: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    The band of ``rings`` rings of faces around the vertices of ``held`` and the weights that
    hold it in :func:`fair_skin` (``anchor``): ``weight`` on the first ring, falling with the
    square of the ring's distance to 0 past the last, so that the faired skin joins the held
    lids without a step. Returns (the band's mask, the weights over all vertices).
    """
    held = np.asarray(held, bool)
    adjacency = _adjacency(len(held), triangles)
    reached = held.copy()
    anchor = np.zeros(len(held))
    band = np.zeros(len(held), bool)
    for k in range(1, rings + 1):
        ring = (adjacency @ reached.astype(np.float64) > 0) & ~reached
        anchor[ring] = weight * (1.0 - (k - 1) / rings) ** 2
        band |= ring
        reached |= ring
    return band, anchor


def skin_fair_region(
    topology,
    skin0: np.ndarray,
    data: np.ndarray,
    ears: np.ndarray,
    rings: int,
    held: np.ndarray | None = None,
) -> np.ndarray:
    """The vertices of a level that :func:`fair_skin` fairs: those of ``skin0`` (a weight per
    vertex of level 0, carried to the level) with data, apart from the outer ears ``ears``
    and the skin within ``rings`` faces of them (which the ears' own fairing takes) and the
    vertices of ``held``."""
    region = (np.asarray(topology.apply(skin0)) > 0.5) & (np.asarray(data) > 0)
    near = np.asarray(ears, bool).copy()
    adjacency = _adjacency(len(region), topology.triangles)
    for _ in range(rings):
        near |= adjacency @ near.astype(np.float64) > 0
    region &= ~near
    if held is not None:
        region &= ~np.asarray(held, bool)
    return region


def relabel_ears(
    vertices: np.ndarray,
    triangles: np.ndarray,
    ear: np.ndarray,
    split: "EarSplit",
    rings: int,
) -> np.ndarray:
    """
    The mask of the fitted mesh's outer ears moved to where the mesh crosses the reference's
    attachment line: within ``rings`` rings of the border of ``ear``, a vertex belongs to the
    ear when its closest point of the ear part is nearer than that of the head part (by
    0.05 mm; else it keeps its label). Of the result, the ear keeps the pieces that hold
    vertices outside the band and the holes it encloses (the concha), and the head the same.

    :func:`pinna` cuts anny's outer ear where it stands off from the head, and a reference's
    mask may hold more of the flat skin around the root, as the front of a game's ear does:
    the head's skin of anny on that strip then had no part to fit but the head beside it, slid
    off the ear onto it and folded into a crack along the front of the ear.
    """
    from scipy.sparse.csgraph import connected_components

    X = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    ear = np.asarray(ear, bool)
    A = _adjacency(len(X), T)
    band = mask_border(T, ear) | mask_border(T, ~ear)
    for _ in range(rings):
        band |= A @ band.astype(np.float64) > 0
    rows = np.nonzero(band)[0]
    de = split.ear.closest(X[rows])[2]
    dh = split.head.closest(X[rows])[2]
    out = ear.copy()
    out[rows[de < dh - 5e-5]] = True
    out[rows[dh < de - 5e-5]] = False
    for label in (True, False):
        # each side keeps its pieces that reach past the band; the others change sides
        own = np.nonzero(out == label)[0]
        _, pieces = connected_components(A[own][:, own], directed=False)
        anchored = np.zeros(pieces.max() + 1, bool)
        anchored[pieces[~band[own]]] = True
        out[own[~anchored[pieces]]] = not label
    return out


def fold_defects(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference,
    region: np.ndarray,
    fold_margin: float = 45.0,
    max_distance: float = 0.02,
) -> np.ndarray:
    """
    Mask (V,) of the vertices of ``region`` where the fit folds and the reference does not:
    those at an edge folded by more than 60 degrees and by ``fold_margin`` more than the
    reference folds within 3 mm of their closest point (``metrics.fold_near``: the reference's
    own creases, such as the rim of an ear, stay), and those whose normal turns away from the
    reference's at their closest point (by more than 90 degrees: a flipped sheet, the inside
    of a crumple, a spike) where the reference does not fold either.
    """
    from opensculptboy.fit.metrics import fold_angles, fold_near

    target = reference if isinstance(reference, Target) else Target(reference)
    X = np.asarray(vertices, dtype=np.float64)
    rows = np.nonzero(np.asarray(region, bool))[0]
    out = np.zeros(len(X), bool)
    if len(rows) == 0:
        return out
    C, N, d = target.closest(X[rows], rows)
    ref = target.reference
    F = fold_near(ref.vertices, ref.folds, C, tree=ref.tree)
    normals = vertex_normals(X, triangles)[rows]
    fold = fold_angles(X, triangles)[rows]
    flipped = ((normals * N).sum(1) < 0.0) & (F < 60.0)
    creased = (fold > 60.0) & (fold > F + fold_margin)
    out[rows] = (flipped | creased) & (d < max_distance)
    return out


def repair_folds(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference,
    region: np.ndarray,
    rounds: int = 3,
    rings: int = 2,
    fairness: float = 10.0,
    max_distance: float = 0.02,
    pool: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """
    Repair the folds and spikes of :func:`fold_defects` within ``region``, locally: the
    defects and ``rings`` rings of their neighbours are faired again (:func:`fair_region`, with
    ``fairness`` and the reverse term among the vertices of the whole region, so that a pointed
    tip, which a coarse mesh draws as a sharp point, stays where the reference has it) while
    the rest is held; up to ``rounds`` times while defects remain. ``pool`` is the reverse
    term's pool (:func:`fair_region`; the region by default). Returns the vertices and the
    number of defects found first.
    """
    X = np.asarray(vertices, dtype=np.float64).copy()
    T = np.asarray(triangles)
    region = np.asarray(region, bool)
    target = reference if isinstance(reference, Target) else Target(reference)
    A = _adjacency(len(X), T)
    first = None
    for _ in range(rounds):
        bad = fold_defects(X, T, target, region, max_distance=max_distance)
        if first is None:
            first = int(bad.sum())
        if not bad.any():
            break
        for _ in range(rings):
            bad = bad | (A @ bad.astype(np.float64) > 0)
        X = fair_region(
            X,
            T,
            target,
            bad & region,
            iterations=2,
            fairness=fairness,
            max_distance=max_distance,
            pool=region if pool is None else pool,
        )
    return X, first or 0


def _reverse_pairs(X, normals, kept, reference, max_distance, normal_threshold):
    """For each vertex of ``X``: 1 when it is the nearest vertex (of ``kept``) of reference
    vertices within ``max_distance`` with agreeing normals, else 0, and their mean. A split
    :class:`Target` pairs the reference's ears with anny's ears alone and the rest with the
    rest, so that the head behind a standing ear never pulls the back of anny's ear, nor the
    ear the head."""
    from scipy.spatial import cKDTree

    if isinstance(reference, Target) and reference.is_split:
        ear = reference.anny_ear
        hit = np.zeros(len(X))
        mean = np.zeros_like(X)
        for part, rows in (
            (reference.split.ear, kept[ear[kept]]),
            (reference.split.head, kept[~ear[kept]]),
        ):
            h, m = _reverse_pairs(
                X, normals, rows, part, max_distance, normal_threshold
            )
            hit += h
            mean += h[:, None] * m
        return hit, mean
    hit = np.zeros(len(X))
    mean = np.zeros_like(X)
    if len(kept) == 0:
        return hit, mean
    d, nearest = cKDTree(X[kept]).query(reference.vertices)
    rows = kept[nearest]
    ok = (d < max_distance) & (
        (normals[rows] * reference.normals).sum(1) > normal_threshold
    )
    skip = getattr(reference, "skip_reverse", None)
    if skip is not None:
        ok &= ~skip
    points = reference.vertices[ok]
    rows = rows[ok]
    count = np.bincount(rows, minlength=len(X)).astype(np.float64)
    for k in range(3):
        mean[:, k] = np.bincount(rows, points[:, k], minlength=len(X))
    hit = (count > 0).astype(np.float64)
    mean /= np.maximum(count, 1.0)[:, None]
    return hit, mean


def follow_skin(
    before: np.ndarray,
    after: np.ndarray,
    excluded: np.ndarray,
    triangles: np.ndarray,
    radius: float = 0.02,
    neighbours: int = 16,
    falloff: float = 0.005,
    inside: float = 0.0005,
    eyes: list | None = None,
) -> np.ndarray:
    """
    Move the excluded vertices with the fitted skin around them:

    - each closed component of excluded vertices (an eyeball) onto the reference's eyeball
      sphere whose centre lies within ``radius`` of it (``eyes``, as
      :attr:`~opensculptboy.fit.reference.Reference.eyes` holds them: moved and scaled about
      its own sphere's centre), or without one by the mean displacement of the fitted
      vertices within ``radius`` of its centre (of the ``neighbours`` nearest fitted vertices
      when none is that close, as for the tongue);
    - each excluded vertex of a component joined to the skin (the inside of the mouth and the
      eye sockets) by the displacement of the ``neighbours`` nearest fitted vertices in space,
      weighted by exp(-d / ``falloff``). Along the mesh, the inside of the mouth hangs from the
      lips' inner margin alone, so when the fit pulls the lips back the Laplacian term leaves
      the gums and the palate in front of the skin; the nearest skin in space carries them
      back with the lips. A vertex of these that still lies in front of the fitted skin by more
      than 0.4 ``inside`` (as anny's own mouth shapes can leave it) goes back to ``inside``
      behind it.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(before)
    t = np.asarray(triangles)
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    out = after.copy()
    kept = np.nonzero(~excluded)[0]
    if len(kept) == 0:
        return out
    tree = cKDTree(before[kept])
    displacement = after - before
    joined = []
    for label in np.unique(labels[excluded]):
        component = labels == label
        if (component & ~excluded).any():
            joined.append(np.nonzero(component & excluded)[0])
            continue
        centre = before[component].mean(0)
        target = None
        if eyes and component.sum() >= 10:
            own, own_radius, _ = fit_sphere(before[component])
            gaps = [np.linalg.norm(np.asarray(c) - own) for c, _ in eyes]
            if min(gaps) < radius and own_radius > 0:
                target = eyes[int(np.argmin(gaps))]
        if target is not None:
            scale = target[1] / own_radius
            out[component] = np.asarray(target[0]) + (before[component] - own) * scale
            continue
        near = tree.query_ball_point(centre, radius)
        if not near:  # a part deep inside (the tongue): its nearest fitted vertices
            near = np.atleast_1d(tree.query(centre, k=min(neighbours, len(kept)))[1])
        out[component] = before[component] + displacement[kept[near]].mean(0)
    if joined:
        rows = np.concatenate(joined)
        k = min(neighbours, len(kept))
        distance, nearest = tree.query(before[rows], k=k)
        distance, nearest = (
            distance.reshape(len(rows), k),
            nearest.reshape(len(rows), k),
        )
        weights = np.exp(-(distance - distance[:, :1]) / falloff)
        weights /= weights.sum(1, keepdims=True)
        moved = np.einsum("nk, nkd -> nd", weights, displacement[kept[nearest]])
        out[rows] = before[rows] + moved
        out[rows] = _behind_skin(out, t, excluded, rows, inside)
    return out


def _behind_skin(vertices, triangles, excluded, rows, inside: float) -> np.ndarray:
    """The vertices ``rows`` moved to ``inside`` behind the skin (the triangles of vertices
    that are not excluded) wherever they lie in front of it."""
    from anny.hair import closest_triangles

    skin = triangles[~excluded[triangles].any(1)]
    if len(skin) == 0:
        return vertices[rows]
    used = np.unique(skin)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    V, T = vertices[used], remap[skin]
    normals = vertex_normals(V, T)
    tri, bary, _ = closest_triangles(vertices[rows], V, T)
    closest = np.einsum("nk, nkd -> nd", bary, V[T[tri]])
    normal = np.einsum("nk, nkd -> nd", bary, normals[T[tri]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    points = vertices[rows]
    # strictly in front: the excluded vertices along the skin's border lie beside it
    front = ((points - closest) * normal).sum(1) > 0.4 * inside
    points = points.copy()
    points[front] = closest[front] - inside * normal[front]
    return points


@dataclasses.dataclass
class LidSeat:
    """
    Which vertices of a level :func:`seat_lids` places, around which eyeball: ``region`` (N,)
    the vertices it moves, ``eye`` (N,) the index of their eyeball in ``eyes`` (-1 for the
    others), ``pool`` (N,) the vertices among which the reverse term finds the nearest of
    each reference vertex, ``eyes`` the reference's eyeball spheres [(centre, radius)] and
    ``gazes`` their gaze directions (unit vectors).
    """

    region: np.ndarray
    eye: np.ndarray
    pool: np.ndarray
    eyes: list
    gazes: list


def lid_region(
    vertices: np.ndarray,
    triangles: np.ndarray,
    eyes: list,
    gazes: list,
    fitted: np.ndarray,
    eyeball: np.ndarray,
    reach: float = 1.35,
    lining_reach: float = 1.6,
    front: float = 100.0,
) -> LidSeat:
    """
    The vertices around the opening of each eye that :func:`seat_lids` places: the skin that
    the refinement fits (``fitted``) within ``reach`` eyeball radii of the reference's eyeball
    centre and ``front`` degrees of its gaze (the lids out to the margin, and the skin of the
    inner corner), and the vertices that it leaves out (the lining of the socket and, on
    ``sculpt``, the eye-corner patch) within ``lining_reach``, but the eyeballs (``eyeball``).
    """
    X = np.asarray(vertices, dtype=np.float64)
    n = len(X)
    used = np.zeros(n, bool)
    used[np.asarray(triangles).ravel()] = True
    fitted = np.asarray(fitted, bool)
    eyeball = np.asarray(eyeball, bool)
    region = np.zeros(n, bool)
    pool = np.zeros(n, bool)
    which = np.full(n, -1)
    best = np.full(n, np.inf)
    for k, ((centre, radius), gaze) in enumerate(zip(eyes, gazes)):
        local = (X - np.asarray(centre)) / radius
        d = np.linalg.norm(local, axis=1)
        cos = (local @ np.asarray(gaze)) / np.maximum(d, 1e-12)
        skin = fitted & (d < reach) & (cos > np.cos(np.radians(front)))
        lining = ~fitted & (d < lining_reach)
        mine = used & ~eyeball & (skin | lining) & (d < best)
        which[mine], best[mine] = k, d[mine]
        region |= mine
        pool |= used & ~eyeball & (d < lining_reach)
    which[~region] = -1
    return LidSeat(region, which, pool, list(eyes), [np.asarray(g) for g in gazes])


def _as_levels(value) -> tuple:
    """A per-level option given as one value for every level, or as a tuple."""
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _per_level(values: tuple, subdiv: int):
    """The value of a per-level option for level ``subdiv`` (the last one for finer levels)."""
    return values[min(subdiv, len(values) - 1)]


def seat_lids(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference,
    seat: LidSeat,
    sphere: float = 0.99,
    margin: float = 0.005,
    iterations: int = 10,
    fairness: float = 20.0,
    weight: float = 10.0,
    reach: float = 0.35,
    sphere_reach: float = 0.25,
    normal_threshold: float = 0.2,
    reverse_normal_threshold: float = 0.5,
    front: float = 100.0,
    clearance: bool = True,
    clearance_rounds: int = 4,
    clearance_weight: float = 10.0,
    clearance_inward: bool = True,
    follow_inside: bool = False,
) -> np.ndarray:
    """
    Place the lids and the corner of each eye (``seat.region``) on what the reference shows
    there: its skin where that lies outside its eyeball's sphere, and the sphere itself where
    the skin goes under it, so that the lids and the corner meet the eyeball flush, as the
    reference's do. Each round solves, over the region's vertices with the others held,

        min sum_i w_i |x_i - c_i|^2 + r_i |x_i - p_i|^2 + fairness |B X|^2

    as :func:`fair_region` does, with data weights ``weight``. The target c_i of a vertex
    more than ``margin`` radii outside the sphere is the closest point of the reference's
    skin when that lies within ``reach`` eyeball radii, more than ``margin`` radii outside
    the sphere, with its normal agreeing with the vertex's to ``normal_threshold``.
    Otherwise it is the point of the sphere, shrunk to ``sphere`` of its radius so that it
    lies under the exported eyeball, along the ray from its centre (within ``sphere_reach``
    radii), and a vertex that lies under the shrunk sphere keeps its place; a vertex that
    once takes the sphere keeps it (under the eyeball its normal may agree with the skin's
    again, and it would swing back out). So the skin comes first: a lid that reaches past
    the reference's margin slides back onto it, and what lies under the reference's tissue
    tucks under the eyeball. With ``follow_inside``, a vertex whose closest point of the
    reference's skin lies inside the sphere (within ``reach``, with its normal agreeing)
    takes that point, so that the lids go under the eyeball as the reference's do.
    p_i is the mean of the reference's skin vertices outside the sphere (by ``margin``,
    within ``front`` degrees of the gaze) whose nearest vertex of ``seat.pool`` x_i is (the
    reverse term, with normals agreeing to ``reverse_normal_threshold``), which pulls the
    surface over the eyeball as far as the reference's tissue reaches; B is the
    bi-Laplacian, which keeps the surface fair (a strong ``fairness`` keeps the line where
    the skin meets the sphere from stepping from vertex to vertex, which folds the corner).
    A vertex with no target follows the fairing.

    With ``clearance``, the region's vertices within the band over the sphere in which the
    exported eyes find outward-facing skin too close (the cornea's shell and the lids'
    clearance, :mod:`opensculptboy.eyes.geometry`) leave it at the end: the reference's skin
    that faces out to the nearer end, the rest under the eyeball, so that the exported
    eyeball keeps its size and nothing lies between it and its cornea; without
    ``clearance_inward`` only the skin that faces out from the eyeball leaves the band (the
    exported eyes count only that skin, and a lid that goes under the eyeball crosses the
    band facing it, as the reference's does). Each vertex takes its side once, as it enters
    the band, and the same faired solve draws it there (with ``clearance_weight`` times the
    data weight, for up to ``clearance_rounds`` rounds), so that its neighbours follow; what
    still lies in the band then moves to the end of its side (all of the band without
    rounds). Moved there alone on a fine level, neighbours along the lid's edge that fell on
    either side of the band's middle stepped by its width, a jagged edge at the inner
    corner.
    """
    import scipy.sparse
    import scipy.sparse.linalg
    from scipy.spatial import cKDTree

    target = reference if isinstance(reference, Target) else Target(reference)
    X = np.asarray(vertices, dtype=np.float64).copy()
    rows = np.flatnonzero(seat.region)
    if len(rows) == 0 or iterations <= 0 or not seat.eyes:
        return X
    others = np.flatnonzero(~seat.region)
    T = np.asarray(triangles)
    L = _mean_laplacian(len(X), T)
    B = (L @ L).tocsr()
    Q = (B.T @ B).tocsr()
    Qrr = Q[rows][:, rows]
    Qro = Q[rows][:, others]
    centres = np.array([np.asarray(c, dtype=np.float64) for c, _ in seat.eyes])
    radii = np.array([float(r) for _, r in seat.eyes])
    # the band over the sphere where the exported eyes count outward-facing skin as too
    # close (the cornea's shell and the lids' clearance, eyes.geometry.bulge_clearance)
    bands = (CORNEA_GAP_MM + LID_CLEARANCE_MM) / (radii * 1000.0)
    ce, re = centres[seat.eye[rows]], radii[seat.eye[rows]]
    mb = bands[seat.eye[rows]]
    # the reference's skin vertices in front of each eyeball, outside its sphere
    P = np.asarray(target.vertices, dtype=np.float64)
    shown, shown_eye = [], []
    for k, (centre, radius) in enumerate(seat.eyes):
        local = (P - centre) / radius
        d = np.linalg.norm(local, axis=1)
        cos = (local @ seat.gazes[k]) / np.maximum(d, 1e-12)
        keep = (d > 1.0 + margin) & (d < 1.6) & (cos > np.cos(np.radians(front)))
        shown.append(np.flatnonzero(keep))
        shown_eye.append(np.full(int(keep.sum()), k))
    shown = np.concatenate(shown)
    shown_eye = np.concatenate(shown_eye)
    pool = np.flatnonzero(seat.pool)
    factor = scipy.sparse.diags(np.full(len(rows), 1e-6)) + fairness * Qrr
    held = np.zeros(len(rows), bool)
    for _ in range(iterations):
        normals = vertex_normals(X, T)
        x = X[rows]
        c, nrm, d = target.closest(x, rows)
        ray = x - ce
        dx = np.linalg.norm(ray, axis=1)
        ray /= np.maximum(dx, 1e-12)[:, None]
        skin_ok = (d < reach * re) & ((normals[rows] * nrm).sum(1) > normal_threshold)
        skin_ok &= np.linalg.norm(c - ce, axis=1) > re * (1.0 + margin)
        skin_ok &= dx > re * (1.0 + margin)
        on_sphere = ce + (sphere * re)[:, None] * ray
        de = np.linalg.norm(x - on_sphere, axis=1)
        # a vertex that once goes to the sphere stays with it: moved under the eyeball, its
        # normal can turn to agree with the skin's again, and it would swing back out
        sphere_ok = (de < sphere_reach * re) & ~skin_ok | held
        held |= sphere_ok
        skin_ok &= ~held
        # what lies under the eyeball stays where it is
        under = ~skin_ok & (dx <= sphere * re)
        C = np.where(skin_ok[:, None], c, np.where(under[:, None], x, on_sphere))
        if follow_inside:
            # the reference's own skin under its eyeball: the lids follow it there
            inside = ~skin_ok & (d < reach * re)
            inside &= (normals[rows] * nrm).sum(1) > normal_threshold
            inside &= np.linalg.norm(c - ce, axis=1) < re
            C[inside] = c[inside]
            under |= inside
        w = weight * (skin_ok | sphere_ok | under)
        # the reverse term: the reference's tissue pulls its nearest vertex of the pool
        r = np.zeros(len(X))
        mean = np.zeros_like(X)
        if len(shown) and len(pool):
            dist, nearest = cKDTree(X[pool]).query(P[shown])
            near = pool[nearest]
            ok = dist < reach * radii[shown_eye]
            ok &= (normals[near] * target.normals[shown]).sum(
                1
            ) > reverse_normal_threshold
            near, points = near[ok], P[shown][ok]
            count = np.bincount(near, minlength=len(X)).astype(np.float64)
            for k in range(3):
                mean[:, k] = np.bincount(near, points[:, k], minlength=len(X))
            r = weight * (count > 0)
            mean /= np.maximum(count, 1.0)[:, None]
        rr = r[rows]
        A = (factor + scipy.sparse.diags(w + rr)).tocsc()
        rhs = (
            w[:, None] * C
            + rr[:, None] * mean[rows]
            + 1e-6 * x
            - fairness * (Qro @ X[others])
        )
        solve = scipy.sparse.linalg.factorized(A)
        X[rows] = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    # the skin within the band over the sphere where the exported eyes find it too close
    # (from just under the sphere, as the exported eyeball's own fit may lie a little inside
    # the reference's): the reference's skin that faces out from the eyeball moves to the
    # nearer end of the band, out to its top or under the eyeball, and the rest under the
    # eyeball (what follows the sphere, and what faces the eyeball), so that the
    # exported eyeball keeps its size (eyes.geometry.bulge_clearance shrinks it for such
    # skin) and nothing lies between the eyeball and its cornea
    if not clearance:
        return X
    low = 1.0 - SPHERE_TOLERANCE
    top = 1.0 + mb
    mid = 0.5 * (low + top)
    under = min(sphere, low)

    def radial():
        ray = X[rows] - ce
        dx = np.linalg.norm(ray, axis=1) / re
        ray /= np.maximum(dx * re, 1e-12)[:, None]
        return ray, dx, (vertex_normals(X, T)[rows] * ray).sum(1)

    # each vertex that enters the band takes a side once (1: out to its top, -1: under the
    # eyeball) and is drawn there, a little past the end, within the faired solve: moved
    # there alone, neighbours along the lid's edge that fell on either side of the middle
    # stepped by the width of the band, which showed as a jagged edge at the inner corner
    side = np.zeros(len(rows), int)
    for _ in range(clearance_rounds):
        ray, dx, facing = radial()
        band = (dx >= low) & (dx < top)
        new = band & (side == 0)
        if not clearance_inward:
            new &= facing > 0.0
        if not new.any() and not (band & (side != 0)).any():
            break
        side[new] = np.where(((facing > 0.0) & (dx >= mid) & skin_ok)[new], 1, -1)
        goal = np.where(side > 0, top + 0.25 * mb, under - 0.25 * SPHERE_TOLERANCE)
        pinned = side != 0
        Cc = np.where(pinned[:, None], ce + (re * goal)[:, None] * ray, C)
        wc = np.where(pinned, clearance_weight * weight, w)
        A = (factor + scipy.sparse.diags(wc + rr)).tocsc()
        rhs = (
            wc[:, None] * Cc
            + rr[:, None] * mean[rows]
            + 1e-6 * X[rows]
            - fairness * (Qro @ X[others])
        )
        solve = scipy.sparse.linalg.factorized(A)
        X[rows] = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    # what the fairing leaves in the band moves to the end of its side (without rounds, all
    # of the band, by the side the rule gives it)
    ray, dx, facing = radial()
    band = (dx >= low) & (dx < top)
    if not clearance_inward:
        band &= facing > 0.0
    out = np.where(side != 0, side > 0, (facing > 0.0) & (dx >= mid) & skin_ok)
    to = np.where(out, top, under)
    X[rows[band]] = ce[band] + (re * to)[band, None] * ray[band]
    return X


def eye_gazes(model, eyes: list, transforms, rotation, translation) -> list:
    """The gaze of each of the reference's eyeballs (``eyes``, in the reference's frame): the
    rest gaze (-Y) turned by its eye bone's skinning transform and the placement; the side
    comes from the eyeball's centre in anny's frame (+X is the figure's left)."""
    labels = list(model.bone_labels)
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    out = []
    for centre, _ in eyes:
        side = "L" if (R.T @ (np.asarray(centre) - t))[0] > 0 else "R"
        bone = f"eye.{side}"
        turn = (
            np.asarray(transforms)[labels.index(bone), :3, :3]
            if bone in labels
            else np.eye(3)
        )
        gaze = R @ turn @ np.array([0.0, -1.0, 0.0])
        out.append(gaze / np.linalg.norm(gaze))
    return out


class _Parameters(torch.nn.Module):
    """The free parameters of the parameter stage."""

    def __init__(
        self,
        model,
        rotation0,
        translation0,
        height0: float,
        shared_targets: str = "face_shapes",
    ):
        super().__init__()
        dtype = model.template_vertices.dtype
        labels = list(model.bone_labels)
        self.free = [i for i, b in enumerate(labels) if b not in FIXED_BONES]
        self.major = [
            k
            for k, i in enumerate(self.free)
            if any(word in labels[i] for word in MAJOR_BONES)
        ]
        self.fingers = [
            k
            for k, i in enumerate(self.free)
            if any(word in labels[i] for word in FINGER_BONES)
        ]
        p = len(model.phenotype_labels)
        phenotype = torch.full((1, p), 0.5, dtype=dtype)
        phenotype[0, list(model.phenotype_labels).index("height")] = height0
        self.phenotype = torch.nn.Parameter(phenotype)
        self.local = torch.nn.Parameter(
            torch.zeros(1, len(model.local_change_labels), dtype=dtype)
        )
        self.face = torch.nn.Parameter(
            torch.zeros(1, len(model.face_shape_labels), dtype=dtype)
        )
        self.rotvec = torch.nn.Parameter(torch.zeros(len(self.free), 3, dtype=dtype))
        self.global_rotvec = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        self.global_translation = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        self.register_buffer("rotation0", torch.as_tensor(rotation0, dtype=dtype))
        self.register_buffer("translation0", torch.as_tensor(translation0, dtype=dtype))
        if shared_targets not in SHARED_TARGETS:
            raise ValueError(
                f"shared_targets must be one of {SHARED_TARGETS}; got {shared_targets!r}."
            )
        shared_local, shared_face = shared_target_controls(model)
        free_local = ~shared_local if shared_targets == "face_shapes" else None
        free_face = ~shared_face if shared_targets == "local_changes" else None
        self.register_buffer(
            "local_free",
            torch.ones(len(model.local_change_labels), dtype=dtype)
            if free_local is None
            else torch.as_tensor(free_local, dtype=dtype),
        )
        self.register_buffer(
            "face_free",
            torch.ones(len(model.face_shape_labels), dtype=dtype)
            if free_face is None
            else torch.as_tensor(free_face, dtype=dtype),
        )
        ranges = [model.face_shape_ranges[n] for n in model.face_shape_labels]
        self.register_buffer(
            "face_lo", torch.tensor([r[0] for r in ranges], dtype=dtype)
        )
        self.register_buffer(
            "face_hi", torch.tensor([r[1] for r in ranges], dtype=dtype)
        )
        self.bone_count = len(labels)

    def pose(self) -> torch.Tensor:
        import roma

        dtype = self.rotvec.dtype
        P = torch.eye(4, dtype=dtype).repeat(1, self.bone_count, 1, 1)
        P[0, self.free, :3, :3] = roma.rotvec_to_rotmat(self.rotvec)
        return P

    def placement(self):
        import roma

        R = roma.rotvec_to_rotmat(self.global_rotvec) @ self.rotation0
        return R, self.translation0 + self.global_translation

    @torch.no_grad()
    def project(self):
        self.phenotype.clamp_(0.0, 1.0)
        self.local.clamp_(-1.0, 1.0)
        self.face.copy_(
            torch.maximum(torch.minimum(self.face, self.face_hi), self.face_lo)
        )


def _initial_placement(model, reference: Reference, regressor, options):
    """Rotation, translation and height phenotype that place anny near the reference."""
    from anny.faces.authoring.nicp import umeyama

    labels = list(model.phenotype_labels)
    best, height0 = None, 0.5
    with torch.no_grad():
        for h in np.linspace(0.0, 1.0, 11):
            phenotype = torch.full((1, len(labels)), 0.5, dtype=model.dtype)
            phenotype[0, labels.index("height")] = float(h)
            rest = model(phenotype_kwargs=phenotype)["rest_vertices"][0].numpy()
            gap = abs(np.ptp(rest[:, 2]) - reference.height())
            if best is None or gap < best:
                best, height0 = gap, float(h)
        phenotype = torch.full((1, len(labels)), 0.5, dtype=model.dtype)
        phenotype[0, labels.index("height")] = height0
        pose = None
        if options.initial_pose is not None:
            # the landmarks of the start pose: a statue's joints lie where its pose puts them
            bones = list(model.bone_labels)
            pose = torch.eye(4, dtype=model.dtype).repeat(1, len(bones), 1, 1)
            free = [i for i, b in enumerate(bones) if b not in FIXED_BONES]
            start = np.asarray(options.initial_pose, dtype=np.float64)[free]
            pose[0, free, :3, :3] = torch.as_tensor(start, dtype=model.dtype)
        out = model(pose_parameters=pose, phenotype_kwargs=phenotype)
    anny_points = _landmark_points(model, out, regressor, reference, None)
    names = [
        k
        for k in reference.landmarks
        if k in anny_points and (reference.auto or LANDMARKS[k][2] > 0)
    ]
    src = np.stack([anny_points[k] for k in names])
    dst = np.stack([reference.landmarks[k] for k in names])
    if reference.auto or len(names) < 3:
        rotation = np.eye(3)
        translation = dst.mean(0) - src.mean(0)
    else:
        rotation, translation, _ = umeyama(src, dst)
    return rotation, translation, height0


def _landmark_points(model, out, regressor, reference, auto_vertices) -> dict:
    """anny's landmarks (numpy, Anny's frame, unplaced) for the names of the reference"""
    vertices = out["vertices"][0].detach().numpy()
    joints = out["bone_poses"][0, :, :3, 3].detach().numpy()
    labels = list(model.bone_labels)
    points = {}
    if reference.auto:
        indices = auto_vertices or auto_landmark_vertices(vertices)
        for name, rows in indices.items():
            points[name] = vertices[rows].mean(0)
        return points
    cranio = regressor.weights_matrix.detach().numpy() @ vertices
    for name in reference.landmarks:
        if name not in LANDMARKS:
            continue
        kind, label, _ = LANDMARKS[name]
        if kind == "bone":
            points[name] = joints[labels.index(label)]
        else:
            points[name] = cranio[regressor.labels.index(label)]
    return points


class _Craniofacial:
    """The craniofacial landmarks as a dense (K, V) matrix on the model's vertices."""

    def __init__(self, model):
        from anny.keypoints import KeypointsRegressor

        regressor = KeypointsRegressor.craniofacial(model)
        self.labels = list(regressor.labels)
        weights = getattr(regressor, "regression_weights", None)
        if weights is None or weights.dim() != 2:
            weights = torch.eye(len(model.template_vertices), dtype=model.dtype)
            weights = regressor({"vertices": weights[None]})[0]
        self.weights_matrix = weights.to(model.dtype)


def fit_parameters(model, reference: Reference, options: FitOptions):
    """Stages 1 and 2: the parameters and the placement. Returns the module and the output."""
    regressor = _Craniofacial(model)
    rotation0, translation0, height0 = _initial_placement(
        model, reference, regressor, options
    )
    params = _Parameters(
        model, rotation0, translation0, height0, options.shared_targets
    )
    if options.initial_pose is not None:
        import roma

        start = np.asarray(options.initial_pose, dtype=np.float64)[params.free]
        with torch.no_grad():
            params.rotvec.copy_(
                roma.rotmat_to_rotvec(torch.as_tensor(start, dtype=params.rotvec.dtype))
            )
    excluded = excluded_vertices(model)
    if options.no_data is not None:
        excluded = excluded | np.asarray(options.no_data, bool)
    fit_rows = np.nonzero(~excluded)[0]
    triangles = model.faces.cpu().numpy()
    rng = np.random.default_rng(options.seed)
    # a reference's open borders (a statue's cut supports and hat) take no pairs: the
    # surface past a cut would pull anny's skin onto the cut's edge
    border = reference.regions.get("border")
    candidates = (
        np.arange(len(reference.vertices))
        if border is None
        else np.nonzero(~np.asarray(border, bool))[0]
    )
    sample = rng.choice(
        candidates,
        min(options.reverse_samples, len(candidates)),
        replace=False,
    )
    sample_points = torch.as_tensor(reference.vertices[sample])
    sample_normals = torch.as_tensor(reference.normals[sample])
    # a reference with ears: anny's outer ears match its ears alone, as in fit_levels
    split_target = None
    if options.ear_split and np.asarray(reference.regions.get("ear", [])).any():
        with torch.no_grad():
            rest = model(phenotype_kwargs=params.phenotype)["rest_vertices"][0]
        anny_ear = pinna(model, rest.numpy(), None, options.pinna_height)
        split_target = Target(
            reference,
            split=EarSplit(reference, options.ear_attachment_reach),
            anny_ear=anny_ear,
        )
        sample_ear = np.asarray(reference.regions["ear"], bool)[sample]
        fit_ear = anny_ear[fit_rows]

    # landmark targets and anny's side of each
    labels = list(model.bone_labels)
    names, kinds, rows, weights, targets = [], [], [], [], []
    auto_vertices = None
    if reference.auto:
        with torch.no_grad():
            out = model(phenotype_kwargs=params.phenotype)
        auto_vertices = auto_landmark_vertices(out["vertices"][0].numpy())
    for name, point in reference.landmarks.items():
        if reference.auto:
            if name not in auto_vertices:
                continue
            kinds.append("vertices")
            rows.append(torch.as_tensor(auto_vertices[name]))
            weights.append(options.auto_landmark_weight)
        elif name in LANDMARKS and LANDMARKS[name][2] > 0:
            kind, label, weight = LANDMARKS[name]
            kinds.append(kind)
            rows.append(
                labels.index(label) if kind == "bone" else regressor.labels.index(label)
            )
            weights.append(options.landmark_weight * weight)
        else:
            continue
        names.append(name)
        targets.append(point)
    lm_targets = torch.as_tensor(np.stack(targets)) if targets else None
    lm_weights = torch.as_tensor(weights, dtype=model.dtype)
    cranio = regressor.weights_matrix

    def forward():
        out = model(
            pose_parameters=params.pose(),
            phenotype_kwargs=params.phenotype,
            local_changes_kwargs=params.local,
            face_shape_kwargs=params.face,
        )
        R, t = params.placement()
        placed = out["vertices"][0] @ R.T + t
        return out, placed, R, t

    def landmark_loss(out, placed, R, t):
        if lm_targets is None:
            return placed.new_zeros(())
        joints = out["bone_poses"][0, :, :3, 3] @ R.T + t
        points = []
        for kind, row in zip(kinds, rows):
            if kind == "bone":
                points.append(joints[row])
            elif kind == "craniofacial":
                points.append(cranio[row] @ placed)
            else:
                points.append(placed[row].mean(0))
        points = torch.stack(points)
        sq = ((points - lm_targets) ** 2).sum(-1)
        return (sq * lm_weights).sum() / lm_weights.sum()

    groups = [
        dict(params=[params.phenotype], lr=0.01),
        dict(params=[params.local], lr=0.01),
        dict(params=[params.face], lr=0.01),
        dict(params=[params.rotvec], lr=0.005),
        dict(params=[params.global_rotvec], lr=0.002),
        dict(params=[params.global_translation], lr=0.002),
    ]
    optimizer = torch.optim.Adam(groups)
    lo, hi = options.distance_limits
    fit_rows_t = torch.as_tensor(fit_rows)
    from scipy.spatial import cKDTree

    for round_ in range(options.rounds):
        major_only = round_ < options.major_rounds
        limit = hi + (lo - hi) * max(0.0, 1.0 - round_ / max(options.rounds - 1, 1))
        with torch.no_grad():
            out, placed, R, t = forward()
        X = placed.numpy()
        if split_target is None:
            C, N, d, tri, bary = reference.closest(X[fit_rows])
            if border is not None:
                on_border = (
                    np.asarray(border, bool)[reference.triangles[tri]] | (bary < 1e-6)
                ).all(1)
                along = np.sqrt(
                    np.maximum(d**2 - ((X[fit_rows] - C) * N).sum(1) ** 2, 0.0)
                )
                d = np.where(on_border & (along > 0.5 * d) & (d > 2e-4), np.inf, d)
        else:
            C, N, d, off = split_target.closest(X[fit_rows], fit_rows, edge=True)
        normals = vertex_normals(X, triangles)[fit_rows]
        ok = (d < limit) & ((normals * N).sum(1) > options.normal_threshold)
        if split_target is not None:
            ok &= ~off
        fwd_rows = fit_rows_t[torch.as_tensor(ok)]
        fwd_points = torch.as_tensor(C[ok])
        fwd_normals = torch.as_tensor(N[ok])
        if split_target is None:
            nearest_d, nearest = cKDTree(X[fit_rows]).query(reference.vertices[sample])
        else:
            nearest_d = np.full(len(sample), np.inf)
            nearest = np.zeros(len(sample), np.int64)
            for on_ref, on_anny in ((sample_ear, fit_ear), (~sample_ear, ~fit_ear)):
                local = np.nonzero(on_anny)[0]
                if len(local) and on_ref.any():
                    dd, nn = cKDTree(X[fit_rows][local]).query(
                        reference.vertices[sample][on_ref]
                    )
                    nearest_d[on_ref], nearest[on_ref] = dd, local[nn]
        rev_ok = nearest_d < limit
        rev_rows = fit_rows_t[torch.as_tensor(nearest[rev_ok])]
        rev_points = sample_points[torch.as_tensor(rev_ok)]
        rev_normals = sample_normals[torch.as_tensor(rev_ok)]
        lm_scale = 1.0 if round_ < options.major_rounds else 0.3
        for step in range(options.steps):
            optimizer.zero_grad()
            out, placed, R, t = forward()
            diff = placed[fwd_rows] - fwd_points
            plane = (diff * fwd_normals).sum(-1)
            data = (plane**2).mean() + 0.1 * (diff**2).sum(-1).mean()
            rdiff = placed[rev_rows] - rev_points
            data = data + 0.5 * ((rdiff * rev_normals).sum(-1) ** 2).mean()
            loss = data + lm_scale * landmark_loss(out, placed, R, t)
            loss = loss + options.local_change_ridge * (params.local**2).sum()
            loss = loss + options.face_shape_ridge * (params.face**2).sum()
            loss = loss + options.pose_ridge * (params.rotvec**2).sum()
            loss.backward()
            if major_only:
                params.local.grad = None
                params.face.grad = None
                mask = torch.zeros(len(params.free), 1, dtype=params.rotvec.dtype)
                mask[params.major] = 1.0
                params.rotvec.grad = params.rotvec.grad * mask
            else:
                params.local.grad = params.local.grad * params.local_free
                params.face.grad = params.face.grad * params.face_free
            if not major_only and round_ < options.finger_rounds:
                mask = torch.ones(len(params.free), 1, dtype=params.rotvec.dtype)
                mask[params.fingers] = 0.0
                params.rotvec.grad = params.rotvec.grad * mask
            optimizer.step()
            params.project()
        _log(
            options,
            f"round {round_}: {1000 * np.sqrt(np.mean(d[ok] ** 2)):.2f} mm rms over "
            f"{ok.mean():.0%} of the vertices (limit {1000 * limit:.0f} mm)",
        )
    with torch.no_grad():
        out, placed, R, t = forward()
    return params, out, R.numpy(), t.numpy()


def _character_from(model, params, name: str) -> Character:
    phenotype = {
        k: round(float(v), 6)
        for k, v in zip(model.phenotype_labels, params.phenotype[0].tolist())
    }
    local = {
        k: round(float(v), 6)
        for k, v in zip(model.local_change_labels, params.local[0].tolist())
        if abs(v) > 1e-6
    }
    face = {
        k: round(float(v), 6)
        for k, v in zip(model.face_shape_labels, params.face[0].tolist())
        if abs(v) > 1e-6
    }
    character = Character(
        name=name,
        topology=_fit_topology(model),
        phenotype=phenotype,
        local_changes=local,
        face_shapes=face,
    )
    pose = Character.pose_from_parameters(model, params.pose().detach())
    if pose["bones"]:
        character.pose = pose
    return character


def _card_rest(model, character: Character) -> tuple[np.ndarray, np.ndarray]:
    """The parametric rest mesh (V0, 3) and the skinning transforms (J, 4, 4) of a card,
    from its rounded values, as the exporters evaluate them."""
    from opensculptboy.detail.layer import bone_transforms

    with torch.no_grad():
        out = model(
            pose_parameters=character.pose_parameters(model),
            phenotype_kwargs=dict(character.phenotype) or None,
            local_changes_kwargs=dict(character.local_changes) or None,
            face_shape_kwargs=dict(character.face_shapes) or None,
        )
    return (
        out["rest_vertices"][0].double().numpy(),
        bone_transforms(out)[0].double().numpy(),
    )


def fit_reference(
    reference: Reference,
    detail: int = 1,
    name: str = "fitted",
    options: FitOptions | None = None,
    model=None,
) -> FitResult:
    """
    Fit anny to a reference and measure the result (see the module documentation).

    Args:
        reference: the target surface in Anny's frame.
        detail: the finest level of the detail layer, 0, 1 or 2; the result holds the
            offsets of every level up to it.
        name: the character's name.
        options: the settings; :class:`FitOptions` by default.
        model: the fit model; :func:`fit_model` by default.
    """
    if detail not in (0, 1, 2):
        raise ValueError(f"detail must be 0, 1 or 2; got {detail!r}.")
    options = options or FitOptions()
    model = model or fit_model(options.topology)
    if _fit_topology(model) != options.topology:
        raise ValueError(
            f"The fit model has the topology {_fit_topology(model)!r} and the options "
            f"{options.topology!r}."
        )
    clock = time.perf_counter()
    params, _, rotation, translation = fit_parameters(model, reference, options)
    character = _character_from(model, params, name)
    seconds = {"parameters": time.perf_counter() - clock}
    return fit_levels(
        reference, character, rotation, translation, detail, options, model, seconds
    )


def _lip_line(
    model, reference, rest0, weights0, indices0, transforms, rotation, translation
):
    """The mouth's box (anny's craniofacial landmarks on the parametric mesh, posed and
    placed) and the line where the reference's closed lips meet (``lips.lip_line``)."""
    from opensculptboy.fit import lips as L_

    posed = R_.pose_vertices(
        rest0, weights0, indices0, transforms, rotation, translation
    )
    cranio = _Craniofacial(model)
    points = cranio.weights_matrix.detach().double().cpu().numpy() @ posed
    named = dict(zip(cranio.labels, points))
    if not all(k in named for k in ("ch.L", "ch.R", "ls", "li")):
        return None, None
    box = L_.MouthBox.from_landmarks(named)
    return box, L_.lip_line(reference.vertices, reference.triangles, box)


def _seal_level(
    offsets,
    parametric,
    topology,
    box,
    line,
    zone,
    cavity,
    jaw,
    pose,
    unpose,
    rings,
    reference=None,
    reach: float = 0.0,
    with_corners: bool = True,
    beyond: float = 0.0,
):
    """Seal the lips of one level (``lips.seal_lips``): the edges of anny's lips found on the
    level's parametric mesh, posed and placed (``lips.lip_edges`` of ``lips.lip_masks``: the
    upper lip is the skin of the mouth that the jaw's opening leaves, the lower lip the skin
    it carries, split at the middle of the mouth around the corners), and the mouth's corners
    (``lips.lip_corners``), moved
    onto ``line``; the mouth's ``zone`` follows, but its skin on the ``reference`` (within
    0.2 mm, the normals agreeing) farther than ``reach`` (metres) from those rows."""
    from opensculptboy.fit import lips as L_

    everything = np.arange(len(parametric))
    posed = pose(parametric, everything)
    roll = zone & ~cavity
    T = np.asarray(topology.triangles)
    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]]])
    edges = edges[roll[edges].all(1)]
    if len(edges) == 0:
        return offsets, L_.Seal(
            everything[:0], everything[:0], np.zeros(len(everything), bool), 0.0
        )
    spacing = float(
        np.median(np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1))
    )
    upper, lower = L_.lip_edges(
        posed, *L_.lip_masks(posed, roll, jaw, box, spacing), box, spacing
    )
    corners = L_.lip_corners(posed, roll, box) if with_corners else everything[:0]
    held = None
    if reference is not None and reach > 0:
        from scipy.spatial import cKDTree

        current = pose(parametric + offsets, everything)
        rows = np.nonzero(zone)[0]
        _, normal, d = reference.closest(current[rows])[:3]
        mine = vertex_normals(current, T)[rows]
        on = (d < 2e-4) & ((normal * mine).sum(1) > 0.5)
        seeds = np.concatenate([upper, lower, corners])
        far = cKDTree(current[seeds]).query(current[rows])[0] > reach
        held = np.zeros(len(everything), bool)
        held[rows[on & far]] = True
    return L_.seal_lips(
        offsets,
        parametric,
        T,
        pose,
        unpose,
        line,
        upper,
        lower,
        zone,
        rings,
        corners=corners,
        held=held,
        beyond=beyond,
    )


def fit_levels(
    reference: Reference,
    character: Character,
    rotation: np.ndarray,
    translation: np.ndarray,
    detail: int = 1,
    options: FitOptions | None = None,
    model=None,
    seconds: dict | None = None,
) -> FitResult:
    """
    Stages 3 and 4 alone: the non-rigid refinement of each level for a card whose parameters
    and placement are already fitted (``x_reference = rotation @ x_anny + translation``), as
    :func:`fit_reference` runs them after the parameters.
    """
    from opensculptboy.detail import detail_topology

    options = options or FitOptions()
    model = model or fit_model(character.topology)
    if _fit_topology(model) != character.topology:
        raise ValueError(
            f"The fit model has the topology {_fit_topology(model)!r} and the card "
            f"{character.topology!r}."
        )
    seconds = dict(seconds or {})
    rest0, transforms = _card_rest(model, character)
    excluded = excluded_vertices(model)
    anchored = PatchAnchors(model, excluded, rest0)
    seat = gazes = eyeball0 = None
    if options.seat_lids and reference.eyes:
        from opensculptboy.export.vrm_tables import eyeball_vertices

        try:
            eyeball0 = np.zeros(len(rest0))
            for rows in eyeball_vertices(model).values():
                eyeball0[rows] = 1.0
            gazes = eye_gazes(model, reference.eyes, transforms, rotation, translation)
            seat = True
        except ValueError:
            seat = None
    errors, offsets = {}, {}
    weights0 = model.vertex_bone_weights.detach().double().numpy()
    indices0 = model.vertex_bone_indices.cpu().numpy()
    previous = np.zeros_like(rest0)
    previous_level = 0
    ears = ear_weights(model)
    split = None
    if options.ear_split and np.asarray(reference.regions.get("ear", [])).any():
        split = EarSplit(reference, options.ear_attachment_reach)
        outer0 = pinna(model, rest0, ears, options.pinna_height).astype(np.float64)
    margin = lip_margin(model, excluded, options.lip_margin_rings).astype(np.float64)
    lining0 = mouth_lining(model, options.lip_margin_rings).astype(np.float64)
    cavity0 = mouth_lining(model, 0).astype(np.float64)
    from opensculptboy.fit.metrics import mouth_zone

    jaw0 = jaw_share(model)
    zone0 = mouth_zone(model, rest0)
    mouth0 = {}  # the mouth that repair_mouth may move, by its reach (per level)
    for reach in set(_as_levels(options.mouth_repair_reach)):
        mask = zone0 if reach is None else zone0 & mouth_lining(model, reach)
        mouth0[reach] = mask.astype(np.float64)
    zone_lips0 = mouth_zone(model, rest0).astype(np.float64)
    lip_box = lip_line_ = None
    lip_rows = {}
    if options.lip_seal and options.mouth_lining == "smooth":
        lip_box, lip_line_ = _lip_line(
            model,
            reference,
            rest0,
            weights0,
            indices0,
            transforms,
            rotation,
            translation,
        )
        _log(
            options,
            "lips: no line where the reference's lips meet"
            if lip_line_ is None
            else f"lips: the reference's lips end along {len(lip_line_.upper)} points, "
            f"{1000 * float(np.median(lip_line_.gap())):.2f} mm apart (median)",
        )
        if (
            lip_line_ is not None
            and float(np.median(lip_line_.gap())) > options.lip_seal_max_gap
        ):
            _log(options, "lips: the reference's lips stand apart; no seal")
            lip_line_ = None
    skin0 = None
    if any(f > 0 for f in options.skin_fairness):
        from opensculptboy.fit import metrics as skin_metrics

        regions_skin = skin_metrics.anny_regions(model, rest0)
        skin0 = np.isin(
            regions_skin,
            [skin_metrics.REGIONS.index(k) for k in options.skin_fair_regions],
        ).astype(np.float64)
        # the lips' inner margin, the lining of the mouth and the lips' roll (the skin the
        # seal of the lips moves) keep to the mouth's own rules
        roll0 = mouth_lining(model, options.lip_seal_rings) > 0
        skin0[(margin > 0) | (lining0 > 0) | roll0] = 0.0
    measures, meshes = {}, {}
    previous_ear = None
    regions0 = reference_region = None
    if options.measure:
        from opensculptboy.fit import metrics

        regions0 = metrics.anny_regions(model, rest0)
    for subdiv in range(detail + 1):
        clock = time.perf_counter()
        topology = detail_topology(model, subdiv, "full", character.topology)
        w, i = topology.skin_weights(weights0, indices0)
        parametric = np.asarray(topology.apply(rest0))
        carried = None
        if subdiv == 0:
            start_rest = parametric
        else:
            up = detail_topology(model, previous_level, "full", character.topology)
            carried = _lift(previous, up, topology)
            start_rest = parametric + carried
        posed = R_.pose_vertices(start_rest, w, i, transforms, rotation, translation)
        mask = np.asarray(topology.apply(excluded.astype(np.float64))) < 0.5
        level_ears = np.asarray(topology.apply(ears))
        regions = None
        if options.measure:
            regions = metrics.level_regions(topology, regions0)
            if reference_region is None:
                reference_region = metrics.reference_regions(reference, posed, regions)
        if subdiv == 0:
            errors["parametric"] = R_.surface_error(
                posed, topology.triangles, reference, mask, seed=options.seed
            )
            if options.measure:
                measures["parametric"] = metrics.measure(
                    posed, topology.triangles, regions, reference, reference_region
                ).summary()
        anny_ear = None
        stiff = level_ears
        if split is not None:
            level_pinna = np.clip(np.asarray(topology.apply(outer0)), 0.0, 1.0)
            anny_ear = level_pinna > 0.5
            if subdiv > 0 and options.ear_relabel_rings > 0:
                if previous_ear is not None:
                    # the mask of the level below, as the offsets are carried
                    lifted = _lift(previous_ear[:, None], up, topology)[:, 0]
                    anny_ear = np.asarray(lifted) > 0.5
                anny_ear = relabel_ears(
                    posed,
                    topology.triangles,
                    anny_ear,
                    split,
                    options.ear_relabel_rings * 2 ** (subdiv - 1),
                )
                level_pinna = anny_ear.astype(np.float64)
                previous_ear = level_pinna
            stiff = level_pinna
        scale = None
        ear_stiffness = (
            options.ear_stiffness if split is None else options.ear_split_stiffness
        )
        if ear_stiffness != 1.0:
            scale = 1.0 + (ear_stiffness - 1.0) * stiff
        data = mask.astype(np.float64)
        if options.no_data is not None:
            lost = np.asarray(topology.apply(np.asarray(options.no_data, np.float64)))
            data *= 1.0 - np.clip(lost, 0.0, 1.0)
        if options.lip_margin_weight != 1.0:
            data *= 1.0 - (1.0 - options.lip_margin_weight) * np.clip(
                np.asarray(topology.apply(margin)), 0.0, 1.0
            )
        # on a detail level, the vertices that come from a patch of the role "anchors" (in
        # part, as the skin along its border) take the rule alone: kept as fitted, the skin
        # along the border moves apart from the patch it carries. The refinement still fits
        # them, so that the reference's points there do not drag the skin around them in
        fit_mask = mask
        if anchored.active and subdiv > 0:
            fit_mask = mask & (anchored.share(topology) == 0)
        threshold = np.where(
            stiff > 0.5,
            max(options.ear_normal_threshold, options.normal_threshold),
            options.normal_threshold,
        )
        target = Target(
            reference,
            options.target_smooth,
            options.target_shading,
            split,
            anny_ear,
            border=reference.regions.get("border"),
        )
        fitted = nicp(
            posed,
            topology.triangles,
            target,
            data,
            options.stiffness if subdiv == 0 else options.detail_stiffness,
            options.nicp_iterations,
            options.nicp_max_distance,
            threshold,
            stiffness_scale=scale,
            reverse=options.reverse,
            attachment_weight=options.ear_attachment_weight if split else 0.0,
        )
        if split is not None:
            region = anny_ear.copy()
            adjacency = _adjacency(len(fitted), topology.triangles)
            for _ in range(options.ear_fair_rings * 2**subdiv):
                region |= adjacency @ region.astype(np.float64) > 0
            region &= data > 0
            fitted = fair_region(
                fitted,
                topology.triangles,
                target,
                region,
                options.ear_fair_iterations,
                options.ear_fairness,
                options.nicp_max_distance,
                pool=data > 0,
            )
            fitted, defects = repair_folds(
                fitted,
                topology.triangles,
                target,
                region,
                options.ear_repair_rounds,
                max_distance=options.nicp_max_distance,
                pool=data > 0,
            )
            _log(options, f"level {subdiv}: {defects} ear fold defects repaired")
        elif options.ear_relax_iterations > 0:
            fitted = relax_along(
                fitted,
                topology.triangles,
                target,
                level_ears * (data > 0),
                options.ear_relax_iterations,
                options.nicp_max_distance,
            )

        def place(points):
            # the vertices of the role "anchors" at the blends of their anchors
            if not anchored.active:
                return points
            rest = R_.unpose(points, w, i, transforms, rotation, translation)
            moved = (
                anchored.apply(
                    rest - parametric, topology, parametric, fit_mask, carried
                )
                + parametric
            )
            rows = anchored.share(topology) > 0
            out = points.copy()
            out[rows] = R_.pose_vertices(
                moved[rows], w[rows], i[rows], transforms, rotation, translation
            )
            return out

        if options.measure:
            before = place(
                follow_skin(
                    posed, fitted, ~mask, topology.triangles, eyes=reference.eyes
                )
            )
            before = metrics.measure(
                before, topology.triangles, regions, reference, reference_region
            ).summary()
        if options.project_distance > 0:
            # not the outer ears, whose folds anny's ear does not share; with a split
            # reference, the skin around them onto the head alone
            keep = (data > 0) & (level_ears < 0.1 if split is None else ~anny_ear)
            fitted = project_onto(
                fitted,
                topology.triangles,
                target,
                keep,
                options.project_distance,
                options.project_normal_threshold,
            )
        fairness = _per_level(options.skin_fairness, subdiv)
        if fairness > 0 and skin0 is not None:
            outer = anny_ear if split is not None else level_ears >= 0.1
            held = (
                anchored.share(topology) > 0
                if anchored.active
                else np.zeros(len(fitted), bool)
            )
            if seat is not None:
                # the lids and the eye corners: seat_lids places and fairs them
                held = held | (
                    lid_region(
                        fitted,
                        topology.triangles,
                        reference.eyes,
                        gazes,
                        mask,
                        np.asarray(topology.apply(eyeball0)) > 0.5,
                        options.lid_reach,
                        options.lid_lining_reach,
                        options.lid_front,
                    ).region
                )
            anchor = None
            if seat is not None and options.skin_fair_lid_rings > 0:
                rings = options.skin_fair_lid_rings * 2**subdiv
                blend = _per_level(_as_levels(options.skin_fair_lid_blend), subdiv)
                if blend > 0:
                    # the band around the lids is faired too, held more the nearer it lies
                    # to them: held fast, it kept the reference's facets and ended in a
                    # step under the lower lid
                    band, anchor = lid_blend(
                        topology.triangles,
                        held,
                        options.skin_fair_lid_blend_rings * 2**subdiv,
                        blend,
                    )
                    ring = held.copy()
                    adjacency = _adjacency(len(fitted), topology.triangles)
                    for _ in range(rings):
                        ring = ring | (adjacency @ ring.astype(np.float64) > 0)
                    held = held | (ring & ~band)
                else:
                    adjacency = _adjacency(len(fitted), topology.triangles)
                    for _ in range(rings):
                        held = held | (adjacency @ held.astype(np.float64) > 0)
            skin_region = skin_fair_region(
                topology,
                skin0,
                data,
                outer,
                options.ear_fair_rings * 2**subdiv,
                held,
            )
            fitted = fair_skin(
                fitted,
                topology.triangles,
                target,
                skin_region,
                fairness,
                options.skin_fair_iterations,
                options.skin_fair_distance,
                anchor=anchor,
            )
            _log(
                options,
                f"level {subdiv}: {int(skin_region.sum())} skin vertices faired",
            )
        fitted = follow_skin(
            posed, fitted, ~mask, topology.triangles, eyes=reference.eyes
        )
        registered = R_.unpose(fitted, w, i, transforms, rotation, translation)
        offsets[subdiv] = anchored.apply(
            R_.rest_offsets(registered, parametric),
            topology,
            parametric,
            fit_mask,
            carried,
        )
        if options.mouth_lining == "smooth":
            level_lining = np.asarray(topology.apply(lining0)) > 0.5
            offsets[subdiv] = smooth_mouth(
                offsets[subdiv], topology.triangles, level_lining
            )
            before_repair = offsets[subdiv]
            level_mouth = (
                np.asarray(
                    topology.apply(
                        mouth0[
                            _per_level(_as_levels(options.mouth_repair_reach), subdiv)
                        ]
                    )
                )
                > 0.5
            )
            held = np.zeros(len(parametric), bool)
            if lip_line_ is not None:
                unsealed = offsets[subdiv]
                offsets[subdiv], seal = _seal_level(
                    offsets[subdiv],
                    parametric,
                    topology,
                    lip_box,
                    lip_line_,
                    np.asarray(topology.apply(zone_lips0)) > 0.5,
                    np.asarray(topology.apply(cavity0)) > 0.5,
                    np.asarray(topology.apply(jaw0)),
                    lambda p, r: R_.pose_vertices(
                        p, w[r], i[r], transforms, rotation, translation
                    ),
                    lambda p, r: R_.unpose(
                        p, w[r], i[r], transforms, rotation, translation
                    ),
                    options.lip_seal_rings * 2**subdiv,
                    reference,
                    _per_level(_as_levels(options.lip_seal_reach), subdiv),
                    options.lip_seal_corners,
                    options.lip_seal_beyond,
                )
                if seal.shift_mm > 1000 * options.lip_seal_max_shift:
                    _log(
                        options,
                        f"level {subdiv}: lips left unsealed (their rows would move "
                        f"{seal.shift_mm:.2f} mm to the line)",
                    )
                    offsets[subdiv] = unsealed
                    seal = None
            if lip_line_ is not None and seal is not None:
                held[seal.upper] = True
                held[seal.lower] = True
                held[seal.corners] = True
                lip_rows[subdiv] = (seal.upper, seal.lower)
                _log(
                    options,
                    f"level {subdiv}: lips sealed ({len(seal.upper)} and "
                    f"{len(seal.lower)} rows moved {seal.shift_mm:.2f} mm)",
                )
            if options.lip_contact > 0:
                jaw = np.asarray(topology.apply(jaw0))
                roll = level_mouth & ~(np.asarray(topology.apply(cavity0)) > 0.5)
                pairs = lip_contact(
                    parametric,
                    roll & (jaw < 0.3),
                    roll & (jaw > 0.6),
                    options.lip_contact,
                )
                offsets[subdiv] = stitch_lips(offsets[subdiv], parametric, pairs)
                _log(options, f"level {subdiv}: {len(pairs)} lip pairs stitched")
                held[np.unique(pairs)] = True
            if options.mouth_repair_rounds > 0 and options.mouth_repair_hold > 0:
                # the lips' visible skin, on the reference with agreeing normals, stays: the
                # closed lips meet there in the reference's crease
                posed_now = R_.pose_vertices(
                    parametric + offsets[subdiv],
                    w,
                    i,
                    transforms,
                    rotation,
                    translation,
                )
                rows = np.nonzero(level_mouth & ~held)[0]
                C, N, d = target.closest(posed_now[rows], rows)[:3]
                normals = vertex_normals(posed_now, topology.triangles)[rows]
                held[
                    rows[(d < options.mouth_repair_hold) & ((normals * N).sum(1) > 0.5)]
                ] = True
            if options.mouth_repair_rounds > 0:
                offsets[subdiv], tears = repair_mouth(
                    offsets[subdiv],
                    parametric,
                    topology.triangles,
                    level_mouth & ~held,
                    options.mouth_repair_rounds,
                    options.mouth_repair_rings,
                )
                _log(options, f"level {subdiv}: {tears} mouth tear vertices repaired")
            corner_fairness = _per_level(
                _as_levels(options.lip_corner_fairness), subdiv
            )
            if corner_fairness > 0 and lip_line_ is not None and seal is not None:
                offsets[subdiv], faired = _fair_corners(
                    offsets[subdiv],
                    parametric,
                    topology.triangles,
                    target,
                    lip_line_.ends,
                    np.asarray(topology.apply(cavity0)) > 0.5,
                    (seal.upper, seal.lower),
                    lambda p, r: R_.pose_vertices(
                        p, w[r], i[r], transforms, rotation, translation
                    ),
                    lambda p, r: R_.unpose(
                        p, w[r], i[r], transforms, rotation, translation
                    ),
                    corner_fairness,
                    options.lip_corner_radius,
                    options.lip_corner_hold,
                    options.skin_fair_iterations,
                    options.skin_fair_distance,
                    options.lip_corner_normal_threshold,
                )
                _log(
                    options,
                    f"level {subdiv}: {faired} vertices at the lips' corners faired",
                )
            level_lining = level_lining | (
                np.abs(offsets[subdiv] - before_repair).max(1) > 0
            )
            fitted = fitted.copy()
            fitted[level_lining] = R_.pose_vertices(
                (parametric + offsets[subdiv])[level_lining],
                w[level_lining],
                i[level_lining],
                transforms,
                rotation,
                translation,
            )
        elif options.mouth_lining != "follow":
            raise ValueError(
                f"mouth_lining is 'smooth' or 'follow'; got {options.mouth_lining!r}."
            )
        if anchored.active:
            rows = anchored.share(topology) > 0
            fitted = fitted.copy()
            fitted[rows] = R_.pose_vertices(
                (parametric + offsets[subdiv])[rows],
                w[rows],
                i[rows],
                transforms,
                rotation,
                translation,
            )
        if seat is not None:
            level_seat = lid_region(
                fitted,
                topology.triangles,
                reference.eyes,
                gazes,
                mask,
                np.asarray(topology.apply(eyeball0)) > 0.5,
                options.lid_reach,
                options.lid_lining_reach,
                options.lid_front,
            )
            fitted = seat_lids(
                fitted,
                topology.triangles,
                Target(reference, options.target_smooth, options.target_shading),
                level_seat,
                sphere=options.lid_sphere,
                margin=options.lid_margin,
                iterations=options.lid_iterations,
                fairness=options.lid_fairness * 16.0**subdiv,
                weight=options.lid_weight,
                front=options.lid_front,
                clearance_rounds=_per_level(options.lid_clearance_rounds, subdiv),
                clearance_inward=_per_level(options.lid_clearance_inward, subdiv),
                follow_inside=_per_level(options.lid_follow_inside, subdiv),
            )
            rows = level_seat.region
            offsets[subdiv] = offsets[subdiv].copy()
            offsets[subdiv][rows] = (
                R_.unpose(
                    fitted[rows], w[rows], i[rows], transforms, rotation, translation
                )
                - parametric[rows]
            )
            fitted[rows] = R_.pose_vertices(
                (parametric + offsets[subdiv])[rows],
                w[rows],
                i[rows],
                transforms,
                rotation,
                translation,
            )
        if options.close_openings > 0:
            fitted, offsets[subdiv] = _close_openings(
                model,
                topology,
                fitted,
                offsets[subdiv],
                parametric,
                reference,
                options,
                (w, i, transforms, rotation, translation),
            )
        errors[subdiv] = R_.surface_error(
            fitted, topology.triangles, reference, mask, seed=options.seed
        )
        if options.measure:
            measures[subdiv] = {
                "before_projection": before,
                "final": metrics.measure(
                    fitted, topology.triangles, regions, reference, reference_region
                ).summary(),
            }
            if lip_box is not None:
                from opensculptboy.fit import lips as L_

                measures[subdiv]["lips"] = L_.lip_slit(
                    fitted,
                    topology.triangles,
                    reference.vertices,
                    reference.triangles,
                    lip_box,
                )
                if subdiv in lip_rows:
                    measures[subdiv]["lips"]["gap"] = L_.lip_gap(
                        parametric + offsets[subdiv], *lip_rows[subdiv]
                    )
                _log(options, f"level {subdiv}: lips {measures[subdiv]['lips']}")
        meshes[subdiv] = (fitted, topology.triangles)
        previous, previous_level = offsets[subdiv], subdiv
        seconds[f"level{subdiv}"] = time.perf_counter() - clock
        _log(
            options,
            f"level {subdiv}: {errors[subdiv]['forward']['mean']:.3f} mm mean "
            f"({seconds[f'level{subdiv}']:.0f} s)",
        )
    return FitResult(
        character=character,
        rotation=np.asarray(rotation),
        translation=np.asarray(translation),
        parametric_rest=rest0,
        offsets=offsets,
        errors=errors,
        seconds=seconds,
        reference=reference,
        measures=measures,
        meshes=meshes,
    )


def cavity_vertices(model) -> np.ndarray:
    """Mask (V,) of the linings of the eye sockets and of the mouth (EXCLUDED_LABELS)."""
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.topology.lookup import base_mesh_rows

    mesh = base_mesh()
    out_base = np.zeros(len(mesh.V), bool)
    for label in EXCLUDED_LABELS:
        out_base[np.unique(mesh.quads[mesh.face_labels == label])] = True
    return out_base[base_mesh_rows(model)]


def _close_openings(
    model, topology, fitted, offsets, parametric, reference, options, skin
):
    """FitOptions.close_openings on one level: the posed vertices and the offsets."""
    from opensculptboy.export.vrm_tables import eyeball_vertices

    w, i, transforms, rotation, translation = skin
    fitted = fitted.copy()
    lining = np.asarray(topology.apply(cavity_vertices(model).astype(np.float64))) > 0.5
    rows = np.nonzero(lining)[0]
    C, _, d, _, _ = reference.closest(fitted[rows])
    near = d < options.close_openings
    fitted[rows[near]] = C[near]
    moved = [rows[near]]
    try:
        eyes = eyeball_vertices(model)
    except ValueError:
        eyes = {}
    for side_rows in eyes.values():
        base = np.zeros(model.template_vertices.shape[0])
        base[side_rows] = 1.0
        mark = np.asarray(topology.apply(base)) > 0.5
        eye = np.nonzero(mark)[0]
        if len(eye):
            centre = fitted[eye].mean(0)
            fitted[eye] = centre + options.eyeball_shrink * (fitted[eye] - centre)
            moved.append(eye)
    changed = np.unique(np.concatenate(moved)) if moved else np.zeros(0, np.int64)
    offsets = offsets.copy()
    if len(changed):
        offsets[changed] = (
            R_.unpose(
                fitted[changed],
                w[changed],
                i[changed],
                transforms,
                rotation,
                translation,
            )
            - parametric[changed]
        )
    return fitted, offsets


def _lift(offsets: np.ndarray, coarse, fine) -> np.ndarray:
    """Offsets of a coarser full level carried to a finer one by the subdivision."""
    steps = fine.operators[len(coarse.operators) :]
    x = offsets
    for operator in steps:
        x = operator.apply(x)
    return np.asarray(x)
