// OpenSculptBoy
// Apache License, Version 2.0
//
// A test page for VRM files (run by test/test_vrm_three.py): it loads a VRM 1.0 or 0.x file with
// three-vrm, the VRM loader that web VTuber apps build on, and gives the test its hooks on window.
// test_vrm_three.py bundles it with esbuild into a temporary folder, so three-vrm never enters the
// viewer page (build.mjs bundles src/main.ts alone).
//
//   index.html?vrm=<url>    loads the file at <url> (or call window.vrmLoad(url))
//
// The hooks (all angles in degrees, all rotations as world quaternions [x, y, z, w]):
//   vrmInfo()                         the meta version, the humanoid bones, the expressions, ...
//   vrmLookAt(yaw, pitch)             turns the look-at; returns the rotations of the raw eye bones
//   vrmExpression(name, weight)       sets an expression weight; returns the weight three-vrm keeps
//   vrmRoll(bone, axis, angle, nodes) turns a normalised humanoid bone about an axis from the rest
//                                     pose; returns the rotations of the glTF nodes ``nodes``
//   vrmVertices(mesh, prim, indices)  the world positions of vertices of a glTF mesh primitive,
//                                     skinned and morphed as the GPU draws them
//   vrmRender()                       draws one frame; returns the draw calls, the covered share and
//                                     the shader programs that fail to link
//   vrmSprings(frames, speed)         runs the spring bones: the figure rests for ``frames`` frames of
//                                     1/60 s, then its hips move forward at ``speed`` m/s for as many;
//                                     returns the joints, the colliders and the largest move (m) of a
//                                     chain's tail from its place on the figure in each phase
//   vrmNormalMaps()                   the materials with a normal map: the mesh's name, whether its
//                                     geometry has vertex tangents, and the normal scale
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMHumanBoneList, type VRM, type VRMHumanBoneName } from '@pixiv/three-vrm';

type Quat = [number, number, number, number];

const canvas = document.getElementById('view') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setClearColor(0x000000, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, canvas.width / canvas.height, 0.01, 100);
const key = new THREE.DirectionalLight(0xffffff, 0.7 * Math.PI);
key.position.set(1, 2, 3);
scene.add(key, new THREE.AmbientLight(0xffffff, 0.3 * Math.PI));

let vrm: VRM | null = null;
let gltf: any = null;
let nodes: THREE.Object3D[] = [];

function current(): VRM {
  if (!vrm) throw new Error('no VRM loaded');
  return vrm;
}

function quat(object: THREE.Object3D): Quat {
  return object.getWorldQuaternion(new THREE.Quaternion()).toArray() as Quat;
}

function update(): void {
  const v = current();
  v.update(0);
  v.scene.updateMatrixWorld(true);
}

async function vrmLoad(url: string) {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  gltf = await loader.loadAsync(url);
  vrm = gltf.userData.vrm as VRM;
  if (!vrm) throw new Error(`three-vrm found no VRM in ${url}`);
  // the glTF node index of each object, as GLTFLoader records it
  nodes = [];
  vrm.scene.traverse((object) => {
    const association = gltf.parser.associations.get(object);
    if (association && association.nodes !== undefined) nodes[association.nodes] = object;
  });
  scene.add(vrm.scene);
  update();
  // the camera frames the whole figure from the side that the file faces
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const centre = box.getCenter(new THREE.Vector3());
  const height = box.max.y - box.min.y;
  const front = vrm.meta.metaVersion === '0' ? -1 : 1;
  camera.position.set(centre.x, centre.y, centre.z + front * 2.2 * height);
  camera.lookAt(centre);
  return vrmInfo();
}

function vrmInfo() {
  const v = current();
  const names = VRMHumanBoneList as readonly VRMHumanBoneName[];
  const expressions = v.expressionManager;
  const meta: any = v.meta;
  let mtoon = 0;
  v.scene.traverse((object: any) => {
    const materials = Array.isArray(object.material) ? object.material : object.material ? [object.material] : [];
    mtoon += materials.filter((m: any) => m.isMToonMaterial).length;
  });
  return {
    metaVersion: meta.metaVersion,
    name: meta.metaVersion === '1' ? meta.name : meta.title,
    humanBones: names.filter((n) => v.humanoid.getRawBoneNode(n)),
    normalizedBones: names.filter((n) => v.humanoid.getNormalizedBoneNode(n)),
    expressions: expressions ? expressions.expressions.map((e) => e.expressionName) : [],
    presets: expressions ? Object.keys(expressions.presetExpressionMap) : [],
    customs: expressions ? Object.keys(expressions.customExpressionMap) : [],
    lookAt: v.lookAt ? { type: (v.lookAt.applier.constructor as any).type, offsetFromHeadBone: v.lookAt.offsetFromHeadBone.toArray() } : null,
    constraints: v.nodeConstraintManager ? v.nodeConstraintManager.constraints.size : 0,
    springJoints: v.springBoneManager ? v.springBoneManager.joints.size : 0,
    mtoonMaterials: mtoon,
    nodes: nodes.length,
  };
}

function vrmLookAt(yaw: number, pitch: number) {
  const v = current();
  if (!v.lookAt) return null;
  v.lookAt.autoUpdate = false;
  v.lookAt.target = undefined;
  v.lookAt.yaw = yaw;
  v.lookAt.pitch = pitch;
  update();
  const eye = (name: VRMHumanBoneName) => {
    const node = v.humanoid.getRawBoneNode(name);
    return node ? quat(node) : null;
  };
  return { leftEye: eye('leftEye'), rightEye: eye('rightEye') };
}

function vrmExpression(name: string, weight: number) {
  const v = current();
  if (!v.expressionManager) return null;
  v.expressionManager.setValue(name, weight);
  update();
  return v.expressionManager.getValue(name);
}

function vrmRoll(bone: VRMHumanBoneName, axis: [number, number, number], angle: number, indices: number[]) {
  const v = current();
  v.humanoid.resetNormalizedPose();
  const node = v.humanoid.getNormalizedBoneNode(bone);
  if (!node) return null;
  node.quaternion.setFromAxisAngle(new THREE.Vector3(...axis).normalize(), THREE.MathUtils.DEG2RAD * angle);
  update();
  return indices.map((i) => (nodes[i] ? quat(nodes[i]) : null));
}

function vrmVertices(mesh: number, primitive: number, indices: number[]) {
  const v = current();
  update();
  let target: THREE.Mesh | null = null;
  v.scene.traverse((object: any) => {
    const association = gltf.parser.associations.get(object);
    if (object.isMesh && association && association.meshes === mesh && association.primitives === primitive) target = object;
  });
  if (!target) return null;
  const t: THREE.Mesh = target;
  const p = new THREE.Vector3();
  return indices.map((i) => t.getVertexPosition(i, p).applyMatrix4(t.matrixWorld).toArray());
}

function vrmRender() {
  current();
  renderer.render(scene, camera);
  const gl = renderer.getContext();
  const error = gl.getError();
  const width = gl.drawingBufferWidth, height = gl.drawingBufferHeight;
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  let covered = 0;
  for (let i = 0; i < pixels.length; i += 4) if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 0) covered++;
  const programs: any[] = (renderer.info as any).programs ?? [];
  const failed = programs.filter((p) => p.diagnostics && !p.diagnostics.runnable).map((p) => p.name);
  return { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, covered: covered / (width * height), glError: error, failedPrograms: failed };
}

function vrmSprings(frames: number, speed: number) {
  const v = current();
  const manager = v.springBoneManager;
  if (!manager) return null;
  v.humanoid.resetNormalizedPose();
  manager.reset();
  update();
  const joints = Array.from(manager.joints);
  const bones = new Set(joints.map((j) => j.bone));
  // the tail of each chain: the child that no joint turns (VRM 1.0), or a joint without a child
  // (VRM 0.x, whose last node turns a virtual tail)
  const tails: THREE.Object3D[] = [];
  for (const j of joints) {
    if (j.child && !bones.has(j.child)) tails.push(j.child);
    else if (!j.child) tails.push(j.bone);
  }
  const hips = v.humanoid.getNormalizedBoneNode('hips')!;
  const raw = v.humanoid.getRawBoneNode('hips')!;
  const at = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3());
  const start = at(raw);
  const rest = tails.map(at);
  const dt = 1 / 60;
  let still = 0;
  let moving = 0;
  for (let f = 0; f < 2 * frames; f++) {
    if (f >= frames) hips.position.z += speed * dt;
    v.update(dt);
    v.scene.updateMatrixWorld(true);
    const shift = at(raw).sub(start);
    const d = Math.max(0, ...tails.map((t, k) => at(t).distanceTo(rest[k].clone().add(shift))));
    if (f < frames) still = Math.max(still, d);
    else moving = Math.max(moving, d);
  }
  v.humanoid.resetNormalizedPose();
  manager.reset();
  update();
  return { joints: joints.length, colliders: manager.colliders.length, tails: tails.length, still, moving };
}

function vrmNormalMaps() {
  const v = current();
  const found: { mesh: string; tangent: boolean; normalScale: number[] }[] = [];
  v.scene.traverse((object: any) => {
    if (!object.isMesh) return;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const m of materials) {
      if (!m || !m.normalMap) continue;
      found.push({ mesh: object.name, tangent: !!object.geometry.attributes.tangent, normalScale: m.normalScale.toArray() });
    }
  });
  return found;
}

Object.assign(window, { vrmLoad, vrmInfo, vrmLookAt, vrmExpression, vrmRoll, vrmVertices, vrmRender, vrmSprings, vrmNormalMaps });

const url = new URLSearchParams(location.search).get('vrm') ?? (window as any).__VRM_URL;
if (url) {
  vrmLoad(url).then(
    (info) => Object.assign(window, { __VRM_INFO: info, __VRM_READY: true }),
    (error) => Object.assign(window, { __VRM_ERROR: String(error && error.stack ? error.stack : error), __VRM_READY: true }),
  );
}
