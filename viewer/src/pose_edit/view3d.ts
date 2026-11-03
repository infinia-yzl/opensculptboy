// OpenSculptBoy
// Apache License, Version 2.0
//
// The scan in the view: a light copy of the statue's scan drawn over the picture, as a ghost (see-through, brighter at
// its silhouette, over the figure) or solid (lit, hidden where the figure stands in front: the overlay compares its
// depth with the depth of the picture's last pass). The magnifier draws the scan again, close up, round the pointer:
// a second camera with a narrower field of view looks at the same spot, into a square of the canvas, and the points'
// layer draws the ring and the point over it.

import type * as THREE_NS from 'three';
import type { PoseHost } from './host.ts';
import type { ScanMesh } from './scan.ts';

const VS = `
uniform mat4 uModel;
varying vec3 vN; varying vec3 vW;
void main() {
  vec4 w = uModel * vec4(position, 1.0);
  vW = w.xyz;
  vN = normalize(mat3(uModel) * normal);
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const FS = `
uniform float uOpacity; uniform float uSolid; uniform vec3 uCam; uniform sampler2D tDepth; uniform vec2 uDepthScale; uniform vec2 uRes;
uniform float uUseDepth; uniform vec3 uTint; uniform vec3 uClip;
varying vec3 vN; varying vec3 vW;
void main() {
  // the magnifier's disc: nothing outside it
  if (uClip.z > 0.0 && distance(gl_FragCoord.xy, uClip.xy) > uClip.z) discard;
  vec3 n = normalize(vN), v = normalize(uCam - vW);
  if (dot(n, v) < 0.0) n = -n;
  float ndv = clamp(dot(n, v), 0.0, 1.0);
  // a soft key from above the camera and a fill, in the cool grey of the screens
  vec3 key = normalize(vec3(0.3, 0.8, 0.5));
  float lit = 0.35 + 0.55 * max(dot(n, key), 0.0) + 0.15 * ndv;
  vec3 c = uTint * lit;
  float a;
  if (uSolid > 0.5) {
    a = 1.0;
    if (uUseDepth > 0.5) {
      float scene = texture2D(tDepth, gl_FragCoord.xy / uRes * uDepthScale).r;
      if (scene < 1.0 && gl_FragCoord.z > scene + 2e-5) discard;
    }
  } else {
    // the ghost: faint across its faces and stronger at its outline, so the figure shows through it
    float rim = pow(1.0 - ndv, 2.0);
    a = uOpacity * (0.45 + 0.55 * rim);
    c = mix(c, vec3(0.86, 0.92, 1.0), 0.35 * rim);
  }
  gl_FragColor = vec4(c * a, a);
}`;

// the magnifier's background: a dark disc over the square of its viewport (clip space, no camera)
const DISC_VS = `void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const DISC_FS = `uniform vec3 uClip;
void main() {
  if (distance(gl_FragCoord.xy, uClip.xy) > uClip.z) discard;
  gl_FragColor = vec4(0.082, 0.09, 0.102, 1.0);
}`;

export type ScanLook = 'ghost' | 'solid';

export class ScanView {
  host: PoseHost;
  THREE: typeof THREE_NS;
  scene: THREE_NS.Scene;
  mesh: THREE_NS.Mesh | null = null;
  material: THREE_NS.ShaderMaterial;
  shown = true;
  look: ScanLook = 'ghost';
  opacity = 0.45;
  private magCam: THREE_NS.PerspectiveCamera;

  constructor(host: PoseHost) {
    this.host = host;
    const T = (this.THREE = host.THREE);
    this.scene = new T.Scene();
    this.material = new T.ShaderMaterial({
      uniforms: {
        uModel: { value: new T.Matrix4() }, uOpacity: { value: this.opacity }, uSolid: { value: 0 }, uCam: { value: new T.Vector3() },
        tDepth: { value: null }, uDepthScale: { value: new T.Vector2(1, 1) }, uRes: { value: new T.Vector2(1, 1) }, uUseDepth: { value: 1 },
        uTint: { value: new T.Color(0.72, 0.78, 0.86) }, uClip: { value: new T.Vector3(0, 0, 0) },
      },
      vertexShader: VS, fragmentShader: FS, transparent: true, premultipliedAlpha: true, depthWrite: true, depthTest: true, side: T.DoubleSide,
      blending: T.CustomBlending, blendEquation: T.AddEquation, blendSrc: T.OneFactor, blendDst: T.OneMinusSrcAlphaFactor,
    });
    this.magCam = new T.PerspectiveCamera();
    const disc = new T.Mesh(new T.PlaneGeometry(2, 2), new T.ShaderMaterial({ uniforms: { uClip: { value: new T.Vector3() } }, vertexShader: DISC_VS, fragmentShader: DISC_FS,
      depthTest: false, depthWrite: false }));
    disc.frustumCulled = false;
    this.discScene = new T.Scene(); this.discScene.add(disc); this.disc = disc;
  }
  private discScene: THREE_NS.Scene;
  private disc: THREE_NS.Mesh;

  setScan(scan: ScanMesh | null) {
    const T = this.THREE;
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh = null; }
    if (!scan) return;
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.BufferAttribute(scan.positions, 3));
    g.setAttribute('normal', new T.BufferAttribute(scan.normals, 3));
    g.setIndex(new T.BufferAttribute(scan.indices, 1));
    g.computeBoundingSphere();
    this.mesh = new T.Mesh(g, this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
  }

  visible(): boolean { return !!this.mesh && this.shown; }

  // the uniforms of a draw with a camera
  private prepare(cam: THREE_NS.Camera, model: THREE_NS.Matrix4, target: THREE_NS.WebGLRenderTarget, useDepth: boolean) {
    const u = this.material.uniforms, d = this.host.sceneDepth();
    u.uModel.value.copy(model);
    u.uOpacity.value = this.opacity;
    u.uSolid.value = this.look === 'solid' ? 1 : 0;
    u.uCam.value.setFromMatrixPosition(cam.matrixWorld);
    u.tDepth.value = d.texture; u.uDepthScale.value.copy(d.uvScale);
    u.uRes.value.set(target.width, target.height);
    u.uUseDepth.value = useDepth && d.texture ? 1 : 0;
    this.material.depthWrite = this.look === 'solid' || true;
  }

  // the scan into the overlay's target (already cleared by the host)
  draw(target: THREE_NS.WebGLRenderTarget, model: THREE_NS.Matrix4) {
    if (!this.visible()) return;
    const r = this.host.renderer;
    this.prepare(this.host.camera, model, target, true);
    r.setRenderTarget(target);
    r.render(this.scene, this.host.camera);
  }

  // the magnifier: the spot under the pointer (client pixels), `zoom` times closer, into a square of `size` CSS pixels
  // centred at (cx, cy) on the canvas; returns the camera it used, for the points' layer
  magnify(target: THREE_NS.WebGLRenderTarget, model: THREE_NS.Matrix4, px: number, py: number, cx: number, cy: number, size: number, zoom: number): THREE_NS.PerspectiveCamera | null {
    if (!this.mesh) return null;
    const cam = this.host.camera, canvas = this.host.canvas, rect = canvas.getBoundingClientRect();
    const dpr = target.width / rect.width;
    const m = this.magCam;
    m.position.copy(cam.position); m.quaternion.copy(cam.quaternion); m.near = cam.near; m.far = cam.far;
    // the studio's camera shifts its picture past the panels with a film offset; the magnifier keeps it
    m.filmGauge = cam.filmGauge; m.filmOffset = cam.filmOffset; m.zoom = cam.zoom;
    // the narrower view: the full view's frustum offset so that the pointer's spot is its centre
    const W = rect.width, H = rect.height;
    m.fov = cam.fov; m.aspect = W / H;
    m.setViewOffset(W * zoom, H * zoom, (px - rect.left) * zoom - size / 2, (py - rect.top) * zoom - size / 2, size, size);
    m.updateProjectionMatrix(); m.updateMatrixWorld();
    const r = this.host.renderer, was = this.look;
    this.look = 'solid';
    this.prepare(m, model, target, false);
    this.look = was;
    const x = Math.round((cx - rect.left - size / 2) * dpr), y = Math.round((rect.bottom - cy - size / 2) * dpr), s = Math.round(size * dpr);
    // a render target draws into its own viewport and scissor (the renderer's are for the canvas)
    const vp = target.viewport.clone(), sc = target.scissor.clone(), test = target.scissorTest;
    target.viewport.set(x, y, s, s); target.scissor.set(x, y, s, s); target.scissorTest = true;
    r.setRenderTarget(target);
    // the square cleared, a dark disc, then the scan within the disc
    r.setClearColor(0x000000, 0); r.clear(true, true, false);
    const clip = [x + s / 2, y + s / 2, s / 2 - 1] as const;
    (this.disc.material as THREE_NS.ShaderMaterial).uniforms.uClip.value.set(...clip);
    r.render(this.discScene, m);
    this.material.uniforms.uClip.value.set(...clip);
    r.render(this.scene, m);
    this.material.uniforms.uClip.value.set(0, 0, 0);
    target.viewport.copy(vp); target.scissor.copy(sc); target.scissorTest = test;
    r.setRenderTarget(target);
    m.clearViewOffset();
    const out = m;
    return out;
  }
}
