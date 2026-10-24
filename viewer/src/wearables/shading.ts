// OpenSculptBoy
// Apache License, Version 2.0
//
// The shader of the garments: glTF's metallic-roughness materials (base colour, metallic-roughness, normal, occlusion
// and emissive textures) under the page's key, rim and environment lights, with the key light's soft shadows. A
// masked material cuts at its alpha cutoff; a blended one drops each fragment with the chance of its transparency, so
// that the page's progressive accumulation averages it.

import { GLSL_COMMON, SKIN_GLSL } from '../shading.ts';

export const GARMENT_VS = /* glsl */`
${SKIN_GLSL}
attribute vec4 gTangent;
varying vec3 vWPos; varying vec3 vN; varying vec2 vUv; varying vec4 vT;
void main() {
  mat4 S = skinMat();
  mat3 M = mat3(modelMatrix) * mat3(S);
  vec4 wp = modelMatrix * (S * vec4(position, 1.0));
  vWPos = wp.xyz; vN = normalize(M * normal); vUv = uv;
#ifdef HAS_NORMAL_MAP
  vT = vec4(normalize(M * gTangent.xyz), gTangent.w);
#else
  vT = vec4(1.0, 0.0, 0.0, 1.0);
#endif
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

export const GARMENT_FS = /* glsl */`
${GLSL_COMMON}
uniform vec4 uBase; uniform float uMetal; uniform float uRough; uniform vec3 uEmissive; uniform float uCutoff; uniform float uNormalScale;
uniform sampler2D uMap; uniform sampler2D uMR; uniform sampler2D uNormalMap; uniform sampler2D uAO; uniform sampler2D uEmissiveMap;
varying vec3 vWPos; varying vec3 vN; varying vec2 vUv; varying vec4 vT;
void main() {
  vec4 base = uBase;
#ifdef HAS_MAP
  base *= texture(uMap, vUv);
#endif
#ifdef ALPHA_MASK
  if (base.a < uCutoff) discard;
#endif
#ifdef ALPHA_BLEND
  if (base.a <= ign(gl_FragCoord.xy + 17.0)) discard;
#endif
  float rough = uRough, metal = uMetal;
#ifdef HAS_MR
  vec4 mr = texture(uMR, vUv); rough *= mr.g; metal *= mr.b;
#endif
  vec3 N = normalize(vN);
  if (!gl_FrontFacing) N = -N;
#ifdef HAS_NORMAL_MAP
  vec3 T = normalize(vT.xyz - N * dot(N, vT.xyz));
  vec3 B = cross(N, T) * vT.w;
  vec3 tn = texture(uNormalMap, vUv).xyz * 2.0 - 1.0;
  tn.xy *= uNormalScale;
  N = normalize(mat3(T, B, N) * tn);
#endif
  rough = clamp(rough, 0.05, 1.0);
  vec3 V = normalize(cameraPosition - vWPos);
  vec3 L = normalize(uKeyDir), Lr = normalize(uRimDir);
  vec2 sh = keyShadow(vWPos, N, 0.8);
  float vis = sh.x * sh.y, ao = 1.0;
#ifdef HAS_AO
  ao = texture(uAO, vUv).r;
#endif
  vec3 diff = base.rgb * (1.0 - metal);
  vec3 col = diff * (uKeyColor * max(dot(N, L), 0.0) * vis + uRimColor * max(dot(N, Lr), 0.0) * 0.3 + shIrr(N) * uEnvDiff * ao);
  float f0 = mix(0.04, 1.0, metal);
  vec3 tint = mix(vec3(1.0), base.rgb, metal);
  col += tint * (PI * uKeyColor * specSphere(N, V, L, rough, f0, uKeyTan) * vis + envSpecular(N, V, rough, f0) * uEnvSpec * ao);
  vec3 em = uEmissive;
#ifdef HAS_EMISSIVE_MAP
  em *= texture(uEmissiveMap, vUv).rgb;
#endif
  col += em;
  gl_FragColor = vec4(min(col, vec3(64.0)), 1.0);
}`;
