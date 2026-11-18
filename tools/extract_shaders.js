// Extracts the GLSL shaders *verbatim* from the HTML animation: runs the scene-construction code
// (lines between the "tank dimensions" comment and stepTracers) with a fake THREE that records every ShaderMaterial.
// usage: node extract_shaders.js <index.html> <output-dir>
const fs = require('fs'), path = require('path'), vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const out = process.argv[3];
const lines = html.split('\n');
const a = lines.findIndex(l => l.includes('/* ---------- dimensions du bassin'));
const b = lines.findIndex(l => l.startsWith('function stepTracers'));
const code = lines.slice(a, b).join('\n');
const mats = [];
const stub = () => new Proxy(function () {}, {
  get: (t, k) => (k === Symbol.toPrimitive ? () => 0 : (k === 'then' ? undefined : stub())),
  apply: () => stub(), construct: () => stub(), set: () => true,
});
const THREE = new Proxy({}, { get: (t, k) => {
  if (k === 'ShaderMaterial') return function (o) { mats.push(o); return stub(); };
  return stub(); } });
const doc = { getElementById: () => stub() };
const ctx = { THREE, document: doc, window: { THREE }, console, Math, Float32Array, Array, Object };
ctx.window.THREE = THREE;
vm.createContext(ctx);
vm.runInContext('(function(){' + code + '\n})()', ctx);
const names = ['surface', 'wall', 'floor', 'ground', 'spray', 'bubbles', 'steam', 'heads', 'segs'];
if (mats.length !== names.length) throw new Error('expected ' + names.length + ' ShaderMaterial, found ' + mats.length);
fs.mkdirSync(out, { recursive: true });
mats.forEach((m, i) => {
  fs.writeFileSync(path.join(out, names[i] + '.vert'), m.vertexShader + '\n');
  fs.writeFileSync(path.join(out, names[i] + '.frag'), m.fragmentShader + '\n');
});
console.log('OK', mats.length, 'materials ->', out);
