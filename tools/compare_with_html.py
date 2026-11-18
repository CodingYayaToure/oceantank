"""Compare the Python render (oceantank) with the WebGL canvas of the HTML animation (Chromium + three.js r128).

Same seeds (Mulberry32 injected into the page), same state (animation start-up, then N steps of 1/30 s),
same resolution and same camera. Both images are placed over the same plain background before comparing.

usage: compare_with_html.py <index.html> <three.min.js> <outdir> [W H] ["0,60,150"] [--portable] [--particles=LIST|none]

  --portable          use the sin()-free hash in both renderers (removes the GPU-dependent noise pattern)
  --particles=a,b     keep only these particle layers: spray,bubbles,steam,heads,segs (default: all)

Requires playwright (+ chromium). The HTML animation itself is not part of this repository.
"""
import io, sys, json, base64
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
sys.path.insert(0, __file__.rsplit('/tools/', 1)[0] + '/src')
from oceantank import Tank, Camera, Timeline, Scene
from oceantank import shaders as _sh

html, three, out = sys.argv[1], sys.argv[2], sys.argv[3]
w = int(sys.argv[4]) if len(sys.argv) > 4 else 960
h = int(sys.argv[5]) if len(sys.argv) > 5 else 540
PORTABLE = "--portable" in sys.argv
if PORTABLE: sys.argv.remove("--portable")
PARTS = {"spray", "bubbles", "steam", "heads", "segs"}
for a_ in list(sys.argv):
    if a_.startswith("--particles="):
        v_ = a_.split("=")[1]
        PARTS = set() if v_ == "none" else set(v_.split(","))
        sys.argv.remove(a_)
NOPART = PARTS != {"spray", "bubbles", "steam", "heads", "segs"}
frames = [int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "0,60,150").split(",")]
SEED = 12345
BG = np.array([2, 10, 28], float)

INIT = """
(function(){var a=%d;Math.random=function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);
t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
window.requestAnimationFrame=function(){return 0;};})();
""" % SEED


def over_bg(rgba_straight):
    a = rgba_straight[..., 3:4] / 255.0
    return rgba_straight[..., :3] * a + BG * (1 - a)


def over_bg_premult(rgba):
    return rgba[..., :3].astype(float) + BG * (1 - rgba[..., 3:4] / 255.0)


# ---- reference: browser ----
refs = []
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    pg = b.new_page(viewport={'width': w, 'height': h})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.add_init_script(INIT)
    pg.route('**/three.min.js', lambda r: r.fulfill(path=three, content_type='application/javascript'))
    pg.route('**/fonts.g*/**', lambda r: r.fulfill(status=200, content_type='text/css', body=''))
    if PORTABLE:   # same portable hash in the HTML (patched copy) as in the Python render
        src = open(html, encoding='utf-8').read()
        n1, n2 = src.count(_sh.HASH_NATIVE_1), src.count(_sh.HASH_NATIVE_2)
        assert n1 == 1 and n2 == 1, (n1, n2)
        src = src.replace(_sh.HASH_NATIVE_1, _sh.HASH_PORTABLE_1).replace(_sh.HASH_NATIVE_2, _sh.HASH_PORTABLE_2)
        html = out + '/index_portable.html'
        open(html, 'w', encoding='utf-8').write(src)
    if NOPART:     # hide the particle layers that were not requested in the HTML
        src = open(html, encoding='utf-8').read()
        for nm in ("spray", "bubbles", "steam"):
            a = nm + ".frustumCulled=false;"
            assert src.count(a) == 1
            if nm not in PARTS:
                src = src.replace(a, a + nm + ".visible=false;")
        a = "[heads,segs].forEach(function(o){o.frustumCulled=false;"
        assert src.count(a) == 1
        src = src.replace(a, a + "if(o===heads&&%s||o===segs&&%s)o.visible=false;" % (
            str('heads' not in PARTS).lower(), str('segs' not in PARTS).lower()))
        html = out + '/index_nopart.html'
        open(html, 'w', encoding='utf-8').write(src)
    pg.goto('file://' + html)
    pg.wait_for_timeout(800)
    pg.evaluate('window.__cyc.manual(true)')
    done = 0
    for f in frames:
        pg.evaluate('(n)=>{const c=window.__cyc;for(let i=0;i<n;i++)c.adv(1/30);c.step(0);}', f - done)
        done = f
        url = pg.evaluate("document.getElementById('scene').toDataURL('image/png')")
        im = Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))).convert('RGBA')
        assert im.size == (w, h), im.size
        refs.append(np.array(im))
    b.close()
if errs:
    print('page errors:', errs[:3])

# ---- Python ----
tl = Timeline(Tank(), seed=SEED, rng='mulberry32')
sc = Scene(Tank(), hash_mode='portable' if PORTABLE else 'native').build_glass().add_ocean()
sc.add_particles(tl)
sc.layers = [ly for ly in sc.layers if ly.name not in {'spray', 'bubbles', 'steam', 'heads', 'segs'} - PARTS]
cam = Camera()
res = []
done = 0
for f, ref in zip(frames, refs):
    for _ in range(f - done):
        tl.advance(1 / 30)
    done = f
    tl.advance(0)
    img = sc.render(tl, cam, size=(w, h), samples=4)
    A, B = over_bg_premult(img), over_bg(ref.astype(float))
    d = np.abs(A - B)
    mae = d.mean()
    psnr = 10 * np.log10(255 ** 2 / max((d ** 2).mean(), 1e-9))
    r = dict(frame=f, mae=round(float(mae), 3), psnr_db=round(float(psnr), 2),
             p_gt4=round(float((d.max(-1) > 4).mean() * 100), 2), p_gt16=round(float((d.max(-1) > 16).mean() * 100), 2),
             max=int(d.max()))
    res.append(r)
    print(json.dumps(r), flush=True)
    Image.fromarray(np.clip(B, 0, 255).astype(np.uint8)).save(f'{out}/html_{f:04d}.png')
    Image.fromarray(np.clip(A, 0, 255).astype(np.uint8)).save(f'{out}/py_{f:04d}.png')
    Image.fromarray(np.clip(d.max(-1) * 8, 0, 255).astype(np.uint8)).save(f'{out}/diffx8_{f:04d}.png')
json.dump(res, open(f'{out}/metrics.json', 'w'), indent=1)
