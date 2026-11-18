"""Generate examples/oceantank_colab.ipynb (requires nbformat)."""
import nbformat as nbf

C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell
nb = nbf.v4.new_notebook()
nb.cells = [
M("# oceantank: glass tank + ocean\n"
  "Renders the \"Cyclone: from birth to collapse\" animation in Python, in Jupyter or Google Colab.\n\n"
  "**Colab**: no GPU needed (Mesa software rendering)."),
M("## 1. Install (once per session)"),
C('''import glob, importlib.util, os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")      # Mesa software rendering: works without a display or GPU

if IN_COLAB:                                             # OpenGL/EGL libraries of the Colab machine
    subprocess.run("apt-get -qq update && apt-get -qq install -y libegl1 libgl1 libgles2 libgl1-mesa-dri",
                   shell=True, check=False)

if importlib.util.find_spec("oceantank") is None:
    whl = glob.glob("oceantank-*.whl")
    if not whl and IN_COLAB:
        from google.colab import files
        files.upload()                                   # pick the oceantank-*.whl file
        whl = glob.glob("oceantank-*.whl")
    if whl:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *whl])
    else:                                                # no wheel found: install from the repository
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/TON_USERNAME/oceantank.git"])
print("oceantank ready")'''),
M("## 2. One image\nThe cycle lasts **32 s**: 0 = formation, ~12 = warm zone, ~25 = leaving towards cold water, 32 = end."),
C('''from oceantank import Viewer

v = Viewer(size=(960, 540))      # glass tank + water + particles
v.frame(at=14)                   # a PIL image: displayed directly'''),
M("## 3. Build it step by step: glass -> water -> particles"),
C('''from IPython.display import display

glass = Viewer(size=(640, 360), ocean=False)
water = Viewer(size=(640, 360), particles=False)
full  = Viewer(size=(640, 360))
for name, viewer in [("1) glass only", glass), ("2) + ocean water", water), ("3) + spray, bubbles, steam", full)]:
    print(name)
    display(viewer.frame(at=14))'''),
M("## 4. Explore: time in the cycle and camera angle"),
C('''import ipywidgets as w
from IPython.display import display

small = Viewer(size=(800, 450))
w.interact(lambda at, az, el: display(small.frame(at=at, az=az, el=el)),
           at=w.FloatSlider(value=14, min=0, max=32, step=1, continuous_update=False),
           az=w.FloatSlider(value=0, min=-1.2, max=1.2, step=0.1, continuous_update=False),
           el=w.FloatSlider(value=0.62, min=0.1, max=1.3, step=0.05, continuous_update=False));'''),
M("## 5. Animate a full cycle (the cyclone crosses the tank)\nAllow 1 to 3 minutes with software rendering."),
C('''from IPython.display import Image as IPImage

small = Viewer(size=(480, 270))
path = small.gif("cycle.gif", duration=32, fps=6)        # use speed=2 to go twice as fast
IPImage(filename=path)'''),
M("## 6. Get the file\nOn Colab this downloads the GIF to your computer."),
C('''if IN_COLAB:
    from google.colab import files
    files.download("cycle.gif")
else:
    print("File saved:", os.path.abspath("cycle.gif"))'''),
M("**Limitation**: the fine caustics pattern depends on the GPU/driver implementation of `sin()` (see the README); "
  "here it is the pattern of an exact `sin()` (Mesa)."),
]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, __file__.rsplit("/tools/", 1)[0] + "/examples/oceantank_colab.ipynb")
print("ok", len(nb.cells))
