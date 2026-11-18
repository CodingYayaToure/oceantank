"""Loading of the animation's GLSL shaders (extracted verbatim from index.html) and GLSL ES 1.00 -> 3.30 adaptation.

The ``shaders/*.vert|frag`` files are the three.js/WebGL ones, unmodified. Only two mechanical adaptations
are made at compile time (keywords ``attribute``/``varying``/``gl_FragColor``); constants tied to the
tank dimensions are substituted if they differ from the default values.
"""
from __future__ import annotations
import re
from pathlib import Path
from .spec import Tank

_DIR = Path(__file__).parent / "shaders"
NAMES = ("surface", "wall", "floor", "ground", "spray", "bubbles", "steam", "heads", "segs")


def _fmt(v: float) -> str:
    s = f"{v:g}"
    return s if ("." in s or "e" in s) else s + "."


def raw(name: str, stage: str) -> str:
    """Original GLSL ES 1.00 source (stage = 'vert' | 'frag')."""
    return (_DIR / f"{name}.{stage}").read_text(encoding="utf-8")


HASH_NATIVE_1 = "float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}"
HASH_NATIVE_2 = "vec2 hash2(vec2 p){p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));return fract(sin(p)*43758.5453);}"
# hash without sin() (Hoskins): same result on every GPU, unlike fract(sin(x)*43758.5453)
HASH_PORTABLE_1 = ("float hash(vec2 p){vec3 q=fract(vec3(p.xyx)*.1031);q+=dot(q,q.yzx+33.33);return fract((q.x+q.y)*q.z);}")
HASH_PORTABLE_2 = ("vec2 hash2(vec2 p){vec3 q=fract(vec3(p.xyx)*vec3(.1031,.1030,.0973));q+=dot(q,q.yzx+33.33);"
                   "return fract((q.xx+q.yz)*q.zy);}")


def parametrize(src: str, tank: Tank, hash_mode: str = "native") -> str:
    """Replace the hard-coded geometric constants with those of ``tank``; ``hash_mode='portable'``
    replaces the sin()-based hash (GPU-dependent) with a deterministic one."""
    if hash_mode == "portable":
        src = src.replace(HASH_NATIVE_1, HASH_PORTABLE_1).replace(HASH_NATIVE_2, HASH_PORTABLE_2)
    elif hash_mode != "native":
        raise ValueError("hash_mode must be 'native' or 'portable'")
    if tank == Tank():
        return src
    hw, hl = tank.half_width, tank.half_length
    src = src.replace("const float D=3.2; const float HW=4.0;",
                      f"const float D={_fmt(tank.depth)}; const float HW={_fmt(hw)};")
    src = src.replace("vec2(15.,4.)", f"vec2({_fmt(hl)},{_fmt(hw)})")
    src = src.replace("step(4.,p.y)", f"step({_fmt(hw)},p.y)").replace("(p.y-4.)", f"(p.y-{_fmt(hw)})")
    src = src.replace("step(abs(p.x),15.3)", f"step(abs(p.x),{_fmt(hl + .3)})")
    return src


def to_330(src: str, stage: str) -> str:
    if stage == "vert":
        src = re.sub(r"\battribute\b", "in", src)
        src = re.sub(r"\bvarying\b", "out", src)
        head = ("#version 330 core\nuniform mat4 modelViewMatrix;\nuniform mat4 projectionMatrix;\n"
                "in vec3 position;\n")
    else:
        src = re.sub(r"\bvarying\b", "in", src)
        src = src.replace("gl_FragColor", "fragColor")
        head = "#version 330 core\nout vec4 fragColor;\n"
    return head + src


# "basic" material (glass edges, panes): equivalent of three.js MeshBasicMaterial
BASIC_VERT = ("#version 330 core\nuniform mat4 modelViewMatrix;\nuniform mat4 projectionMatrix;\n"
              "in vec3 position;\nvoid main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}\n")
BASIC_FRAG = ("#version 330 core\nuniform vec3 uColor;\nuniform float uOpacity;\nout vec4 fragColor;\n"
              "void main(){fragColor=vec4(uColor,uOpacity);}\n")


def sources(name: str, tank: Tank, hash_mode: str = "native") -> tuple[str, str]:
    return (to_330(parametrize(raw(name, "vert"), tank, hash_mode), "vert"),
            to_330(parametrize(raw(name, "frag"), tank, hash_mode), "frag"))
