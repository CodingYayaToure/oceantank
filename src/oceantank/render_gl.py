"""Headless OpenGL 3.3 rendering (moderngl + EGL, e.g. Mesa llvmpipe) of the layers of a ``Scene``."""
from __future__ import annotations
import numpy as np
import moderngl
from . import shaders
from .camera import Camera
from .timeline import Timeline, OceanState

_MODES = {"triangles": moderngl.TRIANGLES, "points": moderngl.POINTS, "lines": moderngl.LINES}


def _has(prog, name):
    try:
        prog[name]
        return True
    except KeyError:
        return False


class GLRenderer:
    def __init__(self, scene, size, samples):
        self.scene, self.size, self.samples = scene, size, samples
        try:
            self.ctx = moderngl.create_standalone_context(backend="egl")
        except Exception:
            self.ctx = moderngl.create_standalone_context()
        w, h = size
        self.fbo = self.ctx.framebuffer(self.ctx.renderbuffer((w, h), samples=samples),
                                        self.ctx.depth_renderbuffer((w, h), samples=samples))
        self.out = self.ctx.simple_framebuffer((w, h))
        self.progs = {}
        for name in shaders.NAMES:
            v, f = shaders.sources(name, scene.tank, scene.hash_mode)
            self.progs[name] = self.ctx.program(vertex_shader=v, fragment_shader=f)
        self.progs["basic"] = self.ctx.program(vertex_shader=shaders.BASIC_VERT, fragment_shader=shaders.BASIC_FRAG)
        self.vaos, self.bufs = [], {}
        for ly in scene.layers:
            prog = self.progs[ly.program]
            content, bufs = [], {}
            for name, arr in ly.attrs.items():
                if not _has(prog, name):
                    continue
                arr = np.ascontiguousarray(arr, np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                b = self.ctx.buffer(arr.tobytes(), dynamic=ly.dynamic is not None)
                bufs[name] = b
                content.append((b, f"{arr.shape[1]}f", name))
            ib = self.ctx.buffer(ly.index.astype(np.uint32).tobytes()) if ly.index is not None else None
            self.vaos.append(self.ctx.vertex_array(prog, content, index_buffer=ib))
            self.bufs[id(ly)] = bufs
        order = list(enumerate(scene.layers))
        self.opaque = [i for i, ly in sorted(order, key=lambda t: (t[1].order, t[0])) if not ly.transparent]
        self.transp = [i for i, ly in sorted(order, key=lambda t: (t[1].order, t[0])) if ly.transparent]

    # -------------------------------------------------------------------------------------------
    def _uniform(self, prog, name, value):
        if _has(prog, name):
            prog[name].value = value

    def _globals(self, prog, state: OceanState, cam_pos, px):
        u = self._uniform
        u(prog, "uT", state.uT); u(prog, "uC", tuple(state.uC)); u(prog, "uI", state.uI)
        u(prog, "uRot", state.uRot); u(prog, "uR", state.uR); u(prog, "uZ", tuple(state.uZ))
        u(prog, "uBurst", state.uBurst); u(prog, "uDis", state.uDis)
        u(prog, "uCam", tuple(float(c) for c in cam_pos)); u(prog, "uPx", px)

    def _draw(self, i, view, proj, state, cam_pos, px):
        ly, vao = self.scene.layers[i], self.vaos[i]
        prog = self.progs[ly.program]
        ctx = self.ctx
        m = np.eye(4)
        m[:3, 3] = ly.offset
        mv = view @ m
        prog["modelViewMatrix"].write(mv.T.astype("f4").tobytes())
        prog["projectionMatrix"].write(proj.T.astype("f4").tobytes())
        if ly.program == "basic":
            prog["uColor"].value = ly.color
            prog["uOpacity"].value = ly.opacity
        else:
            self._globals(prog, state, cam_pos, px)
        if ly.blend == "none":
            ctx.disable(moderngl.BLEND)
        else:
            ctx.enable(moderngl.BLEND)
            ctx.blend_equation = moderngl.FUNC_ADD
            if ly.blend == "normal":     # SRC_ALPHA, 1-SRC_ALPHA (rgb) ; ONE, 1-SRC_ALPHA (a)
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            else:                        # additive: SRC_ALPHA, ONE
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE, moderngl.SRC_ALPHA, moderngl.ONE)
        self.fbo.depth_mask = ly.depth_write
        mode = _MODES[ly.mode]
        if ly.mode != "triangles" or (ly.side == "double" and not ly.transparent):
            ctx.disable(moderngl.CULL_FACE)          # points, lines, opaque double-sided surface (single pass)
            vao.render(mode)
        elif ly.side == "double":
            ctx.enable(moderngl.CULL_FACE)
            for face in ("front", "back"):           # three.js r128: back faces, then front faces
                ctx.cull_face = face
                vao.render(mode)
        else:
            ctx.enable(moderngl.CULL_FACE)
            ctx.cull_face = "back"
            vao.render(mode)

    def render(self, timeline, camera: Camera) -> np.ndarray:
        state = timeline.state if isinstance(timeline, Timeline) else timeline
        w, h = self.size
        aspect = w / h
        view, proj = camera.view(aspect), camera.projection(aspect)
        cam_pos = camera.position(aspect)
        px = h / 720.0
        if isinstance(timeline, Timeline):          # dynamic buffers (tracers)
            for ly in self.scene.layers:
                if ly.dynamic:
                    arr = timeline.head_pos if ly.dynamic == "heads" else timeline.seg_pos
                    self.bufs[id(ly)]["position"].write(np.ascontiguousarray(arr, np.float32).tobytes())
        ctx = self.ctx
        self.fbo.use()
        ctx.viewport = (0, 0, w, h)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        ctx.depth_func = "<="
        self.fbo.depth_mask = True
        self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.line_width = 1.0
        for i in self.opaque + self.transp:
            self._draw(i, view, proj, state, cam_pos, px)
        self.fbo.depth_mask = True
        ctx.copy_framebuffer(self.out, self.fbo)
        raw = np.frombuffer(self.out.read(components=4), np.uint8).reshape(h, w, 4)
        return np.flipud(raw).copy()
