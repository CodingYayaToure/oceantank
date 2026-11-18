# Mathematical, physical and meteorological models

This document states **exactly what is computed** in `oceantank`, where each equation lives in the code, and how it
relates to the physics of tropical cyclones.

> **Scope.** `oceantank` is a *deterministic, model-scale visualisation* of a cyclone crossing a glass tank. It is
> **not** a numerical weather prediction code and it does not integrate the primitive equations. Its cyclone is driven by
> a small **phenomenological** model (sea-surface temperature → intensity) whose structure follows tropical-cyclone
> theory qualitatively. Section 5 lists the physical reference models and states which of them are *not* implemented here.
> All lengths are in **tank units** (the tank is 30 × 8 × 3.2), not metres.

| # | Layer | Nature | Code |
|---|---|---|---|
| 1 | Perspective camera, rasterisation, blending | exact (three.js r128 conventions) | `camera.py`, `scene.py`, `render_gl.py` |
| 2 | Water-surface height field and shading | analytic, hand-built | `shaders/surface.vert`, `surface.frag` |
| 3 | Sea-surface temperature and intensity law | phenomenological, tuned | `timeline.py` |
| 4 | Vortex wind field advecting the tracers | analytic (modified Rankine) | `timeline.py::step_tracers` |
| 5 | Physical reference models | context only, **not implemented** | — |

---

## 1. Rendering mathematics

**Camera.** A vertical field of view $\varphi = 32^\circ$, aspect ratio $a$, near/far planes $n, f$. With
$F = 1/\tan(\varphi/2)$ the non-zero entries of the projection matrix are

```math
P_{11}=\frac{F}{a},\quad P_{22}=F,\quad P_{33}=\frac{f+n}{n-f},\quad P_{34}=\frac{2fn}{n-f},\quad P_{43}=-1
```

and all other entries of $P$ are zero (a $4\times4$ matrix acting on homogeneous coordinates).

The orbit camera sits at distance $d=\left(17/h_f+3.6\right)\ \text{zoom}$ with $h_f=\tan(\varphi/2)\ a$, azimuth
$\alpha$ and elevation $\epsilon$:

```math
\mathbf e = \bigl(d\cos\epsilon\sin\alpha,\  1.3+d\sin\epsilon,\  d\cos\epsilon\cos\alpha\bigr),
\qquad \text{look-at target } (0,\ 1.25,\ 0).
```

**Blending.** With source colour/alpha $(C_s,\alpha_s)$ and destination colour $C_d$:

```math
\text{normal: } C=\alpha_s C_s+(1-\alpha_s)\ C_d,
\qquad
\text{additive: } C=\alpha_s C_s+C_d .
```

Opaque layers (water surface, floor) are drawn first, then transparent layers sorted by `renderOrder`
(ground, back wall, bubbles, side walls, front wall, particles, halos, glass). Transparent double-sided meshes use two passes
(cull front, then cull back). The result is an RGBA image composited over the page's radial-gradient background.

**Anti-aliasing.** 4× MSAA on a multisampled framebuffer, resolved before read-back.

## 2. Water surface

The surface is a plane mesh (660 × 200 segments, about 264 000 triangles) displaced in the vertex shader by an
**analytic height field** $H(x,z,t)$, then shaded per pixel. Notation: $\mathbf d=(x,z)-\mathbf c$ is the offset from the
cyclone centre $\mathbf c=(c_x,c_z)$, $r=\Vert\mathbf d\Vert$, $\theta=\mathrm{atan2}(d_z,d_x)$,
$I$ the intensity, $R$ the vortex radius, $\phi$ a rotation phase, $B\in[0,1]$ the "burst" (rapid-intensification) indicator.

**Ambient sea.** Four travelling sinusoids:

```math
h_\text{amb}=0.075\sin\bigl(0.9x+1.1t+\sin(0.6z+0.4t)\bigr)+0.05\sin(1.3z-1.3t+0.4x)+0.035\sin\bigl(2.1(x+z)+1.9t\bigr)+0.03\sin\bigl(1.3(1.7x-2.6z)-2.4t\bigr).
```

**Cyclone depression and eyewall.** Gaussian profiles in $r$:

```math
H_\text{core}=I\left[-1.75\ e^{-(r/0.6R)^2}+0.26\ e^{-((r-1.1R)/0.5R)^2}+0.20\ e^{-(r/3.2R)^2}\right].
```

The first term is the depression under the eye, the second a raised eyewall ring, the third a broad swell. These amplitudes are
**artistic**, not a hydrostatic or storm-surge solution (see §5).

**Spiral rain bands.** A logarithmic spiral with two arms:

```math
H_\text{spiral}=0.2\ I\ \sin\bigl(2(\theta+\phi)-5\ln(r/R+0.15)\bigr)\ S(r),
\quad S(r)=\mathrm{ss}(0.9R,1.8R,r)\ e^{-r/3.2R},
```

where $\mathrm{ss}(a,b,x)=s^2(3-2s)$, $s=\mathrm{clamp}\bigl((x-a)/(b-a),0,1\bigr)$ is the cubic smoothstep.

**Radiating gravity-wave rings with wall reflections.** A radial wave $\sin(3.4\ r-5.6\ t)$ gives a phase speed
$c=5.6/3.4\approx1.65$ tank units/s, windowed in space. Reflections from the two long walls $z=\pm H_w$ ($H_w=4$) are added by
the **method of images**: the source is mirrored to $z=\pm2H_w-c_z$ and weighted by 0.55:

```math
H_\text{ring}=I\ (0.2+0.4B)\Bigl[g(\mathbf d)+0.55\ g(\mathbf d_{+})+0.55\ g(\mathbf d_{-})\Bigr],
\quad g=\sin(3.4r-5.6t)\ \mathrm{ss}(1.6R,3R,r)\ e^{-r/8.5}.
```

**Wake.** Behind the storm ($b=\max(c_x-x,0)$) a Gaussian-in-$z$ wake amplifies the ambient sea:
$e^{-b/8}\ e^{-[d_z/(1.8+0.22\ b)]^2}$.

**Normals.** Computed by forward finite differences of $H$ with $\varepsilon=0.045$ (three evaluations of $H$ per pixel), then perturbed by a small value-noise term (amplitude 0.10) for micro-ripples:

```math
\mathbf n=\frac{\bigl(-\partial_xH,\ 1,\ -\partial_zH\bigr)}{\Vert\cdot\Vert},\qquad
\partial_xH\approx\frac{H(x+\varepsilon,z)-H(x,z)}{\varepsilon}.
```

**Shading.** Fresnel reflectance in the form $F_r=(1-\mathbf n\cdot\mathbf v)^3$ (a Schlick-type approximation) blends
the water colour towards a sky colour; Blinn–Phong specular highlight
$(\mathbf n\cdot\mathbf h)^{150}$ with half-vector $\mathbf h=(\mathbf l+\mathbf v)/\Vert\mathbf l+\mathbf v\Vert$.

**Caustics.** A cellular (Worley) pattern: $F_2-F_1$, the gap between the distances to the two nearest animated feature
points on a $3\times3$ neighbourhood, passed through $(1-\mathrm{ss}(0,0.28,F_2-F_1))^3$. **Foam and haze** use fractal value
noise (4 octaves, lacunarity 2.03, gain 0.5).

**Thermal tint.** The warm zone is visualised by blending towards an orange tint with weight $\text{warm}(x)$, the same smoothstep
window as in §3, reduced inside the cold wake. Cold water is desaturated by $\text{cold}(x)$.

**Pseudo-random hash.** The shaders use $\mathrm{fract}\bigl(\sin(\mathbf p\cdot\mathbf k)\cdot43758.5453\bigr)$. Because
the argument is large, the result depends on the GPU's `sin` implementation (measured: llvmpipe equals NumPy float32; SwiftShader
differs on 94 % of samples). `hash_mode="portable"` replaces it by a `sin`-free hash identical on every device.

## 3. Sea-surface temperature and intensity law

This is the only "meteorological" model in the package. Let $x$ be the position along the track and
$[Z_0,Z_1]=[-3,\ 5.5]$ the imposed **warm zone**.

**Imposed SST profile** (°C):

```math
\text{SST}(x)=27.6+3.0\ W(x)-3.2\ C(x)-1.1\ S(x)\ W(x),
```

```math
W=\mathrm{ss}(Z_0-2.6,\ Z_0+1.6,\ x)\bigl[1-\mathrm{ss}(Z_1-1.6,\ Z_1+2.6,\ x)\bigr],\quad
C=\mathrm{ss}(7.6,\ 11.4,\ x),\quad
S=\mathrm{ss}(Z_0+1.5,\ Z_1+0.5,\ x).
```

$W$ is the warm anomaly, $C$ the cold-water region at the end of the tank, and the last term is a **cold-wake proxy**
(sea-surface cooling under a storm that has been over the warm patch for a while). It peaks at 30.5 °C near $x=0$ and falls to
24.4 °C at the end of the track.

**Target intensity and relaxation.** The threshold $26.5\ ^\circ$C is the classical genesis threshold of Gray (1968), and
the 4 K normalisation, the exponent 0.9 and the three length scales are tuned constants:

```math
I^{\ast}(x)=\Bigl[\max\Bigl(0,\tfrac{\text{SST}(x)-26.5}{4}\Bigr)\Bigr]^{0.9},
\qquad
\frac{dI}{dx}=\frac{I^{\ast}-I}{\lambda}
```

with the relaxation length $\lambda$ chosen by case:

| Case | Condition | $\lambda$ |
|---|---|---|
| intensification | $I^{\ast}>I$ | 1.5 |
| decay over cold water | $I^{\ast}\le I$ and SST below 26.5 °C | 1.4 |
| slow decay over warm water | $I^{\ast}\le I$ and SST at or above 26.5 °C | 5.2 |

Integrated along the track with explicit Euler on $N_I=1600$ steps of size $\Delta x=22.4/1600=0.014$, starting from
$I_0=0.04$. Since $\Delta x/\lambda\le0.01\ll1$ the scheme is stable, and **refining to 16 000 steps changes the table by at most
$4.5\times10^{-3}$** (peak intensity 0.955, reached near $x\approx1.1$, i.e. slightly downstream of the SST maximum because of the
relaxation lag).

**Time dependence.** With cycle period $T=32$ s and $u=(t \ \mathrm{mod}\  T)/T$:

```math
c_x=X_0+(X_1-X_0)\ u,\quad c_z=0.75\sin(3\pi u),\quad X_{0,1}=\mp11.2\quad(\text{speed }0.7\ \text{units/s}),
```

```math
I_n(u)=I_\text{tab}(u)\bigl[1-\mathrm{ss}(0.97,1,u)\bigr],\qquad I(t)=1.18\ g\ I_n(u),\quad g=1.35\ (\text{gain}),
```

```math
R=R_0\Bigl(1.1-0.28\min\bigl(I_n/0.95,\ 1.1\bigr)+0.45\mathrm{ss}(Z_1-1.5,\ Z_1+6,\ c_x)\Bigr),\quad R_0=1.85,
```

so the storm contracts as it intensifies and expands again over cold water. The spiral phase advances at
$\dot\phi=2.7\ (0.45+0.75\ I)$. Two shader controls are derived from the same table:

```math
B=\mathrm{clamp}\Bigl(\frac{1}{0.26}\frac{dI_n}{dx},\ 0,\ 1\Bigr)\ \ (\text{rapid-intensification burst}),\qquad
D=\mathrm{ss}(8.0,\ 10.6,\ c_x)\ \ (\text{dissipation / eye tearing}).
```

## 4. Vortex wind field and surface tracers

2600 surface tracers are advected by an analytic **modified Rankine vortex** of radius $R_0$ and peak $V_\text{max}=6$:

Solid-body core, $r \lt R_0$:

```math
V(r)=V_\text{max}\ I\ \frac{r}{R_0}
```

Decaying outer wind, $r\ge R_0$:

```math
V(r)=V_\text{max}\ I\ \Bigl(\frac{R_0}{r}\Bigr)^{0.85}e^{-(r-R_0)/11}
```

The velocity is the cyclonic tangential flow plus an inward (frictional) radial component of 22 % of the tangential speed (inflow angle
$\arctan0.22\approx12.4^\circ$) and a weak background turbulence:

```math
\mathbf v=V\Bigl(\tfrac{d_z}{r},-\tfrac{d_x}{r}\Bigr)-0.22\ V\Bigl(\tfrac{d_x}{r},\tfrac{d_z}{r}\Bigr)
+0.06\bigl(\sin(1.3z+t),\ \cos(1.1x-t)\bigr).
```

Positions are advanced with explicit Euler at the simulation step $\Delta t=1/30$ s. A tracer is recycled when its lifetime
(3–12 s) expires, when it falls into the eye ($r<0.45R_0$ and $I>0.3$), or when it leaves the tank. The exponent 0.85 and the
cut-off length 11 are visual choices, not fitted to observations.

**Random numbers.** All particle seeds come from **Mulberry32**, a 32-bit generator, so Python and the original JavaScript
animation consume *exactly* the same stream (the first draws are checked against JavaScript values in the test-suite):

```
a ← a + 0x6D2B79F5;  t ← imul(a ⊕ (a ≫ 15), 1 | a);  t ← (t + imul(t ⊕ (t ≫ 7), 61 | t)) ⊕ t;  return (t ⊕ (t ≫ 14)) / 2³²
```

Spray, bubble and steam particles are not integrated in Python: their seeds go to the GPU and the shaders evaluate their
positions analytically from $t$.

## 5. Physical reference models (context; not implemented in this package)

For orientation, the physics that the phenomenological model above imitates **qualitatively** is the following. None of these
equations is solved by `oceantank`; they are what a physically based intensity solver (see the `itab=` hook below) would provide.

**Moist saturation (Bolton, 1980).** Saturation vapour pressure over water, $T$ in °C:
```math
e_s(T)=6.112\ \exp\Bigl(\frac{17.67\ T}{T+243.5}\Bigr)\ \text{hPa}.
```

**Boundary-layer energy balance and potential intensity (reduced form after Emanuel, 1986).** Surface enthalpy flux feeds the vortex and surface
drag dissipates it, with Carnot efficiency $\varepsilon=(T_s-T_o)/T_o$ between the sea surface and the outflow level:

```math
\frac{d}{dt}\Bigl(\tfrac12V^2\Bigr)=\frac{\eta}{h}\Bigl[\varepsilon\ C_k\ V\ (k_0^{\ast}-k)-C_d\ V^3\Bigr],
\qquad
V_p^2=\frac{C_k}{C_d}\ \varepsilon\ (k_0^{\ast}-k).
```

This gives the relaxation law $dV/dt=a\ (V_p^2-V^2)$ with $a=\eta C_d/h$, whose exact solution is
$V(t)=V_p\tanh\bigl(\mathrm{artanh}(V_0/V_p)+a\ V_p\ t\bigr)$ for $V_0<V_p$. The animation's law
$dI/dx=(I^{\ast}-I)/\lambda$ has the same *form* (relaxation toward an SST-controlled target) but its target is a tuned
function of SST, not a computed $V_p$.

**Gradient-wind balance.** A steady axisymmetric vortex satisfies $V^2/r+fV=\dfrac1\rho\ \partial_rp$; the central pressure drop
underlies the (inverse-barometer) idea that the sea surface bulges up where pressure is low. In reality the response to a
fast cyclone is a storm surge governed by the shallow-water equations with wind stress; here the depression of §2 is a
**visual device** and is not a hydrodynamic solution.

**Axisymmetric shallow water** (rotating, $f$-plane), the natural next step for the ocean layer:
```math
\partial_th+\tfrac1r\partial_r(rhu)=S,\quad
\partial_tu=\tfrac{v^2}{r}+fv-g\ \partial_rh,\quad
\partial_tv=-u\bigl(\partial_rv+\tfrac vr+f\bigr).
```

**Hook for a physical solver.** `Timeline(itab=...)` accepts a table of 1601 normalised intensities along the track.
Any intensity model (for example, a time-dependent solver run along a prescribed track) can drive the water, spiral and tracers
instead of the built-in SST law.

## 6. Numerics, determinism and cost

* **Deterministic:** fixed seeds, fixed step $\Delta t=1/30$ s, the same particle state for a given time
  (`Timeline.run_to(t)` replays from the start of the cycle).
* **Cross-validated:** the same frame rendered by three.js (Chromium, SwiftShader) and by this package agrees to a mean absolute error of
  0.37/255 (PSNR 41 dB) for glass and water when the hash is shared, and 1.3–1.6/255 (30–32 dB) for the full scene.
* **Test-suite:** geometry, matrices, Mulberry32 against JavaScript, timeline determinism, cyclone crossing,
  shader parametrisation, GPU rendering determinism, compositing, notebook API (`pytest`, run on every push by CI).
* **Cost:** the surface fragment shader evaluates $H$ three times per pixel and a $3\times3$ Worley search; on CPU (Mesa llvmpipe)
  a frame takes about 0.5 s at 800×450 and 0.8 s at 1280×720 on 2 CPU cores (measured, including the timeline replay). On a GPU with EGL the same code runs unchanged.

## References

1. Gray, W. M. (1968). Global view of the origin of tropical disturbances and storms. *Mon. Wea. Rev.* **96**, 669–700.
2. Emanuel, K. A. (1986). An air–sea interaction theory for tropical cyclones. Part I. *J. Atmos. Sci.* **43**, 585–605.
3. Rotunno, R. & Emanuel, K. A. (1987). An air–sea interaction theory for tropical cyclones. Part II. *J. Atmos. Sci.* **44**, 542–561.
4. Bolton, D. (1980). The computation of equivalent potential temperature. *Mon. Wea. Rev.* **108**, 1046–1053.
5. Worley, S. (1996). A cellular texture basis function. *Proc. SIGGRAPH '96*, 291–294.
6. Schlick, C. (1994). An inexpensive BRDF model for physically-based rendering. *Computer Graphics Forum* **13**(3), 233–246.
7. Blinn, J. F. (1977). Models of light reflection for computer synthesized pictures. *Proc. SIGGRAPH '77*, 192–198.
