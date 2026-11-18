"""Command line: ``oceantank still out.png``, ``oceantank gif out.gif`` or ``oceantank frames DIR``."""
from __future__ import annotations
import argparse
from PIL import Image
from . import Tank, Camera, Timeline, Scene, over


def build(args):
    tl = Timeline(Tank(), seed=args.seed, rng="mulberry32", start_ready=False)
    sc = Scene(Tank(), hash_mode=args.hash).build_glass()
    if not args.glass_only:
        sc.add_ocean()
        if not args.no_particles:
            sc.add_particles(tl)
    return tl, sc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oceantank", description=__doc__)
    ap.add_argument("mode", choices=["still", "gif", "frames"],
                    help="still: one image | gif: animation | frames: PNG sequence (then ffmpeg -> mp4)")
    ap.add_argument("out", help="output file (still, gif) or directory (frames)")
    ap.add_argument("--size", default=None, help="WxH (default 1280x720 for still/frames, 640x360 for gif)")
    ap.add_argument("--at", type=float, default=12.6, help="still: time in the cycle, in s (0 = formation, 32 = end)")
    ap.add_argument("--start", type=float, default=0.0, help="gif/frames: start time in the cycle (s)")
    ap.add_argument("--duration", type=float, default=32.0, help="gif/frames: simulated duration (s); 32 = one full cycle")
    ap.add_argument("--fps", type=float, default=12.0, help="frames per second of the output")
    ap.add_argument("--speed", type=float, default=1.0, help="speed-up factor (2 = twice as fast)")
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--el", type=float, default=0.62)
    ap.add_argument("--hash", choices=["native", "portable"], default="native")
    ap.add_argument("--glass-only", action="store_true")
    ap.add_argument("--no-particles", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)
    size = args.size or ("640x360" if args.mode == "gif" else "1280x720")
    w, h = (int(v) for v in size.lower().split("x"))
    tl, sc = build(args)
    cam = Camera(az=args.az, el=args.el)
    if args.mode == "still":
        tl.run_to(args.at)
        tl.advance(0)
        Image.fromarray(over(sc.render(tl, cam, (w, h)))).save(args.out)
        print(f"wrote {args.out}  (t = {args.at:g} s, cyclone at x = {tl.state.uC[0]:.1f})")
        return
    tl.run_to(args.start)
    n = max(1, round(args.duration * args.fps / args.speed))
    step = args.speed / args.fps                      # simulated seconds between two frames
    frames = []
    if args.mode == "frames":
        import os
        os.makedirs(args.out, exist_ok=True)
    for k in range(n):
        tl.advance(step)
        img = Image.fromarray(over(sc.render(tl, cam, (w, h))))
        if args.mode == "frames":
            img.save(f"{args.out}/f{k:04d}.png")
        else:
            frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=255))
        if k % 20 == 0:
            print(f"  frame {k + 1}/{n}  (cyclone at x = {tl.state.uC[0]:.1f})", flush=True)
    if args.mode == "gif":
        frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 / args.fps), loop=0)
    print(f"wrote {args.out}  ({n} frames, {n / args.fps:.1f} s)")
    if args.mode == "frames":
        print(f"MP4 : ffmpeg -framerate {args.fps:g} -i {args.out}/f%04d.png -pix_fmt yuv420p tank.mp4")


if __name__ == "__main__":
    main()
