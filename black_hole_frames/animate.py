"""Turn the key frames into an animated MP4 with procedural camera/warp effects."""
import subprocess, numpy as np, imageio_ffmpeg
from PIL import Image

W, H, FPS, SEC, XF = 1280, 720, 24, 6, 12  # XF = crossfade frames
D = "black_hole_frames/"
N = FPS * SEC
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
cx, cy = W / 2, H / 2
dx, dy = xx - cx, yy - cy
r = np.sqrt(dx**2 + dy**2); th = np.arctan2(dy, dx); rmax = r.max()

def load(n):
    im = Image.open(D + n).convert("RGB")
    s = max(W / im.width, H / im.height) * 1.0
    im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    return np.asarray(im.crop((l, t, l + W, t + H))).astype(np.float32)

def sample(img, sx, sy):
    sx = np.clip(sx, 0, W - 1.001); sy = np.clip(sy, 0, H - 1.001)
    x0, y0 = sx.astype(int), sy.astype(int); fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    a, b = img[y0, x0], img[y0, x0 + 1]; c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

def warp(img, zoom=1.0, swirl=0.0, rot=0.0, sx_=1.0, sy_=1.0, ox=0, oy=0):
    # swirl: extra rotation stronger near center (black-hole whirlpool)
    ang = th - rot - swirl * np.exp(-r / (0.35 * rmax))
    rr = r / zoom
    px = cx + rr * np.cos(ang) / sx_ + ox; py = cy + rr * np.sin(ang) / sy_ + oy
    return sample(img, px, py)

def scene(i, f):
    t = f / (N - 1); e = t * t * (3 - 2 * t)
    img = IM[i]
    if i == 0:   # title: emerge from black, whirlpool
        out = warp(img, 1 + 0.25 * e, swirl=1.5 * (1 - e)) * min(1, t * 2.5)
    elif i == 1: # approach: slow push, disk rotating
        out = warp(img, 1 + 0.35 * e, swirl=0.25 * t, rot=0.03 * t)
    elif i == 2: # river of space flowing inward
        out = warp(img, 1 + 0.9 * e, swirl=0.6 * t)
    elif i == 3: # spaghettification: stretch + shake
        s = 1 + 0.5 * e
        out = warp(img, 1.05 + 0.2 * e, sx_=1 / (1 + 0.15 * e), sy_=s,
                   ox=np.sin(f * 1.7) * 6 * e, oy=np.cos(f * 2.3) * 6 * e)
    elif i == 4: # singularity: rotation + pulse
        out = warp(img, 1.1 + 0.05 * np.sin(t * 12), swirl=0.8 * t, rot=0.15 * t)
        out = out * (1 + 0.12 * np.sin(t * 12))
    else:        # hawking: pull out, flicker, fade to black
        out = warp(img, 1.4 - 0.4 * e, rot=-0.05 * t) * (1 + 0.08 * np.sin(f * 0.9))
        out = out * min(1, (1 - t) * 3)
    return out

names = ["01_title.png", "02_approach.png", "03_event_horizon_river.png",
         "04_spaghettification.png", "05_singularity_theories.png", "06_hawking_radiation.png"]
IM = [load(n) for n in names]
ff = imageio_ffmpeg.get_ffmpeg_exe()
p = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                      "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                      "-preset", "medium", D + "black_hole_animation.mp4"],
                     stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
for i in range(len(IM)):
    for f in range(N - (XF if i < len(IM) - 1 else 0)):
        fr = scene(i, f)
        if i > 0 and f < XF:  # crossfade from previous scene's tail
            a = f / XF; fr = fr * a + scene(i - 1, N - XF + f) * (1 - a)
        p.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait(); print("done")
