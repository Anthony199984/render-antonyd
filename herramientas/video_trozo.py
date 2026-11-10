"""Pedacito de video de una máquina: etiqueta los cuadros y codifica con los MISMOS ajustes en todas (se pegan sin recodificar)."""
import imageio_ffmpeg
import os, glob, subprocess, shutil
from PIL import Image, ImageDraw, ImageFont

def armar(job, tz, sal, salida_mp4):
    F = ImageFont.truetype(os.environ.get('FUENTE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fuentes', 'Inter-Black.otf')), 30)
    tmp = sal + '_cuadros'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp); k = 0
    fs = {int(os.path.basename(p)[:5]): p for p in glob.glob(f'{sal}/[0-9]*.png')}
    for i in sorted(fs):
        im = Image.open(fs[i]).convert('RGB'); et = job['etiquetas'][i]
        if et:
            d = ImageDraw.Draw(im); w = d.textlength(et, font=F); x = (im.width - w) / 2; y = im.height - 60
            d.rounded_rectangle((x - 14, y - 8, x + w + 14, y + 42), 12, fill=(0, 0, 0)); d.text((x, y), et, font=F, fill=(255, 255, 255))
        reps = 1
        if i + 1 == job['total'] or job['etiquetas'][i + 1] != et:          # último cuadro de un tramo
            tr = next((t for t in job['tramos'] if t.get('etiqueta') == et), {}); reps += tr.get('repetir_ultimo', 0)
        for _ in range(reps): im.save(f'{tmp}/{k:05d}.png'); k += 1
    if k == 0: return 0
    fps_in, fps_out = job.get('fps', 15), job.get('video_fps', 30)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', str(fps_in), '-i', f'{tmp}/%05d.png',
                    '-vf', f'fps={fps_out},scale=trunc(iw/2)*2:trunc(ih/2)*2:out_color_matrix=bt709:out_range=tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-r', str(fps_out), salida_mp4], check=True)
    shutil.rmtree(tmp); return k
