"""Une los trozos en el video: python3 unir.py trabajos/ACTUAL.json <carpeta_trozos> <carpeta_resultado>"""
import json, os, sys, glob, subprocess, shutil
from PIL import Image, ImageDraw, ImageFont
job = json.load(open(sys.argv[1])); ent = sys.argv[2]; res = sys.argv[3]; os.makedirs(res, exist_ok=True)
F = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fuentes', 'Inter-Black.otf'), 30)
fs = {int(os.path.basename(p)[:5]): p for p in glob.glob(f'{ent}/**/*.png', recursive=True)}
falta = [i for i in range(job['total']) if i not in fs]
tmp = 'cuadros_final'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp); k = 0; ult = None
for i in range(job['total']):
    if i in fs:
        im = Image.open(fs[i]).convert('RGB'); et = job['etiquetas'][i]
        if et:
            d = ImageDraw.Draw(im); w = d.textlength(et, font=F); x = (im.width - w) / 2; y = im.height - 60
            d.rounded_rectangle((x - 14, y - 8, x + w + 14, y + 42), 12, fill=(0, 0, 0)); d.text((x, y), et, font=F, fill=(255, 255, 255))
        ult = im
    if ult is None: continue
    ult.save(f'{tmp}/{k:05d}.png'); k += 1
    fin_tramo = i + 1 == job['total'] or job['etiquetas'][i + 1] != job['etiquetas'][i]
    if fin_tramo:
        tr = next((t for t in job['tramos'] if t.get('etiqueta') == job['etiquetas'][i]), {})
        for _ in range(tr.get('repetir_ultimo', 0)): ult.save(f'{tmp}/{k:05d}.png'); k += 1
nombre = job['nombre']
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(job.get('fps', 15)), '-i', f'{tmp}/%05d.png', '-vf',
                f"fps={job.get('video_fps', 30)},scale=trunc(iw/2)*2:trunc(ih/2)*2", '-c:v', 'libx264', '-preset', 'veryfast',
                '-crf', '20', '-pix_fmt', 'yuv420p', f'{res}/{nombre}.mp4'], check=True)
ts = [float(open(p).read()) for p in glob.glob(f'{ent}/**/tiempo_*.txt', recursive=True)]
inf = f'cuadros {job["total"]}, faltan {len(falta)} {falta[:20]}\ntrozos {len(ts)}; el más lento tardó {max(ts or [0]):.0f} s; suma de todas las máquinas {sum(ts):.0f} s\n'
open(f'{res}/INFORME.txt', 'w').write(inf); print(inf)
