"""Pega los pedacitos de video de cada máquina, sin recodificar: python3 unir.py trabajos/ACTUAL.json <carpeta_trozos> <carpeta_resultado>"""
import imageio_ffmpeg
import json, os, sys, glob, subprocess
job = json.load(open(sys.argv[1])); ent = sys.argv[2]; res = sys.argv[3]; os.makedirs(res, exist_ok=True)
ids = [t['id'] for t in job['trozos']]
mp4 = {i: next(iter(glob.glob(f'{ent}/**/trozo_{i}.mp4', recursive=True)), None) for i in ids}
falta = [i for i in ids if not mp4[i]]
with open('lista.txt', 'w') as f:
    for i in ids:
        if mp4[i]: f.write(f"file '{os.path.abspath(mp4[i])}'\n")
nombre = job['nombre']
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'lista.txt', '-c', 'copy',
                '-movflags', '+faststart', f'{res}/{nombre}.mp4'], check=True)
ts = []
for p in glob.glob(f'{ent}/**/tiempo_*.txt', recursive=True):
    v = open(p).read().split(); ts.append((float(v[0]), float(v[1]) if len(v) > 1 else 0.0))
inf = (f'trozos {len(ids)}, faltan {falta}\n'
       f'render: el más lento {max(r for r, _ in ts):.0f} s, el más rápido {min(r for r, _ in ts):.0f} s, suma {sum(r for r, _ in ts):.0f} s\n'
       f'preparar máquina: el más lento {max(p for _, p in ts):.0f} s\n')
open(f'{res}/INFORME.txt', 'w').write(inf); print(inf)
