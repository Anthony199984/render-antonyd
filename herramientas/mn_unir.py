"""RENDER EN GITHUB CON EL MAX NUEVO · unir los pedazos (sin volver a comprimir) y comprobar el video.
    python3 mn_unir.py trabajos/MN.json salida resultado"""
import sys, os, json, glob, subprocess, imageio_ffmpeg
plan = json.load(open(sys.argv[1])); SAL, RES = sys.argv[2], sys.argv[3]; os.makedirs(RES, exist_ok=True)
ff = imageio_ffmpeg.get_ffmpeg_exe(); inf = [f"# {plan['nombre']}", '']
faltan = []; lista = open('/tmp/lista.txt', 'w')
for t in plan['trozos']:
    v = glob.glob(f"{SAL}/trozo-{t['id']}/trozo_{t['id']}.mp4") or glob.glob(f"{SAL}/*/trozo_{t['id']}.mp4")
    h = glob.glob(f"{SAL}/*/hecho_{t['id']}.json")
    if not v or not h: faltan.append(t['id']); continue
    lista.write(f"file '{os.path.abspath(v[0])}'\n"); inf.append(f"- {t['id']}: cuadros {t['k0']}-{t['k1'] - 1}, {json.load(open(h[0]))['segundos']} s")
lista.close()
if faltan:
    inf.append(f'\nFALTAN pedazos: {faltan} (rehacerlos con mn_plan.py --solo)'); open(f'{RES}/INFORME.md', 'w').write('\n'.join(inf) + '\n'); sys.exit(1)
out = f"{RES}/{plan['nombre']}.mp4"
subprocess.run([ff, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', '/tmp/lista.txt', '-c', 'copy', '-movflags', '+faststart', out], check=True)
r = subprocess.run([ff, '-i', out, '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-'], capture_output=True, text=True).stderr
import re
nf = int(re.findall(r'frame=\s*(\d+)', r)[-1]) if re.findall(r'frame=\s*(\d+)', r) else -1
total = plan['trozos'][-1]['k1'] - plan['trozos'][0]['k0']
inf.append(f"\nVIDEO: {nf} cuadros (deben ser {total}) · {plan['ancho']}x{plan['alto']} · {os.path.getsize(out) / 1e6:.1f} MB")
open(f'{RES}/INFORME.md', 'w').write('\n'.join(inf) + '\n'); print('\n'.join(inf))
sys.exit(0 if nf == total else 1)
