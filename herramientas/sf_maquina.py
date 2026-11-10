"""Lo que hace UNA máquina de GitHub con el video de superfuerza.

    python3 sf_maquina.py trabajos/SF.json <id> <salida>

El plan (SF.json) dice, para esta máquina:
  k0, k1    cuadros del video que le tocan (k0 … k1-1)
  tareas    procesos a lanzar: {"tipo": "escena" | "dentro:<vista>", "cuadros": [lista]}
  paralelo  cuántos procesos a la vez
Modo "medir": solo mide el costo de cada cuadro (no sale video).
Modo "render": lanza las tareas, y al final arma su pedazo de video en orden (cuadros de escena + cuadros por dentro),
con el mismo codificador que tomas/tramo.py (libx264 CRF 18, yuv420p, 60 c/s).
"""
import sys, os, json, time, subprocess, glob
import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = '/home/claude/superfuerza'
FPS, A, H = 60, 1080, 1920
plan = json.load(open(sys.argv[1])); ID = sys.argv[2]; SAL = os.path.abspath(sys.argv[3])
tr = next(t for t in plan['trozos'] if t['id'] == ID)
os.makedirs(SAL, exist_ok=True)
CRUDO = '/tmp/crudo'; os.makedirs(CRUDO, exist_ok=True)
INI = time.time()

# vistas por dentro: (desde, hasta, nombre)  · copiadas de tomas2_sf.INTERIORES con DENTRO=1080 (se comprueba abajo)
INTERIORES = [(8.27, 13.16, 'brazo'), (44.39, 51.0, 'carcel'), (51.0, 52.60, 'tendon'), (58.32, 62.59, 'corazon'), (65.73, 67.95, 'pareja')]
_txt = open(f'{RAIZ}/tomas2_sf.py').read()
assert "INTERIORES = [(8.27, 13.16, f'dentro/cuadros_brazo_{DENTRO}', _FD(30)), (44.39, 51.0, f'dentro/cuadros_carcel_{DENTRO}', 60), (51.0, 52.60, f'dentro/cuadros_tendon_{DENTRO}', _FD(30))," in _txt
assert "(58.32, 62.59, f'dentro/cuadros_corazon_{DENTRO}', 60), (65.73, 67.95, f'dentro/cuadros_pareja_{DENTRO}', _FD(30))]" in _txt


def interior(k):
    """Igual que Rodaje2.interior: (vista, índice del JPG) o None."""
    t = k / FPS
    for a, b, n in INTERIORES:
        if a <= t < b: return n, min(int(round((t - a) * FPS)), int(round((b - a) * FPS)) - 1)
    return None


def correr(tareas, paralelo, medir):
    env = dict(os.environ, MEDIR='1' if medir else '0')
    cola = list(enumerate(tareas)); vivos = []; fallos = []
    while cola or vivos:
        while cola and len(vivos) < paralelo:
            i, t = cola.pop(0); lj = f'/tmp/tarea_{i}.json'; json.dump(t['cuadros'], open(lj, 'w'))
            log = open(f'{SAL}/log_{i}_{t["tipo"].replace(":", "_")}.txt', 'w')
            p = subprocess.Popen([sys.executable, '-u', f'{AQUI}/sf_trabajador.py', t['tipo'], lj, CRUDO if t['tipo'] == 'escena' else SAL],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            vivos.append((p, i, t, log)); print(f'[{time.time() - INI:6.1f}s] lanza tarea {i} {t["tipo"]} ({len(t["cuadros"])} cuadros)', flush=True)
        time.sleep(1)
        for v in list(vivos):
            if v[0].poll() is not None:
                vivos.remove(v); v[3].close()
                print(f'[{time.time() - INI:6.1f}s] termina tarea {v[1]} {v[2]["tipo"]} -> {v[0].returncode}', flush=True)
                if v[0].returncode != 0: fallos.append(v[1]); print(open(v[3].name).read()[-3000:], flush=True)
    return fallos


medir = plan.get('modo') == 'medir'
fallos = correr(tr['tareas'], tr.get('paralelo', 1), medir)
for m in glob.glob(f'{CRUDO}/medida_*.json'): os.replace(m, f'{SAL}/{os.path.basename(m)}')
if fallos: sys.exit(f'FALLARON tareas {fallos}')
T_RENDER = time.time() - INI
if medir:
    open(f'{SAL}/tiempo_{ID}.txt', 'w').write(f'{T_RENDER:.1f} {os.environ.get("T_PREPARAR", "0")}\n'); sys.exit(0)

# ------------------------------------------------------------------ armar el pedazo de video, en orden
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
k0, k1 = tr['k0'], tr['k1']; sal = f'{SAL}/trozo_{ID}.mp4'
p = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{A}x{H}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', sal], stdin=subprocess.PIPE)
n_dentro = 0
for k in range(k0, k1):
    d = interior(k)
    if d is not None:                                   # igual que Rodaje2.cuadro con un cuadro por dentro
        ruta = f'{RAIZ}/dentro/cuadros_{d[0]}_1080/{d[1]:05d}.jpg'
        if not os.path.exists(ruta): sys.exit(f'FALTA el cuadro por dentro {ruta} (cuadro {k})')
        im = np.asarray(Image.open(ruta).convert('RGB').resize((A, H))); n_dentro += 1
    else:
        ruta = f'{CRUDO}/{k:05d}.npy'
        if not os.path.exists(ruta): sys.exit(f'FALTA el cuadro {k}')
        im = np.load(ruta); os.remove(ruta)
    p.stdin.write(np.ascontiguousarray(im).tobytes())
p.stdin.close(); p.wait()
assert p.returncode == 0 and os.path.getsize(sal) > 1000, 'ffmpeg falló'
T_TOTAL = time.time() - INI
open(f'{SAL}/tiempo_{ID}.txt', 'w').write(f'{T_TOTAL:.1f} {os.environ.get("T_PREPARAR", "0")} {T_RENDER:.1f} {k1 - k0} {n_dentro}\n')
print(f'LISTO trozo {ID}: cuadros {k0}–{k1 - 1} ({k1 - k0}, {n_dentro} por dentro) en {T_TOTAL:.0f} s', flush=True)
