"""Lo que hace UNA máquina de GitHub con el video "Si pudieras teletransportarte".

    python3 tp_maquina.py trabajos/TP.json <id> <salida>

Cada trozo es un tramo seguido de cuadros DENTRO de una sola toma (cada toma prepara su propio escenario).
Las tomas con estado (la arena) se llevan desde el principio de la toma sin dibujar (igual que render_tp.py).
Los cuadros van directo a ffmpeg en orden (libx264 CRF 18, yuv420p, 60 c/s, tabla de color BT.709)."""
import sys, os, json, time, subprocess, importlib, math
import numpy as np

plan = json.load(open(sys.argv[1])); ID = sys.argv[2]; SAL = os.path.abspath(sys.argv[3])
tr = next(t for t in plan['trozos'] if t['id'] == ID)
FPS = plan.get('fps', 60); A = plan.get('ancho', 1080); H = A * 16 // 9
os.makedirs(SAL, exist_ok=True)
sys.path.insert(0, '/home/claude/teletransporte'); os.chdir('/home/claude/teletransporte')
import glob, shutil                                   # arreglos de código hechos después de subir el proyecto
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tp_codigo', '*.py')):
    shutil.copy(f, '/home/claude/teletransporte/'); print('código nuevo:', os.path.basename(f), flush=True)
INI = time.time()
from tp_base import escena
import tiempo as TI                                    # el video va en el tiempo NUEVO (voz sin silencios); las tomas, en el viejo
mod = importlib.import_module(f"t{tr['toma']:02d}")
T = mod.Toma(); E = escena(A, H); T.preparar(E)
k0, k1 = tr['k0'], tr['k1']
if hasattr(T, 'avanzar'):                              # el estado (arena) se lleva con los mismos pasos que tendrá el video
    kp = max(0, int(math.floor(TI.nuevo(getattr(T, 't_pre', T.INI)) * FPS))); n = 0
    for k in range(kp, k0):
        tv = TI.viejo(k / FPS)
        if tv >= getattr(T, 't_pre', T.INI): T.avanzar(tv); n += 1
    print(f'[{time.time() - INI:6.1f}s] estado llevado {n} pasos', flush=True)
import imageio_ffmpeg
COLOR = ['-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
SALV = f'{SAL}/trozo_{ID}.mp4'
p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{A}x{H}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', *COLOR, SALV], stdin=subprocess.PIPE)
T_R = time.time()
for k in range(k0, k1):
    t = min(max(TI.viejo(k / FPS), T.INI), T.FIN - 1e-4)
    if hasattr(T, 'avanzar'): T.avanzar(t)
    a = time.time(); img = T.cuadro(E, t)
    if img.shape[:2] != (H, A): raise RuntimeError(f'cuadro {k} mide {img.shape}')
    p.stdin.write(np.ascontiguousarray(img[..., :3]).tobytes())
    if (k - k0) % 20 == 0: print(f'[{time.time() - INI:6.1f}s] cuadro {k} t={t:.3f} {time.time() - a:.2f}s', flush=True)
p.stdin.close(); p.wait()
if p.returncode != 0 or os.path.getsize(SALV) < 1000: sys.exit('ffmpeg falló')
open(f'{SAL}/tiempo_{ID}.txt', 'w').write(f'{time.time() - INI:.1f} {os.environ.get("T_PREPARAR", "0")} {time.time() - T_R:.1f} {k1 - k0} 0\n')
print(f'LISTO trozo {ID}: toma {tr["toma"]} cuadros {k0}–{k1 - 1} en {time.time() - INI:.0f} s', flush=True)
