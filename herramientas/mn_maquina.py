"""RENDER EN GITHUB CON EL MAX NUEVO · lo que hace UNA máquina.
    python3 mn_maquina.py trabajos/MN.json <id> <salida>
Carga el video del proyecto (preparar una vez), dibuja SUS cuadros k0..k1-1 en orden y los va comprimiendo mientras tanto
(libx264 CRF 18, yuv420p, tabla de color BT.709). Un solo proceso por máquina (el render ya usa todos los núcleos).
Con modo "medir" dibuja solo unos cuadros sueltos y anota cuánto tarda cada uno (para repartir mejor)."""
import sys, os, json, time, subprocess, importlib.util, numpy as np
plan = json.load(open(sys.argv[1])); ID = sys.argv[2]; SAL = os.path.abspath(sys.argv[3]); os.makedirs(SAL, exist_ok=True)
tr = next(t for t in plan['trozos'] if t['id'] == ID); A, H = plan['ancho'], plan['alto']; FPS = plan['fps']
INI = time.time(); log = open(f'{SAL}/log_{ID}.txt', 'w')
def dice(m): m = f'[{time.time() - INI:7.1f}s] {m}'; print(m, flush=True); log.write(m + '\n'); log.flush()
sp = importlib.util.spec_from_file_location('video', '/home/claude/' + plan['video']); V = importlib.util.module_from_spec(sp); sp.loader.exec_module(V)
n, fps = V.preparar(A, H); dice(f'escena lista: {n} cuadros a {fps:g} fps')
COLOR = ['-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
if plan.get('modo') == 'medir':
    med = []
    for k in tr['cuadros']:
        t = time.time(); V.cuadro(k); med.append([k, round(time.time() - t, 2)]); dice(f'cuadro {k}: {med[-1][1]} s')
    json.dump(med, open(f'{SAL}/medida_{ID}.json', 'w')); sys.exit(0)
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe(); SALV = f'{SAL}/trozo_{ID}.mp4'
p = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{A}x{H}', '-r', str(FPS), '-i', '-',
                      '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-c:v', 'libx264', '-crf', str(plan.get('crf', 18)),
                      '-pix_fmt', 'yuv420p', *COLOR, SALV], stdin=subprocess.PIPE)
k0, k1 = tr['k0'], tr['k1']
for k in range(k0, k1):
    t = time.time(); im = np.ascontiguousarray(V.cuadro(k)[..., :3], dtype=np.uint8)
    assert im.shape == (H, A, 3), im.shape
    p.stdin.write(im.tobytes())
    if (k - k0) % 10 == 0 or k == k1 - 1: dice(f'cuadro {k} ({k - k0 + 1}/{k1 - k0}) {time.time() - t:.1f} s')
p.stdin.close(); p.wait()
assert p.returncode == 0 and os.path.getsize(SALV) > 1000, 'ffmpeg falló'
json.dump(dict(id=ID, k0=k0, k1=k1, segundos=round(time.time() - INI, 1)), open(f'{SAL}/hecho_{ID}.json', 'w'))
dice(f'LISTO {ID}: cuadros {k0}-{k1 - 1} en {time.time() - INI:.0f} s')
