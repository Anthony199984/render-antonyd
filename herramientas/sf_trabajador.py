"""Un proceso de render de "Si despertaras con superfuerza" (lo lanza sf_maquina.py).

    python3 sf_trabajador.py escena        cuadros.json salida      -> salida/KKKKK.npy  (cuadro k del video, 1080x1920 RGB)
    python3 sf_trabajador.py dentro:brazo  indices.json salida      -> dentro/cuadros_brazo_1080/IIIII.jpg

Hace EXACTAMENTE lo mismo que el render original:
  escena  = tomas/tramo.py   (R = Rodaje2(1080); R.cuadro(k / 60); mismo ajuste de tamaño)
  dentro  = dentro/*.py      (mismo objeto, mismo tiempo de cada cuadro, JPG calidad 92, misma numeración)
Con MEDIR=1 no guarda nada: solo mide el tiempo de cada cuadro (prueba de costo, los cuadros se tiran).
"""
import sys, os, json, time, resource

RAIZ = '/home/claude/superfuerza'
FPS, A = 60, 1080
tipo, lista, sal = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
MEDIR = os.environ.get('MEDIR') == '1'
os.chdir(RAIZ)
os.makedirs(sal, exist_ok=True)
med = {'tipo': tipo, 'cuadros': []}
t_ini = time.time()


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime


if tipo == 'escena':
    sys.path.insert(0, RAIZ + '/tomas'); sys.path.insert(0, RAIZ)       # igual que tomas/tramo.py
    import numpy as np
    import tomas2_sf as T2
    import soma.sin_fugas                                               # noqa: F401
    from PIL import Image
    H = A * 16 // 9; H += H % 2
    R = T2.Rodaje2(A)
    med['arranque'] = time.time() - t_ini
    for k in lista:
        t0, c0 = time.time(), _cpu()
        im = np.asarray(R.cuadro(k / FPS))[..., :3]
        if im.shape[:2] != (H, A): im = np.asarray(Image.fromarray(np.ascontiguousarray(im)).resize((A, H)))
        if not MEDIR:
            tmp = f'{sal}/{k:05d}.tmp.npy'; np.save(tmp, np.ascontiguousarray(im, dtype=np.uint8)); os.replace(tmp, f'{sal}/{k:05d}.npy')
        med['cuadros'].append([k, time.time() - t0, _cpu() - c0])
        print(f'escena {k} {time.time() - t0:.2f}s', flush=True)
else:
    nombre = tipo.split(':')[1]
    # nombre: (programa, clase, comienzo de la vista, numeración)   'rel' = t0 + f*paso (brazo, pareja) · 'abs' = T0 + f/FPS (cárcel, corazón)
    D = {'brazo': ('brazo_sf', 'Brazo', 8.27, 'rel', True), 'tendon': ('brazo_sf', 'Brazo', 51.00, 'rel', True),
         'carcel': ('anim_carcel', 'Toma', 44.39, 'abs', False), 'corazon': ('corazon_sf', 'Pecho', 58.32, 'abs', False),
         'pareja': ('pareja_sf', 'Pareja', 65.73, 'rel', True)}
    prog, clase, a, num, corta = D[nombre]
    sys.path.insert(0, RAIZ + '/dentro')                                 # igual que python3 dentro/<prog>.py
    import importlib
    import numpy as np
    from PIL import Image
    M = importlib.import_module(prog)
    if num == 'abs': assert abs(M.T0 - a) < 1e-9 and M.FPS == 60, (M.T0, M.FPS)
    obj = getattr(M, clase)(A)
    med['arranque'] = time.time() - t_ini
    carpeta = f'{RAIZ}/dentro/cuadros_{nombre}_1080'; os.makedirs(carpeta, exist_ok=True)
    paso = 1 / M.FPS if prog == 'brazo_sf' else 1 / 60
    for f in lista:
        ruta = f'{carpeta}/{f:05d}.jpg'
        if os.path.exists(ruta) and not MEDIR: continue
        t0, c0 = time.time(), _cpu()
        t = (a + f * paso) if num == 'rel' else (M.T0 + f / M.FPS)
        x = np.asarray(obj.cuadro(t))
        if corta: x = x[..., :3]
        if not MEDIR:
            tmp = ruta[:-4] + '.tmp'; Image.fromarray(x).save(tmp, format='JPEG', quality=92); os.replace(tmp, ruta)
        med['cuadros'].append([f, time.time() - t0, _cpu() - c0])
        print(f'dentro {nombre} {f} {time.time() - t0:.2f}s', flush=True)

med['total'] = time.time() - t_ini
med['memoria_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
json.dump(med, open(f'{sal}/medida_{tipo.replace(":", "_")}_{os.getpid()}.json', 'w'))
print('OK', tipo, len(lista), f'{med["total"]:.1f}s', flush=True)
