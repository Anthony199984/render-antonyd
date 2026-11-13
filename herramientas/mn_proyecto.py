"""RENDER EN GITHUB CON EL MAX NUEVO · paso 1: armar el PROYECTO CIFRADO de un video y subirlo (rama proyecto-mn).

    python3 mn_proyecto.py /home/claude/<carpeta>/<video>.py ANCHO ALTO [cuadros_de_prueba]

El video es un archivo con preparar(ancho, alto) -> (cuadros, fps) y cuadro(k) -> imagen (ejemplo:
max_nuevo/render_github_lovely.py). Este paso:
  1. corre el video aquí con strace en unos cuadros (el primero, el del medio y el último) y anota TODOS los archivos de
     /home/claude que abre (código, Max, la ciudad, texturas, animación). Solo eso se sube: nada de más, nada de menos.
  2. los mete en un ZIP y lo CIFRA (AES-256, openssl, clave de 64 caracteres guardada en /home/claude/.clave_render_github,
     que también está como secreto CLAVE_PROYECTO del repo). El repo es público, pero el archivo de Max no se puede abrir
     sin la clave: regla de Anthony, los archivos de Max nunca quedan a la vista.
  3. lo corta en pedazos de 95 MB y los sube a la rama proyecto-mn (cada subida la reemplaza: no se acumulan proyectos).
Deja /tmp/mn_proyecto.json con lo que hace falta para el plan (mn_plan.py)."""
import sys, os, json, subprocess, hashlib, secrets, tempfile, shutil, glob, time
VIDEO, ANCHO, ALTO = os.path.abspath(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
REPO = 'Anthony199984/render-antonyd'; CLAVE = '/home/claude/.clave_render_github'
assert VIDEO.startswith('/home/claude/'), 'el video tiene que estar dentro de /home/claude'
def sh(c, **k): return subprocess.run(c, shell=True, check=True, text=True, capture_output=True, **k).stdout
# ---- 1) qué archivos usa el video
tmp = tempfile.mkdtemp(); prueba = os.path.join(tmp, 'p.py')
open(prueba, 'w').write(f'''import sys, importlib.util, json
sp = importlib.util.spec_from_file_location("video", {VIDEO!r}); V = importlib.util.module_from_spec(sp); sp.loader.exec_module(V)
n, fps = V.preparar({ANCHO}, {ALTO})
ks = sorted(set([0, n // 2, n - 1] + [int(x) for x in {sys.argv[4] if len(sys.argv) > 4 else ''!r}.split(',') if x]))
for k in ks: im = V.cuadro(k); assert im.shape[:2] == ({ALTO}, {ANCHO}), im.shape
json.dump(dict(cuadros=n, fps=fps), open({os.path.join(tmp, "info.json")!r}, "w"))
''')
t0 = time.time()
r = subprocess.run(['strace', '-f', '-e', 'trace=openat,open', '-o', os.path.join(tmp, 'st.txt'), sys.executable, prueba], text=True, capture_output=True)
if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:]); sys.exit('el video falló aquí: no se sube nada')
info = json.load(open(os.path.join(tmp, 'info.json')))
import re
usados = set()
for l in open(os.path.join(tmp, 'st.txt')):
    for f in re.findall(r'"(/home/claude/[^"]+)"', l):
        if '= -1' in l or f.startswith('/home/claude/.'): continue
        if '/__pycache__/' in f:                                  # Python abre la copia compilada: se sube el .py original
            f = f.replace('/__pycache__/', '/').split('.cpython-')[0] + '.py'
        if os.path.isfile(f): usados.add(f)
usados.add(VIDEO)
tam = sum(os.path.getsize(f) for f in usados)
print('el video usa %d archivos (%.0f MB) · %d cuadros a %g fps · prueba en %.0f s' % (len(usados), tam / 1e6, info['cuadros'], info['fps'], time.time() - t0))
# ---- 2) ZIP cifrado
if not os.path.exists(CLAVE):
    open(CLAVE, 'w').write(secrets.token_hex(32)); os.chmod(CLAVE, 0o600); print('clave nueva creada en', CLAVE)
lista = os.path.join(tmp, 'lista.txt'); open(lista, 'w').write('\n'.join(os.path.relpath(f, '/home/claude') for f in sorted(usados)) + '\n')
zipf = os.path.join(tmp, 'mn.zip'); sh(f'cd /home/claude && zip -q -1 {zipf} -@ < {lista}')
enc = os.path.join(tmp, 'mn.enc')
sh(f'openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -in {zipf} -out {enc} -pass file:{CLAVE}'); os.remove(zipf)
h = hashlib.sha256(open(enc, 'rb').read()).hexdigest()
# ---- 3) pedazos a la rama proyecto-mn (repo nuevo y vacío en cada subida: la rama no guarda proyectos viejos)
sub = os.path.join(tmp, 'git'); os.makedirs(sub)
sh(f'cd {sub} && split -b 95000000 -d -a 2 {enc} mn.enc. && rm -f {enc}')
open(os.path.join(sub, 'SHA256.txt'), 'w').write(f'{h}  mn.enc\n')
open(os.path.join(sub, 'LEEME.md'), 'w').write('Proyecto CIFRADO (AES-256) del render en GitHub de Antonydfilms. Sin la clave no se puede abrir.\n')
json.dump(dict(video=os.path.relpath(VIDEO, '/home/claude'), ancho=ANCHO, alto=ALTO, **info, archivos=len(usados), mb=round(tam / 1e6)),
          open(os.path.join(sub, 'proyecto.json'), 'w'), indent=1)
sh(f'cd {sub} && git init -q -b proyecto-mn && git add -A && git -c user.name=antonydfilms -c user.email=render@antonydfilms commit -qm "proyecto cifrado"')
sh(f'cd {sub} && git push -q -f https://github.com/{REPO}.git proyecto-mn:proyecto-mn')
pzs = sorted(glob.glob(os.path.join(sub, 'mn.enc.*')))
print('subido a proyecto-mn: %d pedazos, cifrado %s' % (len(pzs), h[:16]))
json.dump(dict(video=os.path.relpath(VIDEO, '/home/claude'), ancho=ANCHO, alto=ALTO, cuadros=info['cuadros'], fps=info['fps'], sha=h),
          open('/tmp/mn_proyecto.json', 'w'), indent=1)
shutil.rmtree(tmp)
