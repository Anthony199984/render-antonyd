"""Une los pedazos del video de superfuerza (o junta las medidas en modo "medir").

    python3 sf_unir.py trabajos/SF.json <carpeta con los trozo-*> <resultado> [audio_dir]

render: une sin volver a comprimir (igual que render_final.sh) → superfuerza_final_SIN_SONIDO.mp4, comprueba los cuadros,
        y después (tiempo aparte, fuera de la competencia) le pone el sonido: _1080x1920_60fps.mp4 y _SOLO_VOZ.mp4.
"""
import sys, os, json, glob, re, subprocess, time
import imageio_ffmpeg

plan = json.load(open(sys.argv[1])); ENT, SAL = sys.argv[2], sys.argv[3]; AUD = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(SAL, exist_ok=True); ff = imageio_ffmpeg.get_ffmpeg_exe()
inf = [f"# {plan['nombre']}", f"Plan: {len(plan['trozos'])} máquinas · modo {plan.get('modo', 'render')}", '']

tiempos = {}
for f in glob.glob(f'{ENT}/*/tiempo_*.txt'):
    tiempos[re.search(r'tiempo_(\w+)\.txt', f).group(1)] = open(f).read().split()

if plan.get('modo') == 'medir':
    med = [json.load(open(f)) | {'maquina': os.path.basename(os.path.dirname(f))} for f in glob.glob(f'{ENT}/*/medida_*.json')]
    json.dump(med, open(f'{SAL}/medicion.json', 'w'))
    for m in sorted(med, key=lambda m: m['maquina']):
        c = m['cuadros']; w = sum(x[1] for x in c) / max(len(c), 1); u = sum(x[2] for x in c) / max(sum(x[1] for x in c), 1e-9)
        inf.append(f"- {m['maquina']} {m['tipo']}: arranque {m.get('arranque', 0):.1f} s · {len(c)} cuadros · {w:.2f} s/cuadro · "
                   f"uso CPU {u:.2f} núcleos · memoria {m['memoria_mb']:.0f} MB")
    inf += ['', 'Preparar máquina (s): ' + ', '.join(f'{k}={v[1]}' for k, v in sorted(tiempos.items()))]
    open(f'{SAL}/INFORME.md', 'w').write('\n'.join(inf) + '\n'); print('\n'.join(inf)); sys.exit(0)

orden = sorted(plan['trozos'], key=lambda t: t['k0'])
faltan = [t['id'] for t in orden if not glob.glob(f"{ENT}/trozo-{t['id']}/trozo_{t['id']}.mp4")]
if faltan: sys.exit(f'FALTAN trozos: {faltan}')
open('/tmp/lista.txt', 'w').write(''.join(f"file '{os.path.abspath(glob.glob(ENT + '/trozo-' + t['id'] + '/trozo_' + t['id'] + '.mp4')[0])}'\n" for t in orden))
sin = f'{SAL}/superfuerza_final_SIN_SONIDO.mp4'
subprocess.run([ff, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', '/tmp/lista.txt', '-c:v', 'copy', '-an', '-movflags', '+faststart', sin], check=True)
fin_video = time.time()


def contar(f):
    r = subprocess.run([ff, '-i', f, '-map', '0:v', '-c', 'copy', '-f', 'null', '-'], capture_output=True, text=True).stderr
    n = re.findall(r'frame=\s*(\d+)', r); fps = re.search(r'(\d+(?:\.\d+)?) fps', r); res = re.search(r'Video: .*?(\d{3,4})x(\d{3,4})', r)
    return int(n[-1]) if n else -1, (fps.group(1) if fps else '?'), (f'{res.group(1)}x{res.group(2)}' if res else '?')


n, fps, res = contar(sin); total = plan['k_total']
inf += [f"**Video sin sonido:** {n} cuadros (deben ser {total}) · {res} · {fps} FPS · {os.path.getsize(sin) / 1e6:.1f} MB",
        f"VERIFICADO: {'SÍ' if n == total and res == '1080x1920' else 'NO'}", '']
r = [float(v[0]) for v in tiempos.values()]; pr = [float(v[1]) for v in tiempos.values()]
if r: inf += [f'Render por máquina: el más lento {max(r):.0f} s · el más rápido {min(r):.0f} s · suma {sum(r) / 60:.1f} min de máquina',
              f'Preparar máquina: el más lento {max(pr):.0f} s', '']
for t in orden:
    v = tiempos.get(t['id'])
    if v: inf.append(f"- máquina {t['id']}: cuadros {t['k0']}–{t['k1'] - 1} · render {float(v[0]):.0f} s · preparar {v[1]} s · por dentro {v[4]}")

if AUD:                                                             # el sonido, fuera de la competencia
    t0 = time.time()
    for wav, nom in (('mezcla_v2.wav', 'superfuerza_final_1080x1920_60fps.mp4'), ('solo_voz.wav', 'superfuerza_final_SOLO_VOZ.mp4')):
        subprocess.run([ff, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', '/tmp/lista.txt', '-i', f'{AUD}/{wav}', '-map', '0:v', '-map', '1:a',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-shortest', '-movflags', '+faststart', f'{SAL}/{nom}'], check=True)
        n2, _, _ = contar(f'{SAL}/{nom}'); inf.append(f'- {nom}: {n2} cuadros · {os.path.getsize(SAL + "/" + nom) / 1e6:.1f} MB')
    inf.append(f'Sonido puesto en {time.time() - t0:.0f} s (fuera de la competencia)')
open(f'{SAL}/INFORME.md', 'w').write('\n'.join(inf) + '\n'); print('\n'.join(inf))
if n != total: sys.exit('EL VIDEO NO TIENE TODOS LOS CUADROS')
