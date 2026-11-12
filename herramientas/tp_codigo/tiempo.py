"""Voz SIN silencios y el reloj del video.
Las escenas siguen escritas en el tiempo VIEJO (voz_final.wav, 89,2 s, con pausas). Cada silencio de la voz se corta
(queda una pausa natural de 0,24 s) y la escena NO salta: se acelera un poco alrededor del corte (rampa suave), así el video
sigue fluido y dura lo mismo que la voz nueva.
    import tiempo as TI
    TI.viejo(t_nuevo) -> t_viejo      (para dibujar el cuadro)
    TI.nuevo(t_viejo) -> t_nuevo      (para poner efectos de sonido)
    TI.DUR                            (duración nueva del video)
    python3 tiempo.py                 -> audio/voz_corta.wav + audio/cortes.json"""
import json, os, subprocess, re, numpy as np

RA = '/home/claude/teletransporte'
QUEDA = .24                     # pausa que se deja entre frases (con 0,08 la voz sonaba atropellada y el video corría)
QUEDA_LARGA = .45               # la pausa larga del apagón (1,3 s) se queda en un respiro dramático
VEL_MAX = 1.22                  # el video nunca se acelera más que esto alrededor de un corte


def queda(a, b): return QUEDA_LARGA if b - a > .9 else QUEDA
DUR_VIEJA = 89.2


def _silencios():
    p = f'{RA}/audio/cortes.json'
    if os.path.exists(p): return json.load(open(p))['silencios']
    r = subprocess.run(['ffmpeg', '-nostats', '-i', f'{RA}/audio/voz_final.wav', '-af', 'silencedetect=noise=-38dB:d=0.18', '-f', 'null', '-'],
                       capture_output=True, text=True).stderr
    a = [float(x) for x in re.findall(r'silence_start: ([\d.]+)', r)]; b = [float(x) for x in re.findall(r'silence_end: ([\d.]+)', r)]
    return [(x, y) for x, y in zip(a, b)]


SIL = [(a, b) for a, b in _silencios() if (b - a) - queda(a, b) > .03]      # solo se recorta lo que sobra de cada pausa
_o = np.linspace(-1, DUR_VIEJA + 1, 200001)


def _mapa(anchos):
    n = _o.copy()
    for (a, b), W in zip(SIL, anchos):
        r = (b - a) - queda(a, b); c = (a + b) / 2
        n -= r * np.clip((_o - (c - W / 2)) / W, 0, 1)
    return n


_W = [max(2.0, 6.0 * ((b - a) - queda(a, b))) for a, b in SIL]
for _ in range(40):                                    # ensancha las rampas hasta que el video no pase de VEL_MAX
    _n = _mapa(_W); v = np.diff(_o) / np.diff(_n)
    if v.max() <= VEL_MAX: break
    tm = _o[int(np.argmax(v))]
    for i, (a, b) in enumerate(SIL):
        if abs((a + b) / 2 - tm) <= _W[i] / 2: _W[i] *= 1.15
assert np.all(np.diff(_n) > 0)
DUR = float(np.interp(DUR_VIEJA, _o, _n))


def nuevo(t_viejo): return float(np.interp(t_viejo, _o, _n))


def viejo(t_nuevo): return float(np.interp(t_nuevo, _n, _o))


if __name__ == '__main__':
    import wave
    SR = 48000
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'{RA}/audio/voz_final.wav', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True).stdout
    v = np.frombuffer(raw, np.float32)
    partes = []; i0 = 0
    for a, b in SIL:
        q = queda(a, b); ca, cb = int((a + q / 2) * SR), int((b - q / 2) * SR)
        partes.append(v[i0:ca]); i0 = cb
    partes.append(v[i0:])
    f = int(.006 * SR); out = partes[0].copy()
    for p in partes[1:]:                                # empalme con fundido cortito (sin chasquidos)
        if len(out) > f and len(p) > f:
            x = np.linspace(0, 1, f); out[-f:] = out[-f:] * (1 - x) + p[:f] * x; out = np.concatenate([out, p[f:]])
        else: out = np.concatenate([out, p])
    w = wave.open(f'{RA}/audio/voz_corta.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
    json.dump(dict(silencios=_silencios(), quedan=QUEDA), open(f'{RA}/audio/cortes.json', 'w'), indent=1)
    print(f'voz: {len(v) / SR:.2f} s -> {len(out) / SR:.2f} s · video: {DUR_VIEJA} -> {DUR:.2f} s · {len(SIL)} silencios recortados')
    print('máxima aceleración de la escena: %.2fx' % (1 / np.diff(_n).min() * np.diff(_o)[0]))
