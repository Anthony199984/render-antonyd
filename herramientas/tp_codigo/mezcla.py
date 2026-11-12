"""Sonido del video: voz + efectos en su segundo exacto + ambientes por lugar (-24 dB). Sale audio/mezcla.wav (48 kHz estéreo).
Todos los sonidos son Freesound CC0."""
import subprocess, glob, numpy as np, json
import tiempo as TI
SR = 48000; DUR = TI.DUR
SON = '/home/claude/recursos_repo/recursos/out/sonidos'
RA = '/home/claude/teletransporte'


def leer(ruta, ini=0.0, dur=None):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(ini)] + (['-t', str(dur)] if dur else []) + ['-i', ruta, '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-']
    a = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2)
    return a


def son(n):
    return [f for f in glob.glob(f'{SON}/{n}/*') if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.aif'))][0]


def db(x): return 10 ** (x / 20)


mix = np.zeros((int(DUR * SR), 2), np.float32)


def poner(a, t, gan_db, fade_in=.02, fade_out=.05, hasta=None):
    a = a.copy(); n = len(a); t = TI.nuevo(t)                     # todo se escribe en el tiempo viejo y se pasa al nuevo
    if hasta is not None: n = min(n, int((hasta - t) * SR)); a = a[:n]
    fi, fo = int(fade_in * SR), int(fade_out * SR)
    if fi: a[:fi] *= np.linspace(0, 1, min(fi, n))[:, None]
    if fo and n > fo: a[-fo:] *= np.linspace(1, 0, fo)[:, None]
    i = int(t * SR); j = min(len(mix), i + n)
    mix[i:j] += a[:j - i] * db(gan_db)


def cama(n, t0, t1, gan_db=-24, ini=0.0):
    a = leer(son(n), ini)
    need = int((TI.nuevo(t1) - TI.nuevo(t0)) * SR)
    while len(a) < need: a = np.concatenate([a, a])
    a = a / (np.abs(a).max() + 1e-9) * .5
    poner(a[:need], t0, gan_db, .25, .25)


def efecto(n, t, gan_db=-12, ini=0.0, dur=None, normal=True):
    a = leer(son(n), ini, dur)
    if normal: a = a / (np.abs(a).max() + 1e-9) * .7
    poner(a, t, gan_db)


# ------------------------------------------------------------------ ambientes por lugar
cama('tp_amb_cocina', 0.0, 1.55, -26)
cama('tp_amb_tokio', 1.55, 4.7, -22)
cama('tp_amb_cocina', 5.75, 17.5, -26)
cama('tp2_viento', 17.5, 19.0, -24)
cama('tp_amb_tokio', 21.0, 36.3, -24)
cama('tp_amb_dubai', 36.3, 43.7, -23)
cama('tp_amb_mexico', 43.7, 52.05, -24)
cama('tp_zumbido', 52.05, 59.9, -27)
cama('tp_grillos', 61.3, 67.6, -27)
cama('tp_zumbido', 67.6, 69.75, -27)
cama('tp_grillos', 69.75, 73.36, -29)
cama('tp_amb_cocina', 79.4, 80.9, -30)
cama('tp_amb_cocina', 83.35, 88.45, -26)

# ------------------------------------------------------------------ efectos
efecto('tp_whoosh', 1.40, -10, dur=.5)                 # se va (estilo Goku)
efecto('tp_whoosh', 1.62, -12, dur=.5)                 # llega a Tokio
efecto('tp_bolsa', 3.95, -10, dur=.8)                  # cae al piso
efecto('tp_rebobinado', 4.70, -12, ini=2.0, dur=1.05)  # se rebobina hasta la cocina
efecto('tp_whoosh', 5.72, -14, dur=.5)
efecto('tp_enviado', 8.85, -14)                        # el dedo toca la pantalla
efecto('tp_enviado', 9.35, -18)                        # se ilumina Japón
efecto('tp_escaner', 12.35, -16, dur=1.0)
efecto('tp_whoosh', 13.1, -14)
efecto('tp_escaner', 13.6, -24, dur=2.2)
efecto('tp_whoosh', 15.7, -15)
efecto('tp_arena2', 16.45, -24, dur=1.1)
efecto('tp_whoosh', 17.6, -13)
efecto('tp_whoosh', 19.0, -17)
efecto('tp_whoosh', 20.85, -13)
efecto('tp_escaner', 21.2, -22, dur=3.0)
efecto('tp3_cortina2', 30.35, -10, ini=4.6, dur=.7)    # la cortina baja de golpe
efecto('tp_panza', 32.2, -11)
efecto('tp2_viento', 36.3, -12, dur=1.6)               # el horno
efecto('tp_asador', 38.3, -26, dur=3.6)
efecto('tp3_muerde', 41.05, -12)
efecto('tp_whoosh', 42.8, -18)
efecto('tp_whoosh', 45.3, -20)
efecto('tp_whoosh', 47.6, -16)
efecto('tp_enviado', 52.9, -14)
efecto('tp_escaner', 54.2, -18, dur=1.9)
efecto('tp_enviado', 56.15, -12)
efecto('tp_apagon', 58.35, -10, dur=1.5)
efecto('tp3_vela', 62.3, -28, dur=5.3)
efecto('tp_ding', 67.6, -9)
efecto('tp_fluorescente', 67.65, -18, dur=1.2)
efecto('tp_ronquido', 69.75, -22, dur=2.7)
efecto('tp_bolsa', 72.52, -12)
efecto('tp_whoosh', 73.3, -15)
efecto('tp_escoba', 77.6, -22, dur=1.8)
efecto('tp_whoosh', 80.85, -18)
efecto('tp_arena2', 84.4, -30, dur=2.5)

# ------------------------------------------------------------------ voz (ya trae la pausa de 1 s después de "se va la luz")
v = leer(f'{RA}/audio/voz_corta.wav')
v = v / (np.abs(v).max() + 1e-9) * .89
mix[:min(len(v), len(mix))] += v[:len(mix)]
mix[int(TI.nuevo(88.45) * SR):] *= 0                     # corte de todo con el negro
pk = np.abs(mix).max()
if pk > .97: mix *= .97 / pk
import wave
w = wave.open(f'{RA}/audio/mezcla.wav', 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
print('ok', len(mix) / SR, 'pico', float(np.abs(mix).max()))
