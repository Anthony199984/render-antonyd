"""Estilo realista: personajes Mixamo animados + modelos Poly Haven/Poly Pizza + cámaras fluidas.
Prueba de 5 s en la oficina de Benter. Marca integrada: Antonydfilms."""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/fbxtool')
import skin, assetlib
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from gl_engine import matrix, normalize
from effects import minimum_jerk as mj
from benter_scene import screen_image, keyboard_image
REC = '/home/claude/rec'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf'
BRAND = 'Antonydfilms'
import ctypes as C

def lerp(a, b, k): return a + (b - a) * k
def L3(a, b, k): return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * k

def set_mesh(r, name, data):
    """Sube/actualiza una malla dinámica reutilizando su buffer (sin fugas de memoria)."""
    data = np.ascontiguousarray(data, np.float32)
    if not hasattr(r, '_dyn'): r._dyn = {}
    if name not in r._dyn:
        r.add_mesh(name, data); vao = r.meshes[name][0]
        vbo = C.c_uint(); r.glGetIntegerv = getattr(r, 'glGetIntegerv', None)
        r._dyn[name] = vao
        # recuperar el VBO enlazado al VAO recién creado
        g = r.g; g.glGetIntegerv.argtypes = [C.c_uint, C.POINTER(C.c_int)]; v = C.c_int(); g.glGetIntegerv(0x8894, C.byref(v)); r._dyn[name] = (vao, v.value)
        return
    vao, vbo = r._dyn[name]
    r.glBindVertexArray(vao); r.glBindBuffer(0x8892, vbo); r.glBufferData(0x8892, data.nbytes, data.ctypes.data, 0x88E4)
    r.meshes[name] = (vao, len(data)); r.mesh_data[name] = data

class Actor:
    def __init__(s, r, key, char_path, anims, height=1.78):
        s.r = r; s.key = key; s.ch = skin.Character(char_path, height); s.anims = {n: skin.Anim(p) for n, p in anims.items()}; s.tex = []
        for i, p in enumerate(s.ch.parts):
            tex = 'white'
            if p['diff'] is not None:
                im = p['diff'].convert('RGBA')
                if p['opac'] is not None: im.putalpha(p['opac'].resize(im.size))
                if max(im.size) > 1024: im = im.resize((1024, 1024))
                tex = f'{key}_t{i}'; r.add_texture(tex, im); r.set_texture_filter(tex, 'linear')
            s.tex.append((tex, p['opac'] is not None))
    def draw(s, objs, anim, t, M=np.eye(4), blend=None):
        res = s.ch.skin(s.anims[anim], t)
        if blend:
            a2, t2, w = blend
            if w > 0:
                r2 = s.ch.skin(s.anims[a2], t2); res = [((1 - w) * P1 + w * P2, (1 - w) * N1 + w * N2) for (P1, N1), (P2, N2) in zip(res, r2)]
        for i, ((P, N), p) in enumerate(zip(res, s.ch.parts)):
            nm = f'{s.key}_m{i}'; set_mesh(s.r, nm, skin.interleave(P, N, p['uv']))
            objs.append(dict(mesh=nm, model=np.asarray(M, np.float32), texture=s.tex[i][0], color=(1, 1, 1), roughness=.62, skin=.45, ps1=0, transparent=s.tex[i][1], part_id=f'{s.key}{i}'))
    def bone(s, anim, t, name, M=np.eye(4)):
        G = s.ch.globals(s.anims[anim], t); p = G[s.ch.byname[name]][:3, 3]; p = skin.C4[:3, :3] @ p; p = (p - [0, 0, s.ch.zmin]) * s.ch.scale
        return (np.asarray(M) @ np.r_[p, 1])[:3]

def place(objs, M, color=None, **kw):
    out = []
    for o in objs:
        d = dict(o); d['model'] = np.asarray(M, np.float32)
        if color is not None: d['color'] = color
        d.update(kw); out.append(d)
    return out

def tex_image(r, name, im, rep=True):
    r.add_texture(name, im, repeat=rep); r.set_texture_filter(name, 'linear')

def plane_obj(r, name, M, tex, **kw):
    if 'plane' not in r.meshes:
        from gl_engine import plane; r.add_mesh('plane', plane())
    return dict(mesh='plane', model=np.asarray(M, np.float32), texture=tex, color=kw.pop('color', (1, 1, 1)), roughness=kw.pop('roughness', .8), ps1=0, **kw)

def box_obj(r, M, tex='white', **kw):
    tex = kw.pop('texture', tex)
    if 'box' not in r.meshes:
        from gl_engine import box; r.add_mesh('box', box())
    return dict(mesh='box', model=np.asarray(M, np.float32), texture=tex, color=kw.pop('color', (1, 1, 1)), roughness=kw.pop('roughness', .8), ps1=0, **kw)

def brand_label(w=512, h=128):
    im = Image.new('RGB', (w, h), (188, 190, 194)); d = ImageDraw.Draw(im)
    for y in range(h): d.line((0, y, w, y), fill=(178 + int(20 * math.sin(y / h * math.pi)), 180 + int(20 * math.sin(y / h * math.pi)), 186))
    d.rectangle((3, 3, w - 4, h - 4), outline=(120, 122, 128), width=3)
    d.text((w // 2, h // 2), BRAND, font=ImageFont.truetype(FONT, int(h * .5)), fill=(60, 62, 70), anchor='mm'); return im

def painting():
    W, H = 520, 760; im = Image.new('RGB', (W, H)); a = np.zeros((H, W, 3)); yy = np.linspace(0, 1, H)[:, None]
    a[:] = (np.array((236, 190, 120)) * (1 - yy) + np.array((120, 70, 60)) * yy)[:, None, :][:, :, :] if False else 0
    sky = np.array((240, 200, 140.)); low = np.array((90, 60, 50.)); a = sky[None, None] * (1 - yy[:, :, None]) + low[None, None] * yy[:, :, None]; a = np.repeat(a, W, 1)
    im = Image.fromarray(a.astype('uint8')); d = ImageDraw.Draw(im)
    d.ellipse((300, 170, 410, 280), fill=(255, 236, 190))
    d.polygon([(0, 520), (140, 380), (260, 470), (380, 360), (520, 500), (520, 760), (0, 760)], fill=(70, 80, 60))
    for i in range(3):  # caballos al galope, silueta
        x = 80 + i * 140; y = 600 - i * 18; d.ellipse((x, y, x + 90, y + 34), fill=(40, 28, 22)); d.polygon([(x + 80, y + 8), (x + 118, y - 26), (x + 128, y - 14), (x + 92, y + 22)], fill=(40, 28, 22))
        for lx, ang in [(10, -.5), (25, .4), (60, -.4), (75, .5)]: d.line((x + lx, y + 28, x + lx + 30 * math.sin(ang), y + 70), fill=(40, 28, 22), width=6)
    im = im.filter(ImageFilter.GaussianBlur(1.2)); d = ImageDraw.Draw(im)
    for k in range(4000):
        x, y = np.random.randint(0, W), np.random.randint(0, H); c = im.getpixel((x, y)); d.line((x, y, x + 6, y + 2), fill=tuple(min(255, v + 12) for v in c))
    d.text((W - 24, H - 26), BRAND, font=ImageFont.truetype(SERIF, 26), fill=(30, 20, 18), anchor='rm')  # firma del pintor
    return im

class Office:
    def __init__(s, r):
        s.r = r; R = REC
        s.benter = Actor(r, 'benter', f'{R}/z2/personajes/benter.fbx', {
            'teclear': f'{R}/z4/anim_benter/sentado_tecleando.fbx', 'celebrar': f'{R}/z1/anim_benter/sentado_celebrando.fbx',
            'quieto': f'{R}/z4/anim_benter/sentado_quieto.fbx', 'triste': f'{R}/z4/anim_benter/sentado_triste.fbx'})
        g = lambda key, path, **kw: assetlib.register_glb(r, key, path, **kw)[0]
        s.desk = g('desk', f'{R}/z4/oficina/escritorio.glb'); s.lamp = g('lamp', f'{R}/z2/oficina/lampara_escritorio.glb'); s.plant = g('plant', f'{R}/z3/oficina/planta_maceta.glb')
        s.shelf = g('shelf', f'{R}/z4/oficina/estanteria_libros.glb'); s.clock = g('clock', f'{R}/z4/oficina/reloj_pared.glb'); s.frame = g('frame', f'{R}/z4/oficina/cuadro_marco.glb')
        s.pc = g('pc', f'{R}/z4/oficina/computadora_retro.glb', target=.46)
        tex_image(r, 'floor', Image.open(f'{R}/z4/oficina/textura_madera_piso.jpg').convert('RGB').resize((1024, 1024)))
        tex_image(r, 'wallp', Image.open(f'{R}/z4/oficina/textura_pared.jpg').convert('RGB').resize((1024, 1024)))
        tex_image(r, 'kbd', keyboard_image(), False); tex_image(r, 'brand', brand_label(), False); tex_image(r, 'paint', painting(), False)
        tex_image(r, 'crt_screen', screen_image(0), False)
        s.screen_t = None
        # escena estática
        o = []
        o += [plane_obj(r, 'floor', matrix((0, 0, 0), (10, 10, 1)), 'floor', texrepeat=(5, 5), roughness=.55)]
        o += [plane_obj(r, 'w', matrix((0, -2.2, 1.5), (10, 3, 1), (math.pi / 2, 0, 0)), 'wallp', texrepeat=(4, 1.2), roughness=.95)]  # pared detrás del monitor
        o += [plane_obj(r, 'w', matrix((0, 2.6, 1.5), (10, 3, 1), (-math.pi / 2, 0, 0)), 'wallp', texrepeat=(4, 1.2), roughness=.95)]  # pared detrás de Benter
        o += [plane_obj(r, 'w', matrix((-2.4, 0, 1.5), (6, 3, 1), (math.pi / 2, 0, -math.pi / 2)), 'wallp', texrepeat=(3, 1.2), roughness=.95)]
        o += [plane_obj(r, 'w', matrix((2.4, 0, 1.5), (6, 3, 1), (math.pi / 2, 0, math.pi / 2)), 'wallp', texrepeat=(3, 1.2), roughness=.95)]
        o += [plane_obj(r, 'c', matrix((0, 0, 2.9), (10, 10, 1), (math.pi, 0, 0)), 'white', color=(.5, .48, .45))]
        o += place(s.desk, matrix((0, -.70, 0), (1, 1, .91), (0, 0, math.pi)))
        o += place(s.lamp, matrix((-.72, -.95, .72), (1, 1, 1), (0, 0, -.6)))
        o += place(s.plant, matrix((1.35, -1.5, 0)))
        o += place(s.shelf, matrix((-1.6, -2.0, 0)))
        o += place(s.clock, matrix((.9, -2.17, 2.0), (1, 1, 1), (0, 0, 0)))
        o += place(s.frame, matrix((0, -2.17, 1.75), (1.4, 1, 1.4)))
        o += [plane_obj(r, 'p', matrix((0, -2.16, 1.75), (.70, .02, 1.04), (math.pi / 2, 0, 0)) @ matrix(scale=(1, 1, 1)), 'paint', roughness=.9)]
        beige = dict(color=(.80, .75, .63), roughness=.45)
        o += [box_obj(r, matrix((0, -.84, .745), (.30, .28, .05)), **beige), box_obj(r, matrix((0, -.84, .79), (.12, .12, .05)), **beige)]
        o += [box_obj(r, matrix((0, -.88, 1.0), (.36, .30, .32)), **beige)]                       # cuerpo trasero
        o += [box_obj(r, matrix((0, -.71, 1.0 + .135), (.42, .06, .06)), **beige), box_obj(r, matrix((0, -.71, 1.0 - .145), (.42, .06, .08)), **beige)]
        o += [box_obj(r, matrix((.18, -.71, 1.0), (.06, .06, .23)), **beige), box_obj(r, matrix((-.18, -.71, 1.0), (.06, .06, .23)), **beige)]
        o += [box_obj(r, matrix((0, -.735, 1.0), (.31, .02, .24)), color=(.05, .05, .05), roughness=.3)]   # fondo del tubo
        o += [box_obj(r, matrix((.15, -.679, 1.0 - .16), (.025, .004, .012)), color=(.3, 1, .4), emission=2)]  # LED
        # teclado
        o += [box_obj(r, matrix((0, -.42, .728), (.46, .17, .022)), color=(.78, .74, .64), roughness=.6)]
        o += [plane_obj(r, 'k', matrix((0, -.42, .7395), (.44, .155, 1)), 'kbd', roughness=.7)]
        # silla (tapizada)
        o += [box_obj(r, matrix((0, .06, .47), (.5, .48, .07)), color=(.18, .12, .09), roughness=.5), box_obj(r, matrix((0, .32, .78), (.48, .07, .46)), color=(.18, .12, .09), roughness=.5)]
        o += [box_obj(r, matrix((0, .06, .22), (.06, .06, .44)), color=(.12, .12, .12), roughness=.3)]
        s.static = o

    def pc_yaw(s): return 0.0  # pantalla mirando a Benter (+y)

    def screen_M(s, out=0.0):
        # pantalla del modelo: frente hacia +y; centro aproximado
        return matrix((0, -.722 + out, 1.0), (.30, .23, 1), (math.pi / 2, 0, math.pi))

    def update_screen(s, ts):
        tq = round(ts * 30) / 30
        if tq == s.screen_t: return
        s.screen_t = tq; im = screen_image(tq); a = np.ascontiguousarray(np.array(im.convert('RGBA'))[::-1]); r = s.r
        r.glBindTexture(0x0DE1, r.textures['crt_screen']); r.glTexImage2D(0x0DE1, 0, 0x8058, a.shape[1], a.shape[0], 0, 0x1908, 0x1401, a.ctypes.data); r.glGenerateMipmap(0x0DE1)

    def objects(s, anim, ta, ts, blend=None):
        s.update_screen(ts); o = list(s.static)
        o.append(plane_obj(s.r, 's', s.screen_M(.002), 'crt_screen', emission=1.2, roughness=.15))
        o.append(plane_obj(s.r, 'b', matrix((-.05, -.6785, 1.0 - .148), (.12, .03, 1), (math.pi / 2, 0, math.pi)), 'brand', roughness=.3))  # etiqueta de marca en el monitor
        s.benter.draw(o, anim, ta, matrix((0, .02, 0)), blend)
        return o

    def light(s, glow=1.0):
        sc = np.array((0, -.62, 1.0))
        return dict(ambient=.05, p0=(-.62, -.80, 1.35), p1=(1.6, 1.4, 2.2), c0=(1.6, 1.15, .7), c1=(.35, .42, .55), sun=(0, 0, 1), sunpower=0, spot=0, exposure=1.25, vignette=.32,
                    background=(.02, .02, .02), shadow_eye=(-.62, -.80, 1.6), shadow_target=(0, -.2, .6), shadow_extent=2.2, fire=tuple(sc), firecolor=tuple(np.array((.25, 1.2, .45)) * glow))

# ---------------- cámara ----------------
def orbit(center, r0, r1, a0, a1, h0, h1, k, look_h=0):
    a = math.radians(lerp(a0, a1, k)); rr = lerp(r0, r1, k); c = np.asarray(center, float)
    eye = c + np.array((math.sin(a) * rr, -math.cos(a) * rr, lerp(h0, h1, k)))
    return eye, c + (0, 0, look_h)

def orbit_pt(a, r, z, c=(0, 0, 1.05)):
    a = math.radians(a); return np.array((c[0] + math.sin(a) * r, c[1] - math.cos(a) * r, z))

# Toma única: (tiempo, ojo, objetivo, fov, giro)
KEYS = [
    (0.0, orbit_pt(55, 1.75, 1.28), (0, -.02, 1.04), 40, -2.0),
    (1.3, orbit_pt(100, 1.40, 1.30), (0, .0, 1.02), 36, 1.5),
    (2.1, orbit_pt(150, 2.05, 1.78), (0, -.30, 1.00), 40, 2.5),   # se aleja y sube al pasar por la espalda
    (2.9, np.array((.50, .62, 1.50)), (0, -.70, .99), 40, 1.0),    # por encima del hombro, a distancia
    (3.7, np.array((.46, -.30, 1.18)), (0, -.72, 1.00), 36, -1.0),  # junto a su cabeza, hacia la pantalla
    (4.15, np.array((.62, .50, 1.52)), (0, -.40, 1.06), 42, 1.0),  # se aleja hacia atrás: Benter y la pantalla juntos
    (4.6, np.array((1.30, -.02, 1.22)), (0, -.05, 1.10), 40, -1.5), # gira a su lado
    (5.0, np.array((1.10, -.48, .98)), (0, .0, 1.10), 42, -3.0),   # contrapicado de frente: celebra
]
def _cr(p0, p1, p2, p3, u):
    return .5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
def camera_at(t):
    T = [k[0] for k in KEYS]; t = min(max(t, 0), T[-1] - 1e-6); i = max(j for j in range(len(T) - 1) if T[j] <= t)
    u = (t - T[i]) / (T[i + 1] - T[i]); u = u * u * (3 - 2 * u) * .35 + u * .65
    g = lambda idx, f: np.asarray(KEYS[min(max(idx, 0), len(KEYS) - 1)][f], float)
    eye = _cr(g(i - 1, 1), g(i, 1), g(i + 1, 1), g(i + 2, 1), u); us = u * u * (3 - 2 * u); tg = g(i, 2) + (g(i + 1, 2) - g(i, 2)) * us
    fov = lerp(KEYS[i][3], KEYS[i + 1][3], u); roll = lerp(KEYS[i][4], KEYS[i + 1][4], u)
    return eye, tg, fov, roll

def screen_time(t):
    if t < 1.0: return .3
    if t < 3.2: return .7 + (t - 1.0) * .73
    return 2.42 + (t - 3.2) * 1.3

def shot(s, t):
    eye, tg, fov, roll = camera_at(t)
    w = mj(4.0, 4.4, t)
    o = s.objects('teclear', 2.0 + t, screen_time(t), blend=('celebrar', .25 + max(0, t - 4.0), w))
    return o, (tuple(eye), tuple(tg), fov), s.light(1.0 + .3 * mj(3.2, 3.8, t)), roll
