"""Piezas comunes de "Si te cortas un dedo": motor (HQ2), revelado con desenfoque de profundidad, guardado, proyección y luces."""
import sys, os, math, numpy as np
for _p in ('/home/claude/cuerda_proyecto', '/home/claude/endo_proyecto', '/home/claude/anillo_proyecto', '/home/claude/tiburon_proyecto', '/home/claude/dedo_proyecto'):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)                                      # el último (dedo) gana: su tiempos.py
from comun_anillo import E, G, hq, HQ2M, rbchar, AG, RZ, CAM, dibujar, temblor, Rz
import tiempos as TM
from PIL import Image
FPS = TM.FPS
import real_scene as _RS
_ACT = [None]
def activar(r):
    if _ACT[0] is not r:
        assert r.e.eglMakeCurrent(r.display, r.surface, r.surface, r.context); _ACT[0] = r
def _envolver(cls, nombres):
    for n in nombres:
        f = getattr(cls, n)
        def w(self, *a, __f=f, **k): activar(self); return __f(self, *a, **k)
        setattr(cls, n, w)
if not getattr(HQ2M.HQ2, '_multi', False):
    _envolver(HQ2M.HQ2, ['draw', 'mask', 'add_mesh', 'add_texture', 'set_texture_filter'])
    _init0 = HQ2M.HQ2.__init__
    def _init(self, *a, **k): _init0(self, *a, **k); _ACT[0] = self
    HQ2M.HQ2.__init__ = _init; HQ2M.HQ2._multi = True
    _set0 = _RS.set_mesh
    def _set_mesh(r, name, data): activar(r); return _set0(r, name, data)
    _RS.set_mesh = _set_mesh

OUT = '/home/claude/dedo_proyecto/render'
f32 = lambda M: np.asarray(M, np.float32)

def ss(t, a, b):
    if b <= a: return float(t >= b)
    u = min(max((t - a)/(b - a), 0.0), 1.0); return u*u*(3 - 2*u)

def revelar(r, h, f, foco=None, cm=None, k_dof=0.0, near=.01, far=60, exp=1.0, **kw):
    img = hq.develop(h, exposure=exp, bloom_k=kw.get('bloom', .03), grain=.006, vignette=kw.get('vig', .20), sat=kw.get('sat', 1.04),
                     contrast=kw.get('con', 1.06), seed=f)
    if k_dof > 0 and foco is not None:
        dep = r.lin_depth(near, far); fd = np.linalg.norm(np.asarray(cm[0]) - np.asarray(foco))
        img = hq.dof(img, np.maximum(dep, fd*.97), fd, k_dof)
    return img

def guardar(img, out, f):
    os.makedirs(out, exist_ok=True); Image.fromarray(img).save(f'{out}/f{f:04d}.png')

def proyectar(cm, p, W, H):
    e = np.asarray(cm[0], float); t = np.asarray(cm[1], float); f = t - e; f /= np.linalg.norm(f)
    rt = np.cross(f, (0, 0, 1.)); rt /= np.linalg.norm(rt); up = np.cross(rt, f)
    d = np.asarray(p, float) - e; z = d @ f; th = math.tan(math.radians(cm[2])/2)
    return ((d @ rt)/(z*th*W/H) + 1)/2*W, (1 - (d @ up)/(z*th))/2*H, z

def luces_quirofano(r, foco=(0, 0, .93)):
    """Lámpara quirúrgica fuerte desde arriba (sombra casi nula por los varios focos), relleno frío azulado, fondo verde quirúrgico."""
    L = dict(sun=(.15, -.25, .95), sun_col=(.55, .55, .52), shadow_target=tuple(foco), shadow_extent=.6, shadow_eye=tuple(np.array(foco) + np.array((.15, -.25, .95))*4),
             env_top=(.62, .70, .72), env_hor=(.40, .52, .55), env_bot=(.20, .26, .30), env_k=.55,
             bg_top=(.10, .22, .24), bg_hor=(.14, .30, .32), bg_bot=(.05, .10, .12), ao=1.0, ao_rad=.02, near=.002, far=20,
             lights=[dict(pos=(foco[0], foco[1] - .1, foco[2] + 1.2), col=(1.7, 1.7, 1.6)), dict(pos=(foco[0] + .5, foco[1] - .8, foco[2] + .6), col=(.45, .5, .55)),
                     dict(pos=(foco[0] - .6, foco[1] + .4, foco[2] + .5), col=(.25, .3, .35))])
    L['shadow_spots'] = [r.spot_vp(L['lights'][0]['pos'], foco, fov=60)]; L['shadow_soft'] = [9.0]
    return L
