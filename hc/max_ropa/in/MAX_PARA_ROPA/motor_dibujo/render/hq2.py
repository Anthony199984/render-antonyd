"""HQ2: extiende el renderer HQ del motor (PBR GGX, mapas normales, SSAO, revelado ACES) con:
  - sombras de varias luces: hasta 3 focos con mapa de sombra propio (perspectiva, encuadre fijo = sin parpadeo),
    suavizado tipo PCSS (búsqueda de oclusores + filtro cuyo radio crece con la distancia oclusor-receptor);
  - la sombra ortográfica amplia del 'sol' de HQ queda como cascada lejana (todo el escenario).
No cambia nada del motor: subclase con su propio shader de fragmentos."""
import sys, ctypes as C, numpy as np
sys.path.insert(0, '/home/claude/endo_proyecto'); import morgan  # rutas del motor
sys.path.insert(0, '/home/claude/motor/soldadura/proj')
import hq
from gl_engine import perspective, look_at, normalize

NSL = 3; SM = 2048
EXTRA = '''
uniform sampler2D uSL0,uSL1,uSL2; uniform mat4 uLVP[3]; uniform int uNSL; uniform float uLSize[3]; uniform vec2 uNF;
float linz(float z){float n=uNF.x,f=uNF.y;return 2.*n*f/(f+n-(z*2.-1.)*(f-n));}
float pcss(sampler2D sm,mat4 M,vec3 n,vec3 l,float lsize){
 vec4 q=M*vec4(wPos,1.);if(q.w<=0.)return 1.;vec3 p=q.xyz/q.w*.5+.5;if(p.x<0.||p.x>1.||p.y<0.||p.y>1.||p.z>1.)return 1.;
 vec2 tx=1./vec2(textureSize(sm,0));float bias=.00012+.0006*(1.-abs(dot(n,l)));
 float sum=0.,cnt=0.;for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++){float d=texture(sm,p.xy+vec2(x,y)*tx*7.).r;if(d<p.z-bias){sum+=linz(d);cnt+=1.;}}
 if(cnt<.5)return 1.;
 float zr=linz(p.z),zb=sum/cnt;float pen=clamp((zr-zb)/max(zb,.05)*lsize,1.,14.);
 float s=0.;for(int x=-2;x<=2;x++)for(int y=-2;y<=2;y++){vec2 o=(vec2(x,y)+.5*vec2(sin(float(x*7+y)),cos(float(y*5-x))))*tx*pen*.55;s+=(p.z-bias>texture(sm,p.xy+o).r)?0.:1.;}
 return s/25.;}
float spotSh(int i,vec3 n,vec3 l){if(i>=uNSL)return 1.;
 if(i==0)return pcss(uSL0,uLVP[0],n,l,uLSize[0]);if(i==1)return pcss(uSL1,uLVP[1],n,l,uLSize[1]);return pcss(uSL2,uLVP[2],n,l,uLSize[2]);}
'''

def build_fs():
    fs = hq.FS
    fs = fs.replace('float lum(vec3 x)', EXTRA + 'float lum(vec3 x)', 1)
    old = 'Lo+=(dif*(1.-F)*wr+spec*nl)*uLC[i]*att*spot;}'
    assert old in fs
    fs = fs.replace(old, 'float ssh=(i<3)?spotSh(i,n,l):1.;Lo+=(dif*(1.-F)*wr+spec*nl)*uLC[i]*att*spot*ssh;}', 1)
    return fs

U = C.c_uint
class HQ2(hq.HQ):
    def __init__(s, w, h):
        super().__init__(w, h)
        s.pbr = s.program_from(hq.VS, build_fs())
        s.sl_tex = []; s.sl_fbo = []
        for k in range(NSL):
            t = U(); s.glGenTextures(1, C.byref(t)); s.glBindTexture(0x0DE1, t.value)
            s.glTexImage2D(0x0DE1, 0, 0x81A6, SM, SM, 0, 0x1902, 0x1406, None)
            for p_, v_ in ((0x2801, 0x2601), (0x2800, 0x2601), (0x2802, 0x812F), (0x2803, 0x812F)): s.glTexParameteri(0x0DE1, p_, v_)
            f = U(); s.glGenFramebuffers(1, C.byref(f)); s.glBindFramebuffer(0x8D40, f.value)
            s.glFramebufferTexture2D(0x8D40, 0x8D00, 0x0DE1, t.value, 0); s.glDrawBuffer(0); s.glReadBuffer(0)
            s.sl_tex.append(t.value); s.sl_fbo.append(f.value)
        s.glBindFramebuffer(0x8D40, 0); s.slvp = [np.eye(4, dtype=np.float32)] * NSL; s.nf = (.5, 30.)
    def spot_vp(s, pos, target, fov=110):
        """Encuadre fijo de sombra de un foco (perspectiva) hacia una zona del suelo."""
        pos = np.asarray(pos, float); d = np.asarray(target, float) - pos; up = (1, 0, 0) if abs(normalize(d)[2]) > .95 else (0, 0, 1)
        return (perspective(fov, 1.0, *s.nf) @ look_at(pos, target, up)).astype(np.float32)
    def draw(s, objects, camera, L, shadow=True):
        sh = L.get('shadow_spots', [])[:NSL]
        s.glEnable(0x0B71); s.glDepthFunc(0x0203)
        for k, vp in enumerate(sh):                      # pasada de profundidad por foco
            s.slvp[k] = vp; s.glBindFramebuffer(0x8D40, s.sl_fbo[k]); s.glViewport(0, 0, SM, SM); s.glClear(0x0100)
            s.glUseProgram(s.depth_program); s.mat(s.depth_program, 'uVP', vp)
            for o in objects:
                if o.get('mode', 'pbr') != 'pbr' or o.get('transparent') or o.get('emission', 0) > .4 or o.get('noshadow'): continue
                s._obj(o, s.depth_program, True)
        s._nsl = len(sh); s._lsize = L.get('shadow_soft', [6.0] * NSL)
        return super().draw(objects, camera, L, shadow)
    def _globals(s, p, L, eye, sd):
        super()._globals(p, L, eye, sd)
        if p != s.pbr: return
        s.i1(p, 'uNSL', getattr(s, '_nsl', 0)); s.glUniform2f(s.loc(p, 'uNF'), *s.nf)
        a = np.ascontiguousarray(np.stack([m.T for m in s.slvp]).astype(np.float32))
        s.glUniformMatrix4fv(s.loc(p, 'uLVP[0]'), NSL, 0, a.ctypes.data_as(C.POINTER(C.c_float)))
        ls = (C.c_float * NSL)(*[float(x) for x in (list(getattr(s, '_lsize', [6.] * NSL)) + [6.] * NSL)[:NSL]])
        s.g.glUniform1fv.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_float)]; s.g.glUniform1fv(s.loc(p, 'uLSize[0]'), NSL, ls)
        for k in range(NSL):
            s._bind(6 + k, s.sl_tex[k]); s.i1(p, f'uSL{k}', 6 + k)
        s.glActiveTexture(0x84C0)
