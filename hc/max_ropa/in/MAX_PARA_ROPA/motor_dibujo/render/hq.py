"""Motor HQ para Antonydfilms: PBR (GGX), mapas normales, cáusticas animadas, absorción submarina, oclusión ambiental (SSAO),
agua con refracción, materiales aditivos (chispas, arco, rayos), buffer HDR y posproceso (bloom, ACES, grano, aberración).
Hereda de gl_engine.Renderer (EGL sin pantalla) y acepta los mismos diccionarios de objeto, con claves nuevas:
  metal, normalmap, nstrength, mode ('pbr' | 'water' | 'rim' | 'add'), refract, foam, rimpow, caustics (0/1)."""
import sys, math, ctypes as C, numpy as np
sys.path.insert(0, '/home/claude/casino/Casino_Claude_Entrega/casino_project')
from gl_engine import Renderer, perspective, look_at, ortho, normalize
import cv2

VS = '''#version 330 core
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNormal; layout(location=2) in vec2 aUV; layout(location=3) in vec3 aColor;
uniform mat4 uModel,uVP,uLightVP; uniform mat3 uNormal;
out vec3 wPos,normal,color; out vec2 uv; out vec4 lightPos;
void main(){vec4 p=uModel*vec4(aPos,1);wPos=p.xyz;normal=normalize(uNormal*aNormal);uv=aUV;color=aColor;lightPos=uLightVP*p;gl_Position=uVP*p;}'''

FS = '''#version 330 core
in vec3 wPos,normal,color; in vec2 uv; in vec4 lightPos;
uniform sampler2D uTex,uShadow,uNorm,uCaus,uScene,uFoam;
uniform vec4 uColor; uniform vec3 uEye; uniform vec2 uRepeat,uResolution;
uniform float uRough,uMetal,uEmission,uNStr,uHasNorm,uSkin,uMode,uRefract,uFoamK,uRimPow,uCausK,uBump,uSpecK;
uniform int uNL; uniform vec3 uLP[8],uLC[8],uLD[8]; uniform vec2 uLS[8];
uniform vec3 uSunDir,uSunCol; uniform float uShadowOn;
uniform vec3 uEnvTop,uEnvHor,uEnvBot; uniform float uEnvK;
uniform vec3 uCausCol; uniform vec4 uCausP; // escala, t, profundidad(z de superficie), fuerza
uniform vec3 uExt,uInscat; uniform float uFogMax;
out vec4 frag;
const float PI=3.14159265;
float shadowF(vec3 n){vec3 p=lightPos.xyz/lightPos.w*.5+.5;if(p.z>1.||p.x<0.||p.x>1.||p.y<0.||p.y>1.)return 1.;
 vec2 tx=1./vec2(textureSize(uShadow,0));float s=0.;float bias=.0008+.002*(1.-abs(dot(n,uSunDir)));
 for(int x=-2;x<=2;x++)for(int y=-2;y<=2;y++){s+=(p.z-bias>texture(uShadow,p.xy+vec2(x,y)*tx*1.2).r)?0.:1.;}return mix(1.,s/25.,uShadowOn);}
vec3 env(vec3 d){float z=d.z;vec3 c=z>0.?mix(uEnvHor,uEnvTop,pow(z,.6)):mix(uEnvHor,uEnvBot,pow(-z,.5));return c*uEnvK;}
float D_GGX(float nh,float a){float a2=a*a;float d=nh*nh*(a2-1.)+1.;return a2/(PI*d*d+1e-6);}
float G_S(float nv,float nl,float a){float k=(a+1.)*(a+1.)/8.;return nv/(nv*(1.-k)+k)*nl/(nl*(1.-k)+k);}
vec3 F_S(float c,vec3 f0){return f0+(1.-f0)*pow(clamp(1.-c,0.,1.),5.);}
float lum(vec3 x){return dot(x,vec3(.2126,.7152,.0722));}
mat3 tbn(vec3 n,vec2 t){vec3 dp1=dFdx(wPos),dp2=dFdy(wPos);vec2 du1=dFdx(t),du2=dFdy(t);
 vec3 dp2p=cross(dp2,n),dp1p=cross(n,dp1);vec3 T=dp2p*du1.x+dp1p*du2.x;vec3 B=dp2p*du1.y+dp1p*du2.y;
 float inv=inversesqrt(max(max(dot(T,T),dot(B,B)),1e-12));return mat3(T*inv,B*inv,n);}
vec3 fogit(vec3 c){float d=length(uEye-wPos);vec3 tr=exp(-uExt*d);tr=max(tr,vec3(1.-uFogMax));return c*tr+uInscat*(1.-tr);}
void main(){
 vec2 t=uv*uRepeat; vec4 tx=texture(uTex,t); float alpha=tx.a*uColor.a;
 vec3 n=normalize(normal); if(!gl_FrontFacing)n=-n; vec3 v=normalize(uEye-wPos);
 if(uMode>2.5){ // aditivo: chispas, arco, rayos (sin iluminación)
  vec3 c=pow(max(tx.rgb*color*uColor.rgb,vec3(0.)),vec3(2.2))*uEmission*alpha; float d=length(uEye-wPos);vec3 tr=max(exp(-uExt*d*.6),vec3(1.-uFogMax));
  frag=vec4(c*tr,1.);return;}
 if(alpha<.02&&uMode<.5)discard;
 if(uHasNorm>.5){vec3 nt=texture(uNorm,t).xyz*2.-1.;nt.xy*=uNStr;vec3 nn=tbn(n,t)*nt;if(dot(nn,nn)>1e-10)n=normalize(nn);}
 else if(uBump>0.){vec2 ds=1./vec2(textureSize(uTex,0));float h=lum(tx.rgb),hx=lum(texture(uTex,t+vec2(ds.x,0)).rgb),hy=lum(texture(uTex,t+vec2(0,ds.y)).rgb);
  mat3 M=tbn(n,t);n=normalize(n-(hx-h)*uBump*M[0]*20.-(hy-h)*uBump*M[1]*20.);}
 vec3 base=pow(max(tx.rgb*color*uColor.rgb,vec3(0.)),vec3(2.2));
 float rough=clamp(uRough,.04,1.),a=rough*rough; vec3 f0=mix(vec3(.04),base,uMetal); vec3 dif=base*(1.-uMetal);
 if(uMode>.5){dif=vec3(0.);f0=vec3(.02);}
 float nv=clamp(dot(n,v),1e-3,1.); float sh=shadowF(n);
 vec3 Lo=vec3(0.);
 // sol (luz desde la superficie del mar)
 {vec3 l=normalize(uSunDir);float nl=max(dot(n,l),0.);float wr=mix(nl,max(0.,(dot(n,l)+.3)/1.3),uSkin);vec3 h=normalize(l+v);
  vec3 F=F_S(max(dot(h,v),0.),f0);vec3 spec=D_GGX(max(dot(n,h),0.),a)*G_S(nv,nl,a)*F/(4.*nv*max(nl,1e-3)+1e-4);
  Lo+=(dif/PI*(1.-F)*PI*wr+spec*nl)*uSunCol*sh;
  // cáusticas animadas: dos capas que se mueven distinto
  if(uCausK>0.){vec2 cp=wPos.xy*uCausP.x;float c1=texture(uCaus,cp+vec2(uCausP.y*.031,uCausP.y*.017)).r;float c2=texture(uCaus,cp*1.37+vec2(-uCausP.y*.023,uCausP.y*.029)).r;
   float cc=pow(min(c1,c2)*1.6,1.6);float depthk=clamp(1.-(uCausP.z-wPos.z)*.06,.2,1.);Lo+=dif*uCausCol*cc*uCausP.w*uCausK*max(n.z,0.)*sh*depthk;}}
 // luces puntuales / focos
 for(int i=0;i<8;i++){if(i>=uNL)break;vec3 L=uLP[i]-wPos;float d=length(L);vec3 l=L/d;float nl=max(dot(n,l),0.);
  float wr=mix(nl,max(0.,(dot(n,l)+.3)/1.3),uSkin);float att=1./(1.+d*d*1.);
  float spot=1.;if(uLS[i].x>-1.5){float cs=dot(-l,normalize(uLD[i]));spot=smoothstep(uLS[i].y,uLS[i].x,cs);}
  vec3 h=normalize(l+v);vec3 F=F_S(max(dot(h,v),0.),f0);vec3 spec=D_GGX(max(dot(n,h),0.),a)*G_S(nv,nl,a)*F/(4.*nv*max(nl,1e-3)+1e-4);
  Lo+=(dif*(1.-F)*wr+spec*nl)*uLC[i]*att*spot;}
 // ambiente (hemisferio del agua) + reflejos del entorno
 vec3 Fa=F_S(nv,f0); vec3 irr=env(n)*.9; vec3 rd=reflect(-v,n); vec3 refl=mix(env(rd),env(n),rough*rough);
 Lo+=dif*irr*(1.-Fa*.5)+refl*Fa*(1.-rough*.7);
 vec3 rgb=Lo+base*uEmission;
 if(uMode>1.5){ // borde luminoso (burbuja de gas): aditivo con fresnel
  float fr=pow(clamp(1.-nv,0.,1.),uRimPow);vec3 c=uColor.rgb*color*(fr*uEmission+.012*uEmission);frag=vec4(c*alpha,1.);return;}
 if(uMode>.5){ // agua: refracción de la escena + reflejo + espuma
  vec2 sp=gl_FragCoord.xy/uResolution; vec3 vn=n; vec2 off=(vn.xy-dot(vn,v)*v.xy)*uRefract;
  vec3 behind=texture(uScene,clamp(sp+off,.001,.999)).rgb*mix(vec3(1.),uColor.rgb,.35);
  float fr=.02+.98*pow(clamp(1.-nv,0.,1.),5.);vec3 wat=mix(behind,refl,min(1.,fr*.6+uSpecK*.8))+min(Lo*uSpecK,vec3(.35+uSpecK*6.))+vec3(.05,.08,.09)*uSpecK*4.*uEnvK;
  float fm=texture(uFoam,t*1.).r*uFoamK*tx.a; vec3 foamc=vec3(.9,.95,1.)*(irr*1.2+Lo*0.+uSunCol*.2*sh)+vec3(.0);
  // la espuma se ilumina con las luces
  vec3 foamLit=vec3(0.);for(int i=0;i<8;i++){if(i>=uNL)break;vec3 L=uLP[i]-wPos;float d=length(L);foamLit+=uLC[i]/(1.+d*d)*.6;}
  wat=mix(wat,(foamc+foamLit)*.85,clamp(fm,0.,1.));
  frag=vec4(fogit(wat),clamp(uColor.a*(tx.a)+fm*.5,0.,1.));return;}
 frag=vec4(fogit(rgb),alpha);}'''

FSQ_V = '''#version 330 core
out vec2 st;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);st=p;gl_Position=vec4(p*2.-1.,0,1);}'''
# fondo: gradiente de agua según la dirección de la vista
ENV_F = '''#version 330 core
in vec2 st;uniform mat4 uInvVP;uniform vec3 uEnvTop,uEnvHor,uEnvBot,uSunDir;uniform float uEnvK,uSunGlow;out vec4 frag;
void main(){vec4 a=uInvVP*vec4(st*2.-1.,1.,1.);vec4 b=uInvVP*vec4(st*2.-1.,-1.,1.);vec3 d=normalize(a.xyz/a.w-b.xyz/b.w);
 float z=d.z;vec3 c=z>0.?mix(uEnvHor,uEnvTop,pow(z,.6)):mix(uEnvHor,uEnvBot,pow(-z,.5));
 c+=uEnvTop*uSunGlow*pow(max(dot(d,normalize(uSunDir)),0.),8.);frag=vec4(c*uEnvK,1.);}'''
# SSAO sobre profundidad (multiplica el color opaco)
AO_F = '''#version 330 core
in vec2 st;uniform sampler2D uCol,uDepth;uniform mat4 uInvP,uP;uniform float uRad,uAOK;uniform vec2 uResolution;out vec4 frag;
vec3 vpos(vec2 s){float z=texture(uDepth,s).r*2.-1.;vec4 p=uInvP*vec4(s*2.-1.,z,1.);return p.xyz/p.w;}
float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
void main(){vec4 c=texture(uCol,st);float dz=texture(uDepth,st).r;if(dz>=.99999||uAOK<=0.){frag=c;return;}
 vec3 P=vpos(st);vec2 px=1./uResolution;vec3 cr=cross(vpos(st+vec2(px.x,0))-P,vpos(st+vec2(0,px.y))-P);if(dot(cr,cr)<1e-14){frag=c;return;}vec3 N=normalize(cr);
 float occ=0.;float ang=h(gl_FragCoord.xy)*6.2832;int NS=12;
 for(int i=0;i<NS;i++){float fi=(float(i)+.5)/float(NS);float rr=uRad*fi*fi+.02*uRad;float aa=ang+float(i)*2.4;
  vec3 Q=P+vec3(cos(aa),sin(aa),0.)*rr;vec4 q=uP*vec4(Q,1.);vec2 s=q.xy/q.w*.5+.5;if(s.x<0.||s.y<0.||s.x>1.||s.y>1.)continue;
  vec3 S=vpos(s);vec3 dv=S-P;float d=max(length(dv),1e-5);occ+=max(0.,dot(N,dv/d)-.08)*smoothstep(uRad*2.,0.,d);}
 float ao=clamp(1.-occ/float(NS)*1.9,0.,1.);frag=vec4(c.rgb*mix(1.,ao,uAOK),1.);}'''
MASK_F = '''#version 330 core
in vec3 wPos,normal,color;in vec2 uv;in vec4 lightPos;uniform sampler2D uTex;uniform vec4 uColor;uniform vec2 uRepeat;out vec4 frag;
void main(){if(texture(uTex,uv*uRepeat).a<.3)discard;frag=uColor;}'''

U = C.c_uint; I = C.c_int; F = C.c_float; P = C.c_void_p
class HQ(Renderer):
    def __init__(s, w, h):
        super().__init__(w, h)
        for name, (ret, args) in {'glBlitFramebuffer': (None, [I] * 8 + [U, U]), 'glUniform3fv': (None, [I, I, C.POINTER(F)]), 'glUniform2fv': (None, [I, I, C.POINTER(F)]),
                                  'glUniform1i': (None, [I, I]), 'glDepthFunc': (None, [U]), 'glDrawBuffers': (None, [I, C.POINTER(U)])}.items():
            f = getattr(s.g, name); f.restype = ret; f.argtypes = args; setattr(s, name, f)
        s.pbr = s.program_from(VS, FS); s.envp = s.program_from(FSQ_V, ENV_F); s.aop = s.program_from(FSQ_V, AO_F); s.maskp = s.program_from(VS, MASK_F)
        s.cA = s._tex(0x881A, 0x1908, 0x1406); s.cB = s._tex(0x881A, 0x1908, 0x1406); s.cC = s._tex(0x881A, 0x1908, 0x1406)
        s.dT = s._tex(0x81A6, 0x1902, 0x1405, nearest=True)
        s.fA = s._fbo(s.cA, s.dT); s.fAO = s._fbo(s.cB, None); s.fB = s._fbo(s.cB, s.dT); s.fC = s._fbo(s.cC, None)
        v = U(); s.glGenVertexArrays(1, C.byref(v)); s.evao = v.value
        s.hdr = np.empty((h, w, 3), np.float32); s.depth = np.empty((h, w), np.float32)
        s.cdata = {}
        from PIL import Image
        s.add_texture('flatnorm', Image.new('RGBA', (2, 2), (128, 128, 255, 255)))
        s.add_texture('black', Image.new('RGBA', (2, 2), (0, 0, 0, 255)))
    def _tex(s, ifmt, fmt, typ, nearest=False):
        t = U(); s.glGenTextures(1, C.byref(t)); s.glBindTexture(0x0DE1, t.value)
        s.glTexImage2D(0x0DE1, 0, ifmt, s.w, s.h, 0, fmt, typ, None)
        f = 0x2600 if nearest else 0x2601
        for p_, v_ in ((0x2801, f), (0x2800, f), (0x2802, 0x812F), (0x2803, 0x812F)): s.glTexParameteri(0x0DE1, p_, v_)
        return t.value
    def _fbo(s, col, dep):
        f = U(); s.glGenFramebuffers(1, C.byref(f)); s.glBindFramebuffer(0x8D40, f.value)
        s.glFramebufferTexture2D(0x8D40, 0x8CE0, 0x0DE1, col, 0)
        if dep: s.glFramebufferTexture2D(0x8D40, 0x8D00, 0x0DE1, dep, 0)
        b = (U * 1)(0x8CE0); s.glDrawBuffers(1, b)
        assert s.glCheckFramebufferStatus(0x8D40) == 0x8CD5, 'fbo'
        s.glBindFramebuffer(0x8D40, 0); return f.value
    def v3(s, p, n, v): s.glUniform3f(s.loc(p, n), *[float(x) for x in v])
    def i1(s, p, n, v): s.glUniform1i(s.loc(p, n), int(v))
    def _bind(s, unit, tex): s.glActiveTexture(0x84C0 + unit); s.glBindTexture(0x0DE1, tex)
    def _fsq(s): s.glBindVertexArray(s.evao); s.glDrawArrays(0x0004, 0, 3)

    def draw(s, objects, camera, L, shadow=True):
        """Devuelve la imagen HDR lineal (float32 HxWx3). También deja s.depth (z de ventana) y s.VP."""
        eye, target, fov = camera[:3]; up = camera[3] if len(camera) > 3 else (0, 0, 1)
        Pm = perspective(fov, s.w / s.h, L.get('near', .02), L.get('far', 60)); Vm = look_at(eye, target, up); VP = Pm @ Vm; s.VP = VP; s.Pm = Pm; s.Vm = Vm
        s.glEnable(0x0B71); s.glDepthFunc(0x0203)
        # ---- sombra (sol)
        sd = normalize(np.asarray(L.get('sun', (.2, .3, 1)), np.float32)); ctr = np.asarray(L.get('shadow_target', (0, 0, 0)), np.float32); ext = L.get('shadow_extent', 6.0)
        s.lightvp = ortho(-ext, ext, -ext, ext, .1, 40) @ look_at(ctr + sd * 18, ctr, (0, 1, 0) if abs(sd[2]) < .99 else (0, 1, 0))
        s.glBindFramebuffer(0x8D40, s.fbo.value); s.glViewport(0, 0, s.shadow_size, s.shadow_size); s.glClear(0x0100)
        s.glUseProgram(s.depth_program); s.mat(s.depth_program, 'uVP', s.lightvp)
        if shadow:
            for o in objects:
                if o.get('mode', 'pbr') != 'pbr' or o.get('transparent') or o.get('emission', 0) > .4 or o.get('noshadow'): continue
                s._obj(o, s.depth_program, True)
        # ---- fondo + opacos -> A
        s.glBindFramebuffer(0x8D40, s.fA); s.glViewport(0, 0, s.w, s.h); s.glClearColor(0, 0, 0, 1); s.glClear(0x4100)
        s.glDisable(0x0B71); s.glUseProgram(s.envp); s.mat(s.envp, 'uInvVP', np.linalg.inv(VP))
        for k, d in (('uEnvTop', 'env_top'), ('uEnvHor', 'env_hor'), ('uEnvBot', 'env_bot')): s.v3(s.envp, k, L.get('bg' + d[3:], L.get(d, (.1, .3, .5))))
        s.v3(s.envp, 'uSunDir', sd); s.scalar(s.envp, 'uEnvK', L.get('bg_k', 1.0)); s.scalar(s.envp, 'uSunGlow', L.get('sun_glow', .6)); s._fsq()
        s.glEnable(0x0B71)
        p = s.pbr; s.glUseProgram(p); s._globals(p, L, eye, sd)
        s.glBlendFunc(0x0302, 0x0303)
        opq = [o for o in objects if not o.get('transparent') and o.get('mode', 'pbr') == 'pbr']
        for o in opq: s._obj(o, p)
        # ---- AO: A -> B
        s.glBindFramebuffer(0x8D40, s.fAO); s.glDisable(0x0B71); s.glDisable(0x0BE2); s.glUseProgram(s.aop)
        s._bind(0, s.cA); s._bind(1, s.dT); s.i1(s.aop, 'uCol', 0); s.i1(s.aop, 'uDepth', 1)
        s.mat(s.aop, 'uInvP', np.linalg.inv(Pm)); s.mat(s.aop, 'uP', Pm); s.scalar(s.aop, 'uRad', L.get('ao_rad', .18)); s.scalar(s.aop, 'uAOK', L.get('ao', .85))
        s.glUniform2f(s.loc(s.aop, 'uResolution'), s.w, s.h); s._fsq()
        s.glEnable(0x0BE2); s.glEnable(0x0B71)
        # profundidad para DOF (antes de transparentes)
        s.glBindFramebuffer(0x8D40, s.fA); s.glReadPixels(0, 0, s.w, s.h, 0x1902, 0x1406, s.depth.ctypes.data)
        # ---- copia B -> C (escena detrás del agua)
        s.glBindFramebuffer(0x8CA8, s.fB); s.glBindFramebuffer(0x8CA9, s.fC); s.glBlitFramebuffer(0, 0, s.w, s.h, 0, 0, s.w, s.h, 0x4000, 0x2600)
        # ---- transparentes (agua, vidrio, burbujas) sobre B, de atrás hacia delante
        s.glBindFramebuffer(0x8D40, s.fB); s.glViewport(0, 0, s.w, s.h); s.glUseProgram(p); s._globals(p, L, eye, sd)
        s._bind(4, s.cC); s.i1(p, 'uScene', 4); s.glDepthMask(0)
        tr = [o for o in objects if (o.get('transparent') or o.get('mode') in ('water', 'rim')) and o.get('mode') != 'add']
        tr.sort(key=lambda o: -np.linalg.norm(np.asarray(o.get('sortpos', o['model'][:3, 3])) - np.asarray(eye)))
        for o in tr:
            if o.get('mode') == 'rim': s.glBlendFunc(1, 1)
            else: s.glBlendFunc(0x0302, 0x0303)
            s._obj(o, p)
        # ---- aditivos
        s.glBlendFunc(1, 1)
        for o in objects:
            if o.get('mode') == 'add': s._obj(o, p)
        s.glBlendFunc(0x0302, 0x0303); s.glDepthMask(1)
        s.glReadPixels(0, 0, s.w, s.h, 0x1907, 0x1406, s.hdr.ctypes.data)
        s.glBindFramebuffer(0x8D40, 0)
        return np.nan_to_num(s.hdr[::-1], nan=0.0, posinf=50.0)

    def _globals(s, p, L, eye, sd):
        s.mat(p, 'uVP', s.VP); s.mat(p, 'uLightVP', s.lightvp); s.v3(p, 'uEye', eye)
        s.v3(p, 'uSunDir', sd); s.v3(p, 'uSunCol', L.get('sun_col', (1, 1, 1))); s.scalar(p, 'uShadowOn', 1.0)
        for k, d in (('uEnvTop', 'env_top'), ('uEnvHor', 'env_hor'), ('uEnvBot', 'env_bot')): s.v3(p, k, L.get(d, (.1, .3, .5)))
        s.scalar(p, 'uEnvK', L.get('env_k', 1.0))
        s.v3(p, 'uCausCol', L.get('caus_col', (.7, .9, 1))); cp = L.get('caus', (.35, 0, 30, 0)); s.glUniform4f(s.loc(p, 'uCausP'), *[float(x) for x in cp])
        s.v3(p, 'uExt', L.get('ext', (0, 0, 0))); s.v3(p, 'uInscat', L.get('inscat', (0, 0, 0))); s.scalar(p, 'uFogMax', L.get('fogmax', .97))
        lights = L.get('lights', [])[:8]; s.i1(p, 'uNL', len(lights))
        def arr(name, vals, n=3):
            a = np.zeros((8, n), np.float32)
            for i, v in enumerate(vals): a[i] = v
            (s.glUniform3fv if n == 3 else s.glUniform2fv)(s.loc(p, name + '[0]'), 8, np.ascontiguousarray(a).ctypes.data_as(C.POINTER(F)))
        arr('uLP', [l['pos'] for l in lights]); arr('uLC', [l['col'] for l in lights]); arr('uLD', [l.get('dir', (0, 0, -1)) for l in lights])
        arr('uLS', [l.get('cone', (-2, -3)) for l in lights], 2)
        s.glUniform2f(s.loc(p, 'uResolution'), s.w, s.h)
        s._bind(1, s.shadow.value); s.i1(p, 'uShadow', 1); s.i1(p, 'uTex', 0); s.i1(p, 'uNorm', 2); s.i1(p, 'uCaus', 3); s.i1(p, 'uScene', 4); s.i1(p, 'uFoam', 5)
        s._bind(3, s.textures.get(L.get('caus_tex', 'black'), s.textures['black']))

    def _obj(s, o, p, depth=False):
        s.mat(p, 'uModel', o['model']); s._bind(0, s.textures[o.get('texture', 'white')])
        if not depth:
            rep = o.get('texrepeat', (o.get('texscale', 1),) * 2); s.glUniform2f(s.loc(p, 'uRepeat'), *rep)
            nm = np.ascontiguousarray(np.linalg.inv(np.asarray(o['model'], np.float64)[:3, :3]).astype(np.float32))
            s.glUniformMatrix3fv(s.loc(p, 'uNormal'), 1, 0, nm.ctypes.data_as(C.POINTER(F)))
            col = o.get('color', (1, 1, 1)); s.glUniform4f(s.loc(p, 'uColor'), *[float(x) for x in col[:3]], float(col[3]) if len(col) > 3 else 1.)
            mode = o.get('mode', 'pbr'); s.scalar(p, 'uMode', {'pbr': 0, 'water': 1, 'rim': 2, 'add': 3}[mode])
            s.scalar(p, 'uRough', o.get('roughness', .6)); s.scalar(p, 'uMetal', o.get('metal', 0)); s.scalar(p, 'uEmission', o.get('emission', 0))
            s.scalar(p, 'uSkin', o.get('skin', 0)); s.scalar(p, 'uBump', o.get('bump', 0)); s.scalar(p, 'uCausK', o.get('caustics', 1.0))
            s.scalar(p, 'uRefract', o.get('refract', .04)); s.scalar(p, 'uSpecK', o.get('speck', .04)); s.scalar(p, 'uFoamK', o.get('foam', 0)); s.scalar(p, 'uRimPow', o.get('rimpow', 3.0))
            nmap = o.get('normalmap'); s.scalar(p, 'uHasNorm', 1. if nmap else 0.); s.scalar(p, 'uNStr', o.get('nstrength', 1.0))
            s._bind(2, s.textures[nmap] if nmap else s.textures['flatnorm'])
            s._bind(5, s.textures[o.get('foamtex', 'white')])
        vao, cnt = s.meshes[o['mesh']]; s.glBindVertexArray(vao); s.glDrawArrays(0x0004, 0, cnt)

    def mask(s, objects, sel, camera):
        """Máscara 0-255 de los objetos 'sel' (ocultados por los demás opacos), para resaltados y enfoque."""
        ids = set(id(o) for o in sel); eye, target, fov = camera[:3]
        s.glBindFramebuffer(0x8D40, s.fA); s.glViewport(0, 0, s.w, s.h); s.glClearColor(0, 0, 0, 1); s.glClear(0x4100); s.glEnable(0x0B71); s.glDisable(0x0BE2)
        p = s.maskp; s.glUseProgram(p); s.mat(p, 'uVP', s.VP)
        for o in objects:
            if o.get('transparent') or o.get('mode', 'pbr') != 'pbr':
                if id(o) not in ids: continue
            s.glUniform4f(s.loc(p, 'uColor'), *((1., 1., 1., 1.) if id(o) in ids else (0., 0., 0., 1.)))
            rep = o.get('texrepeat', (o.get('texscale', 1),) * 2); s.glUniform2f(s.loc(p, 'uRepeat'), *rep)
            s.mat(p, 'uModel', o['model']); s._bind(0, s.textures[o.get('texture', 'white')])
            nm = np.eye(3, dtype=np.float32); s.glUniformMatrix3fv(s.loc(p, 'uNormal'), 1, 0, nm.ctypes.data_as(C.POINTER(F)))
            vao, cnt = s.meshes[o['mesh']]; s.glBindVertexArray(vao); s.glDrawArrays(0x0004, 0, cnt)
        s.glEnable(0x0BE2); out = np.empty((s.h, s.w, 3), np.float32); s.glReadPixels(0, 0, s.w, s.h, 0x1907, 0x1406, out.ctypes.data)
        s.glBindFramebuffer(0x8D40, 0); return (out[::-1, :, 0] * 255).astype(np.uint8)

    def lin_depth(s, near=.02, far=60):
        z = s.depth[::-1] * 2 - 1; return 2 * near * far / (far + near - z * (far - near))

# ------------------------------------------------------------------------------------------------ posproceso
def aces(x):
    a, b, c, d, e = 2.51, .03, 2.43, .59, .14; return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def bloom(h, k=.06, thr=1.0):
    """Bloom multiescala sobre HDR (lo que pasa de 'thr' brilla con halos de varios tamaños)."""
    br = np.maximum(h - thr * .5, 0) * np.clip((h.max(2, keepdims=True) - thr * .5) / thr, 0, 1)
    acc = np.zeros_like(h); cur = br; H, W = h.shape[:2]
    for i, wgt in enumerate((1.0, .8, .6, .5, .4, .3)):
        cur = cv2.pyrDown(cur); up = cv2.resize(cv2.GaussianBlur(cur, (0, 0), 1.5), (W, H), interpolation=cv2.INTER_LINEAR); acc += up * wgt
    return h + acc * k

def dof(img, depth, focus, strength, near_ok=True):
    """Desenfoque por profundidad (solo para macros de objetos). img uint8, depth en metros."""
    coc = np.clip(np.abs(1 / np.maximum(depth, 1e-3) - 1 / focus) * strength, 0, 1)
    out = img.astype(np.float32); prev = out
    levels = [out] + [cv2.GaussianBlur(out, (0, 0), s_) for s_ in (2, 4.5, 8, 13)]
    idx = coc * (len(levels) - 1); i0 = np.floor(idx).astype(int); fr = (idx - i0)[..., None]; res = np.zeros_like(out)
    for i in range(len(levels)):
        w = ((i0 == i) * (1 - fr[..., 0]) + (i0 + 1 == i) * fr[..., 0])[..., None]; res += levels[i] * w
    return np.clip(res, 0, 255).astype(np.uint8)

def develop(h, exposure=1.0, bloom_k=.05, grain=.012, ca=.0, vignette=.30, sat=1.05, lift=(0, 0, 0), seed=0, contrast=1.04):
    """HDR -> sRGB uint8 con bloom, ACES, saturación, viñeta, aberración cromática y grano de película."""
    h = h * exposure
    if bloom_k > 0: h = bloom(h, bloom_k)
    x = aces(h)
    x = np.power(np.clip(x, 0, 1), 1 / 2.2)
    lum = x @ np.array([.2126, .7152, .0722], np.float32); x = lum[..., None] + (x - lum[..., None]) * sat
    x = (x - .5) * contrast + .5 + np.array(lift, np.float32)
    H, W = x.shape[:2]
    if vignette > 0:
        yy, xx = np.mgrid[:H, :W]; r = ((xx / W - .5) ** 2 * 1.1 + (yy / H - .5) ** 2); x = x * (1 - vignette * r[..., None] * 1.6)
    if ca > 0:
        M = lambda sc: cv2.warpAffine(x[..., 0], np.float32([[sc, 0, W / 2 * (1 - sc)], [0, sc, H / 2 * (1 - sc)]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        r_ = M(1 + ca); b_ = cv2.warpAffine(x[..., 2], np.float32([[1 - ca, 0, W / 2 * ca], [0, 1 - ca, H / 2 * ca]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        x = np.stack([r_, x[..., 1], b_], 2)
    if grain > 0:
        rng = np.random.default_rng(seed); g = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32); g = cv2.resize(g, (W, H))
        x = x + g[..., None] * grain * (1 - np.abs(x - .5))
    return np.clip(x * 255 + .5, 0, 255).astype(np.uint8)
