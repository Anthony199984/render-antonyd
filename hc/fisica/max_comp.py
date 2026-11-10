"""Max (motor hq) con la animación original y con la corregida por física, lado a lado, con piso y mesa."""
import sys, os, json, numpy as np
MD='/home/claude/max_ropa/in/MAX_PARA_ROPA/motor_dibujo'
for p in (f'{MD}/casino_project', f'{MD}/render', f'{MD}/fbxtool', '/home/claude/max_ropa/motor'): sys.path.insert(0,p)
import hq, soma.sin_fugas  # noqa
from soma import Clip
from soma.max_v17 import MaxCaraV17
from soma.cara import PistaCara
from PIL import Image, ImageDraw, ImageFont
cfg=json.loads(sys.argv[1])
import os
if os.environ.get('ROPA')=='1':
    sys.path.insert(0,'/home/claude/ropa'); from ropa_max import max_arreglado; MaxCaraV17=max_arreglado(MaxCaraV17)
elif os.environ.get('PANT_PEGADO')=='1':
    sys.path.insert(0,'/home/claude/mixamo'); from pantalon_pegado import con_pantalon_pegado; MaxCaraV17=con_pantalon_pegado(MaxCaraV17)
m=MaxCaraV17(glb='/home/claude/max_ropa/trabajo/max_v5.glb')
if os.environ.get('SIN_PANT')=='1':
    _pm=m.pose_mats
    def _pm2(*a,**k):
        r=_pm(*a,**k)
        for i,p in enumerate(m.prims):
            if p['mname'].startswith(tuple(os.environ.get('OCULTAR','Pants,Poloche').split(','))) and len(r[i][0]): r[i]=(r[i][0]*0+[0,0,-50],r[i][1])
        return r
    m.pose_mats=_pm2
    print('ocultos:', [p['mname'] for p in m.prims])
for _k,_v in cfg.get('flags',{}).items(): setattr(m,_k,_v)
FL2=cfg.get('flags2')
m.pista=PistaCara([(0,60,'neutral' if 'neutral' in __import__('soma.cara',fromlist=['x']).PRESETS else list(__import__('soma.cara',fromlist=['x']).PRESETS)[0],0.0)],parpadeo=False,vida=False)
W,H=cfg.get('wh',[540,760]); r=hq.HQ(W,H)
tex=[]
for i,p in enumerate(m.prims):
    tt='white'
    if p['image'] is not None:
        tt=f't{i}'; im=p['image'].convert('RGBA')
        if max(im.size)>2048: im=im.resize((2048,2048))
        r.add_texture(tt,im); r.set_texture_filter(tt,'linear')
    tex.append(tt)
def mezcla(P,N,UV):
    a=np.empty((len(P),11),np.float32); a[:,0:3]=P; a[:,3:6]=N; a[:,6:8]=UV; a[:,8:11]=1; return a
def caja(nombre,c,h):
    c=np.array(c,float);h=np.array(h,float); V=[];N=[]
    for ax in range(3):
        for s in (-1,1):
            n=np.zeros(3);n[ax]=s; u=np.zeros(3);u[(ax+1)%3]=1; v=np.cross(n,u)
            ctr=c+n*h[ax]; hu=h[(ax+1)%3]; hv=h[[i for i in range(3) if i!=ax and i!=(ax+1)%3][0]]
            q=[ctr-u*hu-v*hv,ctr+u*hu-v*hv,ctr+u*hu+v*hv,ctr-u*hu+v*hv]
            for t in ((0,1,2),(0,2,3)): V+= [q[t[0]],q[t[1]],q[t[2]]]; N+=[n]*3
    V=np.array(V,np.float32);N=np.array(N,np.float32); r.add_mesh(nombre,mezcla(V,N,np.zeros((len(V),2),np.float32)))
# conversión Y-arriba (BVH) -> motor: se mide con la cadera
c0=Clip(cfg['clips'][0]); m.pose(c0,0.0)
hb=c0.P[0,c0.idx['Hips']]; he=m.joint_pos('Hips'); print('cadera bvh',hb,'motor',he)
def a_motor(p): p=np.asarray(p,float); return np.array([p[0],-p[2],p[1]])
caja('piso',[0,0,-0.05],[30,30,0.05])
for k,(cc,hh) in enumerate(cfg.get('cajas',[])):
    c_=a_motor(cc); h_=np.abs(np.array([hh[0],hh[2],hh[1]])); caja(f'caja{k}',c_,h_)
F=ImageFont.truetype(os.environ.get('FUENTE','/usr/share/fonts/opentype/inter/Inter-Black.otf'),34)
os.makedirs(cfg['out'],exist_ok=True)
for n_,t in enumerate(cfg['tiempos']):
    imgs=[]
    for ci,cl in enumerate(cfg['clips']):
        if FL2 is not None:
            for _k,_v in (cfg.get('flags',{}) if ci==0 else FL2).items(): setattr(m,_k,_v)
        c=Clip(cl)
        if ci==0 and n_==0 and cfg.get('cam_fija') and 'tiempo_cam' in cfg: m.pose(c,float(cfg['tiempo_cam']))  # cámara fija = la del inicio del tramo, aunque el trozo empiece después
        res=m.pose(c,float(t))
        if (ci==0 and not (cfg.get('cam_fija') and n_>0)) or cfg.get('mira_cada'):
            if cfg.get('mira'): tg=m.joint_pos(cfg['mira']).copy()
            else: tg=m.joint_pos('Chest')*0.5+m.joint_pos('Hips')*0.5; tg[2]=cfg.get('alto_tg',0.6)
            ang=np.radians(cfg.get('yaw',90)); dirv=np.array([np.cos(ang),np.sin(ang),0])
            eye=tg+dirv*cfg.get('dist',2.6)+np.array([0,0,cfg.get('alto',0.5)])
        objs=[dict(mesh='piso',model=np.eye(4,dtype=np.float32),texture='white',color=(.78,.8,.84,1),roughness=.8,skin=0)]
        for k in range(len(cfg.get('cajas',[]))): objs.append(dict(mesh=f'caja{k}',model=np.eye(4,dtype=np.float32),texture='white',color=tuple(cfg.get('color_caja',(.5,.32,.18,1))),roughness=.6,skin=0))
        for i,((P,Nn),q) in enumerate(zip(res,m.prims)):
            if len(P)==0 or 'Tongue' in q['mname']: continue
            UV=q['UV'][q['idx']]; UV=np.stack([UV[:,0],1-UV[:,1]],1)
            r.add_mesh(f'm{i}',mezcla(P.astype(np.float32),Nn.astype(np.float32),UV.astype(np.float32)))
            o=dict(mesh=f'm{i}',model=np.eye(4,dtype=np.float32),texture=tex[i],color=(1,1,1),roughness=.5,skin=.6,transparent=(q['alpha']!='OPAQUE'))
            if q['image'] is None: o.update(color=tuple(float(x) for x in q['color'][:4]),skin=0)
            if 'Cornea' in q['mname']: o.update(transparent=True,noshadow=True)
            objs.append(o)
        L=dict(sun=(-0.4,0.3,-0.85),sun_col=(1.2,1.15,1.05),shadow_extent=3.0,env_top=(.7,.75,.85),env_hor=(.55,.6,.7),env_bot=(.3,.3,.32),env_k=.7,ao=1.0,ao_rad=.03,near=.02,far=60,shadow_target=tuple(tg))
        L.update(bg_top=(0.75,0.82,0.92),bg_hor=(0.9,0.92,0.95),bg_bot=(0.9,0.92,0.95))
        hdr=r.draw(objs,(tuple(eye),tuple(tg),cfg.get('fov',40)),L)
        imgs.append(Image.fromarray(hq.develop(hdr,exposure=1.1,grain=0.0,vignette=0.0)))
    S=Image.new('RGB',(2*W+10,H+70),(20,20,20)); S.paste(imgs[0],(0,70)); S.paste(imgs[1],(W+10,70)); dr=ImageDraw.Draw(S)
    dr.text((15,14),cfg.get('t1','ANTES'),font=F,fill=(255,110,110)); dr.text((W+25,14),cfg.get('t2','CON FÍSICA'),font=F,fill=(120,190,255))
    S.save(f"{cfg['out']}/{n_:02d}.png"); print('ok',n_,t,flush=True)
