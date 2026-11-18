uniform float uT; uniform vec2 uC; uniform float uI; uniform float uRot; uniform float uR; uniform vec2 uZ; uniform float uBurst; uniform float uDis;
const float D=3.2; const float HW=4.0;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
vec2 hash2(vec2 p){p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));return fract(sin(p)*43758.5453);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
float fbm(vec2 p){float a=.5,s=0.;for(int i=0;i<4;i++){s+=a*vnoise(p);p=p*2.03+17.1;a*=.5;}return s;}
float caus(vec2 p,float t){vec2 n=floor(p),f=fract(p);float d1=8.,d2=8.;
  for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){vec2 g=vec2(float(i),float(j));vec2 o=.5+.5*sin(t+6.2831*hash2(n+g));
    float d=length(g+o-f);if(d<d1){d2=d1;d1=d;}else if(d<d2){d2=d;}}
  return pow(1.-smoothstep(0.,.28,d2-d1),3.);}
float warm(float x){return smoothstep(uZ.x-2.6,uZ.x+1.6,x)*(1.-smoothstep(uZ.y-1.6,uZ.y+2.6,x));}
float cold(float x){return smoothstep(7.6,11.4,x);}
float amb(vec2 p){float h=0.;
  h+=.075*sin(p.x*.9+uT*1.1+sin(p.y*.6+uT*.4));
  h+=.05*sin(p.y*1.3-uT*1.3+p.x*.4);
  h+=.035*sin((p.x+p.y)*2.1+uT*1.9);
  h+=.03*sin((p.x*1.7-p.y*2.6)*1.3-uT*2.4);
  return h;}
float ring(vec2 d){float r=length(d);return sin(3.4*r-5.6*uT)*smoothstep(1.6*uR,3.*uR,r)*exp(-r/8.5);}
float H(vec2 p){
  vec2 d=p-uC;float r=length(d)+1e-4;
  float back=max(uC.x-p.x,0.);
  float wake=smoothstep(0.,2.,back)*exp(-back/8.)*exp(-pow(d.y/(1.8+.22*back),2.));
  float h=amb(p)*(1.+uI*(1.5*exp(-r*r/30.)+1.6*wake));
  float th=atan(d.y,d.x);float lr=log(r/uR+.15);
  h+=uI*(-1.75*exp(-pow(r/(.6*uR),2.))+.26*exp(-pow((r-1.1*uR)/(.5*uR),2.))+.2*exp(-pow(r/(3.2*uR),2.)));
  h+=uI*.2*sin(2.*(th+uRot)-5.*lr)*smoothstep(.9*uR,1.8*uR,r)*exp(-r/(3.2*uR));
  h+=uI*(.2+.4*uBurst)*(ring(d)+.55*ring(vec2(d.x,2.*HW-p.y-uC.y))+.55*ring(vec2(d.x,-2.*HW-p.y-uC.y)));
  return h;}uniform vec3 uCam; varying vec3 vP;
void main(){
  vec2 p=vP.xz;float e=.045;
  float h0=H(p),hx=H(p+vec2(e,0.)),hz=H(p+vec2(0.,e));
  vec3 n=normalize(vec3(-(hx-h0)/e,1.,-(hz-h0)/e));
  n.xz+=.10*(vec2(vnoise(p*6.+uT*1.3),vnoise(p*6.3-uT*1.1+7.))-.5);n=normalize(n);
  vec3 V=normalize(uCam-vP);
  vec3 Ld=normalize(vec3(-.15,.62,-.77));
  float fres=pow(1.-max(dot(n,V),0.),3.);
  vec3 deep=vec3(.008,.07,.32),mid=vec3(.05,.33,.90),sky=vec3(.50,.78,1.);
  float hn=clamp((h0+.35)/1.3,0.,1.);
  vec3 col=mix(deep,mid,.30+.6*hn);
  col*=.72+.5*max(dot(n,Ld),0.);
  col=mix(col,sky,fres*.75);
  float spec=pow(max(dot(n,normalize(Ld+V)),0.),150.);
  col+=vec3(.9,.97,1.)*spec*.8;
  float rv=ring(p-uC);col+=vec3(.42,.7,1.)*uI*.6*pow(max(rv,0.),1.4);
  col+=vec3(.55,.82,1.)*caus(p*.55+n.xz*.9,uT*.9)*.24;
  float bk=max(uC.x-p.x,0.);
  float wk=smoothstep(0.,2.,bk)*exp(-bk/20.)*exp(-pow((p.y-uC.y)/(1.7+.12*bk),2.))*clamp(uI,0.,1.);
  float wz=warm(p.x)*(1.-.75*wk);float hs=fbm(p*.8+vec2(uT*.18,-uT*.12));
  vec3 hotc=vec3(1.,.40,.09)*(.42+.9*dot(col,vec3(.33)));
  col=mix(col,hotc,wz*(.55+.15*hs));
  col+=vec3(1.,.45,.1)*wz*(.04+.22*hs*hs)*.9;
  col=mix(col,col*vec3(.8,1.05,1.2)+vec3(0.,.04,.08),wk*warm(p.x)*.9);
  col=mix(col,vec3(dot(col,vec3(.33)))*vec3(.82,.98,1.04)+vec3(0.,.02,.03),cold(p.x)*.55);
  vec2 d=p-uC;float r=length(d)+1e-4;float th=atan(d.y,d.x);float lr=log(r/uR+.15);
  float a=th+uRot;
  float arms=.5+.5*sin(2.*a-5.*lr);
  float aw=a-3.4*lr;vec2 q=r*vec2(cos(aw),sin(aw));
  float aw2=th+uRot*1.5-2.4*lr;vec2 q2=r*vec2(cos(aw2),sin(aw2));
  float n1=fbm(q*1.3+3.),n2=fbm(q2*3.+11.);
  float eyeEdge=smoothstep(.52*uR,.86*uR,r);
  float cdo=eyeEdge*(1.-smoothstep(1.25*uR,2.1*uR,r+.8*uR*(n1-.5)));
  float bands=pow(arms,2.2)*smoothstep(1.05*uR,1.8*uR,r)*exp(-pow(r/(3.6*uR),2.))*(.5+1.5*n1);
  float tear=smoothstep(-.3,.5,cos(th-.8));cdo*=1.-uDis*tear;bands*=1.-.65*uDis*tear;
  float f=max(cdo*(1.0+.3*n2),bands*1.3)*(.8+.5*n2);
  float foam=smoothstep(.38,.9,f*min(uI,1.3)*(1.1+.5*uBurst));
  float streak=.5+.5*sin(4.*a-10.*lr+5.*n1);
  vec3 foamCol=mix(vec3(.36,.62,.97),vec3(1.,1.,1.),clamp(.25+.85*streak*(.6+.5*n2),0.,1.));
  foamCol*=.82+.3*max(dot(n,Ld),0.);
  float back=max(uC.x-p.x,0.);
  float wake=smoothstep(0.,2.,back)*exp(-back/8.)*exp(-pow(d.y/(1.8+.22*back),2.));
  float crest=smoothstep(.16,.42,h0-uI*.22*exp(-pow(r/(3.4*uR),2.))+.18*fbm(p*2.5+uT*.3)-.1);
  foam=max(foam,crest*(.2+.6*uI*(wake+exp(-r*r/40.)))*smoothstep(1.4*uR,2.4*uR,r));
  col=mix(col,foamCol,clamp(foam,0.,1.));
  col*=mix(1.,.16,clamp(uI,0.,1.)*exp(-pow(r/(.5*uR),2.)));
  if(!gl_FrontFacing)col=mix(col,vec3(.02,.12,.42),.75);
  gl_FragColor=vec4(col,1.);}
