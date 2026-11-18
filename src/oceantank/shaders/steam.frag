varying float vA;void main(){vec2 c=gl_PointCoord-.5;float d=dot(c,c);if(d>.25)discard;gl_FragColor=vec4(vec3(1.,.72,.45),vA*smoothstep(.25,0.,d));}
