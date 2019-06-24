
#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glext.h>

#include "shaders.h"

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define HEADER \
    "#version 330 core\n" \
    "#extension GL_ARB_separate_shader_objects : enable\n" \
    "#line " STRINGIZE(__LINE__) "\n"

shaders::Shader const shaders::forward_vertex {
    "forward_vertex", GL_VERTEX_SHADER, HEADER R"glsl(

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 colour_in;
layout(location = 0) in vec2 texCoord;

layout(location = 0) smooth out vec3 colour_out;
layout(location = 1) out vec2 texCoordv;


void main() {
    gl_Position = position;
    colour_out = colour_in;
    texCoordv = texCoord;
}

)glsl"
};

shaders::Shader const shaders::forward_fragment{
    "forward_fragment", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 colour_out;

vec3 get_checker_color(vec2 par) {
    if (par.x * par.y < 0) {
      return vec3(1.0);
    } else {
      return vec3(0.0);
    }
}

void main() {
    vec4 trace;
    trace.xy = (texCoordV + 1.0) * 5.0;
    trace.zw = fract(trace.xy) - 0.5;
    colour_out.w = 1.0;
    colour_out.xyz = get_checker_color(trace.zw);
}

)glsl"
};

shaders::Shader const shaders::oceanic{
    "oceanic", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 fragColor;

float waterlevel = 70.0;        // height of the water
float wavegain   = 1.0;       // change to adjust the general water wave level
float large_waveheight = 1.0; // change to adjust the "heavy" waves (set to 0.0 to have a very still ocean :)
float small_waveheight = 1.0; // change to adjust the small waves

vec3 fogcolor    = vec3( 0.5, 0.7, 1.1 );
vec3 skybottom   = vec3( 0.6, 0.8, 1.2 );
vec3 skytop      = vec3(0.05, 0.2, 0.5);
vec3 reflskycolor= vec3(0.025, 0.10, 0.20);
vec3 watercolor  = vec3(0.2, 0.25, 0.3);

vec3 light       = normalize( vec3(  0.1, 0.25,  0.9 ) );

// random/hash function
float hash( float n )
{
  return fract(cos(n)*41415.92653);
}

float rand(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 p){
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);

    float res = mix(
        mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
        mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
    return res;
}

// 3d noise function
float noise( in vec3 x )
{
  vec3 p  = floor(x);
  vec3 f  = smoothstep(0.0, 1.0, fract(x));
  float n = p.x + p.y*57.0 + 113.0*p.z;

  return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
    mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}


mat3 m = mat3( 0.00,  1.60,  1.20, -1.60,  0.72, -0.96, -1.20, -0.96,  1.28 );

// Fractional Brownian motion
float fbm( vec3 p )
{
  float f = 0.5000*noise( p ); p = m*p*1.1;
  f += 0.2500*noise( p ); p = m*p*1.2;
  f += 0.1666*noise( p ); p = m*p;
  f += 0.0834*noise( p );
  return f;
}

mat2 m2 = mat2(1.6,-1.2,1.2,1.6);

// Fractional Brownian motion
float fbm( vec2 p )
{
  float f = 0.5000*noise( p ); p = m2*p;
  f += 0.2500*noise( p ); p = m2*p;
  f += 0.1666*noise( p ); p = m2*p;
  f += 0.0834*noise( p );
  return f;
}

// this calculates the water as a height of a given position
float water( vec2 p )
{
  float height = waterlevel;

  //vec2 shift1 = 0.001*vec2( iTime*160.0*2.0, iTime*120.0*2.0 );
  //vec2 shift2 = 0.001*vec2( iTime*190.0*2.0, -iTime*130.0*2.0 );
  vec2 shift1 = vec2(0.0);
  vec2 shift2 = vec2(0.0);

  // coarse crossing 'ocean' waves...
  float wave = 0.0;
  wave += sin(p.x*0.021  + shift2.x)*4.5;
  wave += sin(p.x*0.0172+p.y*0.010 + shift2.x*1.121)*4.0;
  wave -= sin(p.x*0.00104+p.y*0.005 + shift2.x*0.121)*4.0;
  // ...added by some smaller faster waves...
  wave += sin(p.x*0.02221+p.y*0.01233+shift2.x*3.437)*5.0;
  wave += sin(p.x*0.03112+p.y*0.01122+shift2.x*4.269)*2.5 ;
  wave *= large_waveheight;
  wave -= fbm(p*0.004-shift2*.5)*small_waveheight*24.;
  // ...added by some distored random waves (which makes the water looks like water :)

  float amp = 6.*small_waveheight;
  shift1 *= .3;
  for (int i=0; i<7; i++)
  {
    wave -= abs(sin((noise(p*0.01+shift1)-.5)*3.14))*amp;
    amp *= .51;
    shift1 *= 1.841;
    p *= m2*0.9331;
  }

  height += wave;
  return height;
}

float trace_fog(in vec3 rStart, in vec3 rDirection ) {
  //vec2 shift = vec2( iTime*80.0, iTime*60.0 );
  vec2 shift = vec2(0.0, 0.0);
  float sum = 0.0;
  // use only 12 cloud-layers ;)
  // this improves performance but results in "god-rays shining through clouds" effect (sometimes)...
  float q2 = 0., q3 = 0.;
  for (int q=0; q<10; q++)
  {
    float c = (q2+350.0-rStart.y) / rDirection.y;// cloud distance
    vec3 cpos = rStart + c*rDirection + vec3(831.0, 321.0+q3-shift.x*0.2, 1330.0+shift.y*3.0); // cloud position
    float alpha = smoothstep(0.5, 1.0, fbm( cpos*0.0015 )); // cloud density
	sum += (1.0-sum)*alpha; // alpha saturation
    if (sum>0.98)
        break;
    q2 += 120.;
    q3 += 0.15;
  }

  return clamp( 1.0-sum, 0.0, 1.0 );
}

bool trace(in vec3 rStart, in vec3 rDirection, in float sundot, out float fog, out float dist)
{
    float h = 20.0;
    float t = 0.0;
    float st = 1.0;
    float alpha = 0.1;
    float asum = 0.0;
    vec3 p = rStart;

    for( int j=1000; j<1120; j++ )
    {
      if( t>500.0 )
          st = 2.0;
        else if( t>800.0 )
          st = 5.0;
        else if( t>1000.0 )
          st = 12.0;

        p = rStart + t*rDirection; // calc current ray position

        h = p.y - water(p.xz);

        if( h<0.1 ) // hit the water?
        {
          dist = t;
          fog = asum;
          return true;
        }

        if( p.y>450.0 ) // lost in space? quit...
          break;

        // speed up ray if possible...
        if(rDirection.y > 0.0) // look up (sky!) -> make large steps
          t += 30.0 * st;
        else
          t += max(1.0,1.0*h)*st;
      }

    dist = t;
    fog = asum;
    if (h<10.0)
     return true;
    return false;
}

void main() {
  vec2 xy = texCoordV;
  fragColor = vec4(0.0);
  fragColor.w = 1.0;

  int xsamples = 30;
  int ysamples = 30;

  for (int xx = 0; xx < xsamples; xx++) {
    for (int yy = 0; yy < ysamples; yy++) {

      xy = texCoordV;
      xy.x += (float(xx)) / 480.0 / xsamples;
      xy.y += (float(yy)) / 320.0 / ysamples;

      vec2 s = xy*vec2(1.75,1.0);


      // get camera position and view direction
      float time = 0.0;
      vec3 campos = vec3(500, waterlevel + 30.0, 0.0);
      vec3 camtar = vec3(200, campos.y / 2.0, 600.0);

      float roll = 0.0;
      vec3 cw = normalize(camtar-campos);
      vec3 cp = vec3(sin(roll), cos(roll),0.0);
      vec3 cu = normalize(cross(cw,cp));
      vec3 cv = normalize(cross(cu,cw));
      vec3 rd = normalize( s.x*cu + s.y*cv + 1.6*cw );

      float sundot = clamp(dot(rd,light),0.0,1.0);

      vec3 col = vec3(0.0);
      float fog=0.0, dist=0.0;

      if (!trace(campos,rd,sundot, fog, dist)) {
          float t = pow(1.0-0.7*rd.y, 15.0);
          col = 0.8*(skybottom*t + skytop*(1.0-t));
          // sun
          col += 0.47*vec3(1.6,1.4,1.0)*pow( sundot, 350.0 );
          // sun haze
          col += 0.4*vec3(0.8,0.9,1.0)*pow( sundot, 2.0 );

          //vec2 shift = vec2( iTime*80.0, iTime*60.0 );
          vec2 shift = vec2(0.0, 0.0);
          vec4 sum = vec4(0,0,0,0);
          for (int q=1000; q<1100; q++) // 100 layers
          {
            float c = (float(q-1000)*12.0+350.0-campos.y) / rd.y; // cloud height
            vec3 cpos = campos + c*rd + vec3(831.0, 321.0+float(q-1000)*.15-shift.x*0.2, 1330.0+shift.y*3.0); // cloud position
            float alpha = smoothstep(0.5, 1.0, fbm( cpos*0.0015 ))*.9; // fractal cloud density
            vec3 localcolor = mix(vec3( 1.1, 1.05, 1.0 ), 0.7*vec3( 0.4,0.4,0.3 ), alpha); // density color white->gray
            alpha = (1.0-sum.w)*alpha; // alpha/density saturation (the more a cloud layer's density, the more the higher layers will be hidden)
            sum += vec4(localcolor*alpha, alpha); // sum up weightened color

            if (sum.w>0.98)
              break;
          }
          float alpha = smoothstep(0.7, 1.0, sum.w);
          sum.rgb /= sum.w+0.0001;

          // This is an important stuff to darken dense-cloud parts when in front (or near)
          // of the sun (simulates cloud-self shadow)
          sum.rgb -= 0.6*vec3(0.8, 0.75, 0.7)*pow(sundot,13.0)*alpha;
          // This brightens up the low-density parts (edges) of the clouds (simulates light scattering in fog)
          sum.rgb += 0.2*vec3(1.3, 1.2, 1.0)* pow(sundot,5.0)*(1.0-alpha);

          col = mix( col, sum.rgb , sum.w*(1.0-t) );

          col += vec3(0.5, 0.4, 0.3)*fog;
      } else {
        vec3 wpos = campos + dist*rd; // calculate position where ray meets water

        // calculate water-mirror
        vec2 xdiff = vec2(0.1, 0.0)*wavegain*4.;
        vec2 ydiff = vec2(0.0, 0.1)*wavegain*4.;

        // get the reflected ray direction
        rd = reflect(rd, normalize(vec3(water(wpos.xz-xdiff) - water(wpos.xz+xdiff), 1.0, water(wpos.xz-ydiff) - water(wpos.xz+ydiff))));
        float refl = 1.0-clamp(dot(rd,vec3(0.0, 1.0, 0.0)),0.0,1.0);

        float sh = smoothstep(0.2, 1.0, trace_fog(wpos+20.0*rd,rd))*.7+.3;
        // water reflects more the lower the reflecting angle is...
        float wsky   = refl*sh;     // reflecting (sky-color) amount
        float wwater = (1.0-refl)*sh; // water-color amount

        float sundot = clamp(dot(rd,light),0.0,1.0);

        // watercolor

        col = wsky*reflskycolor; // reflecting sky-color
        col += wwater*watercolor;
        col += vec3(.003, .005, .005) * (wpos.y-waterlevel+30.);

        // Sun
        float wsunrefl = wsky*(0.5*pow( sundot, 10.0 )+0.25*pow( sundot, 3.5)+.75*pow( sundot, 300.0));
        col += vec3(1.5,1.3,1.0)*wsunrefl; // sun reflection

        float fo = 1.0-exp(-pow(0.0003*dist, 1.5));
        vec3 fco = fogcolor + 0.6*vec3(0.6,0.5,0.4)*pow( sundot, 4.0 );
        col = mix( col, fco, fo );

        // add god-rays
        col += vec3(0.5, 0.4, 0.3)*fog;
      }

      fragColor.xyz += col;
    }
  }

  fragColor.xyz /= float(xsamples * ysamples);

}
)glsl"
};

shaders::Shader const shaders::backward_vertex {
    "backward_vertex", GL_VERTEX_SHADER, HEADER R"glsl(

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 barycentric_in;
layout(location = 2) in ivec3 indices_in;

layout(location = 0) smooth out vec3 barycentric_out;
layout(location = 1) flat out ivec3 indices_out;

void main() {
    gl_Position = position;
    barycentric_out = vec3(barycentric_in.x, barycentric_in.y, 1.f - barycentric_in.x - barycentric_in.y);
    indices_out = indices_in;
}

)glsl"
};

shaders::Shader const shaders::backward_fragment{
    "backward_fragment", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 barycentric_in;
layout(location = 1) flat in ivec3 indices_in;

layout(location = 0) out vec4 barycentric_and_depth_out;
layout(location = 1) out vec3 indices_out;  // ** integer-valued textures don't seem to work

void main() {
    barycentric_and_depth_out = vec4(barycentric_in, 1.f / gl_FragCoord.w);  // the 'depth' we use here is exactly the clip-space w-coordinate
    indices_out = indices_in;
}

)glsl"
};
