
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

shaders::Shader const shaders::second_pass_fragment{
    "second_pass_fragment", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 1) out vec4 colour_out;

uniform sampler2D renderedTexture;

void main() {
    colour_out = texture(renderedTexture, (texCoordV + 1.0) / 2.0);
}

)glsl"
};

shaders::Shader const shaders::hill{
    "hill", GL_FRAGMENT_SHADER, HEADER R"glsl(
    
layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 fragColor;

//uniform sampler2D TerrainLookup;
//uniform sampler2D NormalLookup;
uniform sampler2D[2] textures;
uniform float width;
uniform float height;
    
#define MOD2 vec2(3.07965, 7.4235)
float PI  = 4.0*atan(1.0);
vec3 sunLight  = normalize( vec3(  0.35, 0.2,  0.3 ) );
vec3 cameraPos;
vec3 sunColour = vec3(1.0, .75, .6);
const mat2 rotate2D = mat2(1.932, 1.623, -1.623, 1.952);
float gTime = 0.0;
float iTime = 0.0;

//--------------------------------------------------------------------------
// Noise functions...
float Hash( float p )
{
    vec2 p2 = fract(vec2(p) / MOD2);
    p2 += dot(p2.yx, p2.xy+19.19);
    return fract(p2.x * p2.y);
}

//--------------------------------------------------------------------------
float Hash(vec2 p)
{
    p  = fract(p / MOD2);
    p += dot(p.xy, p.yx+19.19);
    return fract(p.x * p.y);
}


//--------------------------------------------------------------------------
float Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    float res = mix(mix( Hash(n+  0.0), Hash(n+  1.0),f.x),
                    mix( Hash(n+ 57.0), Hash(n+ 58.0),f.x),f.y);
    return res;
}

vec2 Voronoi( in vec2 x )
{
    vec2 p = floor( x );
    vec2 f = fract( x );
    float res=100.0,id;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 b = vec2( float(i), float(j) );
        vec2 r = vec2( b ) - f  + Hash( p + b );
        float d = dot(r,r);
        if( d < res )
        {
            res = d;
            id  = Hash(p+b);
        }
    }
    return vec2(max(.4-sqrt(res), 0.0),id);
}


//--------------------------------------------------------------------------

vec2 texture_x_range = vec2(-45.0, 94.0);
vec2 texture_y_range = vec2(4.0, 164.0);
vec2 texture_translation;
vec2 texture_scale;
float height_translation = 2.92;
float height_scale = 47.0;


vec2 Terrain( in vec2 p)
{
    texture_translation.xy = vec2(texture_x_range.x, texture_y_range.x);
    texture_scale.xy = vec2(texture_x_range.x + texture_x_range.y, texture_y_range.x + texture_y_range.y);
    vec2 scaled_p;
    //scaled_p.x = (p.x + 6.0) / 13.0;
    //scaled_p.y = (p.y - 2.0) / 10.0;
    scaled_p = (p - texture_translation) / texture_scale;
    //vec4 h = texture(NormalLookup, scaled_p.yx) * height_scale - height_translation;
    vec4 h = vec4(0.0);
    return vec2(h.x, 0.0);
}

//--------------------------------------------------------------------------
vec2 Map(in vec3 p)
{
    vec2 h = Terrain(p.xz);
    return vec2(p.y - h.x, 0.0);
}

//--------------------------------------------------------------------------
float FractalNoise(in vec2 xy)
{
    float w = .7;
    float f = 0.0;

    for (int i = 0; i < 3; i++)
    {
        f += Noise(xy) * w;
        w = w*0.6;
        xy = 2.0 * xy;
    }
    return f;
}

//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
vec3 GetSky(in vec3 rd)
{
    float sunAmount = max( dot( rd, sunLight), 0.0 );
    float v = pow(1.0-max(rd.y,0.0),6.);
    vec3  sky = mix(vec3(.1, .2, .3), vec3(.32, .32, .32), v);
    sky = sky + sunColour * sunAmount * sunAmount * .25;
    sky = sky + sunColour * min(pow(sunAmount, 800.0)*1.5, .3);
    return clamp(sky, 0.0, 1.0);
}

//--------------------------------------------------------------------------
// Merge grass into the sky background for correct fog colouring...
vec3 ApplyFog( in vec3  rgb, in float dis, in vec3 dir)
{
    float fogAmount = clamp(dis*dis* 0.0000012, 0.0, 1.0);
    return mix( rgb, GetSky(dir), fogAmount );
}

//--------------------------------------------------------------------------
vec3 DE(vec3 p)
{
    float scale = 3.0;
    float base = Terrain(p.xz).x - 1.3;
    p.xz *= 4.0;
    float height = Noise(p.xz*2.0)*.75 + Noise(p.xz)*.35 + Noise(p.xz*.5)*.2;;
    //p.y += height;
    float y = p.y - base-height;
    y = y*y;
    vec2 ret = Voronoi((p.xz*2.5+sin(y*4.0+p.zx*12.3)*.12+vec2(sin(iTime*2.3+1.5*p.z),sin(iTime*3.6+1.5*p.x))*y*.5));
    float f = ret.x * .6 + y * .58;
    return vec3( y - f*1.4, clamp(f * 1.5, 0.0, 1.0), ret.y);
}

//--------------------------------------------------------------------------
// eiffie's code for calculating the aperture size for a given distance...
float CircleOfConfusion(float t)
{
    return max(t * .04, (2.0 / height) * (1.0+t));
}

//--------------------------------------------------------------------------
float Linstep(float a, float b, float t)
{
    return clamp((t-a)/(b-a),0.,1.);
}

//--------------------------------------------------------------------------
vec3 GrassBlades(in vec3 rO, in vec3 rD, in vec3 mat, in float dist)
{
    float d = 0.0;
    // Only calculate cCoC once is enough here...
    float rCoC = CircleOfConfusion(dist*.3);
    float alpha = 0.0;

    vec4 col = vec4(mat*0.15, 0.0);

    for (int i = 0; i < 15; i++)
    {
        if (col.w > .99) break;
        vec3 p = rO + rD * d;

        vec3 ret = DE(p);
        ret.x += 0.5 * rCoC;

        if (ret.x < rCoC)
        {
            alpha = (1.0 - col.y) * Linstep(-rCoC, rCoC, -ret.x);//calculate the mix like cloud density
            // Mix material with white tips for grass...
            vec3 gra = mix(mat, vec3(.35, .35, min(pow(ret.z, 4.0)*35.0, .35)), pow(ret.y, 9.0)*.7) * ret.y;
            col += vec4(gra * alpha, alpha);
        }
        d += max(ret.x * .7, .1);
    }
    if(col.w < .2)
        col.xyz = vec3(0.1, .15, 0.05);
    return col.xyz;
}

//--------------------------------------------------------------------------
// Calculate sun light...
void DoLighting(inout vec3 mat, in vec3 pos, in vec3 normal, in vec3 eyeDir, in float dis)
{
    float h = dot(sunLight,normal);
    mat = mat * sunColour*(max(h, 0.0)+.2);
}

//--------------------------------------------------------------------------
vec3 TerrainColour(vec3 pos, vec3 dir,  vec3 normal, float dis, float type)
{
    vec3 mat;
    if (type == 0.0)
    {
        // Random colour...
        mat = mix(vec3(.0,.3,.0), vec3(.2,.3,.0), Noise(pos.xz*.025));
        // Random shadows...
        float t = FractalNoise(pos.xz * .1)+.5;
        // Do grass blade tracing...
        mat = GrassBlades(pos, dir, mat, dis) * t;
        DoLighting(mat, pos, normal,dir, dis);
    }
    mat = ApplyFog(mat, dis, dir);
    return mat;
}

//--------------------------------------------------------------------------
// Home in on the surface by dividing by two and split...
float BinarySubdivision(in vec3 rO, in vec3 rD, float t, float oldT)
{
    float halfwayT = 0.0;
    for (int n = 0; n < 5; n++)
    {
        halfwayT = (oldT + t ) * .5;
        if (abs(Map(rO + halfwayT*rD)).x < .05)
        {
            t = halfwayT;
        }else
        {
            oldT = halfwayT;
        }
    }
    return t;
}

//--------------------------------------------------------------------------
bool Scene(in vec3 rO, in vec3 rD, out float resT, out float type )
{
    float t;
    float t_inc = 0.0;
    
    t = -(rO.y + 1.0) / rD.y;
    if ((rD.y) > -0.015) t = 80.0;
    float h;
    float st = 1.0;
    vec3 p;
    float old_h;
    bool forward = true;
    for (int j = 0; j < 70; j++) {
        t += t_inc;
        p = rO + t*rD;
        h = Map(p).x;
        t_inc = max(1.0, abs(h)) * sign(h) * st;
        if (h * old_h < 0.0) {
            st /= 2.0;
            forward = false;
        }
        old_h = h;
    }
    type = 0.0;
    resT = t;
    if (abs(h) < 0.05) return true;
    else return false;
}

//--------------------------------------------------------------------------
vec3 CameraPath( float t )
{
    //t = time + t;
    vec2 p = vec2(200.0 * sin(3.54*t), 200.0 * cos(2.0*t) );
    return vec3(p.x+55.0,  12.0+sin(t*.3)*6.5, -94.0+p.y);
} 

//--------------------------------------------------------------------------
vec3 PostEffects(vec3 rgb, vec2 xy)
{
    // Gamma first...
    rgb = pow(rgb, vec3(0.45));

    // Then...
    #define CONTRAST 1.1
    #define SATURATION 1.3
    #define BRIGHTNESS 1.3
    rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
    // Vignette...
    rgb *= .4+0.5*pow(40.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2 );	
    return rgb;
}

//--------------------------------------------------------------------------
void main()
{
    fragColor.x = texture(textures[0], (texCoordV + 1.0) / 2.0).x;
    fragColor.y = texture(textures[1], (texCoordV + 1.0) / 2.0).x;
    fragColor.zw = vec2(0.0);
    return;
    vec2 xy;
    vec2 tex_pl = texCoordV;
    tex_pl.y *= -1.0;
    
    xy = (tex_pl + 1.0) / 2.0;
    
    
    //fragColor.xy = xy;
    //fragColor.zw = vec2(0.0);
    //return;
    
    if (abs(xy.y * height - height / 2.0) / (width / 2.0) >= 0.5625) {
        fragColor = vec4(0.0);
        return;
    }
     vec3 ray_dir;
  ray_dir.x = xy.x * width - width / 2.0;
  ray_dir.y = xy.y * height - height / 2.0;
  ray_dir.z = 0.85 * width;
  ray_dir = normalize(ray_dir);

  float ang1 = 0.0;
  float ang2 = 6.8;
  float ang3 = 0.0;
  cameraPos = vec3(0.0110647, -0.01849679, -1.52716118);
  //cameraPos.y += 10.0;
  //cameraPos.z += 3.0;
  cameraPos.z = -3.0;
  
  float sin1 = sin(ang1);
  float cos1 = cos(ang1);
  float sin2 = sin(ang2);
  float cos2 = cos(ang2);
  float sin3 = sin(ang3);
  float cos3 = cos(ang3);
    
  vec3 ray_dir_p;
    
    // current assumption to align with opensfm reconstruction:
    // should check later whether it is correct
    // d_world = R_z2-y * R_camera^T * d_camera
   
    float r00 = 0.999999573;
    float r01 = -0.0000933038802;
    float r02 = 0.000919791287;
    float r10 = 0.000918443273 ;
    float r11 = -0.0135434586;
    float r12 = -0.999907861;
    float r20 = 0.000105752439 ;
    float r21 = 0.999908279 ;
    float r22 = -0.0135433672;
    
  
    ray_dir_p.x = (r00 * ray_dir.x + r10 * ray_dir.y + r20 * ray_dir.z);
    ray_dir_p.z = (r01 * ray_dir.x + r11 * ray_dir.y + r21 * ray_dir.z);
    ray_dir_p.y = (r02 * ray_dir.x + r12 * ray_dir.y + r22 * ray_dir.z);
    
    //ray_dir_p.x = ray_dir.x;
    //ray_dir_p.z = -ray_dir.z;
    //ray_dir_p.y = -ray_dir.y;
    
    
  vec3 dir = ray_dir_p;


    vec3 col;
    float distance;
    float type;
    bool hit_ground = Scene(cameraPos, dir, distance, type);
    if (hit_ground) {
        fragColor.xyz = vec3(1.0);
        fragColor.y = distance;
        vec3 pos = cameraPos + distance * dir;
        fragColor.z = Terrain(pos.xz).x;
        
        fragColor.xyz = pos;
    }
    else {
        fragColor.xyz = vec3(0.0);
        
    }
    //return;
    
    
    if( !hit_ground )
    {
        // Missed scene, now just get the sky...
        col = GetSky(dir);
    }
    else
    {   
        // Get world coordinate of landscape...
        vec3 pos = cameraPos + distance * dir;
        
        fragColor.xyz = pos;
        fragColor.y = Map(pos).x;
        fragColor.z = Terrain(pos.xz).x;
        //return;
        
        // Get normal from sampling the high definition height map
        // Use the distance to sample larger gaps to help stop aliasing...
        vec2 p = vec2(0.1, 0.0);
        vec3 nor = vec3(0.0,Terrain(pos.xz).x, 0.0);
        vec3 v2 = nor-vec3(p.x,Terrain(pos.xz+p).x, 0.0);
        vec3 v3 = nor-vec3(0.0,Terrain(pos.xz-p.yx).x, -p.x);
        nor = cross(v2, v3);
        nor = normalize(nor);
        

        // Get the colour using all available data...
        col = TerrainColour(pos, dir, nor, distance, type);
    }

    col = PostEffects(col, xy);
    
    #ifdef STEREO
    col *= vec3( isCyan, 1.0-isCyan, 1.0-isCyan );
    #endif
    
    fragColor=vec4(col,1.0);
}
    
)glsl"
};

shaders::Shader const shaders::oceanic{
    "oceanic", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 fragColor;

uniform sampler2D backgroundTexture;
uniform float cam_x;
uniform float cam_y;
uniform float cam_z;
uniform float ang1;
uniform float ang2;
uniform float ang3;
uniform float time;
uniform float light_z;
uniform float width;
uniform float height;

float waterlevel = 70.0;        // height of the water
float wavegain   = 1.0;       // change to adjust the general water wave level
float large_waveheight = 1.0; // change to adjust the "heavy" waves (set to 0.0 to have a very still ocean :)
float small_waveheight = 1.0; // change to adjust the small waves

vec3 fogcolor    = vec3( 0.5, 0.7, 1.1 );
vec3 skybottom   = vec3( 0.6, 0.8, 1.2 );
vec3 skytop      = vec3(0.05, 0.2, 0.5);
vec3 reflskycolor= vec3(0.025, 0.10, 0.20);
vec3 watercolor  = vec3(0.2, 0.25, 0.3);

vec3 light       = normalize( vec3(  0.1, 0.25,  light_z ) );

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

  vec2 shift1 = 0.001*vec2( time*160.0*2.0, time*120.0*2.0 );
  vec2 shift2 = 0.001*vec2( time*190.0*2.0, -time*130.0*2.0 );

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
  vec2 shift = vec2( time*80.0, time*60.0 );
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
  float t = -rStart.y / rDirection.y;
  float st = 0.5;
  float alpha = 0.1;
  float asum = 0.0;
  vec3 p = rStart;
  float old_h = 0.0;
    
  for( int j=1000; j<1020; j++ )
  {
    // some speed-up if all is far away...
    if (t > 500.0) st = 1.0;
    if (t > 800.0) st = 2.0;
    if (t > 1500.0) st = 3.0;

    p = rStart + t*rDirection; // calc current ray position

#if RENDER_GODRAYS
    if (rDirection.y>0. && sundot > 0.001 && t>400.0 && t < 2500.0)
    {
      alpha = sundot * clamp((p.y-waterlevel)/waterlevel, 0.0, 1.0) * st * 0.024*smoothstep(0.80, 1.0, trace_fog(p,light));
      asum  += (1.0-asum)*alpha;
      if (asum > 0.9)
        break;
    }
#endif

    h = p.y - water(p.xz);

    t += max(1.0, abs(h)) * sign(h) * st;
      
    if (old_h * h < 0.0) st /= 2.0;
    
    old_h = h;
  }

  dist = t; 
  fog = asum;
  if (rDirection.y > 0.0) return false;
  return true;
}

void main() {
  vec2 xy = texCoordV;
  fragColor = vec4(0.0);
  fragColor.w = 1.0;

  xy = texCoordV;
  vec4 sample_noise = texture(backgroundTexture, (texCoordV + 1.0) / 2.0);
  xy.x += sample_noise.x / width;
  xy.y += sample_noise.y / height;

  // get camera position and view direction
  vec3 campos;
  campos.x = cam_x;
  campos.y = cam_y;
  campos.z = cam_z;
  
  vec3 ray_dir;
  ray_dir.x = (xy.x + 1.0) * width / 2.0 - width / 2.0;
  ray_dir.y = (xy.y + 1.0) * height / 2.0 - height / 2.0;
  ray_dir.z = 1.73 * width / 2.0;
  ray_dir = normalize(ray_dir);
  
  float sin1 = sin(ang1);
  float cos1 = cos(ang1);
  float sin2 = sin(ang2);
  float cos2 = cos(ang2);
  float sin3 = sin(ang3);
  float cos3 = cos(ang3);
    
  vec3 ray_dir_p;
  ray_dir_p.x = cos2 * cos3 * ray_dir.x + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir.y + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir.z;
  ray_dir_p.y = cos2 * sin3 * ray_dir.x + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir.y + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir.z;
  ray_dir_p.z = -sin2 * ray_dir.x + sin1 * cos2 * ray_dir.y + cos1 * cos2 * ray_dir.z;
  
  vec3 rd = ray_dir_p;

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

      
      vec2 shift = vec2( time*80.0, time*60.0 );
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

  fragColor.xyz = col;
  
}
)glsl"
};


shaders::Shader const shaders::oceanic_still_cloud{
    "oceanic", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 fragColor;

uniform sampler2D backgroundTexture;
uniform float cam_x;
uniform float cam_y;
uniform float cam_z;
uniform float ang1;
uniform float ang2;
uniform float ang3;
uniform float time;
uniform float light_z;
uniform float width;
uniform float height;
uniform float cloud_t;

float waterlevel = 70.0;        // height of the water
float wavegain   = 1.0;       // change to adjust the general water wave level
float large_waveheight = 1.0; // change to adjust the "heavy" waves (set to 0.0 to have a very still ocean :)
float small_waveheight = 1.0; // change to adjust the small waves

vec3 fogcolor    = vec3( 0.5, 0.7, 1.1 );
vec3 skybottom   = vec3( 0.6, 0.8, 1.2 );
vec3 skytop      = vec3(0.05, 0.2, 0.5);
vec3 reflskycolor= vec3(0.025, 0.10, 0.20);
vec3 watercolor  = vec3(0.2, 0.25, 0.3);

vec3 light       = normalize( vec3(  0.1, 0.25,  light_z ) );

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

  vec2 shift1 = 0.001*vec2( time*160.0*2.0, time*120.0*2.0 );
  vec2 shift2 = 0.001*vec2( time*190.0*2.0, -time*130.0*2.0 );

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
  vec2 shift = vec2( cloud_t*80.0, cloud_t*60.0 );
  float sum = 0.0;
  // use only 12 cloud-layers ;)
  // this improves performance but results in "god-rays shining through clouds" effect (sometimes)...
  float q2 = 0., q3 = 0.;
  for (int q=0; q<10; q++)
  {
    float c = (q2+350.0) / rDirection.y;// cloud distance
    vec3 cpos = c*rDirection + vec3(831.0, 321.0+q3-shift.x*0.2, 1330.0+shift.y*3.0); // cloud position
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
  float t = -rStart.y / rDirection.y;
  float st = 0.5;
  float alpha = 0.1;
  float asum = 0.0;
  vec3 p = rStart;
  float old_h = 0.0;
    
  for( int j=1000; j<1020; j++ )
  {
    // some speed-up if all is far away...
    if (t > 500.0) st = 1.0;
    if (t > 800.0) st = 2.0;
    if (t > 1500.0) st = 3.0;

    p = rStart + t*rDirection; // calc current ray position

#if RENDER_GODRAYS
    if (rDirection.y>0. && sundot > 0.001 && t>400.0 && t < 2500.0)
    {
      alpha = sundot * clamp((p.y-waterlevel)/waterlevel, 0.0, 1.0) * st * 0.024*smoothstep(0.80, 1.0, trace_fog(p,light));
      asum  += (1.0-asum)*alpha;
      if (asum > 0.9)
        break;
    }
#endif

    h = p.y - water(p.xz);

    t += max(1.0, abs(h)) * sign(h) * st;
      
    if (old_h * h < 0.0) st /= 2.0;
    
    old_h = h;
  }

  dist = t; 
  fog = asum;
  if (rDirection.y > 0.0) return false;
  return true;
}

void main() {
  vec2 xy = texCoordV;
  fragColor = vec4(0.0);
  fragColor.w = 1.0;


  xy = texCoordV;
  vec4 sample_noise = texture(backgroundTexture, (texCoordV + 1.0) / 2.0);
  xy.x += sample_noise.x / width;
  xy.y += sample_noise.y / height;

  // get camera position and view direction
  vec3 campos;
  campos.x = cam_x;
  campos.y = cam_y;
  campos.z = cam_z;
  
  vec3 ray_dir;
  ray_dir.x = (xy.x + 1.0) * width / 2.0 - width / 2.0;
  ray_dir.y = (xy.y + 1.0) * height / 2.0 - height / 2.0;
  ray_dir.z = 1.73 * width / 2.0;
  ray_dir = normalize(ray_dir);
  
  float sin1 = sin(ang1);
  float cos1 = cos(ang1);
  float sin2 = sin(ang2);
  float cos2 = cos(ang2);
  float sin3 = sin(ang3);
  float cos3 = cos(ang3);
    
  vec3 ray_dir_p;
  ray_dir_p.x = cos2 * cos3 * ray_dir.x + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir.y + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir.z;
  ray_dir_p.y = cos2 * sin3 * ray_dir.x + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir.y + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir.z;
  ray_dir_p.z = -sin2 * ray_dir.x + sin1 * cos2 * ray_dir.y + cos1 * cos2 * ray_dir.z;
  
  vec3 rd = ray_dir_p;

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

      
      vec2 shift = vec2( cloud_t*80.0, cloud_t*60.0 );
      vec4 sum = vec4(0,0,0,0);
      for (int q=1000; q<1100; q++) // 100 layers
      {
        float c = (float(q-1000)*12.0+350.0) / rd.y; // cloud height
        vec3 cpos = c*rd + vec3(831.0, 321.0+float(q-1000)*.15-shift.x*0.2, 1330.0+shift.y*3.0); // cloud position
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

  fragColor.xyz = col;
  
}
)glsl"
};


shaders::Shader const shaders::oceanic_opt_flow{
    "oceanic_opt_flow", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 fragColor;

uniform sampler2D backgroundTexture;
uniform float cam_x;
uniform float cam_y;
uniform float cam_z;
uniform float ang1;
uniform float ang2;
uniform float ang3;
uniform float time;

uniform float dx;
uniform float dy;
uniform float dz;
uniform float dt;
uniform float dang1;
uniform float dang2;
uniform float dang3;

uniform float light_z;
uniform float width;
uniform float height;

float waterlevel = 70.0;        // height of the water
float wavegain   = 1.0;       // change to adjust the general water wave level
float large_waveheight = 0.0; // change to adjust the "heavy" waves (set to 0.0 to have a very still ocean :)
float small_waveheight = 0.0; // change to adjust the small waves

vec3 fogcolor    = vec3( 0.5, 0.7, 1.1 );
vec3 skybottom   = vec3( 0.6, 0.8, 1.2 );
vec3 skytop      = vec3(0.05, 0.2, 0.5);
vec3 reflskycolor= vec3(0.025, 0.10, 0.20);
vec3 watercolor  = vec3(0.2, 0.25, 0.3);

vec3 light       = normalize( vec3(  0.1, 0.25,  light_z ) );

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
float water( vec2 p, float ctime)
{
  return waterlevel - 12.0;
}

bool trace(in vec3 rStart, in vec3 rDirection, in float sundot, out float fog, out float dist)
{
  float h = 20.0;
  float t = -rStart.y / rDirection.y;
  float st = 0.5;
  float alpha = 0.1;
  float asum = 0.0;
  vec3 p = rStart;
  float old_h = 0.0;
    
  for( int j=1000; j<1020; j++ )
  {
    // some speed-up if all is far away...
    if (t > 500.0) st = 1.0;
    if (t > 800.0) st = 2.0;
    if (t > 1500.0) st = 3.0;

    p = rStart + t*rDirection; // calc current ray position

    h = p.y - water(p.xz, time);

    t += max(1.0, abs(h)) * sign(h) * st;
      
    if (old_h * h < 0.0) st /= 2.0;
    
    old_h = h;
  }

  dist = t; 
  fog = asum;
  if (rDirection.y > 0.0) return false;
  return true;
}

void main() {
  vec2 xy = texCoordV;
  fragColor = vec4(0.0);
  fragColor.w = 1.0;


  xy = texCoordV;
  //vec4 sample_noise = texture(backgroundTexture, (texCoordV + 1.0) / 2.0);
  //xy.x += sample_noise.x / width;
  //xy.y += sample_noise.y / height;

  // get camera position and view direction
  vec3 campos;
  campos.x = cam_x;
  campos.y = cam_y;
  campos.z = cam_z;
  
  vec3 ray_dir;
  ray_dir.x = (xy.x + 1.0) * width / 2.0 - width / 2.0;
  ray_dir.y = (xy.y + 1.0) * height / 2.0 - height / 2.0;
  ray_dir.z = 1.73 * width / 2.0;
  ray_dir = normalize(ray_dir);
  
  float sin1 = sin(ang1);
  float cos1 = cos(ang1);
  float sin2 = sin(ang2);
  float cos2 = cos(ang2);
  float sin3 = sin(ang3);
  float cos3 = cos(ang3);
    
  vec3 ray_dir_p;
  ray_dir_p.x = cos2 * cos3 * ray_dir.x + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir.y + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir.z;
  ray_dir_p.y = cos2 * sin3 * ray_dir.x + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir.y + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir.z;
  ray_dir_p.z = -sin2 * ray_dir.x + sin1 * cos2 * ray_dir.y + cos1 * cos2 * ray_dir.z;
  
  vec3 rd = ray_dir_p;

  float sundot = clamp(dot(rd,light),0.0,1.0);

  vec3 col = vec3(0.0);
  float fog=0.0, dist=0.0;
  
  sin1 = sin(ang1 - dang1 * dt);
  cos1 = cos(ang1 - dang1 * dt);
  sin2 = sin(ang2 - dang2 * dt);
  cos2 = cos(ang2 - dang2 * dt);
  sin3 = sin(ang3 - dang3 * dt);
  cos3 = cos(ang3 - dang3 * dt); 
  vec3 old_rd;

  if (!trace(campos,rd,sundot, fog, dist)) {
      old_rd = rd;
  } else {
    vec3 wpos = campos + dist*rd; // calculate position where ray meets water
      
    float intersect_y_old = water(wpos.xz, time - dt);
    intersect_y_old = wpos.y;
    vec3 old_campos = campos - vec3(dx, dy, dz) * dt;
    old_rd = vec3(wpos.x, intersect_y_old, wpos.z) - old_campos;
  }
  
  vec3 old_r_dir;
  old_r_dir.x = cos2 * cos3 * old_rd.x + cos2 * sin3 * old_rd.y - sin2 * old_rd.z;
  old_r_dir.y = (-cos1 * sin3 + sin1 * sin2 * cos3) * old_rd.x + (cos1 * cos3 + sin1 * sin2 * sin3) * old_rd.y + sin1 * cos2 * old_rd.z;
  old_r_dir.z = (sin1 * sin3 + cos1 * sin2 * cos3) * old_rd.x + (-sin1 * cos3 + cos1 * sin2 * sin3) * old_rd.y + cos1 * cos2 * old_rd.z;

  old_r_dir /= old_r_dir.z;
  old_r_dir *= 1.73 * width / 2.0;
      
  vec2 new_coord;
  new_coord.x = old_r_dir.x + width / 2.0;
  new_coord.y = old_r_dir.y + height / 2.0;

  fragColor.xy = new_coord;
  return;
  fragColor = texture(backgroundTexture, vec2(new_coord.x / width, new_coord.y / height));
  
}
)glsl"
};




shaders::Shader const shaders::oceanic_no_cloud{
    "oceanic", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 fragColor;

uniform sampler2D backgroundTexture;
uniform float cam_x;
uniform float cam_y;
uniform float cam_z;
uniform float ang1;
uniform float ang2;
uniform float ang3;
uniform float time;
uniform float light_z;
uniform float width;
uniform float height;

float waterlevel = 70.0;        // height of the water
float wavegain   = 1.0;       // change to adjust the general water wave level
float large_waveheight = 1.0; // change to adjust the "heavy" waves (set to 0.0 to have a very still ocean :)
float small_waveheight = 1.0; // change to adjust the small waves

vec3 fogcolor    = vec3( 0.5, 0.7, 1.1 );
vec3 skybottom   = vec3( 0.6, 0.8, 1.2 );
vec3 skytop      = vec3(0.05, 0.2, 0.5);
vec3 reflskycolor= vec3(0.025, 0.10, 0.20);
vec3 watercolor  = vec3(0.2, 0.25, 0.3);

vec3 light       = normalize( vec3(  0.1, 0.25,  light_z ) );

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

  vec2 shift1 = 0.001*vec2( time*160.0*2.0, time*120.0*2.0 );
  vec2 shift2 = 0.001*vec2( time*190.0*2.0, -time*130.0*2.0 );

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
  return 1.0;
}

bool trace(in vec3 rStart, in vec3 rDirection, in float sundot, out float fog, out float dist)
{
  float h = 20.0;
  float t = -rStart.y / rDirection.y;
  float st = 0.5;
  float alpha = 0.1;
  float asum = 0.0;
  vec3 p = rStart;
  float old_h = 0.0;
    
  for( int j=1000; j<1020; j++ )
  {
    // some speed-up if all is far away...
    if (t > 500.0) st = 1.0;
    if (t > 800.0) st = 2.0;
    if (t > 1500.0) st = 3.0;

    p = rStart + t*rDirection; // calc current ray position

#if RENDER_GODRAYS
    if (rDirection.y>0. && sundot > 0.001 && t>400.0 && t < 2500.0)
    {
      alpha = sundot * clamp((p.y-waterlevel)/waterlevel, 0.0, 1.0) * st * 0.024*smoothstep(0.80, 1.0, trace_fog(p,light));
      asum  += (1.0-asum)*alpha;
      if (asum > 0.9)
        break;
    }
#endif

    h = p.y - water(p.xz);

    t += max(1.0, abs(h)) * sign(h) * st;
      
    if (old_h * h < 0.0) st /= 2.0;
    
    old_h = h;
  }

  dist = t; 
  fog = asum;
  if (rDirection.y > 0.0) return false;
  return true;
}

void main() {
  vec2 xy = texCoordV;
  fragColor = vec4(0.0);
  fragColor.w = 1.0;


  xy = texCoordV;
  vec4 sample_noise = texture(backgroundTexture, (texCoordV + 1.0) / 2.0);
  xy.x += sample_noise.x / width;
  xy.y += sample_noise.y / height;

  // get camera position and view direction
  vec3 campos;
  campos.x = cam_x;
  campos.y = cam_y;
  campos.z = cam_z;
  
  vec3 ray_dir;
  ray_dir.x = (xy.x + 1.0) * width / 2.0 - width / 2.0;
  ray_dir.y = (xy.y + 1.0) * height / 2.0 - height / 2.0;
  ray_dir.z = 1.73 * width / 2.0;
  ray_dir = normalize(ray_dir);
  
  float sin1 = sin(ang1);
  float cos1 = cos(ang1);
  float sin2 = sin(ang2);
  float cos2 = cos(ang2);
  float sin3 = sin(ang3);
  float cos3 = cos(ang3);
    
  vec3 ray_dir_p;
  ray_dir_p.x = cos2 * cos3 * ray_dir.x + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir.y + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir.z;
  ray_dir_p.y = cos2 * sin3 * ray_dir.x + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir.y + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir.z;
  ray_dir_p.z = -sin2 * ray_dir.x + sin1 * cos2 * ray_dir.y + cos1 * cos2 * ray_dir.z;
  
  vec3 rd = ray_dir_p;

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

  fragColor.xyz = col;
  
}
)glsl"
};

shaders::Shader const shaders::oceanic_horizon{
    "oceanic", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 1) in vec2 texCoordV;
layout(location = 0) out vec4 fragColor;

uniform sampler2D backgroundTexture;
uniform float cam_x;
uniform float cam_y;
uniform float cam_z;
uniform float ang1;
uniform float ang2;
uniform float ang3;
uniform float time;
uniform float light_z;
uniform float width;
uniform float height;

float waterlevel = 70.0;        // height of the water
float wavegain   = 1.0;       // change to adjust the general water wave level
float large_waveheight = 1.0; // change to adjust the "heavy" waves (set to 0.0 to have a very still ocean :)
float small_waveheight = 0.0; // change to adjust the small waves

vec3 fogcolor    = vec3( 0.5, 0.7, 1.1 );
vec3 skybottom   = vec3( 0.6, 0.8, 1.2 );
vec3 skytop      = vec3(0.05, 0.2, 0.5);
vec3 reflskycolor= vec3(0.025, 0.10, 0.20);
vec3 watercolor  = vec3(0.2, 0.25, 0.3);

vec3 light       = normalize( vec3(  0.1, 0.25,  light_z ) );

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

  vec2 shift1 = 0.001*vec2( time*160.0*2.0, time*120.0*2.0 );
  vec2 shift2 = 0.001*vec2( time*190.0*2.0, -time*130.0*2.0 );

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
  vec2 shift = vec2( time*80.0, time*60.0 );
  float sum = 0.0;
  // use only 12 cloud-layers ;)
  // this improves performance but results in "god-rays shining through clouds" effect (sometimes)...
  float q2 = 0., q3 = 0.;
  for (int q=0; q<10; q++)
  {
    float c = (q2+350.0) / rDirection.y;// cloud distance
    vec3 cpos = c*rDirection + vec3(831.0, 321.0+q3-shift.x*0.2, 1330.0+shift.y*3.0); // cloud position
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
  float t = -rStart.y / rDirection.y;
  float st = 0.5;
  float alpha = 0.1;
  float asum = 0.0;
  vec3 p = rStart;
  float old_h = 0.0;
    
  for( int j=1000; j<1020; j++ )
  {
    // some speed-up if all is far away...
    if (t > 500.0) st = 1.0;
    if (t > 800.0) st = 2.0;
    if (t > 1500.0) st = 3.0;

    p = rStart + t*rDirection; // calc current ray position

#if RENDER_GODRAYS
    if (rDirection.y>0. && sundot > 0.001 && t>400.0 && t < 2500.0)
    {
      alpha = sundot * clamp((p.y-waterlevel)/waterlevel, 0.0, 1.0) * st * 0.024*smoothstep(0.80, 1.0, trace_fog(p,light));
      asum  += (1.0-asum)*alpha;
      if (asum > 0.9)
        break;
    }
#endif

    h = p.y - water(p.xz);

    t += max(1.0, abs(h)) * sign(h) * st;
      
    if (old_h * h < 0.0) st /= 2.0;
    
    old_h = h;
  }

  dist = t; 
  fog = asum;
  if (rDirection.y > 0.0) return false;
  return true;
}

void main() {
  vec2 xy = texCoordV;
  fragColor = vec4(0.0);
  fragColor.w = 1.0;


  xy = texCoordV;
  vec4 sample_noise = texture(backgroundTexture, (texCoordV + 1.0) / 2.0);
  xy.x += sample_noise.x / width;
  xy.y += sample_noise.y / height;

  // get camera position and view direction
  vec3 campos;
  campos.x = cam_x;
  campos.y = cam_y;
  campos.z = cam_z;
  
  vec3 ray_dir;
  ray_dir.x = (xy.x + 1.0) * width / 2.0 - width / 2.0;
  ray_dir.y = (xy.y + 1.0) * height / 2.0 - height / 2.0;
  ray_dir.z = 1.73 * width / 2.0;
  ray_dir = normalize(ray_dir);
  
  float sin1 = sin(ang1);
  float cos1 = cos(ang1);
  float sin2 = sin(ang2);
  float cos2 = cos(ang2);
  float sin3 = sin(ang3);
  float cos3 = cos(ang3);
    
  vec3 ray_dir_p;
  ray_dir_p.x = cos2 * cos3 * ray_dir.x + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir.y + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir.z;
  ray_dir_p.y = cos2 * sin3 * ray_dir.x + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir.y + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir.z;
  ray_dir_p.z = -sin2 * ray_dir.x + sin1 * cos2 * ray_dir.y + cos1 * cos2 * ray_dir.z;
  
  vec3 rd = ray_dir_p;

  float sundot = clamp(dot(rd,light),0.0,1.0);

  vec3 col = vec3(0.0);
  float fog=0.0, dist=0.0;

  if (!trace(campos,rd,sundot, fog, dist)) {
    col.x = 1.0;
    col.y = pow( sundot, 350.0 );
  } else {
    col.x = 0.0;
    vec3 wpos = campos + dist*rd; // calculate position where ray meets water
    // calculate water-mirror
    vec2 xdiff = vec2(0.1, 0.0)*wavegain*4.;
    vec2 ydiff = vec2(0.0, 0.1)*wavegain*4.;
    rd = reflect(rd, normalize(vec3(water(wpos.xz-xdiff) - water(wpos.xz+xdiff), 1.0, water(wpos.xz-ydiff) - water(wpos.xz+ydiff))));  
    sundot = clamp(dot(rd,light),0.0,1.0);
    float refl = (0.5*pow( sundot, 10.0 )+0.25*pow( sundot, 3.5)+.75*pow( sundot, 300.0));
    col.y = refl;
  }

  fragColor.xyz = col;
  
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
