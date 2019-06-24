
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
layout(location = 1) out vec4 trace;
layout(location = 2) out vec4 trace2;
layout(location = 3) out vec4 trace3;
layout(location = 4) out vec4 trace4;
layout(location = 5) out vec4 trace5;
layout(location = 6) out vec4 trace6;
layout(location = 7) out vec4 trace7;

void main() {
    trace.xy = (texCoordV + 1.0) * 5.0;
    trace.zw = fract(trace.xy) - 0.5;
    colour_out.w = 1.0;
    if (trace.z * trace.w < 0)
      colour_out.xyz = vec3(1.0);
    else
      colour_out.xyz = vec3(0.0);
    trace2.xyzw = trace.xyzw;
    trace3.xyzw = trace.xyzw;
    trace4.xyzw = trace.xyzw;
    trace5.xyzw = trace.xyzw;
    trace6.xyzw = trace.xyzw;
    trace7.xyzw = trace.xyzw;
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
