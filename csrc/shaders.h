
#ifndef SHADERS_H
#define SHADERS_H

#include <string>

namespace shaders
{
    struct Shader {
        std::string name;
        GLenum type;
        std::string source;
    };

    extern Shader const
        forward_vertex,
        forward_fragment,
        backward_vertex,
        backward_fragment,
        second_pass_fragment,
        oceanic,
        oceanic_no_cloud,
        oceanic_still_cloud,
        oceanic_opt_flow,
        oceanic_horizon,
        oceanic_simple_proxy,
        hill,
        texture_debug_minimum;
}

#endif //SHADERS_H
