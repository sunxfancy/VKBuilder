#version 450
layout(set=1,binding=0,std140) uniform Params { vec4 color; mat4 transform; } params;
layout(set=1,binding=1) uniform samplerCube environmentMap;
layout(set=1,binding=2) uniform sampler2D textures[2];
layout(set=1,binding=3,std430) readonly buffer Records { vec4 values[]; } records;
layout(location=0) out vec4 outColor;
void main() {
  outColor = params.transform * params.color + texture(environmentMap, vec3(1,0,0)) +
             texture(textures[0], vec2(0)) + texture(textures[1],vec2(0)) + records.values[0];
}
