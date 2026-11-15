#version 450
layout(set=1,binding=0,std140) uniform Params { ivec4 color; mat4 transform; } params;
void main() { gl_Position = params.transform * vec4(params.color); }
