#version 450
layout(set=1,binding=0,std140,row_major) uniform Params { vec4 color; mat4 transform; } params;
void main() { gl_Position = params.transform * params.color; }
