#version 450
layout(set=1,binding=0,std140) uniform Renamed { vec4 renamedColor; mat4 renamedTransform; } params;
void main() { gl_Position = params.renamedTransform * params.renamedColor; }
