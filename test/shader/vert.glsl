// #version 450

// vec2 positions[3] = vec2[](
//     vec2(0.0, -0.5),
//     vec2(0.5, 0.5),
//     vec2(-0.5, 0.5)
// );

// void main() {
//     gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
// }


#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout (location = 0) out vec3 fragColor;

void main ()
{
	gl_Position = vec4 (inPosition, 0.0, 1.0);
	fragColor = inColor;
}