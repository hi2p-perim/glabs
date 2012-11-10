#version 420 core

layout (location = 0) in vec3 position;
out vec2 vTexCoord;

void main()
{
	vTexCoord = (position.xy + vec2(1.0)) * 0.5;
	gl_Position = vec4(position, 1.0);
}