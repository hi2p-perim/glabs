#version 420 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 mvMatrix;
uniform mat4 projectionMatrix;
uniform float size;

void main()
{
	vec4 eyePos = mvMatrix * gl_in[0].gl_Position;
	float halfSize = size * 0.5;
	vec4 tmp;

	tmp = eyePos;
	tmp.x += halfSize;
	tmp.y += halfSize;
	gl_Position = projectionMatrix * tmp;
	EmitVertex();

	tmp = eyePos;
	tmp.x -= halfSize;
	tmp.y += halfSize;
	gl_Position = projectionMatrix * tmp;
	EmitVertex();

	tmp = eyePos;
	tmp.x += halfSize;
	tmp.y -= halfSize;
	gl_Position = projectionMatrix * tmp;
	EmitVertex();

	tmp = eyePos;
	tmp.x -= halfSize;
	tmp.y -= halfSize;
	gl_Position = projectionMatrix * tmp;
	EmitVertex();

	EndPrimitive();
}
