#version 450

//Accelerated Density Visualization
//Vertex Shader. Projects ellipsoids.

layout(location = 0) in vec3 in_vertex;
layout(location = 1) in mat4 in_transform;

out int gaussIndex;

uniform mat4 projMatrix;
uniform mat4 viewMatrix;

void main() {
	//Correct coordinate system
	vec4 frag_position = in_transform * vec4(in_vertex, 1.0f);
	frag_position = vec4(frag_position.x, frag_position.z, -frag_position.y, frag_position.w);
	gl_Position = projMatrix * viewMatrix  * frag_position;
	gaussIndex = gl_InstanceID;
}