#version 330
layout(location = 0) in vec3 in_vertex;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in float in_color;
layout(location = 3) in mat4 in_transform;
layout(location = 7) in mat4 in_normtrans;

out vec4 frag_position;
out vec3 frag_normal;
out vec3 frag_color;

uniform bool useInColor;
uniform mat4 projMatrix;
uniform mat4 viewMatrix;
uniform vec4 surfaceColor;
uniform sampler1D transferTex;

void main() {
	//Correct coordinate system
	frag_position = in_transform * vec4(in_vertex, 1.0f);
	frag_position = vec4(frag_position.x, frag_position.z, -frag_position.y, frag_position.w);
	gl_Position = projMatrix * viewMatrix  * frag_position;
	frag_normal = mat3(in_normtrans) * in_normal;
	frag_normal = vec3(frag_normal.x, frag_normal.z, -frag_normal.y);
	if (useInColor) {
		frag_color = texture(transferTex, 0.25 + 0.75*in_color).rgb;
	} else {
		frag_color = surfaceColor.rgb;
	}
}