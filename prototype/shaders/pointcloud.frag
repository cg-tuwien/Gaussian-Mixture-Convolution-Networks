#version 330
in vec4 frag_vertex;
out vec3 fragColor;
//uniform vec3 lightPos;
uniform vec4 pointcloudColor;
uniform bool circles;

void main() {
	//float dist = length(frag_vertex.xyz);
	//float coeff = clamp(dist/50.0f, 0.0f, 1.0f);
	//fragColor = vec4(vec3(dist)*vec3(0.8, 0.2, 0.5), 1.0f);
	//fragColor = vec3(1.0f);
	//fragColor = coeff*vec3(1,0,0) + (1-coeff)*vec3(0,1,0);
	fragColor = pointcloudColor.rgb;
	if (circles) {
		if (length(gl_PointCoord - vec2(0.5, 0.5)) > 0.5f) {
			discard;
		}
	}
}