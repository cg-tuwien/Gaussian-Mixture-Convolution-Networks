#version 450

// Accelerated Density Visualization
// Fragment Shader. Calcualtes contribution to pixel sum

flat in int gaussIndex;

out float frag_sum;

uniform mat4 viewMatrix;
uniform mat4 invViewMatrix;
uniform int width;
uniform int height;
uniform float fov;
uniform sampler1D gaussTex;

float sqrt2pi = 2.506628275;

struct Gaussian {
	vec4 mu_alpha;
	mat4 invsigma;
};

layout(std430, binding=0) buffer GaussianMixture {
	Gaussian gaussians[];
} mixture;

float evalGaussian(vec3 rorig, vec3 rdir, Gaussian gauss) {
	mat3 inv = mat3(gauss.invsigma);
	vec3 rs = rdir*inv;
	float sig2 = 1.0 / dot(rs, rdir);
	float mu = dot(rs, gauss.mu_alpha.xyz-rorig) * sig2;
	float sig = sqrt(sig2);
	vec3 pivec = rorig + mu*rdir- gauss.mu_alpha.xyz;
	float gammak = sqrt2pi * gauss.mu_alpha.w * sig * exp(-0.5*dot(pivec*inv, pivec));
	return gammak * texture(gaussTex, mu / sig).r;
}


void main() {
	vec2 pixel_center = gl_FragCoord.xy;
	vec2 pixel_uv = pixel_center / vec2(width, height);
	vec2 d = pixel_uv * 2.0 - 1.0;
	float aspectRatio = float(width) / float(height);	//or just pass aspect ratio

	vec3 origin = vec3(0, 0, 0.0);
	vec3 direction = normalize(vec3(d.x * aspectRatio, d.y, -1/tan(fov/2.0)));	//or just pass -1/tan(fov/2.0)
	vec4 p1 = vec4(origin, 1.0);
	vec4 p2 = vec4(origin + direction, 1.0);
	vec4 vp1 = invViewMatrix * p1;
	vec4 vp2 = invViewMatrix * p2;
	origin = vec3(vp1);
	direction = vec3(normalize(vp2 - vp1));

	origin = vec3(origin.x, -origin.z, origin.y);
	direction = vec3(direction.x, -direction.z, direction.y);

	frag_sum = evalGaussian(origin, direction, mixture.gaussians[gaussIndex]);
}