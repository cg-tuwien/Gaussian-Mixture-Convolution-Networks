//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Reinhold Preiner 2014-2020 
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------


#pragma once

#include <vector>
#include <iostream>
#include "random.hpp"
#include "vec.hpp"


using namespace std;



namespace gms
{
	
#pragma region covariance matrix operators

	// (old) conditioning by diagonal biasing
	inline smat3 conditionCov2(const smat3& cov, vec3& outEvalues, float epsilon = 1e-10f, uint maxIters = 3)
	{
		smat3 newCov = cov;
		outEvalues = newCov.eigenvalues();

		// If we have negative eigenvalues, bias matrix trace
		for (uint i = 0; i < maxIters && outEvalues.x <= 0.0f; ++i)
		{
			float bias = (outEvalues.x == 0.0f) ? epsilon : 1.5f * fabsf(outEvalues.x);
			newCov += smat3(bias, 0, 0, bias, 0, bias);
			outEvalues = newCov.eigenvalues();
		}

		if (outEvalues.x <= 0.0f)
			cout << "Warning: cov still non-psd despite conditioning! det: " << det(newCov) << ", cov: " << newCov.toString() << endl;

		return newCov;
	}
	
	// Conditioning by off diagonal dampening
	inline smat3 conditionCov(const smat3& cov, vec3& outEvalues, float epsilon = 1e-10f, uint maxIters = 3)
	{
		smat3 newCov;
		float abseps = fabsf(epsilon);

		// condition diagonal elements
		newCov.e00 = fmaxf(cov.e00, abseps);
		newCov.e11 = fmaxf(cov.e11, abseps);
		newCov.e22 = fmaxf(cov.e22, abseps);

		// condition off diagonal elements
		float sx = sqrtf(newCov.e00);
		float sy = sqrtf(newCov.e11);
		float sz = sqrtf(newCov.e22);

		for (float rho = 0.99f; rho >= 0; rho -= 0.01f)
		{
			float rxy = rho * sx * sy;
			float rxz = rho * sx * sz;
			float ryz = rho * sy * sz;
			newCov.e01 = clamp(cov.e01, -rxy, rxy);
			newCov.e02 = clamp(cov.e02, -rxz, rxz);
			newCov.e12 = clamp(cov.e12, -ryz, ryz);

			// Check
			outEvalues = cov.eigenvalues();
			if (outEvalues.x > 0.0f)
				break;
		}

		// Check
#if GMS_DEBUG
		if (outEvalues.x <= 0.0f)
		{
			std::cout << "Warning: cov still non-psd despite conditioning! det: " << det(cov) << ", cov: " << cov.toString();
			std::cout << ", evalues: " << outEvalues << ", " << std::endl;
		}
#endif
		return newCov;
	}

	inline smat3 conditionCov(const smat3& cov, float epsilon = 1e-10f, uint maxIters = 3)
	{
		vec3 evd;
		return conditionCov(cov, evd, epsilon, maxIters);
	}
	

	// squared Mahalanobis distance between two points X and Y, given the covariance matrix cov
	inline float SMD(const vec3& X, const vec3& Y, const smat3& cov)
	{
		return dot(X - Y, inverse(cov) * (X - Y));
	}

	// Mahalanobis distance between two points X and Y, given the covariance matrix cov
	inline float MD(const vec3& X, const vec3& Y, const smat3& cov)
	{
		return sqrtf(SMD(X, Y, cov));
	}
#pragma endregion


	// 3D Gaussian distribution with center mu, covariance cov and measure weight
	struct Gaussian
	{
		vec3 mu;		// mean
		smat3 cov;		// covariance
		float weight;	// weight

		
		// Default Ctor
		Gaussian()
		{
		}

		// Ctor
		Gaussian(const vec3& mu, const smat3& cov, float weight = 1.0f) : mu(mu), cov(cov), weight(weight)
		{
		}

		static Gaussian zero() 
		{
			return Gaussian (vec3 (0, 0, 0), smat3::zero(), 0);
		}

		// returns a Gaussian representing the maximum likelihood estimate of two given Gaussians
		static Gaussian MLE(const Gaussian& g0, const Gaussian& g1)
		{
			Gaussian gMLE = Gaussian::zero();

			gMLE.weight = g0.weight + g1.weight;
			//>> gMLE.mu = g0.weight * (g0.mu - g0.mu) + g1.weight * (g1.mu - g0.mu);
			//>> gMLE.cov = g0.weight * (g0.cov + smat3::outer(g0.mu - g0.mu)) + g1.weight * (g1.cov + smat3::outer(g1.mu - g0.mu));
			gMLE.mu = g1.weight * (g1.mu - g0.mu);
			gMLE.cov = g0.weight * g0.cov + g1.weight * (g1.cov + smat3::outer(g1.mu - g0.mu));
			
			if (!isnormal(gMLE.weight))
				return Gaussian::zero();

			// normalize
			float invSumWeights = 1.0f / gMLE.weight;
			gMLE.mu *= invSumWeights;
			gMLE.cov = invSumWeights * gMLE.cov - smat3::outer(gMLE.mu);
			gMLE.mu += g0.mu;

			return gMLE;
		}



		// Gaussian MLE of a mixture
		static Gaussian MLE(const vector<Gaussian>& mixture)
		{
			Gaussian gMLE = Gaussian::zero();
			if (mixture.empty())
				return gMLE;

			// accumulate centralized Gaussian
			for (const Gaussian& g : mixture)
			{
				gMLE.weight += g.weight;
				gMLE.mu += g.weight * (g.mu - mixture[0].mu);
				gMLE.cov += g.weight * (g.cov + smat3::outer(g.mu));
			}
			if (!isnormal(gMLE.weight))
				return Gaussian::zero();

			// normalize
			float invSumWeights = 1.0f / gMLE.weight;
			gMLE.mu *= invSumWeights;
			gMLE.cov = invSumWeights * gMLE.cov - smat3::outer(gMLE.mu);
			gMLE.mu += mixture[0].mu;

			return gMLE;
		}


		// returns the (normalized) probability density function at X, neglecting the weight of this Gaussian
		float pdf(const vec3& X) const
		{
			// TOOD: fuse det and inverse computation
			float d = det(cov);
			vec3 diff = X - mu;
			return expf(-0.5f * dot(diff, inverse(cov) * diff)) / sqrtf(248.0502134f * d);


			// if the above ever causes problems due to bad conditioned matrices, here is an alternative to think about using in the standard implementation:
			/*float mm = cov.maxMagnitude();
			float invMM = 1.0f / mm;
			smat3 condcov = cov * invMM;
			float d = det(condcov) * pow(mm, 3);

			return expf(-0.5f * invMM * dot(diff, inverse(condcov) * diff)) / sqrtf(248.0502134f * d);*/
		}

		// draws a random sample of this Gaussian's pdf
		vec3 sample() const
		{
			vec3 evals;
			mat3 rot;
			cov.eigensystem(evals, rot);

			return mu + rot * vec3(sqrt(evals.x) * random::normal(), sqrt(evals.y) * random::normal(), sqrt(evals.z) * random::normal());
		}

		// returns the gradient of the (normalized) pdf at X, neglecting the weight of this Gaussian
		vec3 gradient(const vec3& X) const
		{
			float d = det(cov);
			vec3 diff = X - mu;
			vec3 icov_diff = inverse(cov) * diff;
			
			return -(expf(-0.5f * dot(diff, icov_diff)) / sqrtf(248.0502134f * d)) * icov_diff;
		}

		
		// returns the Hessian matrix of the (normalized) pdf at X, neglecting the weight of this Gaussian
		smat3 Hessian(const vec3& X) const
		{
			float d = det(cov);
			vec3 diff = X - mu;
			smat3 icov = inverse(cov);
			vec3 icov_diff = icov * diff;

			return (expf(-0.5f * dot(diff, icov_diff)) / sqrtf(248.0502134f * d)) * (smat3::outer(icov_diff) - icov);
		}
		

		// returns the product Gaussian resulting from multiplication of this with another weighted Gaussian (weights are not ignored). 
		// the resulting Gaussian has according mu, cov and weight. (MatrixCookbook)
		Gaussian operator* (const Gaussian& g) const
		{
			smat3 covSum = cov + g.cov;

			//smat3 invCovSum = inverse(covSum);
			//float preFac = 1.0f / sqrtf(248.05021f * det(covSum));  // 2pi ^ 3* det
			float invDetCovSum = 1.0f / det(covSum);
			smat3 invCovSum = adjugate(covSum) * invDetCovSum;
			float preFac = 0.063493636f * sqrtf(invDetCovSum);  // sqrt( 1 / (2pi)^3 )

			smat3 inv_cov0 = inverse(cov);
			smat3 inv_cov1 = inverse(g.cov);
			vec3 d = g.mu - mu;

			Gaussian product;
			product.weight = weight * g.weight * preFac * exp(-0.5f * dot(d, invCovSum * d));
			product.cov = inverse(inv_cov0 + inv_cov1);
			product.mu = product.cov * inv_cov1 * d + mu;

			return product;
		}
	};


#pragma region Gaussian operators

	// Kullback-Leibler divergence between two Gaussians
	inline float KLD(const Gaussian& gc, const Gaussian& gp)
	{
		return 0.5f * (SMD(gc.mu, gp.mu, gp.cov) + trace(inverse(gp.cov) * gc.cov) - 3.0f - log(det(gc.cov) / det(gp.cov)));
	}
	
	// Jensen Shannon Divergence between two Gaussians (symmetrized Kullback-Leibler divergence)
	inline float JSD(const Gaussian& g1, const Gaussian& g2)
	{
		return 0.5f * (KLD(g1, g2) + KLD(g2, g1));
	}

	// Bhattacharyya distance between two Gaussians
	inline float DBhatt(const Gaussian& g1, const Gaussian& g2)
	{
		// 0 <= DBhatt < inf. clamp to zero to avoid negative values due to numerical instabilities
		return max(0.0f,
			0.25f * SMD(g1.mu, g2.mu, g1.cov + g2.cov) + 0.5f * logf(det(g1.cov + g2.cov)) - 0.25f * logf(det(g1.cov) * det(g2.cov)) - 1.03972077f		// dim/2 * ln(2)
		);
	}

	// Bhattacharyya coefficient between two Gaussians
	inline float BC(const Gaussian& g0, const Gaussian& g1)
	{
		return expf(-DBhatt(g0, g1));
	}

	// applies the given linear transformationo T on the Gaussian. The weight is left unaffected.
	inline Gaussian transform(const Gaussian& g, const mat4& T)
	{
		Gaussian gT;

		// for the covariance transformation, only the rotational part is relevant
		mat3 rot = T.toMat3();
		gT.cov = smult(g.cov, rot);
		gT.mu = (T * vec4(g.mu, 1)).xyz();
		gT.weight = g.weight;

		return gT;
	}

	// returns true iff all float elements of this Gaussian (weight, mu, cov) are finite floats (not NaN and not inf)
	inline bool isfinite(const Gaussian& g)
	{
		return isfinite(g.weight) && isfinite(g.mu) && isfinite(g.cov);
	}

	// returns true iff all float elements of this Gaussian (weight, mu, cov) are finite floats (not NaN and not inf) and the weight is > 0
	inline bool isnormal(const Gaussian& g)
	{
		return isfinite(g.mu) && isfinite(g.cov) && isnormal(g.weight) && g.weight > 0;
	}

#pragma endregion

#pragma region higher Gaussians moments
	// first moments
	inline float Ex (const Gaussian& g)  { return g.mu.x; }
	inline float Ey (const Gaussian& g)  { return g.mu.y; }
	inline float Ez (const Gaussian& g)  { return g.mu.z; }
	// second moments
	inline float Ex2 (const Gaussian& g)  { return g.mu.x*g.mu.x + g.cov.e00; }
	inline float Ey2 (const Gaussian& g)  { return g.mu.y*g.mu.y + g.cov.e11; }
	inline float Ez2 (const Gaussian& g)  { return g.mu.z*g.mu.z + g.cov.e22; }
	inline float Exy (const Gaussian& g)  { return g.mu.x*g.mu.y + g.cov.e01; }
	inline float Exz (const Gaussian& g)  { return g.mu.x*g.mu.z + g.cov.e02; }
	inline float Eyz (const Gaussian& g)  { return g.mu.y*g.mu.z + g.cov.e12; }
	// third moments
	inline float Ex3 (const Gaussian& g)  { return g.mu.x*g.mu.x*g.mu.x + 3 * g.mu.x * g.cov.e00; }
	inline float Ey3 (const Gaussian& g)  { return g.mu.y*g.mu.y*g.mu.y + 3 * g.mu.y * g.cov.e11; }
	inline float Ez3 (const Gaussian& g)  { return g.mu.z*g.mu.z*g.mu.z + 3 * g.mu.z * g.cov.e22; }
	inline float Ex2y(const Gaussian& g)  { return g.mu.x*g.mu.x*g.mu.y + g.cov.e00*g.mu.y + 2 * g.cov.e01*g.mu.x; }
	inline float Ey2x(const Gaussian& g)  { return g.mu.y*g.mu.y*g.mu.x + g.cov.e11*g.mu.x + 2 * g.cov.e01*g.mu.y; }
	inline float Ex2z(const Gaussian& g)  { return g.mu.x*g.mu.x*g.mu.z + g.cov.e00*g.mu.z + 2 * g.cov.e02*g.mu.x; }
	inline float Ey2z(const Gaussian& g)  { return g.mu.y*g.mu.y*g.mu.z + g.cov.e11*g.mu.z + 2 * g.cov.e12*g.mu.y; }
	inline float Ez2x(const Gaussian& g)  { return g.mu.z*g.mu.z*g.mu.x + g.cov.e22*g.mu.x + 2 * g.cov.e02*g.mu.z; }
	inline float Ez2y(const Gaussian& g)  { return g.mu.z*g.mu.z*g.mu.y + g.cov.e22*g.mu.y + 2 * g.cov.e12*g.mu.z; }
	inline float Exyz(const Gaussian& g)  { return g.mu.x*g.mu.y*g.mu.z + g.cov.e01*g.mu.z + g.cov.e02*g.mu.y + g.cov.e12*g.mu.x; }
	// fourth moments
	inline float Ex4 (const Gaussian& g)  { return g.mu.x*g.mu.x*g.mu.x*g.mu.x + 3 * g.cov.e00*g.cov.e00 + 6 * g.cov.e00*g.mu.x*g.mu.x; }
	inline float Ey4 (const Gaussian& g)  { return g.mu.y*g.mu.y*g.mu.y*g.mu.y + 3 * g.cov.e11*g.cov.e11 + 6 * g.cov.e11*g.mu.y*g.mu.y; }
	inline float Ez4 (const Gaussian& g)  { return g.mu.z*g.mu.z*g.mu.z*g.mu.z + 3 * g.cov.e22*g.cov.e22 + 6 * g.cov.e22*g.mu.z*g.mu.z; }
	inline float Ex2y2(const Gaussian& g) { return g.mu.x*g.mu.x*g.mu.y*g.mu.y + g.cov.e00*g.cov.e11 + 2 * g.cov.e01*g.cov.e01 + g.cov.e00*g.mu.y*g.mu.y + 4 * g.cov.e01*g.mu.x*g.mu.y + g.cov.e11*g.mu.x*g.mu.x; }
	inline float Ex2z2(const Gaussian& g) { return g.mu.x*g.mu.x*g.mu.z*g.mu.z + g.cov.e00*g.cov.e22 + 2 * g.cov.e02*g.cov.e02 + g.cov.e00*g.mu.z*g.mu.z + 4 * g.cov.e02*g.mu.x*g.mu.z + g.cov.e22*g.mu.x*g.mu.x; }
	inline float Ey2z2(const Gaussian& g) { return g.mu.y*g.mu.y*g.mu.z*g.mu.z + g.cov.e11*g.cov.e22 + 2 * g.cov.e12*g.cov.e12 + g.cov.e11*g.mu.z*g.mu.z + 4 * g.cov.e12*g.mu.y*g.mu.z + g.cov.e22*g.mu.y*g.mu.y; }
	inline float Ex3y(const Gaussian& g)  { return g.mu.x*g.mu.x*g.mu.x*g.mu.y + 3 * g.cov.e00*g.cov.e01 + 3 * g.cov.e00*g.mu.x*g.mu.y + 3 * g.cov.e01*g.mu.x*g.mu.x; }
	inline float Ey3x (const Gaussian& g) { return g.mu.y*g.mu.y*g.mu.y*g.mu.x + 3 * g.cov.e11*g.cov.e01 + 3 * g.cov.e11*g.mu.y*g.mu.x + 3 * g.cov.e01*g.mu.y*g.mu.y; }
#pragma endregion


}	/// end namespace gms

