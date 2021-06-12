//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Reinhold Preiner 2014-2020 
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------


#pragma once


#include <random>
#include "base.hpp"

namespace gms
{
	// direct random number access
	class random
	{
	private:
		random() {}
		random(const random&);
		random& operator= (const random&);

		// singletons
		static std::mt19937& getGenerator()
		{
			static std::mt19937 generator(0);
			return generator;
		}
		static std::normal_distribution<float>& getNormalDist()
		{
			static std::normal_distribution<float> normalDist;
			return normalDist;
		}


	public:
		static void reset()
		{
			getGenerator().seed(0);
			getNormalDist().reset();
		}

		static uint uniform()
		{
			static std::uniform_int_distribution<uint> mUniformDist;
			return mUniformDist(getGenerator());
		}

		static float uniform01()
		{
			static std::uniform_real_distribution<float> mUniformRealDist;
			return mUniformRealDist(getGenerator());
		}

		static float uniform(float rangeMin, float rangeMax)
		{
			return rangeMin + uniform01() * (rangeMax - rangeMin);
		}

		static float normal()
		{
			/*// Box-Muller
			float u1 = uniform01();
			float u2 = uniform01();
			float a = sqrt(fabsf(-2 * log(u1)));
			float b = 6.283185307f * u2;
			float z1 = a * cos(b);
			//float z2 = a * sin(b);
			return z1;*/
			
			return getNormalDist()(getGenerator());
		}

		static vec3 unitvector()
		{
			float a = 0;
			while (a == 0)
				a = uniform(-1, 1);
			return normalize(vec3(a, uniform(-1, 1), uniform(-1, 1)));
		}
	};
}

