//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Reinhold Preiner 2014-2020 
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------


#pragma once


#include <cmath>


#pragma warning (disable : 4267)


namespace gms
{
	using uint = unsigned int;
	
	const float pi = 3.1415926535897932384626433832795f;
	

	// min max abs clamp
	//--------------------------------------------------------------------------------------
	inline float fabsf(float a) { return a < 0 ? -a : a; }
	inline float fminf(float a, float b) { return a < b ? a : b; }
	inline float fmaxf(float a, float b) { return a > b ? a : b; }
	inline float lerp(float a, float b, float t)  { return a + t*(b - a); }
	inline float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }

	#undef max
	#undef min
	inline int abs(int a) { return a < 0 ? -a : a; }
	inline int max(int a, int b) { return a > b ? a : b; }
	inline int min(int a, int b) { return a < b ? a : b; }
	inline int clamp(int f, int a, int b) { return max(a, min(f, b)); }
	inline uint max(uint a, uint b) { return a > b ? a : b; }
	inline uint min(uint a, uint b) { return a < b ? a : b; }
	inline uint clamp(uint f, uint a, uint b) { return max(a, min(f, b)); }
	inline double abs(double a) { return a < 0.0f ? -a : a; }
	inline float abs(float a) { return a < 0.0f ? -a : a; }
	inline float max(float a, float b) { return a > b ? a : b; }
	inline float min(float a, float b) { return a < b ? a : b; }

	
	// converts NAN values to 0, and leaves them unchanged otherwise
    inline float nanto0(float x) { return std::isnan(x) ? 0 : x; }
	
	// signum function returning the sign of val (-1 if negative, +1 if positive, 0 if zero).
	template<typename T> int sgn(T val) {
		return (T(0) < val) - (val < T(0));
	}
		
	// cotangent cot(x) = 1 / tan(x) = cos(x) / sin(x)
	template<typename T> T cot(T angle) {
		return tan(pi / 2 - angle);
	}


}	/// end namespace gms
