//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Reinhold Preiner 2014-2020 
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------

#pragma once

#include "base.hpp"
#include <sstream>
#include <iostream>
#include <limits>


namespace gms
{
	#pragma region vec
	//--------------------------------------------------------------------------------------
	template<int D, class T>
	struct vec
	{
		T e[D] = { 0 };

		vec() {}
		vec(T data[D]) { memcpy(this, data, D * sizeof(T)); }
		vec(std::initializer_list<T> list) { memcpy(this, list.begin(), max(list.size(), (uint)D) * sizeof(T)); }
		void operator=(const vec& rhs) { memcpy(this, &rhs, D * sizeof(T)); }
		bool operator==(const vec& rhs) const { return memcmp(this, &rhs, D * sizeof(T)) == 0; }
		bool operator!=(const vec& rhs) const { return !(*this == rhs); }

		T operator[](uint i) const { /*assert(i < D);*/ return *(e + i); }
		T& operator[](uint i) { /*assert(i < D);*/ return *(e + i); }
	};

	template<int D, class T>
	inline vec<D,T> operator+(const vec<D, T>& a, const vec<D,T> &b)
	{
		vec<D, T> c;
		for (int i = 0; i < D; ++i) c[i] = a[i] + b[i];
		return c;
	}
	template<int D, class T>
	inline vec<D, T> operator-(const vec<D, T>& a, const vec<D, T> &b)
	{
		vec<D,T> c;
		for (int i = 0; i < D; ++i) c[i] = a[i] - b[i];
		return c;
	}
	template<int D, class T>
	inline vec<D, T> operator*(const vec<D, T>& a, T s)
	{
		vec<D, T> c;
		for (int i = 0; i < D; ++i) c[i] = a[i] * s;
		return c;
	}
	template<int D, class T>
	inline vec<D, T> operator*(T s, const vec<D, T>& a)
	{
		vec<D, T> c;
		for (int i = 0; i < D; ++i) c[i] = a[i] * s;
		return c;
	}
	template<int D, class T>
	inline vec<D, T> operator/(const vec<D, T>& a, T s)
	{
		s = T(1) / s;
		vec<D, T> c;
		for (int i = 0; i < D; ++i) c[i] = a[i] * s;
		return c;
	}
	template<int D, class T>
	inline vec<D, T>& operator+=(vec<D, T>& a, const vec<D, T> &b)
	{
		for (int i = 0; i < D; ++i) a[i] += b[i];
		return a;
	}
	template<int D, class T>
	inline vec<D, T>& operator-=(vec<D, T>& a, const vec<D, T> &b)
	{
		for (int i = 0; i < D; ++i) a[i] -= b[i];
		return a;
	}
	template<int D, class T>
	inline vec<D, T>& operator*=(vec<D, T>& a, T s)
	{
		for (int i = 0; i < D; ++i) a[i] *= s;
		return a;
	}
	template<int D, class T>
	inline vec<D, T>& operator/=(vec<D, T>& a, T s)
	{
		s = T(1) / s;
		for (int i = 0; i < D; ++i) a[i] *= s;
		return a;
	}
	template<int D, class T>
	inline vec<D, T>& operator-(vec<D, T>& a)
	{
		for (int i = 0; i < D; ++i) a[i] = -a[i];
		return a;
	}
	template<int D, class T>
	inline std::ostream& operator<< (std::ostream& os, const vec<D,T>& p)
	{
		os << "{";
		for (int i = 0; i < D-1; ++i) 
			os << p[i] << ", ";
		os << p[D-1] << "}";

		return os;
	}


	#pragma region vec2i
	//--------------------------------------------------------------------------------------
	struct vec2i
	{
		vec2i() { x = y = 0; }		
		vec2i(int X, int Y) { x = X; y = Y; }
		void operator=(const vec2i& rhs) { x = rhs.x; y = rhs.y; }

		bool operator==(const vec2i& rhs) const { return x == rhs.x && y == rhs.y; }
		bool operator!=(const vec2i& rhs) const { return x != rhs.x || y != rhs.y; }

		int operator[](uint i) const { /*assert(i < 2);*/ return *(&x + i); }
		int& operator[](uint i) { /*assert(i < 2);*/ return *(&x + i); }

		int x, y;


		struct hasher
		{
			size_t operator()(const vec2i &x) const
			{
				return std::hash<uint>()(x.x) ^ std::hash<uint>()(x.y);
			}

			bool operator()(const vec2i& left, const vec2i& right)
			{
				if (left.x != right.x)	return left.x < right.x;
				return left.y < right.y;
			}
		};
	};

	inline vec2i operator+(const vec2i& a, const vec2i& b)
	{
		return vec2i(a.x + b.x, a.y + b.y);
	}
	inline vec2i operator-(const vec2i& a, const vec2i& b)
	{
		return vec2i(a.x - b.x, a.y - b.y);
	}
	inline vec2i operator*(const vec2i& a, int s)
	{
		return vec2i(a.x*s, a.y*s);
	}
	inline vec2i operator*(int s, const vec2i& a)
	{
		return a*s;
	}
	inline vec2i& operator+=(vec2i& a, const vec2i& b)
	{
		a = a + b;
		return a;
	}
	inline vec2i& operator-=(vec2i& a, const vec2i& b)
	{
		a = a - b;
		return a;
	}
	inline vec2i& operator*=(vec2i& a, int s)
	{
		a = a * s;
		return a;
	}
	inline vec2i operator-(vec2i& a)
	{
		return vec2i(-a.x, -a.y);
	}
	inline std::ostream& operator<< (std::ostream& os, const vec2i& p)
	{
		os << "{" << p.x << ", " << p.y << "}";
		return os;
	}

	inline vec2i abs(const vec2i& a) { return vec2i(abs(a.x), abs(a.y)); }
	//--------------------------------------------------------------------------------------
	#pragma endregion
	



	#pragma region vec2
	//--------------------------------------------------------------------------------------
	struct vec2
	{
		vec2() { x = y = 0; }
		vec2(float X, float Y) { x = X; y = Y; }
		vec2(const vec2i& rhs) { x = (float)rhs.x; y = (float)rhs.y; }

		void operator=(const vec2& rhs) { x = rhs.x; y = rhs.y; }

		bool operator==(const vec2& rhs) const { return x == rhs.x && y == rhs.y; }
		bool operator!=(const vec2& rhs) const { return x != rhs.x || y != rhs.y; }
		
		float operator[](uint i) const { /*assert(i < 2);*/ return *(&x + i); }
		float& operator[](uint i) { /*assert(i < 2);*/ return *(&x + i); }

		float x, y;
	};

	inline vec2 operator+(const vec2& a, const vec2& b)
	{
		return vec2(a.x + b.x, a.y + b.y);
	}
	inline vec2 operator-(const vec2& a, const vec2& b)
	{
		return vec2(a.x - b.x, a.y - b.y);
	}
	inline vec2 operator*(const vec2& a, float s)
	{
		return vec2(a.x*s, a.y*s);
	}
	inline vec2 operator*(const vec2& a, const vec2& b)
	{
		return vec2(a.x*b.x, a.y*b.y);
	}
	inline vec2 operator*(float s, const vec2& a)
	{
		return a*s;
	}
	inline vec2 operator/(const vec2& a, float s)
	{
		float is = 1.0f / s;
		return a * is;
	}
	inline vec2 operator/(float s, const vec2& a)
	{
		return vec2(s / a.x, s / a.y);
	}
	inline vec2& operator+=(vec2& a, const vec2& b)
	{
		a = a + b;
		return a;
	}
	inline vec2& operator-=(vec2& a, const vec2& b)
	{
		a = a - b;
		return a;
	}
	inline vec2& operator*=(vec2& a, float s)
	{
		a = a * s;
		return a;
	}
	inline vec2& operator/=(vec2& a, float s)
	{
		a = a / s;
		return a;
	}
	inline vec2 operator-(const vec2& a)
	{
		return vec2(-a.x, -a.y);
	}
	inline float dot(const vec2& a, const vec2& b)
	{
		return	a.x*b.x + a.y*b.y;
	}
	inline float sqdist(const vec2& a, const vec2& b)
	{
		vec2 d = a - b;
		return dot(d, d);
	}
	inline float dist(const vec2& a, const vec2& b)
	{
		return sqrt(sqdist(a, b));
	}
	inline std::ostream& operator<< (std::ostream& os, const vec2& p)
	{
		os << "{" << p.x << ", " << p.y << "}";
		return os;
	}
	inline float length(const vec2& v)
	{
		return sqrtf(dot(v, v));
	}
	inline vec2 normalize(const vec2& v)
	{
		return v / length(v);
	}
	//--------------------------------------------------------------------------------------
	#pragma endregion



	#pragma region vec3i
	//--------------------------------------------------------------------------------------
	struct vec3i
	{
		vec3i() { x = y = z = 0; }
		vec3i(int X, int Y, int Z) { x = X; y = Y; z = Z; }
		template<class vec3T> explicit vec3i(const vec3T& rhs) { x = (int)rhs.x; y = (int)rhs.y; z = (int)rhs.z; }
		
		const vec3i& operator=(const vec3i& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; return *this; }
		
		bool operator==(const vec3i& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
		bool operator!=(const vec3i& rhs) const { return x != rhs.x || y != rhs.y || z != rhs.z; }

		int operator[](uint i) const { /*assert(i < 3);*/ return *(&x + i); }
		int& operator[](uint i) { /*assert(i < 3);*/ return *(&x + i); }

		int x, y, z;
	};

	inline vec3i operator+(const vec3i& a, const vec3i& b)
	{
		return vec3i(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	inline vec3i operator-(const vec3i& a, const vec3i& b)
	{
		return vec3i(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	inline vec3i operator*(const vec3i& a, int s)
	{
		return vec3i(a.x*s, a.y*s, a.z*s);
	}
	inline vec3i operator*(int s, const vec3i& a)
	{
		return a*s;
	}
	inline vec3i operator/(const vec3i& a, float s)
	{
		return vec3i(int(a.x / s), int(a.y / s), int(a.z / s));
	}
	inline vec3i& operator+=(vec3i& a, const vec3i& b)
	{
		a = a + b;
		return a;
	}
	inline vec3i& operator-=(vec3i& a, const vec3i& b)
	{
		a = a - b;
		return a;
	}
	inline vec3i& operator*=(vec3i& a, int s)
	{
		a = a * s;
		return a;
	}
	inline std::ostream& operator<< (std::ostream& os, const vec3i& p)
	{
		os << "{" << p.x << ", " << p.y << ", " << p.z << "}";
		return os;
	}

	inline vec3i min(const vec3i& a, const vec3i& b) { return vec3i(min(a.x, b.x), min(a.y, b.y), min(a.z, b.y)); }
	inline vec3i max(const vec3i& a, const vec3i& b) { return vec3i(max(a.x, b.x), max(a.y, b.y), max(a.z, b.y)); }
	inline vec3i clamp(const vec3i& x, const vec3i& a, const vec3i& b) { return vec3i(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z)); }
	inline vec3i abs(const vec3i& a) { return vec3i(abs(a.x), abs(a.y), abs(a.z)); }
	//--------------------------------------------------------------------------------------
	#pragma endregion


	

	#pragma region vec3
	//--------------------------------------------------------------------------------------
	template<class T>
	struct vec3_t
	{
		vec3_t() { x = y = z = 0; }
		vec3_t(T X, T Y, T Z) { x = X; y = Y; z = Z; }
		
		// ctors
		vec3_t(const vec3i& rhs) { x = (T)rhs.x; y = (T)rhs.y; z = (T)rhs.z; }
		template<class To>
		vec3_t(const vec3_t<To>& rhs) { x = (T)rhs.x; y = (T)rhs.y; z = (T)rhs.z; }

		const vec3_t& operator=(const vec3_t& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; return *this; }
		
		bool operator==(const vec3_t& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
		bool operator!=(const vec3_t& rhs) const { return x != rhs.x || y != rhs.y || z != rhs.z; }

		T operator[](uint i) const { /*assert(i < 3);*/ return *(&x + i); }
		T& operator[](uint i) { /*assert(i < 3);*/ return *(&x + i); }

		T x, y, z;
	};

	template<class T>
	inline vec3_t<T> operator-(const vec3_t<T>& a)
	{
		return vec3_t<T>(-a.x, -a.y, -a.z);
	}
	template<class T>
	inline vec3_t<T> operator+(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		return vec3_t<T>(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	template<class T> inline vec3_t<T> operator+(const vec3_t<T>& a, const vec3i& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
	template<class T> inline vec3_t<T> operator+(const vec3i& a, const vec3_t<T>& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
	template<class T>
	inline vec3_t<T> operator-(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		return vec3_t<T>(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	template<class T> inline vec3_t<T> operator-(const vec3_t<T>& a, const vec3i& b) { return vec3_t<T>(a.x - b.x, a.y - b.y, a.z - b.z); }
	template<class T> inline vec3_t<T> operator-(const vec3i& a, const vec3_t<T>& b) { return vec3_t<T>(a.x - b.x, a.y - b.y, a.z - b.z); }
	template<class T, class S>
	inline vec3_t<T> operator*(const vec3_t<T>& a, S s)
	{
		return vec3_t<T>(T(a.x*s), T(a.y*s), T(a.z*s));
	}
	template<class T, class S>
	inline vec3_t<T> operator*(S s, const vec3_t<T>& a)
	{
		return a*s;
	}
	// component-wise multiplication
	template<class T>
	inline vec3_t<T> operator*(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		return vec3_t<T>(a.x*b.x, a.y*b.y, a.z*b.z);
	}
	template<class T, class S>
	inline vec3_t<T> operator/(const vec3_t<T>& a, S s)
	{
		T is = 1.0f / s;
		return a * is;
	}
	template<class T>
	inline vec3_t<T> operator/(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		return vec3_t<T>(a.x / b.x, a.y / b.y, a.z / b.z);
	}
	template<class T>
	inline vec3_t<T>& operator+=(vec3_t<T>& a, const vec3_t<T>& b)
	{
		a = a + b;
		return a;
	}
	template<class T>
	inline vec3_t<T>& operator-=(vec3_t<T>& a, const vec3_t<T>& b)
	{
		a = a - b;
		return a;
	}
	template<class T>
	inline vec3_t<T>& operator*=(vec3_t<T>& a, T s)
	{
		a = a * s;
		return a;
	}
	template<class T, class S>
	inline vec3_t<T>& operator/=(vec3_t<T>& a, S s)
	{
		a = a / s;
		return a;
	}
	template<class T>
	inline T dot(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		return	a.x*b.x + a.y*b.y + a.z*b.z;
	}
	template<class T>
	inline T sqdist(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		vec3_t<T> d = a - b;
		return dot(d, d);
	}
	template<class T>
	inline T dist(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		return sqrt(sqdist(a, b));
	}
	template<class T>
	inline std::ostream& operator<< (std::ostream& os, const vec3_t<T>& p)
	{
		os << "{" << p.x << ", " << p.y << ", " << p.z << "}";
		return os;
	}
	template<class T>
	inline T length(const vec3_t<T>& v)
	{
		return sqrt(dot(v, v));
	}
	template<class T>
	inline vec3_t<T> normalize(const vec3_t<T>& v)
	{
		return v / length(v);
	}
	template<class T>
	inline vec3_t<T> cross(const vec3_t<T>& a, const vec3_t<T>& b)
	{
		return vec3_t<T>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}

	template<class T>
	inline vec3_t<T> min(const vec3_t<T>& a, const vec3_t<T>& b) { return vec3_t<T>(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
	template<class T>
	inline vec3_t<T> max(const vec3_t<T>& a, const vec3_t<T>& b) { return vec3_t<T>(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
	template<class T>
	inline vec3_t<T> clamp(const vec3_t<T>& x, const vec3_t<T>& a, const vec3_t<T>& b) { return vec3_t<T>(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z)); }
	template<class T>
	inline vec3_t<T> fabsf(const vec3_t<T>& a) { return vec3_t<T>(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
	template<class T, class S>
	inline vec3_t<T> lerp(const vec3_t<T>& a, const vec3_t<T>& b, S t)  { return b * t + a * (1 - t); }
	//--------------------------------------------------------------------------------------
	#pragma endregion

	typedef vec3_t<float>	vec3;
	typedef vec3_t<float>	vec3f;
	typedef vec3_t<double>	vec3d;




#pragma region vec4i
	//--------------------------------------------------------------------------------------
	struct vec4i
	{
		vec4i() { x = y = z = w = 0; }
		vec4i(int X, int Y, int Z, int W) { x = X; y = Y; z = Z; w = W; }
		vec4i(const vec3i& v3, int W) { x = v3.x; y = v3.y; z = v3.z; w = W; }
		template<class vec3T> explicit vec4i(const vec3T& rhs) { x = (int)rhs.x; y = (int)rhs.y; z = (int)rhs.z; w = (int)rhs.w; }

		const vec4i& operator=(const vec4i& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w; return *this; }

		bool operator==(const vec4i& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w; }
		bool operator!=(const vec4i& rhs) const { return x != rhs.x || y != rhs.y || z != rhs.z || w != rhs.w; }

		int operator[](uint i) const { /*assert(i < 4);*/ return *(&x + i); }
		int& operator[](uint i) { /*assert(i < 4);*/ return *(&x + i); }

		vec3i xyz() const { return vec3i(x, y, z); }

		int x, y, z, w;
	};

	inline vec4i operator+(const vec4i& a, const vec4i& b)
	{
		return vec4i(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}
	inline vec4i operator-(const vec4i& a, const vec4i& b)
	{
		return vec4i(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}
	inline vec4i operator*(const vec4i& a, int s)
	{
		return vec4i(a.x*s, a.y*s, a.z*s, a.w*s);
	}
	inline vec4i operator*(int s, const vec4i& a)
	{
		return a*s;
	}
	inline vec4i operator/(const vec4i& a, float s)
	{
		return vec4i(int(a.x / s), int(a.y / s), int(a.z / s), int(a.w / s));
	}
	inline vec4i& operator+=(vec4i& a, const vec4i& b)
	{
		a = a + b;
		return a;
	}
	inline vec4i& operator-=(vec4i& a, const vec4i& b)
	{
		a = a - b;
		return a;
	}
	inline vec4i& operator*=(vec4i& a, int s)
	{
		a = a * s;
		return a;
	}
	inline vec4i& operator/=(vec4i& a, float s)
	{
		a = a / s;
		return a;
	}
	inline std::ostream& operator<< (std::ostream& os, const vec4i& p)
	{
		os << "{" << p.x << ", " << p.y << ", " << p.z << ", " << p.w << "}";
		return os;
	}

	inline vec4i min(const vec4i& a, const vec4i& b) { return vec4i(min(a.x, b.x), min(a.y, b.y), min(a.z, b.y), min(a.w, b.w)); }
	inline vec4i max(const vec4i& a, const vec4i& b) { return vec4i(max(a.x, b.x), max(a.y, b.y), max(a.z, b.y), max(a.w, b.w)); }
	inline vec4i clamp(const vec4i& x, const vec4i& a, const vec4i& b) { return vec4i(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z), clamp(x.w, a.w, b.w)); }
	inline vec4i abs(const vec4i& a) { return vec4i(abs(a.x), abs(a.y), abs(a.z), abs(a.w)); }
	//--------------------------------------------------------------------------------------
#pragma endregion


	
	#pragma region vec4
	//--------------------------------------------------------------------------------------
	struct vec4
	{
		vec4() { x = y = z = w = 0; }
		vec4(float X, float Y, float Z, float W) { x = X; y = Y; z = Z; w = W; }
		vec4(const vec3& v3, float W) { x = v3.x; y = v3.y; z = v3.z; w = W; }

		void operator=(const vec4& rhs) { x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w; }

		bool operator==(const vec4& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w; }
		bool operator!=(const vec4& rhs) const { return x != rhs.x || y != rhs.y || z != rhs.z || w != rhs.w; }

		float operator[](uint i) const { /*assert(i < 4);*/ return *(&x + i); }
		float& operator[](uint i) { /*assert(i < 4);*/ return *(&x + i); }

		vec3 xyz() const { return vec3(x, y, z); }
		vec2 xy() const  { return vec2(x, y); }

		float x, y, z, w;
	};

	inline vec4 operator+(const vec4& a, const vec4& b)
	{
		return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}
	inline vec4 operator-(const vec4& a, const vec4& b)
	{
		return vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}
	inline vec4 operator*(const vec4& a, float s)
	{
		return vec4(a.x*s, a.y*s, a.z*s, a.w*s);
	}
	inline vec4 operator*(float s, const vec4& a)
	{
		return a*s;
	}
	inline vec4 operator/(const vec4& a, float s)
	{
		float is = 1.0f / s;
		return a * is;
	}
	inline vec4& operator+=(vec4& a, const vec4& b)
	{
		a = a + b;
		return a;
	}
	inline vec4& operator-=(vec4& a, const vec4& b)
	{
		a = a - b;
		return a;
	}
	inline vec4& operator*=(vec4& a, float s)
	{
		a = a * s;
		return a;
	}
	inline vec4& operator/=(vec4& a, float s)
	{
		a = a / s;
		return a;
	}
	inline float dot(const vec4& a, const vec4& b)
	{
		return	a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
	}
	inline std::ostream& operator<< (std::ostream& os, const vec4& p)
	{
		os << "{" << p.x << ", " << p.y << ", " << p.z << ", " << p.w << "}";
		return os;
	}
	inline float length(const vec4& v)
	{
		return sqrtf(dot(v, v));
	}
	inline vec4 normalize(const vec4& v)
	{
		return v / length(v);
	}
	
	inline vec4 clamp(const vec4& x, const vec4& a, const vec4& b) { return vec4(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z), clamp(x.w, a.w, b.w)); }
	//--------------------------------------------------------------------------------------
	#pragma endregion





	#pragma region general functions
	//------------------------------------------------------------------------------------------------------------------------
	inline bool isnan(float v) { return v != v; }
    inline bool isinf(float v) { return std::fpclassify(v) == FP_INFINITE; }
    inline bool isfinite(float v) { return std::fpclassify(v) <= 0; }
    inline bool isnormal(float v) { return std::fpclassify(v) == FP_NORMAL; }

	inline bool isnan(const vec2& v) { return v.x != v.x || v.y != v.y; }
	inline bool isnan(const vec3& v) { return v.x != v.x || v.y != v.y || v.z != v.z; }
	inline bool isnan(const vec4& v) { return v.x != v.x || v.y != v.y || v.z != v.z || v.w != v.w; }

	inline bool isinf(const vec2& v) { return isinf(v.x) || isinf(v.y); }
	inline bool isinf(const vec3& v) { return isinf(v.x) || isinf(v.y) || isinf(v.z); }
	inline bool isinf(const vec4& v) { return isinf(v.x) || isinf(v.y) || isinf(v.z) || isinf(v.w); }

	inline bool isfinite(const vec2& v) { return isfinite(v.x) && isfinite(v.y); }
	inline bool isfinite(const vec3& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
	inline bool isfinite(const vec4& v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(v.w); }
	//------------------------------------------------------------------------------------------------------------------------
	#pragma endregion




	
	
	#pragma region mat3
	//--------------------------------------------------------------------------------------
	// mat3 class - ROW MAJOR!
	class mat3
	{
		float a[3 * 3];

	public:
		// Default constructor
		mat3()
		{}

		// Initializing constructor
		mat3(float e00, float e01, float e02,
			float e10, float e11, float e12,
			float e20, float e21, float e22)
		{
			a[0] = e00;		a[1] = e01;		a[2] = e02;
			a[3] = e10;		a[4] = e11;		a[5] = e12;
			a[6] = e20;		a[7] = e21;		a[8] = e22;
		}

		// identity matrix
		static mat3 identity()
		{
			return mat3(1, 0, 0, 0, 1, 0, 0, 0, 1 );
		}

		// Copy constructor
		mat3(const mat3& m)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] = m.a[k];
		}

		// Copy constructor with a given array (must have a length >= 4*4)
		mat3(const float* m)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] = m[k];
		}

		// Constructor with an initial value for all elements
		explicit mat3(float w)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] = w;
		}

		float& operator[](int k)
		{
			//ASSERT( (k>=0) && (k<3*3) );
			return a[k];
		}

		const float& operator[](int k) const
		{
			//ASSERT( (k>=0) && (k<3*3) );
			return a[k];
		}

		// Matrix access
		float& operator() (int r, int c)
		{
			//ASSERT( (r>=0) && (c>=0) && (r<3) && (c<3) );
			return a[r * 3 + c];
		}

		const float& operator() (int r, int c) const
		{
			//ASSERT( (r>=0) && (c>=0) && (r<3) && (c<3) );
			return a[r * 3 + c];
		}
		
		// Dereferencing operator
		operator float* ()
		{
			return a;
		}

		// Constant dereferencing operator
		operator const float* () const
		{
			return a;
		}

		// Assignment operator
		mat3& operator=(const mat3& m)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] = m.a[k];
			return (*this);
		}

		mat3& operator+=(const mat3& m)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] += m.a[k];
			return *this;
		}

		mat3& operator-=(const mat3& m)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] -= m.a[k];
			return *this;
		}

		mat3& operator*=(float w)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] *= w;
			return *this;
		}

		mat3& operator/=(float w)
		{
			for (int k = 0; k < 3 * 3; ++k) a[k] /= w;
			return *this;
		}
		
		// Matrix Multiplication
		mat3 & operator*=(const mat3& m)
		{
			mat3 result;
			for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			{
				float sum(0);
				for (int k = 0; k < 3; k++)
					sum += a[i * 3 + k] * m.a[k * 3 + j];
				result[i * 3 + j] = sum;
			}
			*this = result;
			return *this;
		}


		mat3 operator+(const mat3& m) const
		{
			mat3 res;
			for (int k = 0; k < 3 * 3; ++k) res[k] = a[k] + m.a[k];
			return res;
		}

		mat3 operator-(const mat3& m) const
		{
			mat3 res;
			for (int k = 0; k < 3 * 3; ++k) res[k] = a[k] - m.a[k];
			return res;
		}

		mat3 operator*(float w) const
		{
			mat3 res;
			for (int k = 0; k < 3 * 3; ++k) res[k] = a[k] * w;
			return res;
		}

		mat3 operator/(float w) const
		{
			float invw = 1.0f / w;
			return (*this) * invw;
		}

		// matrix vector product
		vec3 operator*(const vec3& v) const
		{
			return vec3(
				a[0] * v.x + a[1] * v.y + a[2] * v.z,
				a[3] * v.x + a[4] * v.y + a[5] * v.z,
				a[6] * v.x + a[7] * v.y + a[8] * v.z
				);
		}

		// Product of two matrices
		mat3 operator*(const mat3& m) const
		{
			mat3 res;
			for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			{
				float sum(0);
				for (int k = 0; k < 3; k++)
					sum += a[i * 3 + k] * m.a[k * 3 + j];
				res[i * 3 + j] = sum;
			}
			return res;
		}

		// Unary -
		mat3 operator-() const
		{
			mat3 res;
			for (int k = 0; k < 3 * 3; ++k) res[k] = -a[k];
			return res;
		}

		// Clear the matrix to zero
		void clear()
		{
			for (int k = 0; k < 3 * 3; ++k)
				a[k] = 0.0f;
		}

		// Transpose matrix
		void transpose()
		{
			float help;
			for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			if (i > j)
			{
				help = a[i * 3 + j];
				a[i * 3 + j] = a[j * 3 + i];
				a[j * 3 + i] = help;
			}
		}

		// get i-th row vector (base 0)
		vec3 row(uint i) const
		{
			return vec3((*this)(i, 0), (*this)(i, 1), (*this)(i, 2));
		}

		// get i-th column vector (base 0)
		vec3 col(uint i) const
		{
			return vec3((*this)(0, i), (*this)(1, i), (*this)(2, i));
		}

		// outer product a . b^T
		static mat3 outer(const vec3& a, const vec3& b)
		{
			return mat3(a.x*b.x, a.x*b.y, a.x*b.z, a.y*b.x, a.y*b.y, a.y*b.z, a.z*b.x, a.z*b.y, a.z*b.z);
		}


		float det() const
		{
			return	a[0]*a[4]*a[8] + a[1]*a[5]*a[6] + a[2]*a[3]*a[7] -
					a[2]*a[4]*a[6] - a[1]*a[3]*a[8] - a[0]*a[5]*a[7];
		}

		float trace() const
		{
			return a[0] + a[4] + a[8];
		}

		mat3 inverse() const
		{
			float d = det();
			mat3 r = mat3(
				a[4]*a[8] - a[5]*a[7], a[2]*a[7] - a[1]*a[8], a[1]*a[5] - a[2]*a[4],
				a[5]*a[6] - a[3]*a[8], a[0]*a[8] - a[2]*a[6], a[2]*a[3] - a[0]*a[5],
				a[3]*a[7] - a[4]*a[6], a[1]*a[6] - a[0]*a[7], a[0]*a[4] - a[1]*a[3]
				);
			return r / d;
		}


#pragma region eigensystem functions

		//returns the smallest eigenvector of this matrix using the inverse power method
		vec3 smallestEigenvector(uint nMaxIters = 100)
		{
			// matrix not invertible
			float d = det();
			if (d == 0)
				return vec3(1, 1, 1) / d;

			mat3 A = inverse();
			A = A * A;
			
			// find an initial vector b that is not already an eigenvector of A 
			vec3 b[3];
			b[0] = vec3(1, 0, 0);
			b[1] = vec3(0, 1, 0);
			b[2] = vec3(0, 0, 1);


			// compute 3 eigenvectors
			const float sqEpsilon = 10e-10f; 			
			for (uint i = 0; i < 3; i++)
			{
				for (uint j = 0; j < nMaxIters; j++)
				{
					vec3 bNew = normalize(A * b[j]);
					if (sqdist(bNew, b[j]) < sqEpsilon || !isfinite(bNew))
						i = nMaxIters;

					b[i] = bNew;
				}
			}

			// compute eigenvalues
			float v[3];
			for (uint i = 0; i < 3; i++)
				v[i] = length(A * b[i]);


			// take the one with the largest eigenvalue (largest, because we used the inverse matrix)
			if (v[0] >= v[1] && v[0] >= v[2]) return b[0];
			if (v[1] >= v[0] && v[1] >= v[2]) return b[1];
			return b[2];
		}
#pragma endregion
	};


	inline mat3 transpose(mat3 const & m)
	{
		mat3 result = m;
		result.transpose();
		return result;
	}
	
	inline float det(const mat3& m)
	{
		return	m(0, 0)*m(1, 1)*m(2, 2) + m(0, 1)*m(1, 2)*m(2, 0) + m(0, 2)*m(1, 0)*m(2, 1) -
			m(0, 2)*m(1, 1)*m(2, 0) - m(0, 1)*m(1, 0)*m(2, 2) - m(0, 0)*m(1, 2)*m(2, 1);
	}
		
	inline float trace(const mat3& c)
	{
		return c[0] + c[4] + c[8];
	}

	inline mat3 inverse(const mat3& m)
	{
		float d = det(m);
		mat3 r = mat3(
			m(1, 1)*m(2, 2) - m(1, 2)*m(2, 1), m(0, 2)*m(2, 1) - m(0, 1)*m(2, 2), m(0, 1)*m(1, 2) - m(0, 2)*m(1, 1),
			m(1, 2)*m(2, 0) - m(1, 0)*m(2, 2), m(0, 0)*m(2, 2) - m(0, 2)*m(2, 0), m(0, 2)*m(1, 0) - m(0, 0)*m(1, 2),
			m(1, 0)*m(2, 1) - m(1, 1)*m(2, 0), m(0, 1)*m(2, 0) - m(0, 0)*m(2, 1), m(0, 0)*m(1, 1) - m(0, 1)*m(1, 0)
			);
		return r / d;
	}

	inline mat3 operator* (float s, const mat3& m)
	{
		return m * s;
	}

	// left-multiplication of vector (v^T * m)
	inline vec3 operator* (const vec3& v, const mat3& m)
	{
		return vec3(
			v.x * m(0, 0) + v.y * m(1, 0) + v.z * m(2, 0),
			v.x * m(0, 1) + v.y * m(1, 1) + v.z * m(2, 1),
			v.x * m(0, 2) + v.y * m(1, 2) + v.z * m(2, 2)
		);
	}

	inline std::ostream& operator<< (std::ostream& os, const mat3& m)
	{
		os << "{{" << m(0, 0) << ", " << m(0, 1) << ", " << m(0, 2) << "}, "
			<< "{" << m(1, 0) << ", " << m(1, 1) << ", " << m(1, 2) << "}, "
			<< "{" << m(2, 0) << ", " << m(2, 1) << ", " << m(2, 2) << "}}";

		return os;
	}
	//--------------------------------------------------------------------------------------
	#pragma endregion
	





	#pragma region mat4
	//--------------------------------------------------------------------------------------
	// mat4 class - ROW MAJOR!
	class mat4
	{
		float a[4 * 4];

	public:
		// Default constructor
		mat4()
		{}

		// Initializing constructor
		mat4(float e00, float e01, float e02, float e03,
			float e10, float e11, float e12, float e13,
			float e20, float e21, float e22, float e23,
			float e30, float e31, float e32, float e33)
		{
			a[0] = e00;		a[1] = e01;		a[2] = e02;		a[3] = e03;
			a[4] = e10;		a[5] = e11;		a[6] = e12;		a[7] = e13;
			a[8] = e20;		a[9] = e21;		a[10] = e22;	a[11] = e23;
			a[12] = e30;	a[13] = e31;	a[14] = e32;	a[15] = e33;
		}

		// identity matrix
		static mat4 identity()
		{
			return mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		}

		// Copy constructor
		mat4(const mat4& m)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] = m.a[k];
		}

		// Copy constructor with a given array (must have a length >= 4*4)
		mat4(const float* m)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] = m[k];
		}

		// Constructor with upper left 3x3 matrix provided, optional constant value for remaining elements
		mat4(const mat3& m, float w = 0)
		{
			a[0] = m[0];	a[1] = m[1];	a[2] = m[2];	a[3] = w;
			a[4] = m[3];	a[5] = m[4];	a[6] = m[5];	a[7] = w;
			a[8] = m[6];	a[9] = m[7];	a[10] = m[8];	a[11] = w;
			a[12] = w;		a[13] = w;		a[14] = w;		a[15] = w;
		}

		// Constructor with an initial value for all elements
		explicit mat4(float w)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] = w;
		}

		float& operator[](int k)
		{
			//ASSERT( (k>=0) && (k<4*4) );
			return a[k];
		}

		const float& operator[](int k) const
		{
			//ASSERT( (k>=0) && (k<4*4) );
			return a[k];
		}

		// Matrix access
		float& operator() (int r, int c)
		{
			//ASSERT( (r>=0) && (c>=0) && (r<4) && (c<4) );
			return a[r * 4 + c];
		}

		const float& operator() (int r, int c) const
		{
			//ASSERT( (r>=0) && (c>=0) && (r<4) && (c<4) );
			return a[r * 4 + c];
		}

		// get i-th row vector (base 0)
		vec4 row(uint i) const
		{
			return vec4((*this)(i, 0), (*this)(i, 1), (*this)(i, 2), (*this)(i, 3));
		}

		// get i-th column vector (base 0)
		vec4 col(uint i) const
		{
			return vec4((*this)(0, i), (*this)(1, i), (*this)(2, i), (*this)(3,i));
		}
		
		// Dereferencing operator
		operator float* ()
		{
			return a;
		}

		// Constant dereferencing operator
		operator const float* () const
		{
			return a;
		}

		// Comparison operators
		bool operator==(const mat4& m) const
		{
			for (int k = 0; k < 4 * 4; ++k) 
				if (a[k] != m.a[k])
					return false;
			return true;
		}
		bool operator!=(const mat4& m) const
		{
			return !(*this == m);
		}

		// Assignment operator
		mat4& operator=(const mat4& m)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] = m.a[k];
			return (*this);
		}

		mat4& operator+=(const mat4& m)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] += m.a[k];
			return *this;
		}

		mat4& operator-=(const mat4& m)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] -= m.a[k];
			return *this;
		}

		mat4& operator*=(float w)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] *= w;
			return *this;
		}

		mat4& operator/=(float w)
		{
			for (int k = 0; k < 4 * 4; ++k) a[k] /= w;
			return *this;
		}

		mat3 toMat3() const
		{
			return mat3(a[0], a[1], a[2], a[4], a[5], a[6], a[8], a[9], a[10]);
		}

		// Matrix Multiplication
		mat4 & operator*=(const mat4& m)
		{
			mat4 result;
			for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
			{
				float sum(0);
				for (int k = 0; k < 4; k++)
					sum += a[i * 4 + k] * m.a[k * 4 + j];
				result[i * 4 + j] = sum;
			}
			*this = result;
			return *this;
		}
		
		mat4 operator+(const mat4& m) const
		{
			mat4 res;
			for (int k = 0; k < 4 * 4; ++k) res[k] = a[k] + m.a[k];
			return res;
		}

		mat4 operator-(const mat4& m) const
		{
			mat4 res;
			for (int k = 0; k < 4 * 4; ++k) res[k] = a[k] - m.a[k];
			return res;
		}

		mat4 operator*(float w) const
		{
			mat4 res;
			for (int k = 0; k < 4 * 4; ++k) res[k] = a[k] * w;
			return res;
		}

		mat4 operator/(float w) const
		{
			float invw = 1.0f / w;
			return (*this) * invw;
		}

		// matrix vector product
		vec4 operator*(const vec4& v) const
		{
			return vec4(
				a[0] * v.x + a[1] * v.y + a[2] * v.z + a[3] * v.w,
				a[4] * v.x + a[5] * v.y + a[6] * v.z + a[7] * v.w,
				a[8] * v.x + a[9] * v.y + a[10] * v.z + a[11] * v.w,
				a[12] * v.x + a[13] * v.y + a[14] * v.z + a[15] * v.w
				);
		}

		// Product of two matrices
		mat4 operator*(const mat4& m) const
		{
			mat4 res;
			for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
			{
				float sum(0);
				for (int k = 0; k < 4; k++)
					sum += a[i * 4 + k] * m.a[k * 4 + j];
				res[i * 4 + j] = sum;
			}
			return res;
		}

		// Unary -
		mat4 operator-() const
		{
			mat4 res;
			for (int k = 0; k < 4 * 4; ++k) res[k] = -a[k];
			return res;
		}

		// Clear the matrix to zero
		void clear()
		{
			for (int k = 0; k < 4 * 4; ++k)
				a[k] = 0.0f;
		}

		// Transpose matrix
		void transpose()
		{
			float help;
			for (int i = 0; i < 4; ++i)
			for (int j = 0; j<4; ++j)
			if (i > j)
			{
				help = a[i * 4 + j];
				a[i * 4 + j] = a[j * 4 + i];
				a[j * 4 + i] = help;
			}
		}
	};

	inline mat4 translationMatrix(const vec3& t)
	{
		return mat4(
			1, 0, 0, t.x,
			0, 1, 0, t.y,
			0, 0, 1, t.z,
			0, 0, 0, 1
		);
	}

	inline mat4 transpose(mat4 const & m)
	{
		mat4 result = m;
		result.transpose();
		return result;
	}
	
	inline float det(const mat4& m)
	{
		return	m(3, 0)*m(2, 1)*m(1, 2)*m(0, 3) - m(2, 0)*m(3, 1)*m(1, 2)*m(0, 3) - m(3, 0)*m(1, 1)*m(2, 2)*m(0, 3) + m(1, 0)*m(3, 1)*m(2, 2)*m(0, 3) +
			m(2, 0)*m(1, 1)*m(3, 2)*m(0, 3) - m(1, 0)*m(2, 1)*m(3, 2)*m(0, 3) - m(3, 0)*m(2, 1)*m(0, 2)*m(1, 3) + m(2, 0)*m(3, 1)*m(0, 2)*m(1, 3) +
			m(3, 0)*m(0, 1)*m(2, 2)*m(1, 3) - m(0, 0)*m(3, 1)*m(2, 2)*m(1, 3) - m(2, 0)*m(0, 1)*m(3, 2)*m(1, 3) + m(0, 0)*m(2, 1)*m(3, 2)*m(1, 3) +
			m(3, 0)*m(1, 1)*m(0, 2)*m(2, 3) - m(1, 0)*m(3, 1)*m(0, 2)*m(2, 3) - m(3, 0)*m(0, 1)*m(1, 2)*m(2, 3) + m(0, 0)*m(3, 1)*m(1, 2)*m(2, 3) +
			m(1, 0)*m(0, 1)*m(3, 2)*m(2, 3) - m(0, 0)*m(1, 1)*m(3, 2)*m(2, 3) - m(2, 0)*m(1, 1)*m(0, 2)*m(3, 3) + m(1, 0)*m(2, 1)*m(0, 2)*m(3, 3) +
			m(2, 0)*m(0, 1)*m(1, 2)*m(3, 3) - m(0, 0)*m(2, 1)*m(1, 2)*m(3, 3) - m(1, 0)*m(0, 1)*m(2, 2)*m(3, 3) + m(0, 0)*m(1, 1)*m(2, 2)*m(3, 3);
	}
	
	inline mat4 inverse(const mat4& m)
	{
		float d = det(m);
		mat4 r = mat4(
			m(2, 1)*m(3, 2)*m(1, 3) - m(3, 1)*m(2, 2)*m(1, 3) + m(3, 1)*m(1, 2)*m(2, 3) - m(1, 1)*m(3, 2)*m(2, 3) - m(2, 1)*m(1, 2)*m(3, 3) + m(1, 1)*m(2, 2)*m(3, 3),
			m(3, 1)*m(2, 2)*m(0, 3) - m(2, 1)*m(3, 2)*m(0, 3) - m(3, 1)*m(0, 2)*m(2, 3) + m(0, 1)*m(3, 2)*m(2, 3) + m(2, 1)*m(0, 2)*m(3, 3) - m(0, 1)*m(2, 2)*m(3, 3),
			m(1, 1)*m(3, 2)*m(0, 3) - m(3, 1)*m(1, 2)*m(0, 3) + m(3, 1)*m(0, 2)*m(1, 3) - m(0, 1)*m(3, 2)*m(1, 3) - m(1, 1)*m(0, 2)*m(3, 3) + m(0, 1)*m(1, 2)*m(3, 3),
			m(2, 1)*m(1, 2)*m(0, 3) - m(1, 1)*m(2, 2)*m(0, 3) - m(2, 1)*m(0, 2)*m(1, 3) + m(0, 1)*m(2, 2)*m(1, 3) + m(1, 1)*m(0, 2)*m(2, 3) - m(0, 1)*m(1, 2)*m(2, 3),

			m(3, 0)*m(2, 2)*m(1, 3) - m(2, 0)*m(3, 2)*m(1, 3) - m(3, 0)*m(1, 2)*m(2, 3) + m(1, 0)*m(3, 2)*m(2, 3) + m(2, 0)*m(1, 2)*m(3, 3) - m(1, 0)*m(2, 2)*m(3, 3),
			m(2, 0)*m(3, 2)*m(0, 3) - m(3, 0)*m(2, 2)*m(0, 3) + m(3, 0)*m(0, 2)*m(2, 3) - m(0, 0)*m(3, 2)*m(2, 3) - m(2, 0)*m(0, 2)*m(3, 3) + m(0, 0)*m(2, 2)*m(3, 3),
			m(3, 0)*m(1, 2)*m(0, 3) - m(1, 0)*m(3, 2)*m(0, 3) - m(3, 0)*m(0, 2)*m(1, 3) + m(0, 0)*m(3, 2)*m(1, 3) + m(1, 0)*m(0, 2)*m(3, 3) - m(0, 0)*m(1, 2)*m(3, 3),
			m(1, 0)*m(2, 2)*m(0, 3) - m(2, 0)*m(1, 2)*m(0, 3) + m(2, 0)*m(0, 2)*m(1, 3) - m(0, 0)*m(2, 2)*m(1, 3) - m(1, 0)*m(0, 2)*m(2, 3) + m(0, 0)*m(1, 2)*m(2, 3),

			m(2, 0)*m(3, 1)*m(1, 3) - m(3, 0)*m(2, 1)*m(1, 3) + m(3, 0)*m(1, 1)*m(2, 3) - m(1, 0)*m(3, 1)*m(2, 3) - m(2, 0)*m(1, 1)*m(3, 3) + m(1, 0)*m(2, 1)*m(3, 3),
			m(3, 0)*m(2, 1)*m(0, 3) - m(2, 0)*m(3, 1)*m(0, 3) - m(3, 0)*m(0, 1)*m(2, 3) + m(0, 0)*m(3, 1)*m(2, 3) + m(2, 0)*m(0, 1)*m(3, 3) - m(0, 0)*m(2, 1)*m(3, 3),
			m(1, 0)*m(3, 1)*m(0, 3) - m(3, 0)*m(1, 1)*m(0, 3) + m(3, 0)*m(0, 1)*m(1, 3) - m(0, 0)*m(3, 1)*m(1, 3) - m(1, 0)*m(0, 1)*m(3, 3) + m(0, 0)*m(1, 1)*m(3, 3),
			m(2, 0)*m(1, 1)*m(0, 3) - m(1, 0)*m(2, 1)*m(0, 3) - m(2, 0)*m(0, 1)*m(1, 3) + m(0, 0)*m(2, 1)*m(1, 3) + m(1, 0)*m(0, 1)*m(2, 3) - m(0, 0)*m(1, 1)*m(2, 3),

			m(3, 0)*m(2, 1)*m(1, 2) - m(2, 0)*m(3, 1)*m(1, 2) - m(3, 0)*m(1, 1)*m(2, 2) + m(1, 0)*m(3, 1)*m(2, 2) + m(2, 0)*m(1, 1)*m(3, 2) - m(1, 0)*m(2, 1)*m(3, 2),
			m(2, 0)*m(3, 1)*m(0, 2) - m(3, 0)*m(2, 1)*m(0, 2) + m(3, 0)*m(0, 1)*m(2, 2) - m(0, 0)*m(3, 1)*m(2, 2) - m(2, 0)*m(0, 1)*m(3, 2) + m(0, 0)*m(2, 1)*m(3, 2),
			m(3, 0)*m(1, 1)*m(0, 2) - m(1, 0)*m(3, 1)*m(0, 2) - m(3, 0)*m(0, 1)*m(1, 2) + m(0, 0)*m(3, 1)*m(1, 2) + m(1, 0)*m(0, 1)*m(3, 2) - m(0, 0)*m(1, 1)*m(3, 2),
			m(1, 0)*m(2, 1)*m(0, 2) - m(2, 0)*m(1, 1)*m(0, 2) + m(2, 0)*m(0, 1)*m(1, 2) - m(0, 0)*m(2, 1)*m(1, 2) - m(1, 0)*m(0, 1)*m(2, 2) + m(0, 0)*m(1, 1)*m(2, 2)
		);

		return r / d;
	}
	
	inline std::ostream& operator<< (std::ostream& os, const mat4& m)
	{
		os << "{{" << m(0, 0) << ", " << m(0, 1) << ", " << m(0, 2) << ", " << m(0, 3) << "}, "
			<< "{" << m(1, 0) << ", " << m(1, 1) << ", " << m(1, 2) << ", " << m(1, 3) << "}, "
			<< "{" << m(2, 0) << ", " << m(2, 1) << ", " << m(2, 2) << ", " << m(2, 3) << "}, "
			<< "{" << m(3, 0) << ", " << m(3, 1) << ", " << m(3, 2) << ", " << m(3, 3) << "}}";

		return os;
	}
	//--------------------------------------------------------------------------------------
	#pragma endregion




#pragma region mat2
	//--------------------------------------------------------------------------------------
	// mat2 class - ROW MAJOR!
	class mat2
	{
		float a[2 * 2];

	public:
		// Default constructor
		mat2()
		{}

		// Initializing constructor
		mat2(float e00, float e01,
			 float e10, float e11)
		{
			a[0] = e00;		a[1] = e01;
			a[2] = e10;		a[3] = e11;
		}

		// identity matrix
		static mat2 identity()
		{
			return mat2(1, 0, 0, 1);
		}

		// Copy constructor
		mat2(const mat2& m)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] = m.a[k];
		}

		// Copy constructor with a given array (must have a length >= 2*2)
		mat2(const float* m)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] = m[k];
		}

		// Constructor with an initial value for all elements
		explicit mat2(float w)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] = w;
		}

		float& operator[](int k)
		{
			//ASSERT( (k>=0) && (k<2*2) );
			return a[k];
		}

		const float& operator[](int k) const
		{
			//ASSERT( (k>=0) && (k<2*2) );
			return a[k];
		}

		// Matrix access
		float& operator() (int r, int c)
		{
			//ASSERT( (r>=0) && (c>=0) && (r<2) && (c<2) );
			return a[r * 2 + c];
		}

		const float& operator() (int r, int c) const
		{
			//ASSERT( (r>=0) && (c>=0) && (r<2) && (c<2) );
			return a[r * 2 + c];
		}

		// Dereferencing operator
		operator float* ()
		{
			return a;
		}

		// Constant dereferencing operator
		operator const float* () const
		{
			return a;
		}

		// Assignment operator
		mat2& operator=(const mat2& m)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] = m.a[k];
			return (*this);
		}

		mat2& operator+=(const mat2& m)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] += m.a[k];
			return *this;
		}

		mat2& operator-=(const mat2& m)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] -= m.a[k];
			return *this;
		}

		mat2& operator*=(float w)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] *= w;
			return *this;
		}

		mat2& operator/=(float w)
		{
			for (int k = 0; k < 2 * 2; ++k) a[k] /= w;
			return *this;
		}

		// Matrix Multiplication
		mat2 & operator*=(const mat2& m)
		{
			*this = mat2(
				a[0] * m.a[0] + a[1] * m.a[2], a[0] * m.a[1] + a[1] * m.a[3],
				a[2] * m.a[0] + a[3] * m.a[2], a[2] * m.a[1] + a[3] * m.a[3]);
			return *this;
		}


		mat2 operator+(const mat2& m) const
		{
			mat2 res;
			for (int k = 0; k < 2 * 2; ++k) res[k] = a[k] + m.a[k];
			return res;
		}

		mat2 operator-(const mat2& m) const
		{
			mat2 res;
			for (int k = 0; k < 2 * 2; ++k) res[k] = a[k] - m.a[k];
			return res;
		}

		mat2 operator*(float w) const
		{
			mat2 res;
			for (int k = 0; k < 2 * 2; ++k) res[k] = a[k] * w;
			return res;
		}

		mat2 operator/(float w) const
		{
			float invw = 1.0f / w;
			return (*this) * invw;
		}

		// matrix vector product
		vec2 operator*(const vec2& v) const
		{
			return vec2(
				a[0] * v.x + a[1] * v.y,
				a[2] * v.x + a[3] * v.y
			);
		}

		// Product of two matrices
		mat2 operator*(const mat2& m) const
		{
			return mat2(
				a[0] * m[0] + a[1] * m[2], a[0] * m[1] + a[1] * m[3],
				a[2] * m[0] + a[3] * m[2], a[2] * m[1] + a[3] * m[3]);
		}

		// Unary -
		mat3 operator-() const
		{
			mat3 res;
			for (int k = 0; k < 3 * 3; ++k) res[k] = -a[k];
			return res;
		}

		// Clear the matrix to zero
		void clear()
		{
			for (int k = 0; k < 3 * 3; ++k)
				a[k] = 0.0f;
		}

		// Transpose matrix
		void transpose()
		{
			std::swap(a[1], a[2]);
		}

		// get i-th row vector (base 0)
		vec2 row(uint i) const
		{
			return vec2((*this)(i, 0), (*this)(i, 1));
		}

		// get i-th column vector (base 0)
		vec2 col(uint i) const
		{
			return vec2((*this)(0, i), (*this)(1, i));
		}

		// outer product a . b^T
		static mat2 outer(const vec2& a, const vec2& b)
		{
			return mat2(a.x*b.x, a.x*b.y, a.y*b.x, a.y*b.y);
		}
		

		#pragma region eigenvalue decomposition methods
		// returns the eigenvalues in the elements x,y of a vec2 in ascending order
		// from: http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
		vec2 eigenvalues() const
		{
			// TODO: Change to double if precision problems occur!
			float t = a[0] + a[3];
			float d = a[0]*a[3] - a[1]*a[2];
			float a = t * 0.5f;
			float b = sqrtf(t*t * 0.25f - d);
			return vec2(a - b, a + b);
		}

		// returns the eigenvalues in the elements x,y of out_evalues in ascending order, and their associated eigenvectors in out_evectors
		void eigensystem(vec2& out_evalues, vec2 out_evectors[2]) const
		{
			out_evalues = eigenvalues();

			if (a[2] != 0)
			{
				out_evectors[0] = normalize(vec2(out_evalues.x - a[3], a[2]));
				out_evectors[1] = normalize(vec2(out_evalues.y - a[3], a[2]));
			}
			else if (a[1] != 0)
			{
				out_evectors[0] = normalize(vec2(a[1], out_evalues.x - a[0]));
				out_evectors[1] = normalize(vec2(a[1], out_evalues.y - a[0]));
			}
			else
			{
				// trivial cases -> robust handling
				if (a[0] <= a[3])
				{
					out_evalues = vec2(a[0], a[3]);
					out_evectors[0] = vec2(1, 0);
					out_evectors[1] = vec2(0, 1);
				}
				else
				{
					out_evalues = vec2(a[3], a[0]);
					out_evectors[0] = vec2(0, 1);
					out_evectors[1] = vec2(1, 0);
				}
			}
		}

		// returns the eigenvalues in the elements x,y of out_evalues in ascending order, and their associated 
		// eigenvectors in the columns of out_evectors (in ascending order accordingly)
		void eigensystem(vec2& out_evalues, mat2& out_evectors) const
		{
			vec2 evecs[2];
			eigensystem(out_evalues, evecs);
			out_evectors = mat2(
				evecs[0].x, evecs[1].x,
				evecs[0].y, evecs[1].y
				);
		}

		// returns the eigenvectors in the columns of out_evectors, in ascending order of their respective eigenvalues
		void eigenvectors(vec2 out_evectors[2]) const
		{
			vec2 evalues;
			eigensystem(evalues, out_evectors);
		}

		// returns a matrix containing the eigenvectors in its columns, ascendingly ordered by their respective eigenvalues
		mat2 eigenvectors() const
		{
			vec2 evecs[2];
			eigenvectors(evecs);
			return mat2(
				evecs[0].x, evecs[1].x,
				evecs[0].y, evecs[1].y
				);
		}
		#pragma endregion
	};


	inline mat2 transpose(mat2 const & m)
	{
		mat2 result = m;
		result.transpose();
		return result;
	}

	inline float det(const mat2& m)
	{
		return	m[0] * m[3] - m[1] * m[2];
	}

	inline float trace(const mat2& c)
	{
		return c[0] + c[3];
	}

	inline mat2 inverse(const mat2& m)
	{
		return mat2(m[3], -m[1], -m[2], m[0]) / det(m);
	}

	inline mat2 operator* (float s, const mat2& m)
	{
		return m * s;
	}

	// left-multiplication of vector (v^T * m)
	inline vec2 operator* (const vec2& v, const mat2& m)
	{
		return vec2(
			v.x * m[0] + v.y * m[1],
			v.x * m[2] + v.y * m[3]
		);
	}

	inline std::ostream& operator<< (std::ostream& os, const mat2& m)
	{
		os << "{{" << m(0, 0) << ", " << m(0, 1) << "},"
		   <<  "{" << m(1, 0) << ", " << m(1, 1) << "}}";
		return os;
	}
	//--------------------------------------------------------------------------------------
#pragma endregion



	
#pragma region smat2
	//--------------------------------------------------------------------------------------
	// symmetric 2x2 matrix given by its upper triangle matrix entries
	struct smat2
	{
		float e00, e01, e11;

		smat2() {}

		smat2(float _e00, float _e01, float _e11) :
			e00(_e00), e01(_e01), e11(_e11)
		{}

		// Creates a symmetric 2x2 matrix from the upper right triangle matrix entries of the given 2x2 matrix.
		// Note: This is explicit, since implicit conversion from mat2 to smat2 can raise a lot of errors!
		explicit smat2(const mat2& m) :
			e00(m(0, 0)), e01(m(0, 1)), e11(m(1, 1))
		{
		}

		mat2 toMat2() const
		{
			return mat2(e00, e01, e01, e11);
		}

		smat2 operator+ (const smat2& c) const
		{
			return smat2(e00 + c.e00, e01 + c.e01, e11 + c.e11);
		}

		smat2 operator- (const smat2& c) const
		{
			return smat2(e00 - c.e00, e01 - c.e01, e11 - c.e11);
		}

		smat2 operator* (float s) const
		{
			return smat2(e00*s, e01*s, e11*s);
		}

		smat2 operator/ (float s) const
		{
			float inv_s = 1.0f / s;
			return smat2(e00*inv_s, e01*inv_s, e11*inv_s);
		}

		smat2& operator+= (const smat2& c)
		{
			e00 += c.e00;	e01 += c.e01;	e11 += c.e11;
			return *this;
		}

		smat2& operator-= (const smat2& c)
		{
			e00 -= c.e00;	e01 -= c.e01;	e11 -= c.e11;
			return *this;
		}
		smat2& operator*= (float s)
		{
			e00 *= s;
			e01 *= s;
			e11 *= s;
			return *this;
		}
		smat2& operator/= (float s)
		{
			float inv_s = 1.0f / s;
			return *this *= inv_s;
		}
		vec2 operator* (const vec2& v) const
		{
			return vec2(e00*v.x + e01*v.y, e01*v.x + e11*v.y);
		}
		mat2 operator* (const smat2& c) const
		{
			return mat2(
				e00*c.e00 + e01*c.e01, e00*c.e01 + e01*c.e11,
				e01*c.e00 + e11*c.e01, e01*c.e01 + e11*c.e11
			);
		}
		mat2 operator* (const mat2& a) const
		{
			return mat2(a[0] * e00 + a[2] * e01, a[1] * e00 + a[3] * e01,
						a[0] * e01 + a[2] * e11, a[1] * e01 + a[3] * e11);
		}

		// Unary -
		smat2 operator-() const
		{
			return smat2(-e00, -e01, -e11);
		}

		std::string toString() const
		{
			std::stringstream s;
			s << "{" << e00 << ", " << e01 << ", " << e11 << "}";
			return s.str();
		}

		static smat2 identity()
		{
			return smat2(1, 0, 1);
		};

		static smat2 zero()
		{
			return smat2(0, 0, 0);
		}

		static smat2 outer(const vec2& v)
		{
			return smat2(v.x*v.x, v.x*v.y, v.y*v.y);
		}

		// creates a diagonal matrix using the components of v
		static smat2 diag(const vec2& v)
		{
			return smat2(v.x, 0, v.y);
		}

		// creates a diagonal matrix using the given diagonal entries
		static smat2 diag(float d11, float d22)
		{
			return smat2(d11, 0, d22);
		}

#pragma region eigenvalue decomposition methods
		// returns the eigenvalues in the elements x,y of a vec2 in ascending order
		// from: http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
		vec2 eigenvalues() const
		{
			// TODO: Change to double if precision problems occur!
			float t = e00 + e11;
			float d = e00*e11 - e01*e01;
			float a = t * 0.5f;
			float b = sqrtf(t*t * 0.25f - d);
			return vec2(a - b, a + b);
		}

		// returns the eigenvalues in the elements x,y of out_evalues in ascending order, and their associated eigenvectors in out_evectors
		void eigensystem(vec2& out_evalues, vec2 out_evectors[2]) const
		{
			if (e01 == 0)
			{
				// trivial cases -> robust handling
				if (e00 <= e11)
				{
					out_evalues = vec2(e00, e11);
					out_evectors[0] = vec2(1, 0);
					out_evectors[1] = vec2(0, 1);
				}
				else
				{
					out_evalues = vec2(e11, e00);
					out_evectors[0] = vec2(0, 1);
					out_evectors[1] = vec2(1, 0);
				}
			}
			else
			{
				out_evalues = eigenvalues();
				out_evectors[0] = normalize(vec2(out_evalues.x - e11, e01));
				out_evectors[1] = vec2(out_evectors[0].y, -out_evectors[0].x);	// can just take the orthogonal vector of out_evectors[0], since we work on symmetric matrix
			}
		}

		// returns the eigenvalues in the elements x,y of out_evalues in ascending order, and their associated 
		// eigenvectors in the columns of out_evectors (in ascending order accordingly)
		void eigensystem(vec2& out_evalues, mat2& out_evectors) const
		{
			vec2 evecs[2];
			eigensystem(out_evalues, evecs);
			out_evectors = mat2(
				evecs[0].x, evecs[1].x,
				evecs[0].y, evecs[1].y
			);
		}

		// returns the eigenvectors in the columns of out_evectors, in ascending order of their respective eigenvalues
		void eigenvectors(vec2 out_evectors[2]) const
		{
			vec2 evalues;
			eigensystem(evalues, out_evectors);
		}

		// returns a matrix containing the eigenvectors in its columns, ascendingly ordered by their respective eigenvalues
		mat2 eigenvectors() const
		{
			vec2 evecs[2];
			eigenvectors(evecs);
			return mat2(
				evecs[0].x, evecs[1].x,
				evecs[0].y, evecs[1].y
			);
		}
#pragma endregion
	};



#pragma region smat2	non-member operators

	inline smat2 operator* (float s, const smat2& m)
	{
		return m * s;
	}

	inline float det(const smat2& c)
	{
		return c.e00*c.e11 - c.e01*c.e01;
	}

	inline smat2 inverse(const smat2& c)
	{
		return smat2(c.e11, -c.e01, c.e00) / det(c);
	}

	inline float trace(const smat2& c)
	{
		return c.e00 + c.e11;
	}

	inline std::ostream& operator<< (std::ostream& os, const smat2& cov)
	{
		return (os << cov.toString());
	}

	inline mat2 operator* (const mat2& m, const smat2& sm)
	{
		return mat2(
			m[0] * sm.e00 + m[1] * sm.e01, m[0] * sm.e01 + m[1] * sm.e11,
			m[2] * sm.e00 + m[3] * sm.e01, m[2] * sm.e01 + m[3] * sm.e11
			);
	}

	// returns the symmetric product S.T.S of symmetric matrices S and T. The result is symmetric again.
	inline smat2 smult(const smat2& T, const smat2& S)
	{
		return smat2(S * T * S);
	}

	// returns the symmetric product L.S.L' of a symmetric matrix S and a non-symmetric matrix L. The result is symmetric again.
	inline smat2 smult(const smat2& S, const mat2& L)
	{
		return smat2(L * S * transpose(L));
	}


	inline mat2 operator+ (const smat2& sm, const mat2& m)
	{
		return sm.toMat2() + m;
	}

	inline mat2 operator+ (const mat2& m, const smat2& sm)
	{
		return m + sm.toMat2();
	}

	inline mat2 operator- (const smat2& sm, const mat2& m)
	{
		return sm.toMat2() - m;
	}

	inline mat2 operator- (const mat2& m, const smat2& sm)
	{
		return m - sm.toMat2();
	}

	// left-multiplication of vector (v^T * m)
	inline vec2 operator* (const vec2& v, const smat2& m)
	{
		// similar to right multiplication for symmetric matrices
		return m * v;
	}

#pragma endregion

	//--------------------------------------------------------------------------------------
#pragma endregion





	#pragma region smat3
	//--------------------------------------------------------------------------------------
	// symmetric 3x3 matrix given by its upper triangle matrix entries
	struct smat3
	{
		float e00, e01, e02, e11, e12, e22;
		
		smat3() {}

		smat3(float _e00, float _e01, float _e02, float _e11, float _e12, float _e22) :
			e00(_e00), e01(_e01), e02(_e02), e11(_e11), e12(_e12), e22(_e22)
		{}
		
		// Creates a symmetric 3x3 matrix from the upper right triangle matrix entries of the given 3x3 matrix.
		// Note: This is explicit, since implicit conversion from mat3 to smat3 can raise a lot of errors!
		explicit smat3(const mat3& m) :
			e00(m(0, 0)), e01(m(0, 1)), e02(m(0, 2)), e11(m(1, 1)), e12(m(1, 2)), e22(m(2, 2))
		{
		}

		mat3 toMat3() const
		{
			return mat3(e00, e01, e02, e01, e11, e12, e02, e12, e22);
		}

		smat3 operator+ (const smat3& c) const
		{
			return smat3(e00 + c.e00, e01 + c.e01, e02 + c.e02, e11 + c.e11, e12 + c.e12, e22 + c.e22);
		}

		smat3 operator- (const smat3& c) const
		{
			return smat3(e00 - c.e00, e01 - c.e01, e02 - c.e02, e11 - c.e11, e12 - c.e12, e22 - c.e22);
		}

		smat3 operator* (float s) const
		{
			return smat3(e00*s, e01*s, e02*s, e11*s, e12*s, e22*s);
		}

		smat3 operator/ (float s) const
		{
			float inv_s = 1.0f / s;
			return smat3(e00*inv_s, e01*inv_s, e02*inv_s, e11*inv_s, e12*inv_s, e22*inv_s);
		}

		smat3& operator+= (const smat3& c)
		{
			e00 += c.e00;	e01 += c.e01;	e02 += c.e02;
			e11 += c.e11;	e12 += c.e12;	e22 += c.e22;
			return *this;
		}

		smat3& operator-= (const smat3& c)
		{
			e00 -= c.e00;	e01 -= c.e01;	e02 -= c.e02;
			e11 -= c.e11;	e12 -= c.e12;	e22 -= c.e22;
			return *this;
		}
		smat3& operator*= (float s)
		{
			for (float* e = &e00; e <= &e22; ++e)
				*e *= s;
			return *this;
		}
		smat3& operator/= (float s)
		{
			float inv_s = 1.0f / s;
			return *this *= inv_s;
		}
		vec3 operator* (const vec3& v) const
		{
			return vec3(e00*v.x + e01*v.y + e02*v.z, e01*v.x + e11*v.y + e12*v.z, e02*v.x + e12*v.y + e22*v.z);
		}
		mat3 operator* (const smat3& c) const
		{
			return mat3(
				c.e00*e00 + c.e01*e01 + c.e02*e02, c.e01*e00 + c.e11*e01 + c.e12*e02, c.e02*e00 + c.e12*e01 + c.e22*e02,
				c.e00*e01 + c.e01*e11 + c.e02*e12, c.e01*e01 + c.e11*e11 + c.e12*e12, c.e02*e01 + c.e12*e11 + c.e22*e12,
				c.e00*e02 + c.e01*e12 + c.e02*e22, c.e01*e02 + c.e11*e12 + c.e12*e22, c.e02*e02 + c.e12*e12 + c.e22*e22
				);
		}
		mat3 operator* (const mat3& a) const
		{
			return mat3(a[0] * e00 + a[3] * e01 + a[6] * e02, a[1] * e00 + a[4] * e01 + a[7] * e02, a[2] * e00 + a[5] * e01 + a[8] * e02,
						a[0] * e01 + a[3] * e11 + a[6] * e12, a[1] * e01 + a[4] * e11 + a[7] * e12, a[2] * e01 + a[5] * e11 + a[8] * e12,
						a[0] * e02 + a[3] * e12 + a[6] * e22, a[1] * e02 + a[4] * e12 + a[7] * e22, a[2] * e02 + a[5] * e12 + a[8] * e22);
		}

		// Unary -
		smat3 operator-() const
		{
			return smat3(-e00, -e01, -e02, -e11, -e12, -e22);
		}

		// returns the cofactor a_ij of the matrix (determinant of the submatrix resulting from 
		// deleting the i-th row and j-th column, times a sign. indices are 0-based.
		// https://de.wikipedia.org/wiki/Minor_(Lineare_Algebra))
		float cofactor(uint i, uint j) const
		{
			// compute minor of this matrix
			mat3 M = toMat3();
			for (uint k = 0; k < 3; k++)
				M(i, k) = M(k, j) = 0;
			M(i, j) = 1.0f;

			// return its determinant
			return det(M);
		}

		// returns the largest absolute value of all matrix elements
		float maxMagnitude() const
		{
			return max(abs(e00), max(abs(e01), max(abs(e02), max(abs(e11), max(abs(e12), abs(e22))))));
		}

		std::string toString() const
		{
			std::stringstream s;
			s << "{" << e00 << ", " << e01 << ", " << e02 << ", " << e11 << ", " << e12 << ", " << e22 << "}";
			return s.str();
		}

		static smat3 identity()
		{
			return smat3(1, 0, 0, 1, 0, 1);
		}
		
		static smat3 zero()
		{
			return smat3(0, 0, 0, 0, 0, 0);
		}

		static smat3 outer(const vec3& v) 
		{
			return smat3 (v.x*v.x, v.x*v.y, v.x*v.z, v.y*v.y, v.y*v.z, v.z*v.z);
		}

		// creates a diagonal matrix using the components of v
		static smat3 diag(const vec3& v) 
		{
			return smat3 (v.x, 0, 0, v.y, 0, v.z);
		}

		// creates a diagonal matrix using the given diagonal entries
		static smat3 diag(float d11, float d22, float d33)
		{
			return smat3(d11, 0, 0, d22, 0, d33);
		}

#pragma region eigenvalue decomposition methods
		// Eigenvalue decomposition adopted from:
		// Dave Eberly, 2011. Eigensystems for 3×3 Symmetric Matrices (Revisited). Online, Geometric Tools, 2012
	private:
		int computeRank(mat3& M, float epsilon = 0) const
		{
			// Compute the maximum magnitude matrix entry.
			float abs, save, max = -1;
			int row, col, maxRow = -1, maxCol = -1;
			for (row = 0; row < 3; row++)
			{
				for (col = row; col < 3; col++)
				{
					abs = fabsf(M(row, col));
					if (abs > max)
					{
						max = abs;
						maxRow = row;
						maxCol = col;
					}
				}
			}
			if (max < epsilon)
			{
				// The rank is 0. The eigenvalue has multiplicity 3.
				return 0;
			}

			// The rank is at least 1. Swap the row containing the maximum-magnitude entry with row 0.
			if (maxRow != 0)
			{
				for (col = 0; col < 3; col++)
				{
					save = M(0, col);
					M(0, col) = M(maxRow, col);
					M(maxRow, col) = save;
				}
			}

			/// Row-reduce the matrix.

			//Scale the row containing the maximum to generate a 1-valued pivot.
			float invMax = 1.0f / M(0, maxCol);
			M(0, 0) *= invMax;
			M(0, 1) *= invMax;
			M(0, 2) *= invMax;

			// Eliminate the maxCol column entries in rows 1 and 2.
			if (maxCol == 0)
			{
				M(1, 1) -= M(1, 0)*M(0, 1);
				M(1, 2) -= M(1, 0)*M(0, 2);
				M(2, 1) -= M(2, 0)*M(0, 1);
				M(2, 2) -= M(2, 0)*M(0, 2);
				M(1, 0) = 0;
				M(2, 0) = 0;
			}
			else if (maxCol == 1)
			{
				M(1, 0) -= M(1, 1)*M(0, 0);
				M(1, 2) -= M(1, 1)*M(0, 2);
				M(2, 0) -= M(2, 1)*M(0, 0);
				M(2, 2) -= M(2, 1)*M(0, 2);
				M(1, 1) = 0;
				M(2, 1) = 0;
			}
			else
			{
				M(1, 0) -= M(1, 2)*M(0, 0);
				M(1, 1) -= M(1, 2)*M(0, 1);
				M(2, 0) -= M(2, 2)*M(0, 0);
				M(2, 1) -= M(2, 2)*M(0, 1);
				M(1, 2) = 0;
				M(2, 2) = 0;
			}
			
			// Compute the maximum-magnitude entry of the last two rows of the row-reduced matrix.
			max = -1;
			maxRow = -1;
			maxCol = -1;
			for (row = 1; row < 3; row++)
			{
				for (col = 0; col < 3; col++)
				{
					abs = fabsf(M(row, col));
					if (abs > max)
					{
						max = abs;
						maxRow = row;
						maxCol = col;
					}
				}
			}
			if (max < epsilon)
			{
				// The rank is 1. The eigenvalue has multiplicity 2.
				return 1;
			}

			// If row 2 has the maximum-magnitude entry, swap it with row 1.
			if (maxRow == 2)
			{
				for (col = 0; col < 3; col++)
				{
					save = M(1, col);
					M(1, col) = M(2, col);
					M(2, col) = save;
				}
			}
			// Scale row 1 to generate a 1-valued pivot.
			invMax = 1 / M(1, maxCol);
			M(1, 0) *= invMax;
			M(1, 1) *= invMax;
			M(1, 2) *= invMax;

			// The rank is 2. The eigenvalue has multiplicity 1.
			return 2;
		}
		
		void getComplement2(vec3 u, vec3& v, vec3& w) const
		{
			u = normalize(u);
			if (fabsf(u.x) >= fabsf(u.y))
			{
				float invLength = 1.0f / sqrt(u.x*u.x + u.z*u.z);
				v.x = -u.z*invLength;
				v.y = 0.0f;
				v.z = u.x*invLength;
				w.x = u.y*v.z;
				w.y = u.z*v.x - u.x*v.z;
				w.z = -u.y*v.x;
			}
			else
			{
				float invLength = 1.0f / sqrt(u.y*u.y + u.z*u.z);
				v.x = 0.0f;
				v.y = u.z*invLength;
				v.z = -u.y*invLength;
				w.x = u.y*v.z - u.z*v.y;
				w.y = -u.x*v.z;
				w.z = u.x*v.y;
			}
		}

	public:
		// returns the eigenvalues in the elements x,y,z of a vec3 in ascending order
		vec3 eigenvalues() const
		{
			const double inv3 = 0.33333333333333333333333333333333;
			const double root3 = 1.7320508075688772935274463415059;

			double c0 = e00*e11*e22 + 2 * e01*e02*e12 - e00*e12*e12 - e11*e02*e02 - e22*e01*e01;
			double c1 = e00*e11 - e01*e01 + e00*e22 - e02*e02 + e11*e22 - e12*e12;
			double c2 = e00 + e11 + e22;
			double c2Div3 = c2*inv3;
			double aDiv3 = c1*inv3 - c2Div3*c2Div3;
			if (aDiv3 > 0.0) aDiv3 = 0.0;
			double mbDiv2 = 0.5*c0 + c2Div3*c2Div3*c2Div3 - 0.5*c2Div3*c1;
			double q = mbDiv2*mbDiv2 + aDiv3*aDiv3*aDiv3;		// Note: This line produces an inexact result.
			if (q > 0.0) q = 0.0;
			double magnitude = sqrt(-aDiv3);
			double angle = atan2(sqrt(-q), mbDiv2)*inv3;
			if (angle != angle) angle = 0.0;					// check for NAN. Can happen if q == 0 && mbDiv2 == 0. Although atan2(0,0) does return 0!
			double sn, cs;
			sn = sin(angle);
			cs = cos(angle);

			double evalues[3];
			evalues[0] = c2Div3 + 2 * magnitude*cs;
			evalues[1] = c2Div3 - magnitude*(cs + root3*sn);
			evalues[2] = c2Div3 - magnitude*(cs - root3*sn);

			// Sort eigenvalues in ascending order
			double h;
			if (evalues[2] < evalues[1]) { h = evalues[1];  evalues[1] = evalues[2];  evalues[2] = h; }
			if (evalues[1] < evalues[0]) { h = evalues[0];  evalues[0] = evalues[1];  evalues[1] = h; }
			if (evalues[2] < evalues[1]) { h = evalues[1];  evalues[1] = evalues[2];  evalues[2] = h; }
			return vec3(float(evalues[0]), float(evalues[1]), float(evalues[2]));
		}

		// returns the eigenvalues in the elements x,y,z of out_evalues in ascending order, and their associated eigenvectors in out_evectors
		void eigensystem(vec3& out_evalues, vec3 out_evectors[3]) const
		{
			// condition matrix by normalizing by element of largest magnitude
			float emax = 0;
			for (const float *eptr = &e00; eptr != &e22 + 1; ++eptr)
				if (fabsf(*eptr) > emax)
					emax = fabsf(*eptr);
			smat3 cn = (*this) / emax;
			
			vec3 evalues;				// pivoted eigen values
            float eps = 2 * std::numeric_limits<float>::min();	// Note: if epsilon is zero, small numeric instabilities can pretty fast result in a false rank-estimation, which can result in bad eigenvectors
			{
				evalues = cn.eigenvalues();

				// Compute eigenvectors
				mat3 M(cn.e00 - evalues.x, cn.e01, cn.e02, cn.e01, cn.e11 - evalues.x, cn.e12, cn.e02, cn.e12, cn.e22 - evalues.x);

				int rank0 = computeRank(M, eps);
				if (rank0 == 0)
				{
					// evalue[0] = evalue[1] = evalue[2]
					out_evectors[0] = vec3(1, 0, 0);
					out_evectors[1] = vec3(0, 1, 0);
					out_evectors[2] = vec3(0, 0, 1);
				}
				else if (rank0 == 1 || evalues.x == evalues.y)			// EXPERIMENTAL: The term right of the OR
				{
					// evalue[0] = evalue[1] < evalue[2]
					getComplement2(normalize(M.row(0)), out_evectors[0], out_evectors[1]);
					out_evectors[2] = cross(out_evectors[0], out_evectors[1]);
				}
				else	// rank0 == 2
				{
					out_evectors[0] = normalize(cross(normalize(M.row(0)), normalize(M.row(1))));
					M = mat3(cn.e00 - evalues.y, cn.e01, cn.e02, cn.e01, cn.e11 - evalues.y, cn.e12, cn.e02, cn.e12, cn.e22 - evalues.y);

					int rank1 = computeRank(M, eps);
					if (rank1 == 1)
					{
						// evalue[0] < evalue[1] = evalue[2]
						getComplement2(out_evectors[0], out_evectors[1], out_evectors[2]);
					}
					else
					{
						// evalue[0] < evalue[1] < evalue[2]
						out_evectors[1] = normalize(cross(M.row(0), M.row(1)));
						out_evectors[2] = cross(out_evectors[0], out_evectors[1]);
					}
				}
			}
			// scale back to obtain proper eigenvalues
			out_evalues = evalues * emax;
		}

		// returns the eigenvalues in the elements x,y,z of out_evalues in ascending order, and their associated 
		// eigenvectors in the columns of out_evectors (in ascending order accordingly)
		void eigensystem(vec3& out_evalues, mat3& out_evectors) const
		{
			vec3 evecs[3];
			eigensystem(out_evalues, evecs);
			out_evectors = mat3(
				evecs[0].x, evecs[1].x, evecs[2].x,
				evecs[0].y, evecs[1].y, evecs[2].y,
				evecs[0].z, evecs[1].z, evecs[2].z
				);
		}
		
		// returns the eigenvectors in the columns of out_evectors, in ascending order of their respective eigenvalues
		void eigenvectors(vec3 out_evectors[3]) const
		{
			vec3 evalues;
			eigensystem(evalues, out_evectors);
		}

		// returns a matrix containing the eigenvectors in its columns, ascendingly ordered by their respective eigenvalues
		mat3 eigenvectors() const
		{
			vec3 evecs[3];
			eigenvectors(evecs);
			return mat3(
				evecs[0].x, evecs[1].x, evecs[2].x,
				evecs[0].y, evecs[1].y, evecs[2].y,
				evecs[0].z, evecs[1].z, evecs[2].z
			);
		}
				
		#pragma endregion
	};

	
	#pragma region smat3	non-member operators

	inline smat3 operator* (float s, const smat3& m)
	{
		return m * s;
	}

	inline float det(const smat3& c)
	{
		return -c.e02*c.e02*c.e11 + 2 * c.e01*c.e02*c.e12 - c.e00*c.e12*c.e12 - c.e01*c.e01*c.e22 + c.e00*c.e11*c.e22;
	}

	// returns the adjugate matrix (also cofactor matrix or classical adjoint) of the matrix c.
	// The adjugate is proportional to the inverse of a symmetric matrix: adj(c) = det(c) * inverse(c)
	inline smat3 adjugate(const smat3& c)
	{
		return smat3(c.e11*c.e22 - c.e12*c.e12, c.e02*c.e12 - c.e01*c.e22, c.e01*c.e12 - c.e02*c.e11, c.e00*c.e22 - c.e02*c.e02, c.e02*c.e01 - c.e00*c.e12, c.e00*c.e11 - c.e01*c.e01);
	}
	
	inline smat3 inverse(const smat3& c)
	{
		return adjugate(c) / det(c);
	}

	inline float trace(const smat3& c)
	{
		return c.e00 + c.e11 + c.e22;
	}

	inline std::ostream& operator<< (std::ostream& os, const smat3& cov)
	{
		return os << "{" << cov.e00 << ", " << cov.e01 << ", " << cov.e02 << ", " << cov.e11 << ", " << cov.e12 << ", " << cov.e22 << "}";
	}

	inline mat3 operator* (const mat3& m, const smat3& sm)
	{
		mat3 res;
		for (uint r = 0; r < 3; ++r)
		{
			res(r, 0) = m(r, 0)*sm.e00 + m(r, 1)*sm.e01 + m(r, 2)*sm.e02;
			res(r, 1) = m(r, 0)*sm.e01 + m(r, 1)*sm.e11 + m(r, 2)*sm.e12;
			res(r, 2) = m(r, 0)*sm.e02 + m(r, 1)*sm.e12 + m(r, 2)*sm.e22;
		}
		return res;
	}

	// returns the symmetric product S.T.S of symmetric matrices S and T. The result is symmetric again.
	inline smat3 smult(const smat3& T, const smat3& S)
	{
		return smat3(S * T * S);
	}

	// returns the symmetric product L.S.L' of a symmetric matrix S and a non-symmetric matrix L. The result is symmetric again.
	inline smat3 smult(const smat3& S, const mat3& L)
	{
		return smat3(L * S * transpose(L));
	}
	

	inline mat3 operator+ (const smat3& sm, const mat3& m)
	{
		return sm.toMat3() + m;
	}

	inline mat3 operator+ (const mat3& m, const smat3& sm)
	{
		return m + sm.toMat3();
	}

	inline mat3 operator- (const smat3& sm, const mat3& m)
	{
		return sm.toMat3() - m;
	}

	inline mat3 operator- (const mat3& m, const smat3& sm)
	{
		return m - sm.toMat3();
	}

	// left-multiplication of vector (v^T * m)
	inline vec3 operator* (const vec3& v, const smat3& m)
	{
		// similar to right multiplication for symmetric matrices
		return m * v;
	}

	// returns the cholesky decomposition L from sm, defined as sm = LL'. L is a triangular matrix whose elements are stored in the elements of the returned smat3.
	// TODO: create a class tmat3 for triangular matrix, omitting certain matrix operations, and derive smat3 from it!
	inline smat3 chol(const smat3& sm)
	{
		smat3 c = sm;

		c.e00 = sqrtf(c.e00);
		float inv_e00 = 1.0f / c.e00;

		c.e01 *= inv_e00;
		c.e11 = sqrtf(c.e11 - c.e01 * c.e01);

		c.e02 *= inv_e00;
		c.e12 = (c.e12 - c.e01 * c.e02) / c.e11;
		c.e22 = sqrtf(c.e22 - c.e02 * c.e02 - c.e12 * c.e12);

		return c;
	}


	#pragma endregion

	//--------------------------------------------------------------------------------------
	#pragma endregion



	

#pragma region smat4
	//--------------------------------------------------------------------------------------
	// symmetric 4x4 matrix given by its upper triangle matrix entries
	struct smat4
	{
		float e00, e01, e02, e03, e11, e12, e13, e22, e23, e33;

		smat4() {}

		smat4(float _e00, float _e01, float _e02, float _e03, float _e11, float _e12, float _e13, float _e22, float _e23, float _e33) :
			e00(_e00), e01(_e01), e02(_e02), e03(_e03), e11(_e11), e12(_e12), e13(_e13), e22(_e22), e23(_e23), e33(_e33)
		{}

		// Creates a symmetric 4x4 matrix from the upper right triangle matrix entries of the given 3x3 matrix.
		// Note: This is explicit, since implicit conversion from mat4 to smat4 can raise a lot of errors!
		explicit smat4(const mat4& m) :
			e00(m(0, 0)), e01(m(0, 1)), e02(m(0, 2)), e03(m(0, 3)), e11(m(1, 1)), e12(m(1, 2)), e13(m(1, 3)), e22(m(2, 2)), e23(m(2, 3)), e33(m(3, 3))
		{
		}
		
		mat4 toMat4() const
		{
			return mat4(e00, e01, e02, e03, e01, e11, e12, e13, e02, e12, e22, e23, e03, e13, e23, e33);
		}

		smat4 operator+ (const smat4& c) const
		{
			return smat4(e00 + c.e00, e01 + c.e01, e02 + c.e02, e03 + c.e03, e11 + c.e11, e12 + c.e12, e13 + c.e13, e22 + c.e22, e23 + c.e23, e33 + c.e33);
		}

		smat4 operator- (const smat4& c) const
		{
			return smat4(e00 - c.e00, e01 - c.e01, e02 - c.e02, e03 - c.e03, e11 - c.e11, e12 - c.e12, e13 - c.e13, e22 - c.e22, e23 - c.e23, e33 - c.e33);
		}

		smat4 operator* (float s) const
		{
			return smat4(e00*s, e01*s, e02*s, e03*s, e11*s, e12*s, e13*s, e22*s, e23*s, e33*s);
		}
		
		smat4 operator/ (float s) const
		{
			float inv_s = 1.0f / s;
			return smat4(e00 * inv_s, e01 * inv_s, e02 * inv_s, e03 * inv_s, e11 * inv_s, e12 * inv_s, e13 * inv_s, e22 * inv_s, e23 * inv_s, e33 * inv_s);
		}

		smat4& operator+= (const smat4& c)
		{
			e00 += c.e00;	e01 += c.e01;	e02 += c.e02;	e03 += c.e03;
			e11 += c.e11;	e12 += c.e12;	e13 += c.e13;	
			e22 += c.e22;	e23 += c.e23;
			e33 += c.e33;
			return *this;
		}

		smat4& operator-= (const smat4& c)
		{
			e00 -= c.e00;	e01 -= c.e01;	e02 -= c.e02;	e03 -= c.e03;
			e11 -= c.e11;	e12 -= c.e12;	e13 -= c.e13;
			e22 -= c.e22;	e23 -= c.e23;
			e33 -= c.e33;
			return *this;
		}
		smat4& operator*= (float s)
		{
			for (float* e = &e00; e <= &e33; ++e)
				*e *= s;
			return *this;
		}
		smat4& operator/= (float s)
		{
			float inv_s = 1.0f / s;
			return *this *= inv_s;
		}
		vec4 operator* (const vec4& v) const
		{
			return vec4(e00*v.x + e01*v.y + e02*v.z + e03*v.w, e01*v.x + e11*v.y + e12*v.z + e13*v.w, e02*v.x + e12*v.y + e22*v.z + e23*v.w, e03*v.x + e13*v.y + e23*v.z + e33*v.w);
		}
		mat4 operator* (const smat4& c) const
		{
			return mat4(
				c.e00*e00 + c.e01*e01 + c.e02*e02 + c.e03*e03,	 c.e01*e00 + c.e11*e01 + c.e12*e02 + c.e13*e03,	 c.e02*e00 + c.e12*e01 + c.e22*e02 + c.e23*e03,	 c.e03*e00 + c.e13*e01 + c.e23*e02 + c.e33*e03,
				c.e00*e01 + c.e01*e11 + c.e02*e12 + c.e03*e13,	 c.e01*e01 + c.e11*e11 + c.e12*e12 + c.e13*e13,	 c.e02*e01 + c.e12*e11 + c.e22*e12 + c.e23*e13,	 c.e03*e01 + c.e13*e11 + c.e23*e12 + c.e33*e13,
				c.e00*e02 + c.e01*e12 + c.e02*e22 + c.e03*e23,	 c.e01*e02 + c.e11*e12 + c.e12*e22 + c.e13*e23,	 c.e02*e02 + c.e12*e12 + c.e22*e22 + c.e23*e23,	 c.e03*e02 + c.e13*e12 + c.e23*e22 + c.e33*e23,
				c.e00*e03 + c.e01*e13 + c.e02*e23 + c.e03*e33,	 c.e01*e03 + c.e11*e13 + c.e12*e23 + c.e13*e33,	 c.e02*e03 + c.e12*e13 + c.e22*e23 + c.e23*e33,	 c.e03*e03 + c.e13*e13 + c.e23*e23 + c.e33*e33
			);
		}
		mat4 operator* (const mat4& a) const
		{
			return mat4(
				a[0]*e00 + a[4]*e01 + a[8]*e02 + a[12]*e03,   a[1]*e00 + a[5]*e01 + a[9]*e02 + a[13]*e03,   a[2]*e00 + a[6]*e01 + a[10]*e02 + a[14]*e03,   a[3]*e00 + a[7]*e01 + a[11]*e02 + a[15]*e03,
				a[0]*e01 + a[4]*e11 + a[8]*e12 + a[12]*e13,   a[1]*e01 + a[5]*e11 + a[9]*e12 + a[13]*e13,   a[2]*e01 + a[6]*e11 + a[10]*e12 + a[14]*e13,   a[3]*e01 + a[7]*e11 + a[11]*e12 + a[15]*e13,
				a[0]*e02 + a[4]*e12 + a[8]*e22 + a[12]*e23,   a[1]*e02 + a[5]*e12 + a[9]*e22 + a[13]*e23,   a[2]*e02 + a[6]*e12 + a[10]*e22 + a[14]*e23,   a[3]*e02 + a[7]*e12 + a[11]*e22 + a[15]*e23,
				a[0]*e03 + a[4]*e13 + a[8]*e23 + a[12]*e33,   a[1]*e03 + a[5]*e13 + a[9]*e23 + a[13]*e33,   a[2]*e03 + a[6]*e13 + a[10]*e23 + a[14]*e33,   a[3]*e03 + a[7]*e13 + a[11]*e23 + a[15]*e33
				); 
		}

		// Unary -
		smat4 operator-() const
		{
			return smat4(-e00, -e01, -e02, -e03, -e11, -e12, -e13, -e22, -e23, -e33);
		}
		
		// returns the cofactor a_ij of the matrix (determinant of the submatrix resulting from 
		// deleting the i-th row and j-th column, times a sign. indices are 0-based.
		// https://de.wikipedia.org/wiki/Minor_(Lineare_Algebra))
		float cofactor(uint i, uint j) const
		{
			// compute minor of this matrix
			mat4 M = toMat4();
			for (uint k = 0; k < 4; k++)
				M(i, k) = M(k, j) = 0;
			M(i, j) = 1.0f;

			// return its determinant
			return det(M);
		}

		// returns the largest absolute value of all matrix elements
		float maxMagnitude() const
		{
			float mm = 0;
			for (const float* e = &e00; e <= &e33; ++e)
				mm = max(mm, abs(*e));
			return mm;
		}

		std::string toString() const
		{
			std::stringstream s;
			s << "{" << e00 << ", " << e01 << ", " << e02 << ", " << e03 << ", " << e11 << ", " << e12 << ", " << e13 << ", " << e22 << ", " << e23 << ", " << e33 << "}";
			return s.str();
		}

		static smat4 identity()
		{
			return smat4(1, 0, 0, 0, 1, 0, 0, 1, 0, 1);
		}

		static smat4 zero()
		{
			return smat4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		}

		static smat4 outer(const vec4& v)
		{
			return smat4(v.x*v.x, v.x*v.y, v.x*v.z, v.x*v.w, v.y*v.y, v.y*v.z, v.y*v.w, v.z*v.z, v.z*v.w, v.w*v.w);
		}

		// creates a diagonal matrix using the components of v
		static smat4 diag(const vec4& v)
		{
			return smat4(v.x, 0, 0, 0, v.y, 0, 0, v.z, 0, v.w);
		}

		// creates a diagonal matrix using the given diagonal entries
		static smat4 diag(float d00, float d11, float d22, float d33)
		{
			return smat4(d00, 0, 0, 0, d11, 0, 0, d22, 0, d33);
		}
	};
	

#pragma region smat4	non-member operators

	inline smat4 operator* (float s, const smat4& m)
	{
		return m * s;
	}

	//inline float det(const smat3& c)
	//{
	//	return -c.e02*c.e02*c.e11 + 2 * c.e01*c.e02*c.e12 - c.e00*c.e12*c.e12 - c.e01*c.e01*c.e22 + c.e00*c.e11*c.e22;
	//}

	// returns the adjugate matrix (also cofactor matrix or classical adjoint) of the matrix c.
	// The adjugate is proportional to the inverse of a symmetric matrix: adj(c) = det(c) * inverse(c)
	//inline smat3 adjugate(const smat3& c)
	//{
	//	return smat3(c.e11*c.e22 - c.e12*c.e12, c.e02*c.e12 - c.e01*c.e22, c.e01*c.e12 - c.e02*c.e11, c.e00*c.e22 - c.e02*c.e02, c.e02*c.e01 - c.e00*c.e12, c.e00*c.e11 - c.e01*c.e01);
	//}

	//inline smat3 inverse(const smat3& c)
	//{
	//	return adjugate(c) / det(c);
	//}

	inline float trace(const smat4& c)
	{
		return c.e00 + c.e11 + c.e22 + c.e33;
	}

	inline std::ostream& operator<< (std::ostream& os, const smat4& m)
	{
		return (os << m.toString());
	}

	inline mat4 operator* (const mat4& m, const smat4& sm)
	{
		mat4 res;
		for (uint r = 0; r < 4; ++r)
		{
			res(r, 0) = m(r, 0)*sm.e00 + m(r, 1)*sm.e01 + m(r, 2)*sm.e02 + m(r, 3)*sm.e03;
			res(r, 1) = m(r, 0)*sm.e01 + m(r, 1)*sm.e11 + m(r, 2)*sm.e12 + m(r, 3)*sm.e13;
			res(r, 2) = m(r, 0)*sm.e02 + m(r, 1)*sm.e12 + m(r, 2)*sm.e22 + m(r, 3)*sm.e23;
			res(r, 3) = m(r, 0)*sm.e03 + m(r, 1)*sm.e13 + m(r, 2)*sm.e23 + m(r, 3)*sm.e33;
		}
		return res;
	}

	// returns the symmetric product S.T.S of symmetric matrices S and T. The result is symmetric again.
	inline smat4 smult(const smat4& T, const smat4& S)
	{
		return smat4(S * T * S);
	}

	// returns the symmetric product L.S.L' of a symmetric matrix S and a non-symmetric matrix L. The result is symmetric again.
	inline smat4 smult(const smat4& S, const mat4& L)
	{
		return smat4(L * S * transpose(L));
	}


	inline mat4 operator+ (const smat4& sm, const mat4& m)
	{
		return sm.toMat4() + m;
	}

	inline mat4 operator+ (const mat4& m, const smat4& sm)
	{
		return m + sm.toMat4();
	}

	inline mat4 operator- (const smat4& sm, const mat4& m)
	{
		return sm.toMat4() - m;
	}

	inline mat4 operator- (const mat4& m, const smat4& sm)
	{
		return m - sm.toMat4();
	}

	// left-multiplication of vector (v^T * m)
	inline vec4 operator* (const vec4& v, const smat4& m)
	{
		// similar to right multiplication for symmetric matrices
		return m * v;
	}

	// TODO: THIS FUNCTION IS NOT TESTED!
	// returns the cholesky decomposition L from sm, defined as sm = LL'. L is a triangular matrix whose elements are stored in the elements of the returned smat4.
	// TODO: create a class tmat4 for triangular matrix, omitting certain matrix operations, and derive smat4 from it!
	inline smat4 chol(const smat4& sm)
	{
		smat4 c = sm;

		c.e00 = sqrtf(c.e00);
		float inv_e00 = 1.0f / c.e00;

		c.e01 *= inv_e00;
		c.e11 = sqrtf(c.e11 - c.e01 * c.e01);

		c.e02 *= inv_e00;
		c.e12 = (c.e12 - c.e01 * c.e02) / c.e11;
		c.e22 = sqrtf(c.e22 - c.e02 * c.e02 - c.e12 * c.e12);

		c.e03 *= inv_e00;
		c.e13 = (c.e13 - c.e01 * c.e03) / c.e11;
		c.e23 = (c.e23 - c.e02 * c.e03 - c.e12 * c.e13) / c.e22;
		c.e33 = sqrtf(c.e33 - c.e03 * c.e03 - c.e13 * c.e13 - c.e23 * c.e23);

		return c;
	}

#pragma endregion

	//--------------------------------------------------------------------------------------
#pragma endregion


	
	#pragma region general functions
	//--------------------------------------------------------------------------------------
	inline bool isfinite(const smat2& m)
	{
		return isfinite(m.e00) && isfinite(m.e01) && isfinite(m.e11);
	}
	inline bool isfinite(const smat3& m)
	{
		const float* element = &m.e00;
		for (uint i = 0; i < 6; i++, element++) 
			if (!isfinite(*element))
				return false;
		return true;
	}
	inline bool isfinite(const mat3& m)
	{
		for (uint i = 0; i < 9; i++)
			if (!isfinite(m[i]))
				return false;
		return true;
	}

	inline float sqrt(float x) { return sqrtf(x); }
	inline double sqrt(double x) { return std::sqrt(x); }
	inline vec2  sqrt(const vec2& v) { return vec2(sqrt(v.x), sqrt(v.y)); }
	inline vec3  sqrt(const vec3& v) { return vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z)); }

	//--------------------------------------------------------------------------------------
	#pragma endregion


}	// end namespace gms
