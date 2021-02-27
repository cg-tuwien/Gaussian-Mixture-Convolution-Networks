//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Reinhold Preiner 2014-2020 
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------

#pragma once

#include "vec.hpp"
#include <vector>
#include <limits>


namespace gms
{
	// AABB struct
	struct BBox
	{
		vec3 pmin;
		vec3 pmax;

		BBox() :
            pmin(vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max())),
            pmax(vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()))
		{}

		BBox(const vec3& pMin, const vec3& pMax) :
			pmin(pMin), pmax(pMax)
		{}

		BBox(const std::vector<vec3>& points)
		{
			compute(points);
		}

		void reset()		
        {
            pmin = vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            pmax = vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
		}

		// extend bounding box to cover new point p
		void extend(const vec3& p)
		{
			pmax = max(pmax, p);
			pmin = min(pmin, p);
		}

		// extend bounding box to cover new bbox
		void extend(const BBox& bbox)
		{
			pmax = max(pmax, bbox.pmax);
			pmin = min(pmin, bbox.pmin);
		}

		void compute(const std::vector<vec3>& points)
		{
			reset();
			for (const vec3& p : points)
				extend(p);			
		}

		void compute(const std::vector<BBox>& bboxes)
		{
			reset();
			for (const BBox& bbox : bboxes)
				extend(bbox);
		}

		vec3 dim() const 
		{
			vec3 dim = fabsf(pmax - pmin);
			if (isinf(dim.x))	dim.x = 0;
			if (isinf(dim.y))	dim.y = 0;
			if (isinf(dim.z))	dim.z = 0;
			return dim;
		}

		vec3 center() const 
		{
			return (pmin + pmax) * 0.5f;
		}

		std::vector<vec3> getCorners() const 
		{
			std::vector<vec3> c;
			c.push_back(pmin);
			c.push_back(vec3(pmax.x, pmin.y, pmin.z));
			c.push_back(vec3(pmin.x, pmax.y, pmin.z));
			c.push_back(vec3(pmax.x, pmax.y, pmin.z));
			c.push_back(vec3(pmin.x, pmin.y, pmax.z));
			c.push_back(vec3(pmax.x, pmin.y, pmax.z));
			c.push_back(vec3(pmin.x, pmax.y, pmax.z));
			c.push_back(pmax);
			return c;
		}

		// returns true if this bbox touches or shares a common volume with the given sphere.
		bool intersectsSphere(const vec3& center, float r) const
		{
			float dist_squared = r * r;
			vec3 dmin = center - pmin;
			vec3 dmax = center - pmax;
			
			if (center.x < pmin.x) dist_squared -= dmin.x * dmin.x;
			else if (center.x > pmax.x) dist_squared -= dmax.x * dmax.x;
			if (center.y < pmin.y) dist_squared -= dmin.y * dmin.y;
			else if (center.y > pmax.y) dist_squared -= dmax.y * dmax.y;
			if (center.z < pmin.z) dist_squared -= dmin.z * dmin.z;
			else if (center.z > pmax.z) dist_squared -= dmax.z * dmax.z;
			
			return dist_squared >= 0;		// set to > to exclude touch or zero-radius sphere cases
		}

		// returns true, if this bbox contains a given point
		bool contains(const vec3& p) const
		{
			return (p.x > pmin.x && p.y > pmin.y && p.z > pmin.z && p.x < pmax.x && p.y < pmax.y && p.z < pmax.z);
		}

		// returns the length of the bbox diagonal
		float diagonal() const
		{
			return length(pmax - pmin);
		}
	};

	inline BBox transform(const BBox& box, const mat4& trafoMatrix)
	{
		return BBox((trafoMatrix * vec4(box.pmin, 1)).xyz(), (trafoMatrix * vec4(box.pmax, 1)).xyz());
	}




	// Sphere
	struct Sphere
	{
		vec3 center;
		float radius;

		Sphere() : center(vec3(0, 0, 0)), radius(0)
		{}

		Sphere(const vec3& c, float r) : center(c), radius(r)
		{}


		// returns true if this sphere contains the given point p.
		// if closed is true, a point on the sphere's boundary is considered to be contained (operator <=), otherwise not (operator <).
		bool contains(const vec3& p, bool closed = false) const
		{
			float sqDist = dot(p - center, p - center);
			float sqRad = radius * radius;
			return closed ? sqDist <= sqRad : sqDist < sqRad;
		}

		// returns true if this sphere contains the given point s.
		// if closed is true, a sphere s touching this sphere's boundary is considered to be contained (operator <=), otherwise not (operator <).
		bool contains(const Sphere s, bool closed = false) const
		{
			float d = dist(center, s.center) + s.radius;
			return closed ? d <= radius : d < radius;
		}

		// returns true if this sphere intersects the given sphere s.
		// if closed is true, the intersection definition incorporates spheres with singular contact point or points that lie on the sphere's boundary (operator <=).
		// if closed is false, such border-cases are not incorporated and false is returned (operator <).
		bool intersects(const Sphere& s, bool closed = false) const
		{
			float sqDist = dot(s.center - center, s.center - center);		// sq distance between sphere centers
			float sqRsum = (s.radius + radius) * (s.radius + radius);		// sq sum of radii
			return closed ? sqDist <= sqRsum : sqDist < sqRsum;
		}


		// returns true if the sphere intersects the plane given in Hessian normal form by unit normal vector n and origin distance d. 
		// if closed is true, the intersection incorporates boundary points of the sphere (operator <=), i.e., when the sphere touches the plane in a single boundary point
		// otherwise, boundary points are excluded of the intersection definition (operator <).
		bool intersectsPlane(const vec3& n, float d, bool closed = false) const
		{
			float dist = abs(dot(center, n) - d);	// distance of sphere center to its closest point on the plane
			return closed ? dist <= radius : dist < radius;
		}

		// takes a line parameterized by an origin x and a normalized (!) direction vector v and returns the parameter interval of its intersection with the sphere.
		// in case of an intersection, the start and end of the interval is encoded in the x- and y-component of the vec2, respectively (with x <= y). 
		// otherwise, a vec2 with x > y is returned. Note: if the line touches the sphere at a single point, a vec2 with x == y is returned.
		vec2 getLineIntersectionInterval(const vec3& x, const vec3& v) const
		{
			float qd = dot(center - x, v);				// parametric position of the closest line point to the sphere center
			vec3 q = x + v * qd;						// closest point on line to center 
			float d2 = dot(q - center, q - center);
			float r2 = radius * radius;

			if (d2 > r2)
                return vec2(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());

			float s = sqrtf(r2 - d2);				// distance of q to sphere boundary along the line

			return vec2(qd - s, qd + s);
		}


		// returns the smallest sphere circumscribing a triangle given by the three points
		// (from: http://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d)
		static Sphere circumscribing(const vec3& p0, const vec3& p1, const vec3& p2)
		{
			vec3 v01 = p1 - p0;
			vec3 v02 = p2 - p0;
			vec3 n = cross(v01, v02);
			vec3 p0_to_center = (cross(n, v01) * dot(v02, v02) + cross(v02, n) * dot(v01, v01)) / (2 * dot(n, n));
			
			return Sphere(p0 + p0_to_center, length(p0_to_center));
		}
	};



	// Frustum
	struct Frustum
	{
		// each plane is represented in Hessian normal form: (nx, ny, nz, d)
		// ordering: front, back, left, right, top, bottom, with outward pointing normals
		vec4 planes[6];		

		// returns true if the given sphere is fully contained or partially intersecting the frustum
		bool intersectsSphere(const Sphere& sphere) const
		{
			// compute the distance of the sphere to each of the bounding planes
			for (uint i = 0; i < 6; ++i) 
			{
				// distance to this plane
				float d = dot(planes[i].xyz(), sphere.center) - planes[i].w;

				// sphere entirely outside plane
				if (d > sphere.radius)
					return false;
			}

			// otherwise, sphere is fully contained 
			return true;
		}

	};
}

