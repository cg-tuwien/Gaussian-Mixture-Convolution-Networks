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
#include <unordered_map>
#include "vec.hpp"
#include "geom.hpp"
#include "pointindex.hpp"
#include <memory>
using std::vector;
using std::shared_ptr;


namespace gms
{

	class PointSet : public vector<vec3>
	{
	public:
		using ptr = shared_ptr<PointSet>;

	private:
		typedef unordered_map<string, shared_ptr<vector<float>>> FloatAttributes;
		typedef unordered_map<string, shared_ptr<vector<vec3>>>  Vec3Attributes;
		typedef unordered_map<string, shared_ptr<vector<vec4>>>  Vec4Attributes;

		FloatAttributes mAttribsFloat;
		Vec3Attributes  mAttribsVec3;
		Vec4Attributes  mAttribsVec4;
				
	public:
		PointSet()
		{
		}

		// copy constructor
		PointSet(const PointSet& ps)
		{
			*this = ps;
		}
				
		
		PointSet(const PointSet& ps, uint first, uint count)
		{
			uint cnt = min((uint)ps.size(), count);

			*this = vector<vec3>(ps.begin(), ps.begin() + cnt);
			for (auto attrib : ps.mAttribsFloat)	mAttribsFloat[attrib.first] = make_shared<vector<float>>(ps.attribFloat(attrib.first)->begin(), ps.attribFloat(attrib.first)->begin() + cnt);
			for (auto attrib : ps.mAttribsVec3)		mAttribsVec3[attrib.first] = make_shared<vector<vec3>>(ps.attribVec3(attrib.first)->begin(), ps.attribVec3(attrib.first)->begin() + cnt);
			for (auto attrib : ps.mAttribsVec4)		mAttribsVec4[attrib.first] = make_shared<vector<vec4>>(ps.attribVec4(attrib.first)->begin(), ps.attribVec4(attrib.first)->begin() + cnt);
		}  

		
		PointSet(const vector<vec3>& points) : vector<vec3>(points)
		{
		}
		
		// prepare empty point set with nPoints
		PointSet(uint nPoints) : vector<vec3>(nPoints)
		{
		}

		~PointSet()
		{
		}

		const PointSet& operator=(const PointSet& ps)
		{
			if (this == &ps)
				return *this;

			// copy points
			clear();
			for (const vec3& p : ps)
				push_back(p);

			for (auto attrib : ps.mAttribsFloat)	mAttribsFloat[attrib.first] = make_shared<vector<float>>(*ps.attribFloat(attrib.first));
			for (auto attrib : ps.mAttribsVec3)		mAttribsVec3[attrib.first]  = make_shared<vector<vec3>>(*ps.attribVec3(attrib.first));
			for (auto attrib : ps.mAttribsVec4)		mAttribsVec4[attrib.first]  = make_shared<vector<vec4>>(*ps.attribVec4(attrib.first));
			
			return *this;
		}
		

		void append(const PointSet& ps, uint first = 0, int count = -1)
		{
			if (count == -1)
				count = (uint)ps.size();

			// init similar attribute arrays if not given yet
			for (auto attrib : ps.mAttribsFloat)	if (attribFloat(attrib.first) == nullptr)	mAttribsFloat[attrib.first] = make_shared<vector<float>>(size(), 0.0f);
			for (auto attrib : ps.mAttribsVec3)		if (attribVec3(attrib.first) == nullptr)	mAttribsVec3[attrib.first] = make_shared<vector<vec3>>(size(), vec3(0, 0, 0));
			for (auto attrib : ps.mAttribsVec4)		if (attribVec4(attrib.first) == nullptr)	mAttribsVec4[attrib.first] = make_shared<vector<vec4>>(size(), vec4(0, 0, 0, 0));

			// copy points
			for (uint i = first; i < first + count; i++)
			{
				const vec3& p = ps[i];
				push_back(p);

				for (auto attrib : ps.mAttribsFloat)	mAttribsFloat[attrib.first]->push_back(attrib.second->at(i));
				for (auto attrib : ps.mAttribsVec3)		mAttribsVec3[attrib.first]->push_back(attrib.second->at(i));
				for (auto attrib : ps.mAttribsVec4)		mAttribsVec4[attrib.first]->push_back(attrib.second->at(i));
			}
		}

				
		// Returns a pointer to the float attribute list of the given name. if the attribute doesn't exists, nullptr is returned.
		shared_ptr<vector<float>> attribFloat(const string& name)
		{
			if (mAttribsFloat.find(name) != mAttribsFloat.end())
				return mAttribsFloat.at(name);
			return nullptr;
		}
		const shared_ptr<vector<float>> attribFloat(const string& name) const
		{
			if (mAttribsFloat.find(name) != mAttribsFloat.end())
				return mAttribsFloat.at(name);
			return nullptr;
		}

		shared_ptr<vector<vec3>> attribVec3(const string& name)
		{
			if (mAttribsVec3.find(name) != mAttribsVec3.end())
				return mAttribsVec3.at(name);
			return nullptr;
		}
		const shared_ptr<vector<vec3>> attribVec3(const string& name) const
		{
			if (mAttribsVec3.find(name) != mAttribsVec3.end())
				return mAttribsVec3.at(name);
			return nullptr;
		}

		shared_ptr<vector<vec4>> attribVec4(const string& name)
		{
			if (mAttribsVec4.find(name) != mAttribsVec4.end())
				return mAttribsVec4.at(name);
			return nullptr;
		}
		const shared_ptr<vector<vec4>> attribVec4(const string& name) const
		{
			if (mAttribsVec4.find(name) != mAttribsVec4.end())
				return mAttribsVec4.at(name);
			return nullptr;
		}
		
		void setAttrib(const string& name, shared_ptr<vector<float>> attribute)
		{
			mAttribsFloat[name] = attribute;
		}
		void setAttrib(const string& name, shared_ptr<vector<vec3>> attribute)
		{
			mAttribsVec3[name] = attribute;
		}
		void setAttrib(const string& name, shared_ptr<vector<vec4>> attribute)
		{
			mAttribsVec4[name] = attribute;
		}
			
		vector<string> getFloatAttributeNames() const
		{
			vector<string> names;
			for (auto kv : mAttribsFloat) names.push_back(kv.first);
			return names;
		}
		vector<string> getVec3AttributeNames() const
		{
			vector<string> names;
			for (auto kv : mAttribsVec3) names.push_back(kv.first);
			return names;
		}
		vector<string> getVec4AttributeNames() const
		{
			vector<string> names;
			for (auto kv : mAttribsVec4) names.push_back(kv.first);
			return names;
		}

		const FloatAttributes& getFloatAttribs() const { return mAttribsFloat; }
		const Vec3Attributes&  getVec3Attribs() const  { return mAttribsVec3; }
		const Vec4Attributes&  getVec4Attribs() const  { return mAttribsVec4; }
		

		// removes all points and its attributes that have non-zero flag in the removeFlags vector
		template<class Integer>
		void remove(const vector<Integer>& removeFlags)
		{
			PointSet reducedPointSet;
			
			// init similar attribute arrays
			for (auto attrib : mAttribsFloat)	reducedPointSet.mAttribsFloat[attrib.first] = make_shared<vector<float>>();
			for (auto attrib : mAttribsVec3)	reducedPointSet.mAttribsVec3[attrib.first] = make_shared<vector<vec3>>();
			for (auto attrib : mAttribsVec4)	reducedPointSet.mAttribsVec4[attrib.first] = make_shared<vector<vec4>>();
			
			for (uint i = 0; i < size(); i++)
			{
				if (!removeFlags[i])
				{
					reducedPointSet.push_back(at(i));

					for (auto attrib : mAttribsFloat)	reducedPointSet.mAttribsFloat[attrib.first]->push_back(attrib.second->at(i));
					for (auto attrib : mAttribsVec3)	reducedPointSet.mAttribsVec3[attrib.first]->push_back(attrib.second->at(i));
					for (auto attrib : mAttribsVec4)	reducedPointSet.mAttribsVec4[attrib.first]->push_back(attrib.second->at(i));
				}
			}

			*this = reducedPointSet;
		}
		

		// transforms the point positions so that its bounding box is centered in the origin
		// returns the previous bounding box center
		vec3 centralize()
		{
			BBox bbox(*this);
			vec3 c = bbox.center();
			
			for (vec3& p : *this)
				p -= c;

			return c;
		}


		// translation by vector t
		void translate(const vec3& t)
		{
			for (vec3& p : *this)
				p += t;
		}

		// scale by vector s around origin
		void scale(const vec3& s)
		{
			for (vec3& p : *this)
			{
				p.x *= s.x;
				p.y *= s.y;
				p.z *= s.z;
			}
		}
	};

}	/// end namespace gms

