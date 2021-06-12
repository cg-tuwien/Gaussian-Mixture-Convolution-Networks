//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Reinhold Preiner 2014-2020 
//				 Simon Fraiss 2021
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------


#pragma once

#include "vec.hpp"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include "parallel.hpp"

using namespace std;


namespace gms
{

	// index offsets for neighbor cells in a grid
	static const vec3i neighborOffsets[27] = {
		vec3i(-1, -1, -1), vec3i(0, -1, -1), vec3i(1, -1, -1),
		vec3i(-1, 0, -1), vec3i(0, 0, -1), vec3i(1, 0, -1),
		vec3i(-1, 1, -1), vec3i(0, 1, -1), vec3i(1, 1, -1),
		vec3i(-1, -1, 0), vec3i(0, -1, 0), vec3i(1, -1, 0),
		vec3i(-1, 0, 0), vec3i(0, 0, 0), vec3i(1, 0, 0),
		vec3i(-1, 1, 0), vec3i(0, 1, 0), vec3i(1, 1, 0),
		vec3i(-1, -1, 1), vec3i(0, -1, 1), vec3i(1, -1, 1),
		vec3i(-1, 0, 1), vec3i(0, 0, 1), vec3i(1, 0, 1),
		vec3i(-1, 1, 1), vec3i(0, 1, 1), vec3i(1, 1, 1)
	};



	// primitive 3D hash grid
	class PointIndex
	{
	private:
		// hash function for a vec3i
		struct cellHasher
		{
			static const size_t bucket_size = 10;	// mean bucket size that the container should try not to exceed
			static const size_t min_buckets = 1024;	// minimum number of buckets, power of 2, >0

			cellHasher() {}

			size_t operator()(const vec3i &x) const
			{
				return hash<uint>()(x.x) ^ hash<uint>()(x.y) ^ hash<uint>()(x.z);
			}

			bool operator()(const vec3i& left, const vec3i& right) const
			{
				if (left.x != right.x)	return left.x < right.x;
				if (left.y != right.y)	return left.y < right.y;
				return left.z < right.z;
			}
		};

		typedef unordered_map<vec3i, vec2i, cellHasher> HashGrid;


		HashGrid mGrid;
		const vector<vec3>* mPoints;
		vector<uint> mIndices;
		vec3 mBBmin;
		vec3 mBBmax;
		vec3 mBBsize;		// world space dimensions
		vec3i mGridSize;	// grid dimensions
		float mCellSize;


	private:
		struct gridCoordPred
		{
			const vector<vec3>* mPoints;
			const vector<vec3i>* mGridCoords;

			gridCoordPred()
			{}

			gridCoordPred(const vector<vec3>& points, const vector<vec3i>& gridCoords)
				: mPoints(&points), mGridCoords(&gridCoords)
			{}

			/// compares ordering of indices based on the grid coordinate of their associating points
			bool operator()(const uint& a, const uint& b) const
			{
				const vec3i& left = mGridCoords->at(a);
				const vec3i& right = mGridCoords->at(b);

				if (left.x != right.x)	return left.x < right.x;
				if (left.y != right.y)	return left.y < right.y;
				if (left.z != right.z)	return left.z < right.z;
				return a < b;			// in case of identic coordinates, sort by index
			}
		};

		struct distancePred
		{
			vec3 mQueryPoint;
			const vector<vec3>* mPoints;

			distancePred() {}
			distancePred(const vector<vec3>* points, const vec3& queryPoint) : mPoints(points), mQueryPoint(queryPoint) {}

			bool operator()(const uint& a, const uint& b) const
			{
				return sqdist(mQueryPoint, (*mPoints)[a]) < sqdist(mQueryPoint, (*mPoints)[b]);
			}
		};

	public:
		PointIndex()
		{
		}

		PointIndex(const vector<vec3>& points, float maxSearchRadius)
		{
			create(points, maxSearchRadius);
		}


		vector<vec3i> getCellCoords() const
		{
			vector<vec3i> coords(mGrid.size());
			uint i = 0;
			for (const auto& k : mGrid)
				coords[i++] = k.first;
			return coords;
		}


		void create(const vector<vec3>& points, float maxSearchRadius)
		{
			assert(!points.empty());	// can't create index on empty set

			mGrid.clear();
			mPoints = &points;

			// compute bounding box (with epsilon space border)
            mBBmin = vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            mBBmax = vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

			for (const vec3& pt : points)
			{
				mBBmin = min(mBBmin, pt);
				mBBmax = max(mBBmax, pt);
			}

			// create dim cells of size maxSearchRadius, and adapt bbox
			mCellSize = maxSearchRadius;
			mBBsize = mBBmax - mBBmin;
			mGridSize = vec3i(mBBsize / mCellSize) + vec3i(1, 1, 1);
			mBBsize = vec3(mGridSize) * mCellSize;
			vec3 halfSize = mBBsize * 0.5;
			vec3 center = (mBBmax + mBBmin) * 0.5f;
			mBBmin = center - halfSize;
			mBBmax = center + halfSize;

			
			// compute grid coordinates for each point
			vector<vec3i> gridCoords(points.size());
			#pragma omp parallel for
			for (int i = 0; i < (int)points.size(); i++)
				gridCoords[i] = getGridCoord(points[i]);


			// create point index buffer sorted by their grid coordinates
			mIndices.resize(points.size());
			#pragma omp parallel for
			for (int i = 0; i < (int)mIndices.size(); ++i) mIndices[i] = i;
			//sort(mIndices.begin(), mIndices.end(), gridCoordPred(points, gridCoords));
			parallel::sort(mIndices, gridCoordPred(points, gridCoords));	// faster than serial sort
			

			// populate grid
			vec3i currentGridCoord = getGridCoord(points[mIndices[0]]);
			uint currentFirst = 0;
			for (uint i_ = 1; i_ < mIndices.size(); ++i_)
			{
				// next point index and associate gridCoord
				uint i = mIndices[i_];
				const vec3i& gridCoord = gridCoords[i];

				// if we have a new gridCoord, finish current cell at currentGridCoord first
				if (gridCoord != currentGridCoord)
				{
					mGrid[currentGridCoord] = vec2i(currentFirst, i_ - currentFirst);
					currentGridCoord = gridCoord;
					currentFirst = i_;
				}
			}
			mGrid[currentGridCoord] = vec2i(currentFirst, mIndices.size() - currentFirst);		// finish index list for last cell
		}


		// retrieve grid coordinates of point p
		vec3i getGridCoord(const vec3& p) const
		{
			return min(vec3i((p - mBBmin) / mCellSize), mGridSize - vec3i(1, 1, 1));
		}

		// retrieve the side length of a grid cell. This equals the maximum reliable search radius
		float cellSize()
		{
			return mCellSize;
		}


		// approximate k nearest neighbor search within a maximum radius of mCellSize.
		// if a neighbor's distance is not within the 3x3x3 neighboring cells, it is not returned.
		// all previous content in outIndices will be cleared.
		void annSearch(const vec3& queryPoint, uint k, vector<uint>& outIndices)
		{
			outIndices.clear();
			radiusSearch(queryPoint, sqrtf(12 * mCellSize * mCellSize), outIndices);	// sqrt(12) = diameter of two cells
			// sort by distance
			sort(outIndices.begin(), outIndices.end(), distancePred(mPoints, queryPoint));
			// remove any indices beyond k
			outIndices.erase(outIndices.begin() + min(k, (uint)outIndices.size()), outIndices.end());
		}


		// queries all indices within a radius ball around queryPoint and write them to the vector outIndices
		// all previous content in outIndices will be cleared.
		void radiusSearch(const vec3& queryPoint, float radius, vector<uint>& outIndices) const
		{
			const float sqradius = radius * radius;

			outIndices.clear();
			vec3i c = getGridCoord(queryPoint);

			// visit each neighbor cell and process points in there
			for (uint i = 0; i < 27; ++i)
			{
				// find n in the hash grid
				vec3i n = c + neighborOffsets[i];
				auto pos = mGrid.find(n);
				if (pos != mGrid.end())
				{
					// search point list of neighbor cell for in-range points
					const vec2i& indexRange = pos->second;
					
					for (int i_ = indexRange.x; i_ < indexRange.x + indexRange.y; ++i_)
						if (sqdist(queryPoint, mPoints->at(mIndices[i_])) < sqradius)
							outIndices.push_back(mIndices[i_]);
				}
			}
		}

		float nearestDistSearch(const vec3& queryPoint, int forbiddenindex=-1) const
		{
			vec3i c = getGridCoord(queryPoint);

			float minsqdist = numeric_limits<float>::infinity();

			// visit each neighbor cell and process points in there
			for (uint i = 0; i < 27; ++i)
			{
				// find n in the hash grid
				vec3i n = c + neighborOffsets[i];
				auto pos = mGrid.find(n);
				if (pos != mGrid.end())
				{
					// search point list of neighbor cell for in-range points
					const vec2i& indexRange = pos->second;

					for (int i_ = indexRange.x; i_ < indexRange.x + indexRange.y; ++i_) {
						if (mIndices[i_] != forbiddenindex)
						{
							float sqd = sqdist(queryPoint, mPoints->at(mIndices[i_]));
							if (sqd < minsqdist)
								minsqdist = sqd;
						}
					}
				}
			}
			if (minsqdist == numeric_limits<float>::infinity())
			{
				for (int i = 0; i < this->mPoints->size(); ++i)
				{
					if (i != forbiddenindex)
					{
						float sqd = sqdist(queryPoint, mPoints->at(i));
						if (sqd < minsqdist)
							minsqdist = sqd;
					}
				}
			}
			return minsqdist;
		}

		vector<float> nearestKDistSearch(const vec3& queryPoint, int k, int forbiddenindex = -1, vector<size_t>* indizes = nullptr) const
		{
			vec3i c = getGridCoord(queryPoint);

			vector<float> minsqdists = vector<float>(k, numeric_limits<float>::infinity());
			if (indizes)
			{
				*indizes = vector<size_t>(k, -1);
			}

			// visit each neighbor cell and process points in there
			for (uint i = 0; i < 27; ++i)
			{
				// find n in the hash grid
				vec3i n = c + neighborOffsets[i];
				auto pos = mGrid.find(n);
				if (pos != mGrid.end())
				{
					// search point list of neighbor cell for in-range points
					const vec2i& indexRange = pos->second;

					for (int i_ = indexRange.x; i_ < indexRange.x + indexRange.y; ++i_) {
						if (mIndices[i_] != forbiddenindex)
						{
							float sqd = sqdist(queryPoint, mPoints->at(mIndices[i_]));
							for (int gi = 0; gi < k; ++gi)
							{
								if (sqd < minsqdists[gi])
								{
									for (int gj = k - 1; gj > gi; --gj)
									{
										minsqdists[gj] = minsqdists[gj - 1];
										if (indizes) (*indizes)[gj] = (*indizes)[gj - 1];
									}
									minsqdists[gi] = sqd;
									if (indizes) (*indizes)[gi] = mIndices[i_];
									break;
								}
							}
						}
					}
				}
			}
			if (minsqdists[k - 1] == numeric_limits<float>::infinity())
			{
				for (int i = 0; i < this->mPoints->size(); ++i)
				{
					if (i != forbiddenindex)
					{
						float sqd = sqdist(queryPoint, mPoints->at(i));
						for (int gi = 0; gi < k; ++gi)
						{
							if (sqd < minsqdists[gi])
							{
								for (int gj = k - 1; gj > gi; --gj)
								{
									minsqdists[gj] = minsqdists[gj - 1];
									if (indizes) (*indizes)[gj] = (*indizes)[gj - 1];
								}
								minsqdists[gi] = sqd;
								if (indizes) (*indizes)[gi] = mIndices[i];
								break;
							}
						}
					}
				}
			}
			return minsqdists;
		}


		struct NeighborProcessor
		{
			virtual void operator() (uint nIndex, const vec3& nPos, float squaredDist) {}

			virtual void finalize () {}
		};


		void processNeighbors(const vec3& queryPoint, float radius, NeighborProcessor& nproc) const
		{
			const float sqradius = radius * radius;

			vec3i c = getGridCoord(queryPoint);

			// visit each neighbor cell and process points in there
			for (uint i = 0; i < 27; ++i)
			{
				// find n in the hash grid
				vec3i n = c + neighborOffsets[i];
				auto pos = mGrid.find(n);
				if (pos != mGrid.end())
				{
					// search point list of neighbor cell for in-range points
					const vec2i& indexRange = pos->second;
					for (int i_ = indexRange.x; i_ < indexRange.x + indexRange.y; ++i_)
					{
						uint i = mIndices[i_];
						float squaredDist = (queryPoint == mPoints->at(i)) ? 0.0f : sqdist(queryPoint, mPoints->at(i));
						if (squaredDist <= sqradius)
						{
							nproc(i, mPoints->at(i), squaredDist);
						}
					}
				}
			}

			nproc.finalize();
		}


		struct PointProcessor
		{
			virtual void operator() (uint index, const vec3& pos, const vector<uint>& neighbors) {}
		};


		void processCell(const vec3i& cellCoord, PointProcessor& proc) const
		{
			auto pos = mGrid.find(cellCoord);
			if (pos == mGrid.end())
				return;

			// get set of neighbors
			vector<uint> neighbors;
			for (uint i = 0; i < 27; ++i)
			{
				auto pos = mGrid.find(cellCoord + neighborOffsets[i]);
				if (pos != mGrid.end())
				{
					const vec2i& indexRange = pos->second;
					for (int i_ = indexRange.x; i_ < indexRange.x + indexRange.y; ++i_)
						neighbors.push_back(mIndices[i_]);
				}
			}

			// process points in cell
			const vec2i& indexRange = pos->second;
			for (int i_ = indexRange.x; i_ < indexRange.x + indexRange.y; ++i_)
			{
				uint i = mIndices[i_];
				proc(i, mPoints->at(i), neighbors);
			}
		}



		// radius search for a list of queryPoints with common radius.
		// all previous content in outIndices will be cleared.
		void radiusSearch(const vector<vec3>& queryPoints, float radius, vector<vector<uint>> outIndices) const
		{
			outIndices.resize(queryPoints.size());
			for (uint i = 0; i < queryPoints.size(); ++i)
				radiusSearch(queryPoints[i], radius, outIndices[i]);
		}


		// radius search for a list of queryPoints with individual radii.
		// all previous content in outIndices will be cleared.
		void radiusSearch(const vector<vec3>& queryPoints, const vector<float>& radii, vector<vector<uint>> outIndices) const
		{
			assert(queryPoints.size() == radii.size());

			outIndices.resize(queryPoints.size());
			for (uint i = 0; i < queryPoints.size(); ++i)
				radiusSearch(queryPoints[i], radii[i], outIndices[i]);
		}

	};	/// class PointIndex


}	/// end namespace gms
