//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Reinhold Preiner 2014-2020 
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------

#pragma once

#include <cstring>
#include <vector>
#include <omp.h>



// algorithms for omp parallelization
namespace gms
{
	class parallel
	{
	public:
		template<typename T>
		static void prefixSum(std::vector<T>& a)
		{
			int num_threads, sub_len;
			std::vector<T> partial, temp;

#pragma omp parallel shared(num_threads, sub_len, partial, temp)
			{
#pragma omp single
				{
					num_threads = omp_get_num_threads();
					sub_len = (a.size() - 1) / num_threads + 1;
					partial.resize(num_threads);
					temp.resize(num_threads);

					//cout << "l: " << sub_len << endl;
					//cout << "i: ";	for (T x : a) cout << x << " "; cout << endl;
				}

				int tid = omp_get_thread_num();

				// compute prefix sum on sub arrays
				int i = 0;
				for (i = tid * sub_len + 1; i < (tid + 1) * sub_len && i < (int)a.size(); i++)
					a[i] += a[i - 1];
				partial[tid] = a[i - 1];

#pragma omp barrier

				for (i = 1; i < num_threads; i <<= 1)
				{
					if (tid >= i)
						temp[tid] = partial[tid] + partial[tid - i];
#pragma omp barrier

#pragma omp single
                    std::memcpy(&partial[0] + 1, &temp[0] + 1, sizeof(T) * (num_threads - 1));
				}


				/*#pragma omp barrier
				#pragma omp single
				{
				cout << "t: ";	for (T x : a) cout << x << " "; cout << endl;
				cout << "p: ";	for (T x : partial) cout << x << " "; cout << endl;
				}
				#pragma omp barrier*/

				// update original array
				int last = min((tid + 1) * sub_len, (int)a.size());
				for (i = tid * sub_len; i < last; i++)
					a[i] += partial[tid] - a[last - 1];
			}

			/*cout << "r: ";	for (T x : a) cout << x << " "; cout << endl;
			T sum = 0;
			cout << "R: ";	for (T x : A) cout << (sum += x) << " "; cout << endl;*/
		}


		// returns the number of packed elements (i.e., the packed size)
		template<typename T>
		static unsigned pack(const std::vector<T>& inVector, const std::vector<int>& selectionMask, std::vector<T>& outVector)
		{
			assert(inVector.size() == selectionMask.size());

			if (outVector.size() < inVector.size())
				outVector.resize(inVector.size());

			// compute prefixSum
			std::vector<uint> prefix_sum(inVector.size(), 0);
#pragma omp parallel for
			for (int i = 0; i < (int)prefix_sum.size(); i++)
				if (selectionMask[i])
					prefix_sum[i] = 1;

			prefixSum(prefix_sum);

			// pack
#pragma omp parallel for
			for (int i = 0; i < (int)inVector.size(); i++)
				if (selectionMask[i])
					outVector[prefix_sum[i] - 1] = inVector[i];

			return prefix_sum.back();
		}



		template<typename T, class Functor>
		static T reduce(const std::vector<T>& a, const Functor& func)
		{
			int num_threads, sub_len;
			T* partial;

#pragma omp parallel shared(num_threads, sub_len, partial)
			{
#pragma omp single
				{
					num_threads = omp_get_num_threads();
					sub_len = (a.size() - 1) / num_threads + 1;
					partial = new T[num_threads];
				}

				int tid = omp_get_thread_num();

				// initialize partial with first element
				if (tid * sub_len < a.size())
					partial[tid] = a[tid * sub_len];

				// compute reduction on sub arrays
				for (int i = tid * sub_len + 1; i < (tid + 1) * sub_len && i < a.size(); i++)
					partial[tid] = func(a[i], partial[tid]);
			}

			T erg = partial[0];
			for (int i = 0; i < num_threads && i * sub_len < a.size(); i++)
				erg = func(partial[i], erg);

			delete[] partial;
			return erg;
		}



		template<class T, class Functor>
		static void sort(std::vector<T>& a, const Functor& func)
		{
			int numThreads;

#pragma omp parallel shared(numThreads)
#pragma omp single
			{
				numThreads = omp_get_num_threads();
			}
			int subn = (a.size() - 1) / numThreads + 1;

#pragma omp parallel
			{
				int tid = omp_get_thread_num();

				int first = tid * subn;
				int last = min((tid + 1) * subn, (int)a.size());

				if (first < (int)a.size())
				{
					std::sort(a.begin() + first, a.begin() + last, func);
				}
			}

			// parallel merge
			//---------------------------------------------------------------------------
			//for (auto x : a) cout << x << " "; cout << endl << endl;

			std::vector<T> temp(a.size());

			for (uint worksize = subn; worksize < a.size(); worksize *= 2)
			{
#pragma omp parallel for
				for (int t = 0; t < numThreads; t += 2)
				{
					uint start0 = t * worksize;

					// only merge if this thread's region is not out of range
					if (start0 < a.size())
					{
						uint start1 = min((t + 1) * worksize, a.size());
						uint end1 = min((t + 2) * worksize, a.size());

						merge(a, temp, start0, start1, end1, func);
					}
				}
				a.swap(temp);

				//for (auto x : a) cout << x << " "; cout << endl;
			}
		}

		template<class T, class Functor>
		static void merge(const std::vector<T>& a, std::vector<T>& b, unsigned start0, unsigned start1, unsigned end1, const Functor& func)
		{
			unsigned p0 = start0, p1 = start1;
			for (unsigned i = start0; i < end1; i++)
			{
				if (p1 >= end1)			b[i] = a[p0++];
				else if (p0 >= start1)	b[i] = a[p1++];
				else
				{
					// both pointers still within their valid range
					if (func(a[p0], a[p1]))
						b[i] = a[p0++];
					else
						b[i] = a[p1++];
				}
			}
		}
	};
}

