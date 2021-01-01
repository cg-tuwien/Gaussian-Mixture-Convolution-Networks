#ifndef NDEBUG

#include <catch2/catch.hpp>

#include "util/containers.h"
#include "util/algorithms.h"

using namespace gpe;

namespace  {
struct UnitTests {
    UnitTests() {
    }

    void testBitSet() {
        BitSet<32> bs;
        for (unsigned i = 0; i < 32; i++)
            REQUIRE(bs.isSet(i) == false);

        bs.set1(10);
        bs.set1(15);
        bs.set1(20);

        for (unsigned i = 0; i < 32; i++) {
            if (i != 10 && i != 15 && i != 20)
                REQUIRE(bs.isSet(i) == false);
            else
                REQUIRE(bs.isSet(i) == true);
        }

        for (unsigned i = 0; i < 32; i++)
            bs.set1(i);

        for (unsigned i = 0; i < 32; i++)
            REQUIRE(bs.isSet(i) == true);

        bs.set0(15);
        bs.set0(16);
        bs.set0(17);

        for (unsigned i = 0; i < 32; i++) {
            if (i != 15 && i != 16 && i != 17)
                REQUIRE(bs.isSet(i) == true);
            else
                REQUIRE(bs.isSet(i) == false);
        }

        for (unsigned i = 0; i < 32; i++)
            bs.set0(i);

        for (unsigned i = 0; i < 32; i++)
            REQUIRE(bs.isSet(i) == false);
    }

    template<typename T, uint32_t N>
    bool isValidHeap(const ArrayHeap<T, N>& heap) {
        for (unsigned i = 1; i < heap.m_data.size(); ++i) {
            if (heap.m_data[i] < heap.m_data[heap.parentIndex(i)])
                return false;
        }
        return true;
    }

    void testHeap() {

        {
            ArrayHeap<int, 2> h = Array<int, 2>{10, 5};
            REQUIRE(isValidHeap(h));
            REQUIRE(h.m_data[0] == 5);
            REQUIRE(h.m_data[1] == 10);
            REQUIRE(h.replaceRoot(8) == 5);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(12) == 8);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.m_data[0] == 10);
            REQUIRE(h.m_data[1] == 12);
        }
        {
            ArrayHeap<int, 3> h({10, 2, 5});
            REQUIRE(h.m_data[0] == 2);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(8) == 2);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(12) == 5);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(12) == 8);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(12) == 10);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(12) == 12);
            REQUIRE(isValidHeap(h));
        }
        {
            ArrayHeap<int, 8> h({10, 2, 5, 18, 12, 9, 4, 2});
            REQUIRE(h.m_data[0] == 2);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 2);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 2);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 4);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 5);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 9);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 10);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 12);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 18);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 100);
            REQUIRE(isValidHeap(h));
            REQUIRE(h.replaceRoot(100) == 100);
            REQUIRE(isValidHeap(h));
        }
    }

    void testArrayVectorConversion() {
        Vector<int, 8> v = {0, 1, 2, 3, 4};
        REQUIRE(v.size() == 5);
        Array<int, 8> a = to_array(v, -1);
        REQUIRE(a[0] == 0);
        REQUIRE(a[1] == 1);
        REQUIRE(a[2] == 2);
        REQUIRE(a[3] == 3);
        REQUIRE(a[4] == 4);
        REQUIRE(a[5] == -1);
        REQUIRE(a[6] == -1);
        REQUIRE(a[7] == -1);
    }

    void testArrayOuterProduct() {
        Array<int, 3> m1 = {1, 2, 3};
        Array<int, 4> m2 = {10, 20, 30, 40};
        Array2d<int, 3, 4> result = outer_product(m1, m2, gpe::functors::plus<int>);
        REQUIRE(result[0][0] == 11); REQUIRE(result[0][1] == 21); REQUIRE(result[0][2] == 31); REQUIRE(result[0][3] == 41);
        REQUIRE(result[1][0] == 12); REQUIRE(result[1][1] == 22); REQUIRE(result[1][2] == 32); REQUIRE(result[1][3] == 42);
        REQUIRE(result[2][0] == 13); REQUIRE(result[2][1] == 23); REQUIRE(result[2][2] == 33); REQUIRE(result[2][3] == 43);
    }

    void testArrayTransform1d() {
        Array<int, 3> m1 = {1, 2, 3};
        Array<int, 3> result = transform(m1, [](int v) { return v * 10; });
        REQUIRE(result[0] == 10);
        REQUIRE(result[1] == 20);
        REQUIRE(result[2] == 30);
    }

    void testArrayTransform2d() {
        Array2d<int, 2, 3> m1;
        m1[0] = (Array<int, 3>{1, 2, 3});
        m1[1] = (Array<int, 3>{4, 5, 6});
        Array2d<int, 2, 3> result = transform(m1, [](int v) { return v + 10; });
        REQUIRE(result[0][0] == 11); REQUIRE(result[0][1] == 12); REQUIRE(result[0][2] == 13);
        REQUIRE(result[1][0] == 14); REQUIRE(result[1][1] == 15); REQUIRE(result[1][2] == 16);
    }

    void testArrayCwiseFun1d() {
        Array<int, 3> m1 = {1, 2, 3};
        Array<int, 3> m2 = {10, 20, 30};
        Array<int, 3> result = cwise_fun(m1, m2, functors::plus<int>);
        REQUIRE(result[0] == 11);
        REQUIRE(result[1] == 22);
        REQUIRE(result[2] == 33);
    }

    void testArrayCwiseFun2d() {
        Array2d<int, 2> m1;
        m1[0] = (Array<int, 2>{1, 2});
        m1[1] = (Array<int, 2>{3, 4});
        Array2d<int, 2> m2;
        m2[0] = (Array<int, 2>{10, 100});
        m2[1] = (Array<int, 2>{1000, 10000});
        Array2d<int, 2> result = cwise_fun(m1, m2, functors::times<int>);
        REQUIRE(result[0][0] == 10); REQUIRE(result[0][1] == 200);
        REQUIRE(result[1][0] == 3000); REQUIRE(result[1][1] == 40000);
    }

    void testArrayCwiseFun2dX1d() {
        Array2d<int, 2, 3> m1;
        m1[0] = (Array<int, 3>{1, 2, 3});
        m1[1] = (Array<int, 3>{4, 5, 6});
        Array<int, 2> v = {1, 2};
        Array2d<int, 2, 3> result = cwise_fun(m1, v, functors::times<int>);
        REQUIRE(result[0][0] == 1); REQUIRE(result[0][1] == 2); REQUIRE(result[0][2] == 3);
        REQUIRE(result[1][0] == 8); REQUIRE(result[1][1] == 10); REQUIRE(result[1][2] == 12);
    }

    void testArrayCwiseFun1dX2d() {
        Array<int, 3> v = {1, 2, 10};
        Array2d<int, 2, 3> m1;
        m1[0] = (Array<int, 3>{1, 2, 3});
        m1[1] = (Array<int, 3>{4, 5, 6});
        Array2d<int, 2, 3> result = cwise_fun(v, m1, functors::times<int>);
        REQUIRE(result[0][0] == 1); REQUIRE(result[0][1] == 4); REQUIRE(result[0][2] == 30);
        REQUIRE(result[1][0] == 4); REQUIRE(result[1][1] == 10); REQUIRE(result[1][2] == 60);
    }

    void testArrayReduce1d() {
        Array<int, 3> m1 = {1, 2, 3};
        int result = reduce(m1, int(0), functors::plus<int>);
        REQUIRE(result == 6);
    }

    void testArrayReduce1dBool() {
        {
            Array<int, 3> m1 = {1, 2, 3};
            bool result = reduce(m1, int(1), functors::logical_and<int>);
            REQUIRE(result);
        }
        {
            Array<int, 3> m1 = {0, 2, 3};
            bool result = reduce(m1, int(1), functors::logical_and<int>);
            REQUIRE(!result);
        }
    }

    void testArrayReduce2d() {
        Array2d<int, 2> m1;
        m1[0] = (Array<int, 2>{1, 2});
        m1[1] = (Array<int, 2>{3, 4});
        int result = reduce(m1, int(1), functors::times<int>);
        REQUIRE(result == 1*2*3*4);
    }

    void testArrayReduceRows() {
        Array2d<int, 2, 3> m1;
        m1[0] = (Array<int, 3>{1, 2, 3});
        m1[1] = (Array<int, 3>{3, 4, 5});
        Array<int, 2> result = reduce_rows(m1, int(1), functors::times<int>);
        REQUIRE(result[0] == 1*2*3);
        REQUIRE(result[1] == 3*4*5);
    }

    void testArrayReduceCols() {
        Array2d<int, 2, 3> m1;
        m1[0] = (Array<int, 3>{1, 2, 3});
        m1[1] = (Array<int, 3>{3, 4, 5});
        Array<int, 3> result = reduce_cols(m1, int(1), functors::times<int>);
        REQUIRE(result[0] == 1*3);
        REQUIRE(result[1] == 2*4);
        REQUIRE(result[2] == 3*5);
    }

    void testOuterProduct() {
        Vector<int, 3> m1 = {1, 2, 3};
        Vector<int, 4> m2 = {10, 20, 30, 40};
        Vector2d<int, 3, 4> result = outer_product(m1, m2, gpe::functors::plus<int>);
        REQUIRE(result[0][0] == 11); REQUIRE(result[0][1] == 21); REQUIRE(result[0][2] == 31); REQUIRE(result[0][3] == 41);
        REQUIRE(result[1][0] == 12); REQUIRE(result[1][1] == 22); REQUIRE(result[1][2] == 32); REQUIRE(result[1][3] == 42);
        REQUIRE(result[2][0] == 13); REQUIRE(result[2][1] == 23); REQUIRE(result[2][2] == 33); REQUIRE(result[2][3] == 43);
    }

    void testTransform1d() {
        Vector<int, 3> m1 = {1, 2, 3};
        Vector<int, 3> result = transform(m1, [](int v) { return v * 10; });
        REQUIRE(result[0] == 10);
        REQUIRE(result[1] == 20);
        REQUIRE(result[2] == 30);
    }

    void testTransform2d() {
        Vector2d<int, 2, 3> m1;
        m1.push_back(Vector<int, 3>{1, 2, 3});
        m1.push_back(Vector<int, 3>{4, 5, 6});
        Vector2d<int, 2, 3> result = transform(m1, [](int v) { return v + 10; });
        REQUIRE(result[0][0] == 11); REQUIRE(result[0][1] == 12); REQUIRE(result[0][2] == 13);
        REQUIRE(result[1][0] == 14); REQUIRE(result[1][1] == 15); REQUIRE(result[1][2] == 16);
    }

    void testCwiseFun1d() {
        Vector<int, 3> m1 = {1, 2, 3};
        Vector<int, 3> m2 = {10, 20, 30};
        Vector<int, 3> result = cwise_fun(m1, m2, functors::plus<int>);
        REQUIRE(result[0] == 11);
        REQUIRE(result[1] == 22);
        REQUIRE(result[2] == 33);
    }

    void testCwiseFun2d() {
        Vector2d<int, 2> m1;
        m1.push_back(Vector<int, 2>{1, 2});
        m1.push_back(Vector<int, 2>{3, 4});
        Vector2d<int, 2> m2;
        m2.push_back(Vector<int, 2>{10, 100});
        m2.push_back(Vector<int, 2>{1000, 10000});
        Vector2d<int, 2> result = cwise_fun(m1, m2, functors::times<int>);
        REQUIRE(result[0][0] == 10); REQUIRE(result[0][1] == 200);
        REQUIRE(result[1][0] == 3000); REQUIRE(result[1][1] == 40000);
    }

    void testCwiseFun2dX1d() {
        Vector2d<int, 2, 3> m1;
        m1.push_back(Vector<int, 3>{1, 2, 3});
        m1.push_back(Vector<int, 3>{4, 5, 6});
        Vector<int, 2> v = {1, 2};
        Vector2d<int, 2, 3> result = cwise_fun(m1, v, functors::times<int>);
        REQUIRE(result[0][0] == 1); REQUIRE(result[0][1] == 2); REQUIRE(result[0][2] == 3);
        REQUIRE(result[1][0] == 8); REQUIRE(result[1][1] == 10); REQUIRE(result[1][2] == 12);
    }

    void testCwiseFun1dX2d() {
        Vector<int, 3> v = {1, 2, 10};
        Vector2d<int, 2, 3> m1;
        m1.push_back(Vector<int, 3>{1, 2, 3});
        m1.push_back(Vector<int, 3>{4, 5, 6});
        Vector2d<int, 2, 3> result = cwise_fun(v, m1, functors::times<int>);
        REQUIRE(result[0][0] == 1); REQUIRE(result[0][1] == 4); REQUIRE(result[0][2] == 30);
        REQUIRE(result[1][0] == 4); REQUIRE(result[1][1] == 10); REQUIRE(result[1][2] == 60);
    }

    void testReduce1d() {
        Vector<int, 3> m1 = {1, 2, 3};
        int result = reduce(m1, int(0), functors::plus<int>);
        REQUIRE(result == 6);
    }

    void testReduce1dBool() {
        {
            Vector<int, 3> m1 = {1, 2, 3};
            bool result = reduce(m1, int(1), functors::logical_and<int>);
            REQUIRE(result);
        }
        {
            Vector<int, 3> m1 = {0, 2, 3};
            bool result = reduce(m1, int(1), functors::logical_and<int>);
            REQUIRE(!result);
        }
    }

    void testReduce2d() {
        Vector2d<int, 2> m1;
        m1.push_back(Vector<int, 2>{1, 2});
        m1.push_back(Vector<int, 2>{3, 4});
        int result = reduce(m1, int(1), functors::times<int>);
        REQUIRE(result == 1*2*3*4);
    }

    void testReduceRows() {
        Vector2d<int, 2, 3> m1;
        m1.push_back(Vector<int, 3>{1, 2, 3});
        m1.push_back(Vector<int, 3>{3, 4, 5});
        Vector<int, 2> result = reduce_rows(m1, int(1), functors::times<int>);
        REQUIRE(result[0] == 1*2*3);
        REQUIRE(result[1] == 3*4*5);
    }

    void testReduceCols() {
        Vector2d<int, 2, 3> m1;
        m1.push_back(Vector<int, 3>{1, 2, 3});
        m1.push_back(Vector<int, 3>{3, 4, 5});
        Vector<int, 3> result = reduce_cols(m1, int(1), functors::times<int>);
        REQUIRE(result[0] == 1*3);
        REQUIRE(result[1] == 2*4);
        REQUIRE(result[2] == 3*5);
    }
};
} // anonymous namespace

TEST_CASE("algorithms") {
    UnitTests t;

    SECTION("bit set") {
        t.testBitSet();
    }
    SECTION("heap") {
        t.testHeap();
    }
    SECTION("conversion") {
        t.testArrayVectorConversion();
    }
    SECTION("arrays") {
        t.testArrayOuterProduct();
        t.testArrayTransform1d();
        t.testArrayTransform2d();
        t.testArrayCwiseFun1d();
        t.testArrayCwiseFun2d();
        t.testArrayCwiseFun2dX1d();
        t.testArrayCwiseFun1dX2d();
        t.testArrayReduce1d();
        t.testArrayReduce1dBool();
        t.testArrayReduce2d();
        t.testArrayReduceRows();
        t.testArrayReduceCols();
    }
    SECTION("vectors") {
        t.testOuterProduct();
        t.testTransform1d();
        t.testTransform2d();
        t.testCwiseFun1d();
        t.testCwiseFun2d();
        t.testCwiseFun2dX1d();
        t.testCwiseFun1dX2d();
        t.testReduce1d();
        t.testReduce1dBool();
        t.testReduce2d();
        t.testReduceRows();
        t.testReduceCols();
    }
}


#endif // not NDEBUG
