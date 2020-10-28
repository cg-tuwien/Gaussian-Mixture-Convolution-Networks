#include "containers.h"

namespace gpe {
namespace  {


static struct UnitTests {
    UnitTests() {
        testBitSet();
        testOuterProduct();
        testCwiseFun1d();
        testCwiseFun2d();
        testReduce1d();
        testReduce1dBool();
        testReduce2d();
    }

    void testBitSet() {
        BitSet<32> bs;
        for (unsigned i = 0; i < 32; i++)
            assert(bs.isSet(i) == false);

        bs.set1(10);
        bs.set1(15);
        bs.set1(20);

        for (unsigned i = 0; i < 32; i++) {
            if (i != 10 && i != 15 && i != 20)
                assert(bs.isSet(i) == false);
            else
                assert(bs.isSet(i) == true);
        }

        for (unsigned i = 0; i < 32; i++)
            bs.set1(i);

        for (unsigned i = 0; i < 32; i++)
            assert(bs.isSet(i) == true);

        bs.set0(15);
        bs.set0(16);
        bs.set0(17);

        for (unsigned i = 0; i < 32; i++) {
            if (i != 15 && i != 16 && i != 17)
                assert(bs.isSet(i) == true);
            else
                assert(bs.isSet(i) == false);
        }

        for (unsigned i = 0; i < 32; i++)
            bs.set0(i);

        for (unsigned i = 0; i < 32; i++)
            assert(bs.isSet(i) == false);
    }

    void testOuterProduct() {
        Vector<int, 3> m1 = {1, 2, 3};
        Vector<int, 4> m2 = {10, 20, 30, 40};
        Vector2d<int, 3, 4> result = outer_product(m1, m2, functors::plus<int>);
        assert(result[0][0] == 11); assert(result[0][1] == 21); assert(result[0][2] == 31); assert(result[0][3] == 41);
        assert(result[1][0] == 12); assert(result[1][1] == 22); assert(result[1][2] == 32); assert(result[1][3] == 42);
        assert(result[2][0] == 13); assert(result[2][1] == 23); assert(result[2][2] == 33); assert(result[2][3] == 43);
    }

    void testCwiseFun1d() {
        Vector<int, 3> m1 = {1, 2, 3};
        Vector<int, 3> m2 = {10, 20, 30};
        Vector<int, 3> result = cwise_fun(m1, m2, functors::plus<int>);
        assert(result[0] == 11);
        assert(result[1] == 22);
        assert(result[2] == 33);
    }

    void testCwiseFun2d() {
        Vector2d<int, 2> m1;
        m1.push_back(Vector<int, 2>{1, 2});
        m1.push_back(Vector<int, 2>{3, 4});
        Vector2d<int, 2> m2;
        m2.push_back(Vector<int, 2>{10, 100});
        m2.push_back(Vector<int, 2>{1000, 10000});
        Vector2d<int, 2> result = cwise_fun(m1, m2, functors::times<int>);
        assert(result[0][0] == 10); assert(result[0][1] == 200);
        assert(result[1][0] == 3000); assert(result[1][1] == 40000);
    }

    void testReduce1d() {
        Vector<int, 3> m1 = {1, 2, 3};
        int result = reduce(m1, int(0), functors::plus<int>);
        assert(result == 6);
    }

    void testReduce1dBool() {
        {
            Vector<int, 3> m1 = {1, 2, 3};
            bool result = reduce(m1, int(1), functors::logical_and<int>);
            assert(result);
        }
        {
            Vector<int, 3> m1 = {0, 2, 3};
            bool result = reduce(m1, int(1), functors::logical_and<int>);
            assert(!result);
        }
    }

    void testReduce2d() {
        Vector2d<int, 2> m1;
        m1.push_back(Vector<int, 2>{1, 2});
        m1.push_back(Vector<int, 2>{3, 4});
        int result = reduce(m1, int(1), functors::times<int>);
        assert(result == 1*2*3*4);
    }
} unit_tests;

} // anonymous namespace
} // namespace gpe
