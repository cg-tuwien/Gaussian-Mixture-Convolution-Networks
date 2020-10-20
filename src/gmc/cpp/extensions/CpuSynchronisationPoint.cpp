#include "CpuSynchronisationPoint.h"

gpe::detail::CpuSynchronisationPoint& gpe::detail::CpuSynchronisationPoint::instance()
{
    static CpuSynchronisationPoint instance;
    return instance;
}

void gpe::detail::CpuSynchronisationPoint::synchronise()
{
    assert(instance().m_barrier);
    instance().m_barrier->arrive_and_wait();
}

void gpe::detail::CpuSynchronisationPoint::setThreadCount(int n)
{
    instance().m_barrier = std::make_unique<yamc::barrier<>>(n);
}
