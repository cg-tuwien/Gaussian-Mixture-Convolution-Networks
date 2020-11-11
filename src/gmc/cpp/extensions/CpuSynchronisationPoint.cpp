#include "CpuSynchronisationPoint.h"
#include <thread>

gpe::detail::CpuSynchronisationPoint& gpe::detail::CpuSynchronisationPoint::instance()
{
    static CpuSynchronisationPoint instance;
    return instance;
}

void gpe::detail::CpuSynchronisationPoint::synchronise(unsigned sync_id)
{
    auto barrier = instance().getBarrier(sync_id);
    assert(barrier != nullptr);
    barrier->arrive_and_wait();
}

void gpe::detail::CpuSynchronisationPoint::setThreadCount(unsigned n)
{
    assert(n > 0);
    instance().m_threadCount = n;
    instance().m_barriers.clear();
}

yamc::barrier<>* gpe::detail::CpuSynchronisationPoint::getBarrier(unsigned sync_id)
{
    std::unique_lock<decltype(m_mutex)> mutex_lock(m_mutex);
    for (const auto& barier_pairs : m_barriers) {
        if (barier_pairs.first == sync_id)
            return barier_pairs.second.get();
    }
    m_barriers.emplace_back(sync_id, std::make_unique<yamc::barrier<>>(m_threadCount));
    return m_barriers.back().second.get();
}
