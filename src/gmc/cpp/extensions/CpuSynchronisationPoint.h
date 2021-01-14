#ifndef CPUSYNCHRONISATIONPOINT_H
#define CPUSYNCHRONISATIONPOINT_H

#include <memory>
#include <vector>
#include <yamc_barrier.hpp>

namespace gpe {
namespace detail {

class CpuSynchronisationPoint
{
    CpuSynchronisationPoint() = default;
    static CpuSynchronisationPoint& instance();
public:
    CpuSynchronisationPoint(const CpuSynchronisationPoint&) = delete;
    void operator=(const CpuSynchronisationPoint&) = delete;
    /// sync_id is required because we need seperate bariers for each code location due to spurious wakeups (http://blog.vladimirprus.com/2005/07/spurious-wakeups.html)
    static void synchronise(unsigned sync_id);
    static void setThreadCount(unsigned n);
private:
    yamc::barrier<>* getBarrier(unsigned sync_id);
    std::mutex m_mutex;
    std::vector<std::pair<unsigned, std::unique_ptr<yamc::barrier<>>>> m_barriers;
    unsigned m_threadCount = 0;
};

} // namespace detail
} // namespace gpe

#endif // CPUSYNCHRONISATIONPOINT_H

