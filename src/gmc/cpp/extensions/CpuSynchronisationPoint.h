#ifndef CPUSYNCHRONISATIONPOINT_H
#define CPUSYNCHRONISATIONPOINT_H

#include <memory>
#include <yamc_barrier.hpp>

namespace gpe {
namespace detail {

class CpuSynchronisationPoint
{
    CpuSynchronisationPoint() : m_barrier(nullptr) {}
    static CpuSynchronisationPoint& instance();
public:
    CpuSynchronisationPoint(const CpuSynchronisationPoint&) = delete;
    void operator=(const CpuSynchronisationPoint&) = delete;
    static void synchronise();
    static void setThreadCount(int n);
private:
    std::unique_ptr<yamc::barrier<>> m_barrier;
};

} // namespace detail
} // namespace gpe

#endif // CPUSYNCHRONISATIONPOINT_H

