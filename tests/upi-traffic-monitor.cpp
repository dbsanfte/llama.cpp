#include "upi-traffic-monitor.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <cstring>
#include <iomanip>

UpiTrafficMonitor::UpiTrafficMonitor() 
    : has_upi_support_(false), has_numa_stats_(false), num_numa_nodes_(0) {
}

UpiTrafficMonitor::~UpiTrafficMonitor() {
    for (int fd : upi_device_fds_) {
        if (fd >= 0) {
            close(fd);
        }
    }
}

bool UpiTrafficMonitor::initialize() {
    // Get number of NUMA nodes
    num_numa_nodes_ = numa_num_configured_nodes();
    
    // Try UPI hardware monitoring first
    has_upi_support_ = setup_upi_monitoring();
    
    // Always enable NUMA stats as fallback
    has_numa_stats_ = (access("/proc/vmstat", R_OK) == 0);
    
    return has_upi_support_ || has_numa_stats_;
}

bool UpiTrafficMonitor::setup_upi_monitoring() {
    // Try to access UPI uncore devices
    for (int i = 0; i < 3; ++i) {  // Typically 0-2 UPI links
        std::string device_path = "/sys/bus/event_source/devices/uncore_upi_" + std::to_string(i);
        if (access(device_path.c_str(), R_OK) == 0) {
            // UPI device exists, try to open performance counter
            struct perf_event_attr attr = {};
            attr.type = PERF_TYPE_RAW;
            attr.size = sizeof(struct perf_event_attr);
            attr.config = 0x2;  // Generic UPI traffic event
            attr.disabled = 1;
            attr.exclude_kernel = 0;
            attr.exclude_hv = 0;
            
            int fd = syscall(SYS_perf_event_open, &attr, -1, 0, -1, 0);
            if (fd >= 0) {
                upi_device_fds_.push_back(fd);
            } else {
                upi_device_fds_.push_back(-1);
            }
        } else {
            upi_device_fds_.push_back(-1);
        }
    }
    
    // Enable any valid counters
    bool any_enabled = false;
    for (int fd : upi_device_fds_) {
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
            any_enabled = true;
        }
    }
    
    return any_enabled;
}

uint64_t UpiTrafficMonitor::read_upi_counter(int device_index) {
    if (device_index >= (int)upi_device_fds_.size() || upi_device_fds_[device_index] < 0) {
        return 0;
    }
    
    uint64_t count = 0;
    ssize_t bytes_read = read(upi_device_fds_[device_index], &count, sizeof(count));
    return (bytes_read == sizeof(count)) ? count : 0;
}

uint64_t UpiTrafficMonitor::read_numa_foreign_accesses() {
    std::ifstream file("/proc/vmstat");
    if (!file.is_open()) {
        return 0;
    }
    
    std::string line;
    uint64_t numa_foreign = 0;
    
    while (std::getline(file, line)) {
        if (line.find("numa_foreign") == 0) {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> numa_foreign;
            break;
        }
    }
    
    return numa_foreign;
}

uint64_t UpiTrafficMonitor::read_numa_local_accesses() {
    std::ifstream file("/proc/vmstat");
    if (!file.is_open()) {
        return 0;
    }
    
    std::string line;
    uint64_t numa_local = 0;
    
    while (std::getline(file, line)) {
        if (line.find("numa_local") == 0) {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> numa_local;
            break;
        }
    }
    
    return numa_local;
}

uint64_t UpiTrafficMonitor::get_current_timestamp_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

UpiTrafficMonitor::TrafficSnapshot UpiTrafficMonitor::take_snapshot() {
    TrafficSnapshot snapshot = {};
    snapshot.timestamp_ns = get_current_timestamp_ns();
    
    // Try UPI hardware counters first
    if (has_upi_support_) {
        uint64_t total_upi_traffic = 0;
        for (size_t i = 0; i < upi_device_fds_.size(); ++i) {
            uint64_t counter_val = read_upi_counter(i);
            total_upi_traffic += counter_val;
        }
        snapshot.total_cross_node_transfers = total_upi_traffic;
        snapshot.total_local_node_transfers = 0;  // UPI only measures cross-node
    }
    
    // Use NUMA statistics (always available as fallback)
    if (has_numa_stats_) {
        snapshot.total_cross_node_transfers = read_numa_foreign_accesses();
        snapshot.total_local_node_transfers = read_numa_local_accesses();
    }
    
    // Per-node statistics if available
    snapshot.per_node_remote_accesses.resize(num_numa_nodes_, 0);
    for (int node = 0; node < num_numa_nodes_; ++node) {
        std::string numa_stat_path = "/sys/devices/system/node/node" + std::to_string(node) + "/numastat";
        std::ifstream file(numa_stat_path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (line.find("other_node") == 0) {
                    std::istringstream iss(line);
                    std::string key;
                    uint64_t value;
                    iss >> key >> value;
                    snapshot.per_node_remote_accesses[node] = value;
                    break;
                }
            }
        }
    }
    
    return snapshot;
}

double UpiTrafficMonitor::calculate_imbalance_change(const TrafficSnapshot& before, const TrafficSnapshot& after) {
    double before_imbalance = before.get_imbalance_percentage();
    double after_imbalance = after.get_imbalance_percentage();
    return after_imbalance - before_imbalance;
}

bool UpiTrafficMonitor::validate_numa_optimization(const TrafficSnapshot& before, const TrafficSnapshot& after, 
                                                   double max_imbalance_percent) {
    double imbalance_change = calculate_imbalance_change(before, after);
    
    // NUMA optimization should reduce cross-node traffic, so imbalance should decrease or stay low
    if (imbalance_change > max_imbalance_percent) {
        return false;  // Imbalance increased too much
    }
    
    // Additional check: absolute imbalance shouldn't exceed threshold
    double final_imbalance = after.get_imbalance_percentage();
    if (final_imbalance > max_imbalance_percent + 10.0) {  // Allow 10% extra for baseline
        return false;
    }
    
    return true;
}

std::string UpiTrafficMonitor::get_traffic_report(const TrafficSnapshot& before, const TrafficSnapshot& after) {
    std::ostringstream report;
    
    // Calculate deltas
    uint64_t cross_node_delta = (after.total_cross_node_transfers >= before.total_cross_node_transfers) ?
        (after.total_cross_node_transfers - before.total_cross_node_transfers) : 0;
    uint64_t local_node_delta = (after.total_local_node_transfers >= before.total_local_node_transfers) ?
        (after.total_local_node_transfers - before.total_local_node_transfers) : 0;
    
    double time_diff_ms = (after.timestamp_ns - before.timestamp_ns) / 1000000.0;
    
    report << std::fixed << std::setprecision(2);
    report << "🔍 UPI TRAFFIC ANALYSIS (" << time_diff_ms << " ms)\n";
    report << "=====================================\n";
    
    if (has_upi_support_) {
        report << "📊 Hardware UPI Counters:\n";
        report << "  Cross-node transfers: " << cross_node_delta << "\n";
    }
    
    if (has_numa_stats_) {
        report << "📊 NUMA Statistics:\n";
        report << "  Foreign accesses: " << cross_node_delta << "\n";
        report << "  Local accesses: " << local_node_delta << "\n";
        report << "  Imbalance before: " << before.get_imbalance_percentage() << "%\n";
        report << "  Imbalance after: " << after.get_imbalance_percentage() << "%\n";
        report << "  Imbalance change: " << calculate_imbalance_change(before, after) << "%\n";
    }
    
    // Per-node breakdown
    if (!after.per_node_remote_accesses.empty()) {
        report << "📊 Per-Node Remote Accesses:\n";
        for (size_t i = 0; i < after.per_node_remote_accesses.size(); ++i) {
            uint64_t node_delta = (after.per_node_remote_accesses[i] >= before.per_node_remote_accesses[i]) ?
                (after.per_node_remote_accesses[i] - before.per_node_remote_accesses[i]) : 0;
            report << "  Node " << i << ": " << node_delta << "\n";
        }
    }
    
    return report.str();
}
