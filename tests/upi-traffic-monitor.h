#pragma once

#include <vector>
#include <cstdint>
#include <string>

/**
 * UPI Traffic Monitor for NUMA optimization validation
 * 
 * Monitors inter-node traffic to ensure NUMA optimizations actually reduce
 * cross-node communication. Uses Intel uncore PMU when available, fallback
 * to NUMA memory statistics for validation.
 */
class UpiTrafficMonitor {
public:
    struct TrafficSnapshot {
        uint64_t timestamp_ns;
        uint64_t total_cross_node_transfers;
        uint64_t total_local_node_transfers;
        std::vector<uint64_t> per_node_remote_accesses;
        
        // Calculate imbalance percentage
        double get_imbalance_percentage() const {
            if (total_local_node_transfers == 0) return 100.0;
            return (double)total_cross_node_transfers / 
                   (double)(total_cross_node_transfers + total_local_node_transfers) * 100.0;
        }
    };
    
    UpiTrafficMonitor();
    ~UpiTrafficMonitor();
    
    // Initialize UPI monitoring (returns false if not supported)
    bool initialize();
    
    // Take a snapshot of current UPI traffic
    TrafficSnapshot take_snapshot();
    
    // Compare two snapshots and return imbalance change
    double calculate_imbalance_change(const TrafficSnapshot& before, const TrafficSnapshot& after);
    
    // Validate that imbalance didn't increase beyond threshold
    bool validate_numa_optimization(const TrafficSnapshot& before, const TrafficSnapshot& after, 
                                   double max_imbalance_percent = 5.0);
    
    // Get human-readable traffic report
    std::string get_traffic_report(const TrafficSnapshot& before, const TrafficSnapshot& after);
    
private:
    bool has_upi_support_;
    bool has_numa_stats_;
    int num_numa_nodes_;
    std::vector<int> upi_device_fds_;
    
    // UPI hardware monitoring
    bool setup_upi_monitoring();
    uint64_t read_upi_counter(int device_index);
    
    // Fallback NUMA statistics monitoring
    uint64_t read_numa_foreign_accesses();
    uint64_t read_numa_local_accesses();
    uint64_t get_current_timestamp_ns();
};
