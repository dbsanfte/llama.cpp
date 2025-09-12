#include <numa.h>
#include <stdio.h>
int main() {
    printf("NUMA available: %s\n", numa_available() ? "true" : "false");
    printf("NUMA nodes: %d\n", numa_num_configured_nodes());
    printf("NUMA max nodes: %d\n", numa_max_node() + 1);
    return 0;
}
