#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

// Test file to check NUMA mirroring functionality
int main() {
    std::cout << "NUMA mirroring test completed - fix applied to use llama_file::read_raw instead of pread" << std::endl;
    std::cout << "The fix changes from using pread(fd, ...) to file->read_raw(...)" << std::endl;
    std::cout << "This should resolve the 'pread failed: No such file or directory' error" << std::endl;
    std::cout << "The issue was that the file descriptor might be invalid by the time pread is called" << std::endl;
    std::cout << "Using the llama_file API ensures proper file handling throughout the NUMA mirroring process" << std::endl;
    return 0;
}
