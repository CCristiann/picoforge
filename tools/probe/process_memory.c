/* Read-only macOS process/system counters. Byte counts are not interchangeable:
 * RSS, charged physical footprint and Metal allocations describe different sets.
 * Build: clang -std=c11 -Wall -Wextra -Wconversion tools/probe/process_memory.c
 *        -o build/process_memory
 * Usage: build/process_memory PID (or no PID to sample this probe).
 */
#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <limits.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    long pid = getpid();
    if (argc == 2) {
        char *end;
        errno = 0;
        pid = strtol(argv[1], &end, 10);
        if (errno || !*argv[1] || *end || pid <= 0 || pid > INT_MAX) return 2;
    } else if (argc != 1) return 2;
    struct rusage_info_v4 r = {0};
    struct xsw_usage swap = {0};
    int pressure = 0;
    size_t size = sizeof(swap);
    if (proc_pid_rusage((int)pid, RUSAGE_INFO_V4, (rusage_info_t *)&r) != 0 ||
        sysctlbyname("vm.swapusage", &swap, &size, NULL, 0) != 0) {
        perror("process_memory");
        return 1;
    }
    size = sizeof(pressure);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure,
                    &size, NULL, 0) != 0) {
        perror("memory pressure");
        return 1;
    }
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t status = host_statistics64(host, HOST_VM_INFO64,
                                            (host_info64_t)&vm, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (status != KERN_SUCCESS) return 1;
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 1;
    printf("{\"pid\":%ld,\"monotonic_s\":%.9f,"
           "\"resident_bytes\":%" PRIu64 ",\"phys_footprint_bytes\":%" PRIu64 ","
           "\"lifetime_peak_phys_footprint_bytes\":%" PRIu64 ","
           "\"disk_read_bytes\":%" PRIu64 ",\"disk_write_bytes\":%" PRIu64 ","
           "\"system_swap_used_bytes\":%" PRIu64 ",\"system_pressure_level\":%d,"
           "\"system_swapins_pages\":%" PRIu64 ",\"system_swapouts_pages\":%" PRIu64 ","
           "\"page_bytes\":%ld,\"system_wired_pages\":%u,"
           "\"system_compressor_pages\":%u}\n",
           pid, (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9,
           r.ri_resident_size, r.ri_phys_footprint,
           r.ri_lifetime_max_phys_footprint,
           r.ri_diskio_bytesread, r.ri_diskio_byteswritten,
           (uint64_t)swap.xsu_used, pressure, vm.swapins, vm.swapouts,
           sysconf(_SC_PAGESIZE), vm.wire_count, vm.compressor_page_count);
    return 0;
}
