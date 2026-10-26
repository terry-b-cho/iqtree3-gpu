// Empty-kernel launch latency on the benchmark card (author action R14, round 8).
//
// Two quantities are reported, both in microseconds per launch:
//   sync   one launch of an empty <<<1,1>>> kernel followed by cudaDeviceSynchronize(), timed per
//          iteration on the host with std::chrono::steady_clock. This is the fixed cost a caller pays
//          when it launches a kernel and waits for it, which is what the engine does for every
//          likelihood call (launch, then read the result back).
//   async  N back-to-back launches followed by one synchronisation; per-launch time = total / N.
//          This is the launch throughput when nothing waits, a lower bound on the sync figure.
// Output is a TSV on stdout preceded by '#' header lines naming the device, driver, runtime and host.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unistd.h>

__global__ void empty_kernel() {}

static double pct(std::vector<double>& v, double p) {
    size_t k = (size_t)(p * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 100000;
    const int WARM = 1000;
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    char host[256]; gethostname(host, sizeof host);
    printf("# device\t%s\n# driver\t%d\n# runtime\t%d\n# host\t%s\n# n_per_variant\t%d\n",
           prop.name, drv, rt, host, N);
    for (int i = 0; i < WARM; ++i) { empty_kernel<<<1, 1>>>(); }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "warm-up failed\n"); return 1; }

    using clk = std::chrono::steady_clock;
    std::vector<double> t; t.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto a = clk::now();
        empty_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        auto b = clk::now();
        t.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    double mean = 0; for (double x : t) mean += x; mean /= t.size();
    double mn = *std::min_element(t.begin(), t.end());
    printf("variant\tn\tmedian_us\tp25_us\tp75_us\tmean_us\tmin_us\n");
    printf("sync\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", N, pct(t, 0.5), pct(t, 0.25), pct(t, 0.75), mean, mn);

    // async: repeat the block 20 times so a median over blocks is reportable
    std::vector<double> blk;
    const int B = 20, per = N / B;
    for (int b = 0; b < B; ++b) {
        auto a = clk::now();
        for (int i = 0; i < per; ++i) empty_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        auto e = clk::now();
        blk.push_back(std::chrono::duration<double, std::micro>(e - a).count() / per);
    }
    double bmean = 0; for (double x : blk) bmean += x; bmean /= blk.size();
    printf("async\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", per * B, pct(blk, 0.5), pct(blk, 0.25), pct(blk, 0.75),
           bmean, *std::min_element(blk.begin(), blk.end()));
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
