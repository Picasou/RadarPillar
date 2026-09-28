#pragma once

#include <thread>

/*
双线程分段执行(板上无 libgomp, OpenMP 不可用, std::thread 等效):
body(begin, end) 两段覆盖 [0, n); n < grain 时单线程(线程创建开销不划算);
未定义 FRONT_THREADS 时编译为纯串行(归因阶梯 B1/B2 用)
*/
#if defined(FRONT_THREADS)
template <class F>
inline void run_two_threads(int n, int grain, F&& body) {
    if (n < grain) { body(0, n); return; }
    const int mid = n / 2;
    std::thread worker(body, 0, mid);   // 前半段子线程, 后半段当前线程
    body(mid, n);
    worker.join();
}
#else
template <class F>
inline void run_two_threads(int n, int /*grain*/, F&& body) { body(0, n); }
#endif
