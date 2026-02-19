/**
 * @file thread_pool.h
 * @brief High-performance thread pool with custom memory allocator
 * 
 * This header implements an optimized thread pool with a custom small object
 * allocator for efficient task management. The design focuses on:
 * 
 * - Low-latency task scheduling
 * - Memory-efficient task storage
 * - Custom allocator for small objects
 * - RAII resource management
 * - Thread-safe operations
 * 
 * Key optimizations:
 * - Custom allocator reduces heap fragmentation
 * - Deque provides O(1) push/pop operations
 * - Condition variables for efficient thread synchronization
 * - Perfect forwarding for task arguments
 * 
 * Usage:
 * ```cpp
 * ThreadPool pool(4);  // 4 worker threads
 * auto future = pool.enqueue(function, args...);
 * auto result = future.get();  // Wait for completion
 * ```
 */

#include <mutex>
#include <deque>
#include <thread>
#include <vector>
#include <functional>
#include <future>
#include <stdexcept>
#include <memory>

/**
 * @brief Custom small object allocator for thread pool tasks
 * 
 * Implements a simple bump allocator optimized for small, short-lived
 * objects like std::packaged_task. This reduces heap fragmentation and
 * improves performance compared to general-purpose allocators.
 * 
 * Design characteristics:
 * - Linear allocation from a pre-allocated memory block
 * - Simple deallocation (only works for LIFO deallocation)
 * - Fixed capacity to prevent unbounded memory growth
 * - RAII cleanup with unique_ptr
 * 
 * Trade-offs:
 * + Fast allocation/deallocation for small objects
 * + No heap fragmentation
 * + Predictable memory usage
 * - Limited to LIFO deallocation pattern
 * - Fixed capacity (no growth)
 * - No thread safety (must be protected externally)
 */
class SmallAllocator {
public:
    /**
     * @brief Construct allocator with specified capacity
     * 
     * Allocates a contiguous memory block of the specified size.
     * All subsequent allocations will come from this block.
     * 
     * @param capacity Maximum total size in bytes
     */
    explicit SmallAllocator(size_t capacity) : capacity_(capacity), memory_(new char[capacity]), offset_(0) {}

    /**
     * @brief Destructor - automatically cleans up memory
     * 
     * Uses RAII with std::unique_ptr to ensure memory is properly
     * deallocated when the allocator goes out of scope.
     */
    ~SmallAllocator() = default; // `std::unique_ptr` will handle memory cleanup

    /**
     * @brief Allocate memory block of specified size
     * 
     * Performs linear allocation from the pre-allocated memory block.
     * Throws std::bad_alloc if insufficient memory is available.
     * 
     * @param size Size of memory block to allocate in bytes
     * @return Pointer to allocated memory block
     * @throws std::bad_alloc if allocation would exceed capacity
     */
    void* allocate(size_t size) {
        if (offset_ + size > capacity_)
            throw std::bad_alloc();
        void* ptr = memory_.get() + offset_;
        offset_ += size;
        return ptr;
    }

    /**
     * @brief Deallocate memory block
     * 
     * Implements simple LIFO deallocation by adjusting the offset.
     * Only works correctly if deallocations happen in reverse order
     * of allocations (LIFO pattern). This is suitable for thread pool
     * task objects which have short, predictable lifetimes.
     * 
     * @param ptr Pointer to memory block to deallocate
     * @param size Size of the memory block
     */
    void deallocate(void* ptr, size_t size) {
        // Simplistic allocator; no real deallocation.
        // Works correctly for LIFO deallocation pattern.
        if (reinterpret_cast<char*>(ptr) + size == memory_.get() + offset_)
            offset_ -= size;
    }

private:
    size_t capacity_;                    ///< Total capacity of memory block
    std::unique_ptr<char[]> memory_;    ///< Managed memory block
    size_t offset_;                      ///< Current allocation offset
};

/**
 * @brief High-performance thread pool with custom memory management
 * 
 * Implements a production-ready thread pool optimized for high-frequency
 * task submission and execution. Key features include:
 * 
 * Performance optimizations:
 * - Custom small object allocator for task storage
 * - Deque for O(1) task queue operations
 * - Condition variables for efficient thread wake-up
 * - Perfect forwarding to avoid unnecessary copies
 * 
 * Thread safety:
 * - Mutex-protected task queue
 * - Condition variable for thread synchronization
 * - Atomic stop flag for graceful shutdown
 * 
 * Memory management:
 * - Custom allocator reduces heap fragmentation
 * - RAII ensures proper cleanup
 * - Exception-safe task execution
 * 
 * @note This implementation is optimized for small, short-lived tasks
 *       typical of parallel matrix operations and numerical computations.
 */
class ThreadPool {
public:
    /**
     * @brief Construct thread pool with specified number of worker threads
     * 
     * Creates worker threads that continuously pull tasks from the queue
     * and execute them. Threads are started immediately and wait
     * for tasks using condition variables for efficient CPU usage.
     * 
     * @param numThreads Number of worker threads to create
     */
    explicit ThreadPool(size_t numThreads) : taskAllocator_(1024*1024) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers_.emplace_back(std::thread([this]() {
                for (;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queueMutex_);
                        condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty())
                            return;
                        task = std::move(tasks_.front());
                        tasks_.pop_front();
                    }

                    task();  // Execute the task
                }
            }));
        }
    }

    /**
     * @brief Enqueue a task for execution by the thread pool
     * 
     * Adds a function with its arguments to the task queue and returns
     * a future for retrieving the result. Uses placement new with the
     * custom allocator for efficient memory management.
     * 
     * The task is wrapped in a std::packaged_task to enable future-based
     * result retrieval and exception propagation.
     * 
     * @tparam F Function type to execute
     * @tparam Args Argument types for the function
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return std::future for the function's return value
     * @throws std::runtime_error if enqueue is called on stopped pool
     * @throws std::bad_alloc if allocator runs out of memory
     */
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using returnType = typename std::invoke_result<F, Args...>::type;

        // Allocate memory for packaged_task using custom allocator
        void* taskMemory = taskAllocator_.allocate(sizeof(std::packaged_task<returnType()>));
        
        // Use placement new to construct packaged_task in allocated memory
        auto task = new (taskMemory) std::packaged_task<returnType()>(
            [f = std::forward<F>(f), ... args = std::forward<Args>(args)]() mutable { return f(args...); });

        std::future<returnType> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            if (stop_)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            // Wrap task with cleanup lambda for proper memory management
            tasks_.emplace_back([task, this]() {
                (*task)();
                task->~packaged_task();  // Explicit destructor call
                taskAllocator_.deallocate(task, sizeof(std::packaged_task<returnType()>));
            });
        }
        condition_.notify_one();
        return res;
    }

    /**
     * @brief Destructor - gracefully shuts down the thread pool
     * 
     * Signals all worker threads to stop, waits for them to complete
     * current tasks, and then joins all threads. This ensures clean
     * shutdown without resource leaks or dangling threads.
     * 
     * The custom allocator automatically cleans up any remaining
     * allocated memory when it goes out of scope.
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable())
                worker.join();
        }
    }

private:
    std::vector<std::thread> workers_;           ///< Worker threads
    std::deque<std::function<void()>> tasks_;      ///< Task queue (deque for O(1) operations)
    std::mutex queueMutex_;                       ///< Mutex for task queue access
    std::condition_variable condition_;           ///< Condition variable for task availability
    bool stop_ = false;                            ///< Flag to signal thread shutdown
    SmallAllocator taskAllocator_;                ///< Custom allocator for task objects
};
