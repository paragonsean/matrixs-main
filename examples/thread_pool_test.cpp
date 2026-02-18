#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "thread_pool.h" // Include your ThreadPool implementation

// Function to generate a vector of random numbers
std::vector<int> generateRandomVector(size_t size, int min = 1, int max = 10000) {
    static std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(min, max);

    std::vector<int> result(size);
    for (auto& num : result) {
        num = distribution(generator);
    }
    return result;
}

// Function to check if a number is prime
bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

// Main function
int main() {
    const size_t numVectors = 25;
    const size_t vectorSize = 10000;
    const size_t numThreads = 4;

    // Generate vectors of random numbers
    std::vector<std::vector<int>> vectors(numVectors);
    for (auto& vec : vectors) {
        vec = generateRandomVector(vectorSize);
    }

    // Measure execution time with ThreadPool
    ThreadPool pool(numThreads);
    std::vector<std::future<std::vector<int>>> futures;

    auto start = std::chrono::high_resolution_clock::now();

    for (auto& vec : vectors) {
        futures.emplace_back(pool.enqueue([vec]() {
            std::vector<int> primes;
            for (int num : vec) {
                if (isPrime(num)) {
                    primes.push_back(num);
                }
            }
            return primes;
        }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        std::cout << "Vector " << i + 1 << " primes found: " << futures[i].get().size() << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> threadPoolTime = end - start;
    std::cout << "Execution time with ThreadPool: " << threadPoolTime.count() << " seconds\n";

    // Measure execution time sequentially
    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numVectors; ++i) {
        std::vector<int> primes;
        for (int num : vectors[i]) {
            if (isPrime(num)) {
                primes.push_back(num);
            }
        }
        std::cout << "Sequential Vector " << i + 1 << " primes found: " << primes.size() << std::endl;
    }

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sequentialTime = end - start;
    std::cout << "Execution time sequentially: " << sequentialTime.count() << " seconds\n";

    return 0;
}
