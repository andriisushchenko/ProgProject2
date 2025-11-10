//Compiler - gcc
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <iomanip>
#include <functional>

std::vector<double> generate_data(size_t N, int seed, double dis_start,double dis_end) {
    std::vector<double> data(N);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(dis_start, dis_end);
    for (auto& x : data) {
        x = dis(gen);
    }
    return data;
}

double fast_op(double x) {
    return x + 1.0;
}

double slow_op(double x) {
    double s = 0.0;
    for (int i = 0; i < 100; ++i) {
        s += std::sin(static_cast<double>(i));
    }
    return s + x;
};

double measure_transform(const std::vector<double>& data, auto op) {
    std::vector<double> result(data.size());
    auto start = std::chrono::high_resolution_clock::now();
    std::transform(data.cbegin(), data.cend(), result.begin(), op);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double measure_policy_transform(const std::vector<double>& data, auto op, auto policy) {
    std::vector<double> result(data.size());
    auto start = std::chrono::high_resolution_clock::now();
    std::transform(policy, data.cbegin(), data.cend(), result.begin(), op);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double measure_custom_parallel_transform(const std::vector<double>& data, auto op, int K) {
    if (K <= 0) K = 1;
    size_t N = data.size();
    std::vector<double> result(N);
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    size_t chunk_size = N / K;
    size_t remainder = N % K;

    size_t current_start = 0;
    for (int i = 0; i < K; ++i) {
        size_t current_end = current_start + chunk_size + (i < remainder ? 1 : 0);
        threads.emplace_back([&, current_start, current_end]() {
            std::transform(data.cbegin() + current_start, data.cbegin() + current_end,
                           result.begin() + current_start, op);
        });
        current_start = current_end;
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void measure_and_print_custom_parallel(const std::vector<double>& data,
                                       std::function<double(double)> op,
                                       const std::vector<int>& K_values,
                                       unsigned int num_cores) {
    std::cout << std::left << std::setw(10) << "K" << std::setw(15) << "Time (seconds)" << std::endl;

    double min_time = std::numeric_limits<double>::max();
    int best_K = 1;

    for (int K : K_values) {
        double time_custom = measure_custom_parallel_transform(data, op, K);
        std::cout << std::left << std::setw(10) << K << std::setw(15) << time_custom << std::endl;
        if (time_custom < min_time) {
            min_time = time_custom;
            best_K = K;
        }
    }

    std::cout << "Best K: " << best_K << std::endl;
    std::cout << "Relation to processor threads: " << best_K << " / " << num_cores << " = " << static_cast<double>(best_K) /
        num_cores << std::endl << std::endl;
}

void run_and_print_all_transforms(const std::vector<double>& data,
                        const std::vector<std::pair<std::string,
                        std::function<double(double)>>>& ops,
                        const std::vector<int>& K_values,
                        unsigned int num_cores)
{
    for (const auto& [op_name, op] : ops) {
        std::cout << "Operation: " << op_name << std::endl;

        // Transform no policy
        double time = measure_transform(data, op);
        std::cout << "Sequential transform (no policy): " << time << " seconds" << std::endl;

        // Transform seq policy
        double time_seq_policy = measure_policy_transform(data, op, std::execution::seq);
        std::cout << "Transform with seq policy: " << time_seq_policy << " seconds" << std::endl;

        // Transform par policy
        double time_par_policy = measure_policy_transform(data, op, std::execution::par);
        std::cout << "Transform with par policy: " << time_par_policy << " seconds" << std::endl;

        // Transform par_unseq policy
        double time_par_unseq_policy = measure_policy_transform(data, op, std::execution::par_unseq);
        std::cout << "Transform with par_unseq policy: " << time_par_unseq_policy << " seconds" << std::endl;

        // Custom Transform
        std::cout << "Custom parallel transform:" << std::endl;
        measure_and_print_custom_parallel(data, op, K_values, num_cores);

        std::cout << "-------------------------------------" << std::endl;
    }
}

int main() {
    std::vector<size_t> sizes = {10'000, 100'000, 1'000'000};
    unsigned int num_cores = std::thread::hardware_concurrency();
    std::cout << "Number of processor threads: " << num_cores << std::endl << std::endl;

    std::vector<int> K_values = {1, 2, 4, 8, 16, 32};

    std::vector<std::pair<std::string, std::function<double(double)>>> ops = {
        {"fast", fast_op},
        {"slow", slow_op}
    };

    for (const auto& size : sizes) {
        auto data = generate_data(size, 42, 0.0, 1.0);
        std::cout << "Data size: " << size << std::endl;

        run_and_print_all_transforms(data, ops, K_values, num_cores);
    }


    return 0;
}