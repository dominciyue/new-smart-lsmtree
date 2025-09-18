#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

void simulate_work(int id, int duration_ms = 100) {
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  std::cout << "Task " << id << " completed by thread "
            << std::this_thread::get_id() << std::endl;
}

//-------------------------------------------------------------------------------------------------
// 线程池 (Thread Pool)
//-------------------------------------------------------------------------------------------------
class ThreadPool {
public:
  ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty()) {
              return;
            }
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <class F> void enqueue(F &&f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
      worker.join();
    }
  }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

void test_thread_pool() {
  ThreadPool pool(4); // 创建一个包含4个工作线程的线程池

  std::atomic<int> completed_tasks_count(0);

  for (int i = 0; i < 10; ++i) {
    pool.enqueue([i, &completed_tasks_count] {
      std::cout << "Start Job " << i << " on thread "
                << std::this_thread::get_id() << std::endl;
      simulate_work(i, 50 + i * 10);
      std::cout << "Finish Job " << i << std::endl;
      completed_tasks_count++;
    });
  }

  while (completed_tasks_count < 10) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cout << "All Task Finished" << std::endl;
}

int main() {
  test_thread_pool();

  return 0;
}