# Project阶段5：并行化

## Task

本阶段的任务如课上所讲，需要你从lsmtree的get/put/delete过程中，找到一个比较耗费时间、并且可以并行化去做的逻辑。
用你学过的并行手段使其并行，提高该逻辑阶段的效率。

在报告中，你需要叙述：

1. 你要并行化的代码/逻辑是哪些，这些代码的运行在get/put/delete操作中的(平均)耗时是多少? 为了避免偶然性，你需要做大量操作(>10k次)，然后计算平均每次操作中的运行时间。
2. 你的并行化设计是什么?
3. 应用了你的并行设计后，运行时间降低了多少?

## 常见的并行场景

1. 单任务内的并行：例子：mapreduce，代码见 mapreduce-parallel.cc，在例子中并行统计了一篇文章中各个word的count

2. 多任务之间的并行：例子：并行读取文件：代码见 readfile-parallel.cc，在例子中并行读取了一个10000000个long int的二进制文件

### 线程池example

```c++
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
```

可以将任务插入到线程池中避免频繁的线程创建和销毁。
