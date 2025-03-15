# DSAI3202

Sequential	7.1991
Threading (1 thread per function)	7.4002
Advanced Threading (2 threads per function)	7.2848
Multiprocessing (1 process per function)	3.8263
Multiprocessing (2 processes per function)	3.8263
Computing Speedup:
Threading (1 thread per function) resulted in a speedup of approximately 0.973, indicating no improvement over sequential execution.
Advanced threading (2 threads per function) resulted in a speedup of 0.988, which is also very close to sequential execution.
Multiprocessing (1 process per function) achieved a speedup of 1.882, meaning it was almost twice as fast as sequential execution.
Multiprocessing (2 processes per function) also had a speedup of 1.882, showing that adding more processes did not improve performance further.
Computing efficiency:
Threading with 2 threads had an efficiency of 0.487, which is quite low.
Advanced threading with 4 threads had an efficiency of 0.247, showing that adding more threads does not necessarily improve performance due to the Global Interpreter Lock (GIL).
Multiprocessing with 2 processes had an efficiency of 0.941, meaning the CPU was utilized effectively.
Multiprocessing with 4 processes had an efficiency of 0.470, which was lower but still better than threading.
Speedup Using Amdahl’s Law:
With 2 threads, the predicted speedup is 1.82.
With 4 threads, the speedup is 1.41.
With 2 processes, the speedup is also 1.82.
With 4 processes, the speedup remains 1.41, showing diminishing returns when adding more processes.
Speedup Using Gustafson’s Law:
With 2 threads, the predicted speedup is 1.9.
With 4 threads, the speedup is 3.7.
With 2 processes, the speedup is 1.9.
With 4 processes, the speedup is 3.7, suggesting better scalability for large workloads.
Conclusions:
Threading provided no significant improvement due to Python’s Global Interpreter Lock (GIL), which prevents true parallel execution for CPU-bound tasks.
Multiprocessing showed nearly 2× improvement, proving to be more effective for CPU-intensive computations.
Efficiency decreased as more threads or processes were added, meaning that after a certain point, adding more parallelism doesn’t help.
Amdahl’s Law shows the limits of speedup, emphasizing that parallel execution can only help up to a certain point.
Gustafson’s Law suggests that for larger problems, multiprocessing can scale better, making it a preferred approach for CPU-bound workloads.
For real-world applications, multiprocessing should be used over threading when dealing with CPU-intensive tasks.
