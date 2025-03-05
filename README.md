# DSAI3202

1. Which synchronization metric did you use for each of the tasks?
For synchronization, we used the following metrics:

Sensor Simulation (simulate_sensor) → Used threading.RLock() to ensure that multiple sensors do not update latest_temperatures at the same time.
Data Processing (process_temperatures) → Used Queue to store temperature readings safely, preventing race conditions.
Display Update (update_display) → Used threading.Condition() to synchronize updates every 5 seconds without conflicts.
Overall Thread Management → Used daemon=True to allow background threads to terminate when the main program exits.


2. Why did the professor not ask you to compute metrics?
The professor likely did not ask for explicit performance metrics (such as execution time or memory usage) because:

Focus on Parallelism & Synchronization

The main goal is to ensure proper thread synchronization and real-time updates, rather than optimizing execution time.
Simple Data Processing

The program processes only temperature values (basic numerical computations), making performance overhead minimal.
Simulation-Based Approach

Since temperature readings are randomly generated, measuring real-world accuracy or efficiency is less meaningful.



