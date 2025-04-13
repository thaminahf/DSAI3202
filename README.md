Sequential Execution:
This method had the longest runtime since each hyperparameter combination was tested one after the other.

Threaded Execution:
Using threading reduced the overall runtime by evaluating multiple combinations in parallel. However, it's not ideal for CPU-intensive tasks due to inherent limitations.

Multiprocessing Execution:
This approach significantly improved execution time by distributing combinations across different CPU cores, enabling faster and more efficient processing.

Summary:
Transitioning from sequential to threaded and then to multiprocessing execution led to noticeable improvements in speed. Among them, multiprocessing was the most effective for parallel processing.

Threaded Execution Results:

Best RMSE: Lowest RMSE achieved during hyperparameter tuning with threading.

Best MAPE: Lowest MAPE achieved during the same process.

Multiprocessing Execution Results:

Best RMSE: Lowest RMSE achieved during hyperparameter tuning with multiprocessing.

Best MAPE: Lowest MAPE achieved during the same process.

Key Insights:

Both threading and multiprocessing yielded nearly identical RMSE and MAPE values, as the underlying model training remained unchanged.

Multiprocessing achieved these results in less time due to superior parallelization across multiple CPU cores.

In conclusion, while performance metrics were similar for both methods, multiprocessing proved to be faster and more efficient.


