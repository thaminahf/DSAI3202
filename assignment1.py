import time
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Semaphore, Process, Queue

# Function to compute square
def square(n):
    return n * n

# Generate a list of 10^6 random numbers
NUMBERS = [random.randint(1, 100) for _ in range(10**6)]

# Sequential execution
def sequential_execution():
    start = time.time()
    results = [square(n) for n in NUMBERS]
    end = time.time()
    print(f"Sequential Execution Time: {end - start:.4f} seconds")

# **Fixed Multiprocessing - Uses a Limited Number of Processes**
def multiprocess_individual():
    start = time.time()
    num_workers = min(10, multiprocessing.cpu_count())  # Use a max of 10 processes
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(square, NUMBERS)
    end = time.time()
    print(f"Multiprocessing (Limited) Execution Time: {end - start:.4f} seconds")

# Multiprocessing Pool (map and apply)
def multiprocessing_pool():
    with multiprocessing.Pool(processes=10) as pool:
        start = time.time()
        results = pool.map(square, NUMBERS)
        end = time.time()
        print(f"Multiprocessing Pool (map) Execution Time: {end - start:.4f} seconds")

def multiprocessing_apply():
    with multiprocessing.Pool(processes=10) as pool:
        start = time.time()
        results = [pool.apply(square, args=(n,)) for n in NUMBERS[:1000]]  # Only process 1000 numbers for efficiency
        end = time.time()
        print(f"Multiprocessing Pool (apply) Execution Time: {end - start:.4f} seconds")

# **Fixed Process Pool Executor**
def process_pool_executor():
    num_workers = min(10, multiprocessing.cpu_count())  # Limit workers to prevent excessive overhead
    chunk_size = len(NUMBERS) // num_workers  # Process numbers in chunks
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(square, NUMBERS, chunksize=chunk_size))  # Efficient chunk processing
    end = time.time()
    print(f"ProcessPoolExecutor Execution Time: {end - start:.4f} seconds")

# **Fixed Connection Pool Class (Proper Semaphore Handling)**
class ConnectionPool:
    def __init__(self, size=3):
        self.semaphore = Semaphore(size)
        self.pool = Queue()  # Use a Queue for proper synchronization
        for i in range(size):
            self.pool.put(f"Conn{i}")  # Create unique connection names

    def get_connection(self):
        self.semaphore.acquire()  # Block if no connections are available
        return self.pool.get()  # Get an available connection

    def release_connection(self, conn):
        self.pool.put(conn)  # Put connection back into the pool
        self.semaphore.release()  # Release the semaphore

# **Simulating Database Access (Fixed Synchronization)**
def access_database(pool, process_id):
    print(f"Process {process_id} is waiting for a connection...")
    conn = pool.get_connection()  # Correctly acquire a unique connection
    print(f"Process {process_id} acquired {conn}")
    time.sleep(random.uniform(1, 3))  # Simulate work
    pool.release_connection(conn)  # Correctly release the connection
    print(f"Process {process_id} released {conn}")

# **Running the Tests**
if __name__ == "__main__":
    sequential_execution()
    multiprocess_individual()
    multiprocessing_pool()
    multiprocessing_apply()
    process_pool_executor()

    # **Fixed Semaphore Test**
    pool = ConnectionPool(size=3)  # Limit to 3 connections
    processes = []

    for i in range(10):  # 10 processes trying to access 3 connections
        p = Process(target=access_database, args=(pool, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
