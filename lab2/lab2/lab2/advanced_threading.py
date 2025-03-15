from sequential import generate_and_add_numbers
from sequential import generate_and_join_letters
import multiprocessing
import random
import time
import threading
print("Starting the two threads program for generate numbers")
total_start_time = time.time()

thread_numbers = threading.Thread(target=generate_and_add_numbers, args=[int(1e7)])
thread_letters = threading.Thread(target=generate_and_add_numbers, args=[int(1e7)])

thread_numbers.start()
thread_letters.start()

thread_numbers.join()
thread_letters.join()

total_end_time = time.time()
print("Exiting two threads program for generate numbers")
execution_time = total_end_time - total_start_time
print(f"It took {execution_time}s to execute the tasks with threads.")

