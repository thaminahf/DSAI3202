from sequential import generate_and_add_numbers
import multiprocessing
import random
import time
import threading
print("Starting the two process program for generate letters")
total_start_time = time.time()

process_numbers = multiprocessing.Process(target=generate_and_add_numbers, args=[int(1e7)])
process_letters = multiprocessing.Process(target=generate_and_add_numbers, args=[int(1e7)])

process_numbers.start()
process_letters.start()

process_numbers.join()
process_letters.join()

total_end_time = time.time()
print("Exiting two process program for generate letters")
process_execution_time = total_end_time - total_start_time
print(f"It took {process_execution_time}s to execute the tasks with processes.")

