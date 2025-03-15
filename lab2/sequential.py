import multiprocessing
import random
import time
import threading
def generate_and_add_numbers(n: int = 1000):
    total = 0
    for i in range(n):
        total += random.randint(0,1000000)
    return total

def generate_and_join_letters(n: int = 1000):
    letters = ''
    for i in range(n):
        letters += chr(random.randint(33, 126))  # Generates random characters
    return letters
print("Starting the Program")
total_start_time = time.time()

generate_and_add_numbers(int(1e7))
generate_and_join_letters(int(1e7))

total_end_time = time.time()
print("Exiting the Program")
sequential_execution_time = total_end_time - total_start_time
print(f"It took {sequential_execution_time}s to execute the tasks.")
