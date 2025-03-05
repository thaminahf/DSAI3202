#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import threading
from queue import Queue

temperature_averages = {}
temp_queue = Queue()
lock = threading.RLock()

def process_temperatures():
    global temperature_averages
    while True:
        with lock:
            if not temp_queue.empty():
                temps = list(temp_queue.queue)
                temperature_averages['average'] = sum(temps) / len(temps)
        time.sleep(5)

processor_thread = threading.Thread(target=process_temperatures, daemon=True)
processor_thread.start()

