#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import time
import threading

latest_temperatures = {}
lock = threading.RLock()

def simulate_sensor(sensor_id):
    global latest_temperatures
    while True:
        temp = random.randint(15, 40)
        with lock:
            latest_temperatures[sensor_id] = temp
        time.sleep(1)

sensor_threads = [threading.Thread(target=simulate_sensor, args=(i,), daemon=True) for i in range(3)]
for thread in sensor_threads:
    thread.start()

