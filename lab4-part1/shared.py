#!/usr/bin/env python
# coding: utf-8

# In[1]:


import threading

# Shared dictionary to store sensor readings
latest_temperatures = {}

# Lock for thread safety
lock = threading.RLock()

def simulate_sensor(sensor_id):
    """Simulates a temperature sensor that updates the shared dictionary."""
    import random
    import time

    while True:
        temp = round(random.uniform(20.0, 30.0), 2)  # Generate random temp
        with lock:  # Ensure thread-safe access
            latest_temperatures[sensor_id] = temp
            print(f"Sensor {sensor_id}: {temp}°C")
        time.sleep(2)  # Simulate sensor delay


# In[ ]:





# In[ ]:




