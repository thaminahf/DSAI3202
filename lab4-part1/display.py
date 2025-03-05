#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
from IPython.display import clear_output

def initialize_display():
    print("Current temperatures:")
    print("Latest Temperatures: Sensor 0: --°C  Sensor 1: --°C  Sensor 2: --°C")
    print("Sensor 1 Average: --°C")
    print("Sensor 2 Average: --°C")
    print("Sensor 3 Average: --°C")

def update_display(latest_temperatures, temperature_averages):
    clear_output(wait=True)
    print("Current temperatures:")
    for sensor, temp in latest_temperatures.items():
        print(f"Latest Temperatures: Sensor {sensor}: {temp}°C")
    print(f"Sensor Averages: {temperature_averages.get('average', '--')}°C")

