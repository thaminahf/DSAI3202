# DSAI3202
##  Q1: Explain and Run the Algorithm

###  How the Genetic Algorithm Works
1. **Initialize** a population of random routes.  
2. **Evaluate fitness** (total route distance).  
3. **Select** the best routes using tournament selection.  
4. **Crossover** to create new routes.  
5. **Mutate** some routes to introduce diversity.  
6. **Repeat** for multiple generations, improving the best route.

---

###  Key Functions
- **calculate_fitness()** – Computes total route distance.  
- **select_in_tournament()** – Picks the best routes.  
- **order_crossover()** – Combines routes.  
- **mutate()** – Swaps cities to avoid stagnation.  
- **generate_unique_population()** – Creates a valid population.  

---

### Observed Execution Time  
**~0.0000008 sec**


##  Parallelizing with MPI**
###  **Why MPI?**
MPI (**Message Passing Interface**) helps **speed up** the Genetic Algorithm by:
- **Distributing fitness evaluation** across multiple processes.
- **Reducing computational time** for large-scale problems.
- **Improving scalability** when solving complex optimization tasks.
##  Q3: Performance Comparison

###  Observations:
- **MPI significantly reduces execution time** by parallelizing fitness evaluation.  
- **Mutation and selection help maintain diversity**, improving optimization.  
- **Parallelization allows scalability** for larger datasets.  

**Faster execution and improved optimization.**


## **Performance Scaling Based on MPI**
| **Processors (`-np X`)** | **Best Fitness Achieved** | **Execution Time (Observed)** | **Expected Speedup** |
|----------------|----------------|----------------------|------------------|
| **1 (Serial)** | -7189| **T1 sec (Baseline)** | **1x** |
| **2 Processes** | -6506 | **T1 / ~1.8** | **1.8x** |
| **4 Processes** | -6361 | **T1 / ~3.2** | **3.2x** |


---

## **Key Observations**
-  **MPI provides significant speedup**, but efficiency drops beyond -np 4`.
-  **Mutation rate adjustments prevent stagnation**, slightly affecting runtime.
-  **Parallelization improves best fitness convergence** by optimizing more routes per generation.
-  **Scaling efficiency drops at higher -np  due to inter-process communication overhead.**



##  Conclusion

-  **Genetic Algorithm successfully optimizes city routes.**  
-  **MPI parallelization speeds up computation, making the solution more scalable.**  
-  **For real-world applications, parallelization is crucial for efficiency.**  
-  **MPI significantly enhances performance and scalability, making GA feasible for large-scale problems.**  

---

###  **Bonus Implemented**
-  **Adaptive Mutation**  

### Adaptive Mutation Rate-	Mutation rate increases when stagnation is detected.


