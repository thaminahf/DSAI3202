# DSAI3202

1. What happens if more processes try to access the pool than available connections?

Extra processes must wait until a connection is released.
The semaphore limits access to the set number of connections.
As soon as a process releases a connection, a waiting process acquires it.

2. How does the semaphore prevent race conditions and ensure safe access?

The semaphore acts as a lock, allowing only a limited number of processes to access connections at a time.
Prevents multiple processes from grabbing the same connection, avoiding conflicts.
Ensures orderly resource allocation, preventing race conditions.


