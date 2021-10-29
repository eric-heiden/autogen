# Vectorized function calls

The `Generated` class provides vectorized functions for the forward and backward passes of the traced function, which allows to run the function on multiple inputs in parallel.

When using the vectorized mode, the input is split into *global* and *local* memory. The global memory is shared between the parallel threads, while the local memory is split up into segments for each thread.