# CUDA Notes

## Basics
### Terminology
* **Kernel** - some function/set of instructions to be run on the GPU, i.e. on every thread
  * Generally initialized like
    ```c++
    __global__ void FUNCNAME_kernel(const float* input, float* result, other params){
        // some function here
    }
    ```
* **Thread** - basic unit of computation, many threads on GPU
  * Can get thread index with `threadIdx.x`, `threadIdx.y`, and `threadIdx.z`. 
* **Block** - Some number of threads. Shares memory and can be synchronized
  * Generally max number of threads per block is 1024. Process bigger inputs by calling more blocks. Generally, threads per block is set to `dim3 threadsPerBlock(16,16);`. 
  * Can declare shared memory with `__shared__ DTYPE VARNAME;`
* **Grid** - Set of blocks/threads that execute over a set of data (I think). Gets initialized when you do a `func_kernel<<<num_blocks, num_grids>>>(params)` call. 
* **Warp** - CUDA scheduler assigns 32 threads at a time to a processing unit. All 32 threads must be on same block, which is also why `threadsPerBlock` is usually set to some multiple of 32. 
* **Memory**
  * DRAM - CPU (12gb/s, tbs on server)
  * HBM - GPU main memory (1tb/s, 40gb) 
  * SRAM - Shared memory between threads (19tb/s, 20mb)
  * L1/L2 cache - you don't have access to this, so ignore it for now
* **Streaming Multiprocessor** - basic compute unit on GPU. Has compute, register, decoder, etc. 
##

## CUDA in PyTorch

Basic CUDA Kernel (to be imported in PyTorch):

* In the CUDA File: 
    ```c++
    __global__ void FUNCNAME_kernel(const float* input, float* result, int height, int width){
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (row_idx < height && col_idx < width){
            int ndx = row_idx * width + col_idx;
            //SOME FUNCTION HERE; 
        }

    }
    torch::Tensor FUNCNAME(torch::Tensor input){
        int height = input.size(0);
        int width = input.size(1);

        auto result = torch::empty_like(input);
        dim3 threads_per_block(16,16);
        
        dim3 num_blocks((width + threads_per_block.x - 1)/threads_per_block.x, (height + threads_per_block.y - 1)/threads_per_block.y);

        FUNCNAME_kernel<<<num_blocks, threads_per_block>>>(input.data_ptr<float>(), result.data_ptr<float>(), height, width);

        return result;
    }
    ```
* In the C++ File:
    ```c++
    torch::Tensor FUNCNAME(torch::Tensor input);
    ```
* In the Python file: 
    ```python
    import torch
    from torch.utils.cpp_extension import load_inline

    with open('./CUDAFILE.cu') as f:
        cuda_source = f.readlines()

    with open('./CPPFILE.cpp') as f:
        cpp_source = f.readlines()

    extension_name = load_inline(
        name = 'extension_name',
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ['FUNCNAME'],
        with_cuda = True,
        extra_cuda_cflags = ['-O2'], # compiler optimizations, can't find documentation on this right now 
        build_directory = "./dir/to/build/",
    )
    ```

  

## Using CUDA to Optimize ML 

Can't really find that much on this right now. The initial workflow seems to be something like:
1. Try `torch.compile`
2. Try Triton
3. Try CUDA



## CUDA Optimizations
[From here](https://www.youtube.com/watch?v=SGhfUhlowB4&ab_channel=CUDAMODE)

[and here](https://www.youtube.com/watch?v=2NgpYFdsduY&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&ab_channel=CoffeeBeforeArch)
1. Coalesced Global Memory Access
   * Can index like matrices/tensors, but in reality memory stored sequentially. try to access things sequentially instead of hopping around
   * `ncu` command line tools, can help with determining things like block size and others
2. Maximize Occupancy
   * Tile quantization - matrix dimensions not divisible by thread block tile size
   * Wave quantization - number of tiles not divisible by number of SM on GPU
   * https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html - has sizes of different matrices to get optimal performance
   * Can sometimes increase time for certain operations by like 4x. 
   * Padding tends to be important in PyTorch as a result
   * `cudaOccupancyMaxPotentialBlockSize` function can print out recommended blocksize and grid sizes
3. Memory or compute bound
   * `ncu` userful for this
   * [Roofline model](https://en.wikipedia.org/wiki/Roofline_model#:~:text=The%20roofline%20model%20is%20an,benefit%20and%20priority%20of%20optimizations.). tldr for smaller workloads, can be memory bound. for larger workloads, usually bound by performance (need better algorithms).  
   * useful to think how many operations done per byte (item) read
   * bandwidth bound kernels: fuse, quantize, compile
   * compute bound kernels: write a better algorithm
4. Control Divergence
   * If statements can block threads since each warp executes same set of instructions. Processing for if statements freeze the threads that don't apply to first case, and then unfreeze and do their case later. 
5. Tiling reused data (matmul, flash attention)
6. Privatization
   * local copies of data
   * copying global data to private variable and then operating on that private/shared variable might be a lot faster than performing the operation directly on the global variable
7. Thread Coarsening
   * for memory-bound applications, doing more work per thread can make it faster (might go against intuition to have each thread do as little work as possible). 
   * coarsening by a factor of 2 can still improve speed of operation by more than 2x
8. Do better math/algorithms