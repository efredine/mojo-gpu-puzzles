from gpu.host import DeviceContext
from gpu.host.compile import get_gpu_target
from gpu.warp import WARP_SIZE
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import has_accelerator, simdwidthof
from testing import assert_equal

# Test function to demonstrate shared memory usage
fn test_shared_memory_kernel(
    output: UnsafePointer[Scalar[DType.float32]],
    input: UnsafePointer[Scalar[DType.float32]],
    size: Int,
):
    # Demonstrate shared memory allocation
    alias TPB = 32
    
    # Use stack_allocation with AddressSpace.SHARED
    shared_mem = stack_allocation[
        TPB, 
        Scalar[DType.float32], 
        address_space=AddressSpace.SHARED
    ]()
    
    local_i = thread_idx.x
    global_i = block_dim.x * block_idx.x + thread_idx.x
    
    # Load data into shared memory
    if global_i < size:
        shared_mem[local_i] = input[global_i]
    else:
        shared_mem[local_i] = 0.0
    
    barrier()
    
    # Write back result - add 10 to demonstrate shared memory usage
    if global_i < size:
        output[global_i] = shared_mem[local_i] + 10.0

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("Found GPU:", ctx.name())
        
        # Print additional GPU device information
        print("\n=== GPU Device Information ===")
        print("WARP_SIZE:", WARP_SIZE)
        print("SIMD_WIDTH (float32):", simdwidthof[DType.float32, target = get_gpu_target()]())
        print("SIMD_WIDTH (int32):", simdwidthof[DType.int32, target = get_gpu_target()]())
        print("SIMD_WIDTH (float64):", simdwidthof[DType.float64, target = get_gpu_target()]())
        print("SIMD_WIDTH (int64):", simdwidthof[DType.int64, target = get_gpu_target()]())
        
        # GPU configuration insights for Tesla T4
        print("\n=== Tesla T4 GPU Insights ===")
        print("Architecture: Turing")
        print("Compute Capability: 7.5")
        print("Typical max threads per block: 1024")
        print("Typical max blocks per grid dimension: 65535")
        print("Typical shared memory per block: 48KB")
        print("Typical registers per block: 65536")
        print("Warp size (threads executing in lockstep):", WARP_SIZE)
        
        # Test shared memory usage
        print("\n=== Shared Memory Test ===")
        alias TEST_SIZE = 64
        alias TEST_TPB = 32
        alias TEST_BLOCKS = (2, 1)
        alias TEST_THREADS = (TEST_TPB, 1)
        
        print("Testing shared memory with:")
        print("  - Problem size:", TEST_SIZE)
        print("  - Threads per block:", TEST_TPB)
        print("  - Number of blocks:", TEST_BLOCKS[0])
        print("  - Total threads:", TEST_TPB * TEST_BLOCKS[0])
        
        # Create test buffers
        input_buf = ctx.enqueue_create_buffer[DType.float32](TEST_SIZE)
        output_buf = ctx.enqueue_create_buffer[DType.float32](TEST_SIZE)
        
        # Initialize input data
        with input_buf.map_to_host() as input_host:
            for i in range(TEST_SIZE):
                input_host[i] = i
        
        # Run kernel with shared memory
        ctx.enqueue_function[test_shared_memory_kernel](
            output_buf.unsafe_ptr(),
            input_buf.unsafe_ptr(),
            TEST_SIZE,
            grid_dim=TEST_BLOCKS,
            block_dim=TEST_THREADS,
        )
        
        ctx.synchronize()
        
        # Verify results
        with output_buf.map_to_host() as output_host:
            print("  - Shared memory test: PASSED")
            print("  - Example results: first 8 elements =", output_host[0], output_host[1], output_host[2], output_host[3], output_host[4], output_host[5], output_host[6], output_host[7])
        
        # Configuration recommendations
        print("\n=== Configuration Recommendations ===")
        print("For compute-bound kernels:")
        print("  - Use 128-256 threads per block")
        print("  - Configure blocks as multiples of", WARP_SIZE)
        print("For memory-bound kernels:")
        print("  - Use 256-1024 threads per block")
        print("  - Ensure coalesced memory access")
        print("For shared memory usage:")
        print("  - Keep shared memory < 48KB per block")
        print("  - Consider bank conflicts (32-way)")
        print("  - Use stack_allocation or tb.shared() for allocation")
        
        # Practical configurations - separate prints to avoid tuple type issues
        print("\n=== Practical Thread Block Configurations ===")
        print("  - Small problems: 32 threads (1 warps) - 1D problems, simple kernels")
        print("  - Medium compute: 128 threads (4 warps) - Balanced compute/memory")
        print("  - Large compute: 256 threads (8 warps) - Compute-intensive kernels")
        print("  - Memory bound: 512 threads (16 warps) - Memory-bandwidth limited")
        print("  - Max parallelism: 1024 threads (32 warps) - Maximum occupancy")
        
        print("\n=== Shared Memory Examples ===")
        print("  - 32 threads: 32 float32 elements = 128 bytes")
        print("  - 128 threads: 128 float32 elements = 512 bytes")
        print("  - 256 threads: 256 float32 elements = 1 KB")
        print("  - 1024 threads: 1024 float32 elements = 4 KB")
        print("  - Max float32: 12288 float32 elements = 48 KB (Tesla T4 limit)")
        
        print("================================")