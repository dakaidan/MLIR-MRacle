// LB (load buffering), relaxed: minimal 2-thread litmus
// P0: r0 = atomic_load(x, relaxed); atomic_store(y, 1, relaxed)
// P1: r1 = atomic_load(y, relaxed); atomic_store(x, 1, relaxed)
// r0/r1 are forwarded to shared telemetry z/w; (r0,r1) = (0,0) is forbidden
// under SC ordering but observable under relaxed ordering.
module {
  func.func @main() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %x = memref.alloc() : memref<1xi64>
    %y = memref.alloc() : memref<1xi64>
    %z = memref.alloc() : memref<1xi64>
    %w = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %z[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %w[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          omp.atomic.write %y = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %z = %r0 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.section {
          %r1_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r1_mem[%c0] : memref<1xi64>
          omp.atomic.read %r1_mem = %y memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r1 = memref.load %r1_mem[%c0] : memref<1xi64>
          omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %w = %r1 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %z_val = memref.load %z[%c0] : memref<1xi64>
    %w_val = memref.load %w[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %y : memref<1xi64>
    memref.dealloc %z : memref<1xi64>
    memref.dealloc %w : memref<1xi64>
    return %z_val, %w_val : i64, i64
  }
}
