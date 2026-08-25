// CoRW+poonceonce+Once: read-write coherence, relaxed
// P0: r0 = read(x); atomic_write(x, 1)
// P1: atomic_write(x, 2)
// x and r0 of P0 observable post-join; bad outcome (x=2 /\ 0:r0=2)
module {
  func.func @main() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64

    %x = memref.alloc() : memref<1xi64>
    %t0r0 = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t0r0[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %t0r0 = %r0 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.section {
          // P1
          omp.atomic.write %x = %c2_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %x_val = memref.load %x[%c0] : memref<1xi64>
    %t0r0_val = memref.load %t0r0[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %t0r0 : memref<1xi64>
    return %x_val, %t0r0_val : i64, i64
  }
}
