// MP+fencewmbonceonce+fencermbonceonce: message passing, wmb/rmb fences
// P0: atomic_write(buf, 1); omp.flush; atomic_write(flag, 1)
// P1: r0 = read(flag); omp.flush; r1 = read(buf)
// r0/r1 of P1 observable via telemetry; bad outcome (1:r0=1 /\ 1:r1=0)
module {
  func.func @main() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %buf = memref.alloc() : memref<1xi64>
    %flag = memref.alloc() : memref<1xi64>
    %t1r0 = memref.alloc() : memref<1xi64>
    %t1r1 = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %buf[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %flag[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t1r0[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t1r1[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0
          omp.atomic.write %buf = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.flush
          omp.atomic.write %flag = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.section {
          // P1
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %flag memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          omp.flush
          %r1_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r1_mem[%c0] : memref<1xi64>
          omp.atomic.read %r1_mem = %buf memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r1 = memref.load %r1_mem[%c0] : memref<1xi64>
          omp.atomic.write %t1r0 = %r0 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %t1r1 = %r1 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %t1r0_val = memref.load %t1r0[%c0] : memref<1xi64>
    %t1r1_val = memref.load %t1r1[%c0] : memref<1xi64>
    memref.dealloc %buf : memref<1xi64>
    memref.dealloc %flag : memref<1xi64>
    memref.dealloc %t1r0 : memref<1xi64>
    memref.dealloc %t1r1 : memref<1xi64>
    return %t1r0_val, %t1r1_val : i64, i64
  }
}
