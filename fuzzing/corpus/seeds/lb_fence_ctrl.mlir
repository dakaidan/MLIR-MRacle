// LB+fencembonceonce+ctrlonceonce: load-buffering via control dependency + full fence
// P0: r0 = read(x); if (r0 == 1) atomic_write(y, 1)
// P1: r0 = read(y); omp.flush; atomic_write(x, 1)
// r0 of P0/P1 observable via telemetry; bad outcome (0:r0=1 /\ 1:r0=1)
module {
  func.func @main() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %x = memref.alloc() : memref<1xi64>
    %y = memref.alloc() : memref<1xi64>
    %t0r0 = memref.alloc() : memref<1xi64>
    %t1r0 = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t0r0[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t1r0[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          %cond_r0_1 = arith.cmpi eq, %r0, %c1_i64 : i64
          scf.if %cond_r0_1 {
            omp.atomic.write %y = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          }
          omp.atomic.write %t0r0 = %r0 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.section {
          // P1
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %y memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          omp.flush
          omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %t1r0 = %r0 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %t0r0_val = memref.load %t0r0[%c0] : memref<1xi64>
    %t1r0_val = memref.load %t1r0[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %y : memref<1xi64>
    memref.dealloc %t0r0 : memref<1xi64>
    memref.dealloc %t1r0 : memref<1xi64>
    return %t0r0_val, %t1r0_val : i64, i64
  }
}
