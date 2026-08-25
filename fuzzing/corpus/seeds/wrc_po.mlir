// WRC+poonceonces+Once: write, read, read chain (co + fr), relaxed
// P0: atomic_write(x, 1)
// P1: r0 = read(x); atomic_write(y, 1)
// P2: r0 = read(y); r1 = read(x)
// bad outcome (1:r0=1 /\ 2:r0=1 /\ 2:r1=0)
module {
  func.func @main() -> (i64, i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %x = memref.alloc() : memref<1xi64>
    %y = memref.alloc() : memref<1xi64>
    %t1r0 = memref.alloc() : memref<1xi64>
    %t2r0 = memref.alloc() : memref<1xi64>
    %t2r1 = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t1r0[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t2r0[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %t2r1[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0
          omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.section {
          // P1
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          omp.atomic.write %y = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %t1r0 = %r0 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.section {
          // P2
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %y memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          %r1_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r1_mem[%c0] : memref<1xi64>
          omp.atomic.read %r1_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r1 = memref.load %r1_mem[%c0] : memref<1xi64>
          omp.atomic.write %t2r0 = %r0 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %t2r1 = %r1 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %t1r0_val = memref.load %t1r0[%c0] : memref<1xi64>
    %t2r0_val = memref.load %t2r0[%c0] : memref<1xi64>
    %t2r1_val = memref.load %t2r1[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %y : memref<1xi64>
    memref.dealloc %t1r0 : memref<1xi64>
    memref.dealloc %t2r0 : memref<1xi64>
    memref.dealloc %t2r1 : memref<1xi64>
    return %t1r0_val, %t2r0_val, %t2r1_val : i64, i64, i64
  }
}
