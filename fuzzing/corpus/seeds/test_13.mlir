// test_13: two relaxed loads vs store (read-read coherence)
// forall (x == 1)
module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %x = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: r0 = atomic_load(x, relaxed);
          //     r1 = atomic_load(x, relaxed);
          %r0 = memref.atomic_rmw addi %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
          %r1 = memref.atomic_rmw addi %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.section {
          // P1: atomic_store(x, 1, relaxed);
          memref.atomic_rmw assign %c1_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %x_val = memref.load %x[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    return %x_val : i64
  }
}
