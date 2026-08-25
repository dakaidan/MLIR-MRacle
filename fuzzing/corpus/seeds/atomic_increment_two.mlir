module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %counter = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %counter[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: r0 = atomic_fetch_add(&counter, 1);
          %r0 = memref.atomic_rmw addi %c1_i64, %counter[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.section {
          // P1: r0 = atomic_fetch_add(&counter, 1);
          %r0_1 = memref.atomic_rmw addi %c1_i64, %counter[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %counter_val = memref.load %counter[%c0] : memref<1xi64>
    memref.dealloc %counter : memref<1xi64>
    return %counter_val : i64
  }
}
