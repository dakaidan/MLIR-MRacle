// test_8: relaxed store buffering (all four outcomes allowed)
// forall ((0:r0 == 0 /\ 1:r0 == 0 /\ x == 1 /\ y == 1) \/
//         (0:r0 == 0 /\ 1:r0 == 1 /\ x == 1 /\ y == 1) \/
//         (0:r0 == 1 /\ 1:r0 == 0 /\ x == 1 /\ y == 1) \/
//         (0:r0 == 1 /\ 1:r0 == 1 /\ x == 1 /\ y == 1))
module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %x = memref.alloc() : memref<1xi64>
    %y = memref.alloc() : memref<1xi64>
    memref.store %c0_i64, %x[%c0] : memref<1xi64>
    memref.store %c0_i64, %y[%c0] : memref<1xi64>

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: r0 = atomic_load(y, relaxed);
          //     atomic_store(x, 1, relaxed);
          %r0 = memref.atomic_rmw addi %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
          memref.atomic_rmw assign %c1_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.section {
          // P1: r0 = atomic_load(x, relaxed);
          //     atomic_store(y, 1, relaxed);
          %r0 = memref.atomic_rmw addi %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
          memref.atomic_rmw assign %c1_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %result = memref.load %x[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %y : memref<1xi64>
    return %result : i64
  }
}
