// test_4: message passing, acquire/release
// forall (0:r0 == 0 /\ 1:r0 == 0 /\ x == 0 /\ y == 0)
module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %y = memref.alloc() : memref<1xi64>
    %x = memref.alloc() : memref<1xi64>
    memref.store %c0_i64, %y[%c0] : memref<1xi64>
    memref.store %c0_i64, %x[%c0] : memref<1xi64>

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: r0 = atomic_load(y, acquire);
          //     if (r0 == 1) { *x = 1; }
          %r0 = memref.atomic_rmw addi %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
          %cond0 = arith.cmpi eq, %r0, %c1_i64 : i64
          scf.if %cond0 {
            memref.store %c1_i64, %x[%c0] : memref<1xi64>
          }
          omp.terminator
        }
        omp.section {
          // P1: r0 = *x;
          //     if (r0 == 1) { atomic_store(y, 1, release); }
          %r0 = memref.load %x[%c0] : memref<1xi64>
          %cond0 = arith.cmpi eq, %r0, %c1_i64 : i64
          scf.if %cond0 {
            memref.atomic_rmw assign %c1_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %result = memref.load %x[%c0] : memref<1xi64>
    memref.dealloc %y : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    return %result : i64
  }
}
