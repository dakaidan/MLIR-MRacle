// test_2: non-atomic store x=2 then x=1 (coherence), single thread
// forall (x == 1)
module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64

    %x = memref.alloc() : memref<1xi64>
    memref.store %c0_i64, %x[%c0] : memref<1xi64>

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: *x = 2; *x = 1;
          memref.store %c2_i64, %x[%c0] : memref<1xi64>
          memref.store %c1_i64, %x[%c0] : memref<1xi64>
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %result = memref.load %x[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    return %result : i64
  }
}
