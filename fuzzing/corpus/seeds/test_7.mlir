// test_7: CAS vs relaxed store
// forall (x == 1)
module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64

    %x = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: CAS(x, &r0, 2, relaxed, relaxed)
          omp.atomic.compare memory_order(relaxed) %x : memref<1xi64> {
          ^bb0(%xval: i64):
            %cmp = arith.cmpi eq, %xval, %c0_i64 : i64
            %sel = arith.select %cmp, %c2_i64, %xval : i64
            omp.yield (%sel : i64)
          }
          omp.terminator
        }
        omp.section {
          // P1: atomic_store(x, 1, relaxed);
          omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
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
