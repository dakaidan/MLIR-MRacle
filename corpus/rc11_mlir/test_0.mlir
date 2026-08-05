// test_0: non-atomic store -> load, single thread
// forall (0:r0 == 1 /\ x == 1)
module {
  func.func @main() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %x = memref.alloc() : memref<1xi64>
    memref.store %c0_i64, %x[%c0] : memref<1xi64>

    %p0_r0_mem = memref.alloc() : memref<1xi64>

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: int r0 = 0; *x = 1; r0 = *x;
          memref.store %c0_i64, %p0_r0_mem[%c0] : memref<1xi64>
          memref.store %c1_i64, %x[%c0] : memref<1xi64>
          %r0 = memref.load %x[%c0] : memref<1xi64>
          memref.store %r0, %p0_r0_mem[%c0] : memref<1xi64>
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %r0_val = memref.load %p0_r0_mem[%c0] : memref<1xi64>
    %x_val = memref.load %x[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %p0_r0_mem : memref<1xi64>
    return %r0_val, %x_val : i64, i64
  }
}
