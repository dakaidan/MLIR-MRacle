// test_51: x store/load pair with guarded y store (made atomic)
// forall (x == 1 /\ y == 0)
module {
  func.func @main() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64

    %x = memref.alloc() : memref<1xi64>
    %y = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0
          %r0_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
          omp.atomic.read %r0_mem = %y memory_order(acquire) : memref<1xi64>, memref<1xi64>, i64
          %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
          %cond_r0_1 = arith.cmpi eq, %r0, %c1_i64 : i64
          scf.if %cond_r0_1 {
            %r1_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %r1_mem[%c0] : memref<1xi64>
            omp.atomic.read %r1_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %r1 = memref.load %r1_mem[%c0] : memref<1xi64>
            %cond_r1_1 = arith.cmpi eq, %r1, %c1_i64 : i64
            scf.if %cond_r1_1 {
              omp.atomic.write %x = %c2_i64 memory_order(relaxed) : memref<1xi64>, i64
            }
          }
          omp.terminator
        }
        omp.section {
          // P1
          %r1_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %r1_mem[%c0] : memref<1xi64>
          omp.atomic.read %r1_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %r1 = memref.load %r1_mem[%c0] : memref<1xi64>
          %cond_r1_2 = arith.cmpi eq, %r1, %c2_i64 : i64
          scf.if %cond_r1_2 {
            %r0_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %r0_mem[%c0] : memref<1xi64>
            omp.atomic.read %r0_mem = %x memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %r0 = memref.load %r0_mem[%c0] : memref<1xi64>
            omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
            %cond_r0_1 = arith.cmpi eq, %r0, %c1_i64 : i64
            scf.if %cond_r0_1 {
              omp.atomic.write %y = %c1_i64 memory_order(release) : memref<1xi64>, i64
            }
          } else {
            omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %x_val = memref.load %x[%c0] : memref<1xi64>
    %y_val = memref.load %y[%c0] : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %y : memref<1xi64>
    return %x_val, %y_val : i64, i64
  }
}
