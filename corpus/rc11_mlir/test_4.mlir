// test_4: message passing, acquire/release
// forall (0:r0 == 0 /\ 1:r0 == 0 /\ x == 0 /\ y == 0)
module {
  func.func @main() -> (i64, i64, i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %y = memref.alloc() : memref<1xi64>
    %x = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
    memref.store %c0_i64, %x[%c0] : memref<1xi64>

    %p0_r0_mem = memref.alloc() : memref<1xi64>
    %p1_r0_mem = memref.alloc() : memref<1xi64>

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: int r0 = 0; r0 = atomic_load(y, acquire);
          //     if (r0 == 1) { *x = 1; }
          memref.store %c0_i64, %p0_r0_mem[%c0] : memref<1xi64>
          %r0 = memref.atomic_rmw addi %c0_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
          %cond0 = arith.cmpi eq, %r0, %c1_i64 : i64
          scf.if %cond0 {
            memref.store %c1_i64, %x[%c0] : memref<1xi64>
          }
          memref.store %r0, %p0_r0_mem[%c0] : memref<1xi64>
          omp.terminator
        }
        omp.section {
          // P1: int r0 = 0; r0 = *x;
          //     if (r0 == 1) { atomic_store(y, 1, release); }
          memref.store %c0_i64, %p1_r0_mem[%c0] : memref<1xi64>
          %r0 = memref.load %x[%c0] : memref<1xi64>
          %cond0 = arith.cmpi eq, %r0, %c1_i64 : i64
          scf.if %cond0 {
            memref.atomic_rmw assign %c1_i64, %y[%c0] : (i64, memref<1xi64>) -> i64
          }
          memref.store %r0, %p1_r0_mem[%c0] : memref<1xi64>
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %p0_r0 = memref.load %p0_r0_mem[%c0] : memref<1xi64>
    %p1_r0 = memref.load %p1_r0_mem[%c0] : memref<1xi64>
    %x_val = memref.load %x[%c0] : memref<1xi64>
    %y_val = memref.load %y[%c0] : memref<1xi64>
    memref.dealloc %y : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %p0_r0_mem : memref<1xi64>
    memref.dealloc %p1_r0_mem : memref<1xi64>
    return %p0_r0, %p1_r0, %x_val, %y_val : i64, i64, i64, i64
  }
}
