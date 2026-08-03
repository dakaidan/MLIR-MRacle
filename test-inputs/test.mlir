module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %counter = memref.alloc() : memref<1xi64>
    %zero = arith.constant 0 : i64
    memref.store %zero, %counter[%c0] : memref<1xi64>

    %lb   = arith.constant 0 : index
    %ub   = arith.constant 1000 : index
    %step = arith.constant 1 : index
    %inc  = arith.constant 1 : i64

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%lb) to (%ub) step (%step) {
          %old = memref.atomic_rmw addi %inc, %counter[%c0] : (i64, memref<1xi64>) -> i64
          memref.store %old, %counter[%c0] : memref<1xi64>   // <-- added non‑atomic store
          omp.yield
        }
      }
      omp.terminator
    }

    %result = memref.load %counter[%c0] : memref<1xi64>
    memref.dealloc %counter : memref<1xi64>
    return %result : i64
  }
}