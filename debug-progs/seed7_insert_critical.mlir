module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() : memref<1xi64>
    %0 = memref.atomic_rmw assign %c0_i64, %alloc[%c0] : (i64, memref<1xi64>) -> i64
    omp.parallel {
      omp.sections {
        omp.section {
          omp.critical {
            %2 = memref.atomic_rmw addi %c1_i64, %alloc[%c0] : (i64, memref<1xi64>) -> i64
            omp.terminator
          }
          omp.terminator
        }
        omp.section {
          omp.critical {
            %2 = memref.atomic_rmw addi %c1_i64, %alloc[%c0] : (i64, memref<1xi64>) -> i64
            omp.terminator
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    %1 = memref.load %alloc[%c0] : memref<1xi64>
    memref.dealloc %alloc : memref<1xi64>
    return %1 : i64
  }
}
