module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %counter = memref.alloc() : memref<1xi64>
    %a = arith.constant 5 : i64
    %b = arith.constant 10 : i64
    %c = arith.constant 15 : i64
    omp.parallel {
      omp.sections {
        omp.section {
          %x = memref.atomic_rmw addi %a, %counter[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.section {
          %y = memref.atomic_rmw addi %b, %counter[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.section {
          %z = memref.atomic_rmw addi %c, %counter[%c0] : (i64, memref<1xi64>) -> i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    %r = memref.load %counter[%c0] : memref<1xi64>
    return %r : i64
  }
}
