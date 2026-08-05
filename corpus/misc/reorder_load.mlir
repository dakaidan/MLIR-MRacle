module {
  memref.global "private" @flags : memref<2xi64> = dense<[10, 20]>

  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %out = memref.alloc() : memref<1xi64>
    %zero = arith.constant 0 : i64
    memref.store %zero, %out[%c0] : memref<1xi64>

    %k1 = arith.constant 2 : i64
    %k2 = arith.constant 3 : i64

    omp.parallel {
      omp.sections {
        omp.section {
          %g = memref.get_global @flags : memref<2xi64>
          %x = memref.load %g[%c0] : memref<2xi64>
          %y = memref.load %g[%c1] : memref<2xi64>
          %sum = arith.addi %x, %y : i64
          %prod = arith.muli %sum, %k1 : i64
          memref.store %prod, %out[%c0] : memref<1xi64>
          omp.terminator
        }
        omp.section {
          %g2 = memref.get_global @flags : memref<2xi64>
          %p = memref.load %g2[%c0] : memref<2xi64>
          %q = memref.load %g2[%c1] : memref<2xi64>
          %diff = arith.subi %q, %p : i64
          %mul = arith.muli %diff, %k2 : i64
          memref.store %mul, %out[%c0] : memref<1xi64>
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %result = memref.load %out[%c0] : memref<1xi64>
    memref.dealloc %out : memref<1xi64>
    return %result : i64
  }
}
