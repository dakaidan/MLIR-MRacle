module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %counter = memref.alloc() : memref<1xi64>
    %zero = arith.constant 0 : i64
    memref.store %zero, %counter[%c0] : memref<1xi64>

    // Bump this to make each thread repeat its add multiple times.
    %num_iters = arith.constant 1 : index
    %lb   = arith.constant 0 : index
    %step = arith.constant 1 : index

    %add0 = arith.constant 5 : i64
    %add1 = arith.constant 10 : i64
    %add2 = arith.constant 15 : i64

    omp.parallel {
      omp.sections {
        omp.section {
          scf.for %i = %lb to %num_iters step %step {
            %val = memref.load %counter[%c0] : memref<1xi64>
            %new = arith.addi %val, %add0 : i64
            memref.store %new, %counter[%c0] : memref<1xi64>
          }
          omp.terminator
        }
        omp.section {
          scf.for %i = %lb to %num_iters step %step {
            %val = memref.load %counter[%c0] : memref<1xi64>
            %new = arith.addi %val, %add1 : i64
            memref.store %new, %counter[%c0] : memref<1xi64>
          }
          omp.terminator
        }
        omp.section {
          scf.for %i = %lb to %num_iters step %step {
            %val = memref.load %counter[%c0] : memref<1xi64>
            %new = arith.addi %val, %add2 : i64
            memref.store %new, %counter[%c0] : memref<1xi64>
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %result = memref.load %counter[%c0] : memref<1xi64>
    memref.dealloc %counter : memref<1xi64>
    return %result : i64
  }
}