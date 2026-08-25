// parallel partial sums: 4 threads contend pairwise on two shared counters
// P0/P1: r = atomic_fetch_add(&sums[0], 5); if (r == 5) atomic_fetch_add(&sums[0], 7)
// P2/P3: r = atomic_fetch_add(&sums[1], 5); if (r == 5) atomic_fetch_add(&sums[1], 7)
// the guard fires for exactly the thread that observes its peer's contribution,
// so sums[0] == 17 /\ sums[1] == 17 is deterministic
module {
  func.func @main() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %five = arith.constant 5 : i64
    %seven = arith.constant 7 : i64

    %sums = memref.alloc() : memref<2xi64>
    memref.atomic_rmw assign %c0_i64, %sums[%c0] : (i64, memref<2xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %sums[%c1] : (i64, memref<2xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0
          %r0 = memref.atomic_rmw addi %five, %sums[%c0] : (i64, memref<2xi64>) -> i64
          %cond0 = arith.cmpi eq, %r0, %five : i64
          scf.if %cond0 {
            memref.atomic_rmw addi %seven, %sums[%c0] : (i64, memref<2xi64>) -> i64
          }
          omp.terminator
        }
        omp.section {
          // P1
          %r1 = memref.atomic_rmw addi %five, %sums[%c0] : (i64, memref<2xi64>) -> i64
          %cond1 = arith.cmpi eq, %r1, %five : i64
          scf.if %cond1 {
            memref.atomic_rmw addi %seven, %sums[%c0] : (i64, memref<2xi64>) -> i64
          }
          omp.terminator
        }
        omp.section {
          // P2
          %r2 = memref.atomic_rmw addi %five, %sums[%c1] : (i64, memref<2xi64>) -> i64
          %cond2 = arith.cmpi eq, %r2, %five : i64
          scf.if %cond2 {
            memref.atomic_rmw addi %seven, %sums[%c1] : (i64, memref<2xi64>) -> i64
          }
          omp.terminator
        }
        omp.section {
          // P3
          %r3 = memref.atomic_rmw addi %five, %sums[%c1] : (i64, memref<2xi64>) -> i64
          %cond3 = arith.cmpi eq, %r3, %five : i64
          scf.if %cond3 {
            memref.atomic_rmw addi %seven, %sums[%c1] : (i64, memref<2xi64>) -> i64
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %s0 = memref.load %sums[%c0] : memref<2xi64>
    %s1 = memref.load %sums[%c1] : memref<2xi64>
    memref.dealloc %sums : memref<2xi64>
    return %s0, %s1 : i64, i64
  }
}
