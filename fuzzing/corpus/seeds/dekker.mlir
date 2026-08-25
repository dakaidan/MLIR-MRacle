// dekker: Dekker's mutual-exclusion algorithm, 2 threads, bounded spin-wait
// Each thread sets its flag, checks the other's flag, and while the other's
// flag is raised gives way on turn. The critical section writes x and bumps
// the shared entry counter. After the parallel region both threads must have
// exited (flags cleared) and entered the critical section exactly once.
// forall (flag1 == 0 /\ flag2 == 0 /\ turn == 1 /\ entries == 2)
module {
  func.func @main() -> (i64, i64, i64, i64) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64

    %flag1 = memref.alloc() : memref<1xi64>
    %flag2 = memref.alloc() : memref<1xi64>
    %turn = memref.alloc() : memref<1xi64>
    %x = memref.alloc() : memref<1xi64>
    %entries = memref.alloc() : memref<1xi64>
    memref.atomic_rmw assign %c0_i64, %flag1[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %flag2[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %turn[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %x[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %entries[%c0] : (i64, memref<1xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0
          omp.atomic.write %flag1 = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          %f2_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %f2_mem[%c0] : memref<1xi64>
          omp.atomic.read %f2_mem = %flag2 memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %f2 = memref.load %f2_mem[%c0] : memref<1xi64>
          scf.while (%i = %c0_i64, %fv = %f2) : (i64, i64) -> (i64, i64) {
            %ge1 = arith.cmpi sge, %fv, %c1_i64 : i64
            %lt8 = arith.cmpi slt, %i, %c8_i64 : i64
            %cont = arith.andi %ge1, %lt8 : i1
            scf.condition(%cont) %i, %fv : i64, i64
          } do {
          ^bb0(%i: i64, %fv: i64):
            %t_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %t_mem[%c0] : memref<1xi64>
            omp.atomic.read %t_mem = %turn memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %t = memref.load %t_mem[%c0] : memref<1xi64>
            %t_ne0 = arith.cmpi ne, %t, %c0_i64 : i64
            scf.if %t_ne0 {
              omp.atomic.write %flag1 = %c0_i64 memory_order(relaxed) : memref<1xi64>, i64
              omp.atomic.write %flag1 = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
            }
            %f2b_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %f2b_mem[%c0] : memref<1xi64>
            omp.atomic.read %f2b_mem = %flag2 memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %f2b = memref.load %f2b_mem[%c0] : memref<1xi64>
            %i_next = arith.addi %i, %c1_i64 : i64
            scf.yield %i_next, %f2b : i64, i64
          }
          omp.atomic.write %x = %c0_i64 memory_order(relaxed) : memref<1xi64>, i64
          memref.atomic_rmw addi %c1_i64, %entries[%c0] : (i64, memref<1xi64>) -> i64
          omp.atomic.write %turn = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %flag1 = %c0_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.section {
          // P1
          omp.atomic.write %flag2 = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          %f1_mem = memref.alloca() : memref<1xi64>
          memref.store %c0_i64, %f1_mem[%c0] : memref<1xi64>
          omp.atomic.read %f1_mem = %flag1 memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
          %f1 = memref.load %f1_mem[%c0] : memref<1xi64>
          scf.while (%i = %c0_i64, %fv = %f1) : (i64, i64) -> (i64, i64) {
            %ge1 = arith.cmpi sge, %fv, %c1_i64 : i64
            %lt8 = arith.cmpi slt, %i, %c8_i64 : i64
            %cont = arith.andi %ge1, %lt8 : i1
            scf.condition(%cont) %i, %fv : i64, i64
          } do {
          ^bb0(%i: i64, %fv: i64):
            %t_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %t_mem[%c0] : memref<1xi64>
            omp.atomic.read %t_mem = %turn memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %t = memref.load %t_mem[%c0] : memref<1xi64>
            %t_ne1 = arith.cmpi ne, %t, %c1_i64 : i64
            scf.if %t_ne1 {
              omp.atomic.write %flag2 = %c0_i64 memory_order(relaxed) : memref<1xi64>, i64
              omp.atomic.write %flag2 = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
            }
            %f1b_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %f1b_mem[%c0] : memref<1xi64>
            omp.atomic.read %f1b_mem = %flag1 memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %f1b = memref.load %f1b_mem[%c0] : memref<1xi64>
            %i_next = arith.addi %i, %c1_i64 : i64
            scf.yield %i_next, %f1b : i64, i64
          }
          omp.atomic.write %x = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          memref.atomic_rmw addi %c1_i64, %entries[%c0] : (i64, memref<1xi64>) -> i64
          omp.atomic.write %turn = %c1_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.atomic.write %flag2 = %c0_i64 memory_order(relaxed) : memref<1xi64>, i64
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %f1_v = memref.load %flag1[%c0] : memref<1xi64>
    %f2_v = memref.load %flag2[%c0] : memref<1xi64>
    %t_v = memref.load %turn[%c0] : memref<1xi64>
    %e_v = memref.load %entries[%c0] : memref<1xi64>
    memref.dealloc %flag1 : memref<1xi64>
    memref.dealloc %flag2 : memref<1xi64>
    memref.dealloc %turn : memref<1xi64>
    memref.dealloc %x : memref<1xi64>
    memref.dealloc %entries : memref<1xi64>
    return %f1_v, %f2_v, %t_v, %e_v : i64, i64, i64, i64
  }
}
