// safestack: 3-thread Treiber-style lock-free stack with a shared counter
// The stack holds 3 elements (next[0]=1, next[1]=2, next[2]=-1, head=0).
// Each thread pops one element via compare-and-swap on head, writes its
// thread id into val[elem], then pushes the element back. Every successful
// pop is matched by a push, so the shared element counter must return to its
// initial value of 3 after the parallel region.
// forall (count == 3)
module {
  func.func @main() -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %cm1_i64 = arith.constant -1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c8_i64 = arith.constant 8 : i64

    %head = memref.alloc() : memref<1xi64>
    %count = memref.alloc() : memref<1xi64>
    %next = memref.alloc() : memref<3xi64>
    %val = memref.alloc() : memref<3xi64>
    memref.atomic_rmw assign %c0_i64, %head[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c3_i64, %count[%c0] : (i64, memref<1xi64>) -> i64
    memref.atomic_rmw assign %c1_i64, %next[%c0] : (i64, memref<3xi64>) -> i64
    memref.atomic_rmw assign %c2_i64, %next[%c1] : (i64, memref<3xi64>) -> i64
    memref.atomic_rmw assign %cm1_i64, %next[%c2] : (i64, memref<3xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %val[%c0] : (i64, memref<3xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %val[%c1] : (i64, memref<3xi64>) -> i64
    memref.atomic_rmw assign %c0_i64, %val[%c2] : (i64, memref<3xi64>) -> i64

    omp.parallel {
      omp.sections {
        omp.section {
          // P0: pop one element, tag it, push it back
          %elem = memref.alloca() : memref<1xi64>
          memref.store %cm1_i64, %elem[%c0] : memref<1xi64>
          scf.while (%i = %c0_i64) : (i64) -> i64 {
            %ev = memref.load %elem[%c0] : memref<1xi64>
            %lt0 = arith.cmpi slt, %ev, %c0_i64 : i64
            %lt32 = arith.cmpi slt, %i, %c32_i64 : i64
            %cont = arith.andi %lt0, %lt32 : i1
            scf.condition(%cont) %i : i64
          } do {
          ^bb0(%i: i64):
            %c_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %c_mem[%c0] : memref<1xi64>
            omp.atomic.read %c_mem = %count memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %cv = memref.load %c_mem[%c0] : memref<1xi64>
            %cgt1 = arith.cmpi sgt, %cv, %c1_i64 : i64
            scf.if %cgt1 {
              %h_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h_mem[%c0] : memref<1xi64>
              omp.atomic.read %h_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h1 = memref.load %h_mem[%c0] : memref<1xi64>
              %h1_idx = arith.index_cast %h1 : i64 to index
              %n1 = memref.atomic_rmw assign %cm1_i64, %next[%h1_idx] : (i64, memref<3xi64>) -> i64
              %n1ge0 = arith.cmpi sge, %n1, %c0_i64 : i64
              scf.if %n1ge0 {
                omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
                ^bb0(%hv: i64):
                  %eq = arith.cmpi eq, %hv, %h1 : i64
                  %sel = arith.select %eq, %n1, %hv : i64
                  omp.yield (%sel : i64)
                }
                %h2_mem = memref.alloca() : memref<1xi64>
                memref.store %c0_i64, %h2_mem[%c0] : memref<1xi64>
                omp.atomic.read %h2_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
                %h2 = memref.load %h2_mem[%c0] : memref<1xi64>
                %ok = arith.cmpi eq, %h2, %n1 : i64
                scf.if %ok {
                  memref.atomic_rmw addi %cm1_i64, %count[%c0] : (i64, memref<1xi64>) -> i64
                  memref.store %h1, %elem[%c0] : memref<1xi64>
                } else {
                  memref.atomic_rmw assign %n1, %next[%h1_idx] : (i64, memref<3xi64>) -> i64
                }
              }
            }
            %inext = arith.addi %i, %c1_i64 : i64
            scf.yield %inext : i64
          }
          %ev = memref.load %elem[%c0] : memref<1xi64>
          %ev_idx = arith.index_cast %ev : i64 to index
          %ge0 = arith.cmpi sge, %ev, %c0_i64 : i64
          scf.if %ge0 {
            memref.atomic_rmw assign %c0_i64, %val[%ev_idx] : (i64, memref<3xi64>) -> i64
            %h_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %h_mem[%c0] : memref<1xi64>
            omp.atomic.read %h_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %h1 = memref.load %h_mem[%c0] : memref<1xi64>
            memref.atomic_rmw assign %h1, %next[%ev_idx] : (i64, memref<3xi64>) -> i64
            omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
            ^bb0(%hv: i64):
              %eq = arith.cmpi eq, %hv, %h1 : i64
              %sel = arith.select %eq, %ev, %hv : i64
              omp.yield (%sel : i64)
            }
            scf.while (%j = %c0_i64) : (i64) -> i64 {
              %h2_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h2_mem[%c0] : memref<1xi64>
              omp.atomic.read %h2_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h2 = memref.load %h2_mem[%c0] : memref<1xi64>
              %notdone = arith.cmpi ne, %h2, %ev : i64
              %lt8 = arith.cmpi slt, %j, %c8_i64 : i64
              %cont = arith.andi %notdone, %lt8 : i1
              scf.condition(%cont) %j : i64
            } do {
            ^bb0(%j: i64):
              %h3_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h3_mem[%c0] : memref<1xi64>
              omp.atomic.read %h3_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h3 = memref.load %h3_mem[%c0] : memref<1xi64>
              memref.atomic_rmw assign %h3, %next[%ev_idx] : (i64, memref<3xi64>) -> i64
              omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
              ^bb0(%hv: i64):
                %eq = arith.cmpi eq, %hv, %h3 : i64
                %sel = arith.select %eq, %ev, %hv : i64
                omp.yield (%sel : i64)
              }
              %jnext = arith.addi %j, %c1_i64 : i64
              scf.yield %jnext : i64
            }
            memref.atomic_rmw addi %c1_i64, %count[%c0] : (i64, memref<1xi64>) -> i64
          }
          omp.terminator
        }
        omp.section {
          // P1: pop one element, tag it, push it back
          %elem = memref.alloca() : memref<1xi64>
          memref.store %cm1_i64, %elem[%c0] : memref<1xi64>
          scf.while (%i = %c0_i64) : (i64) -> i64 {
            %ev = memref.load %elem[%c0] : memref<1xi64>
            %lt0 = arith.cmpi slt, %ev, %c0_i64 : i64
            %lt32 = arith.cmpi slt, %i, %c32_i64 : i64
            %cont = arith.andi %lt0, %lt32 : i1
            scf.condition(%cont) %i : i64
          } do {
          ^bb0(%i: i64):
            %c_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %c_mem[%c0] : memref<1xi64>
            omp.atomic.read %c_mem = %count memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %cv = memref.load %c_mem[%c0] : memref<1xi64>
            %cgt1 = arith.cmpi sgt, %cv, %c1_i64 : i64
            scf.if %cgt1 {
              %h_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h_mem[%c0] : memref<1xi64>
              omp.atomic.read %h_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h1 = memref.load %h_mem[%c0] : memref<1xi64>
              %h1_idx = arith.index_cast %h1 : i64 to index
              %n1 = memref.atomic_rmw assign %cm1_i64, %next[%h1_idx] : (i64, memref<3xi64>) -> i64
              %n1ge0 = arith.cmpi sge, %n1, %c0_i64 : i64
              scf.if %n1ge0 {
                omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
                ^bb0(%hv: i64):
                  %eq = arith.cmpi eq, %hv, %h1 : i64
                  %sel = arith.select %eq, %n1, %hv : i64
                  omp.yield (%sel : i64)
                }
                %h2_mem = memref.alloca() : memref<1xi64>
                memref.store %c0_i64, %h2_mem[%c0] : memref<1xi64>
                omp.atomic.read %h2_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
                %h2 = memref.load %h2_mem[%c0] : memref<1xi64>
                %ok = arith.cmpi eq, %h2, %n1 : i64
                scf.if %ok {
                  memref.atomic_rmw addi %cm1_i64, %count[%c0] : (i64, memref<1xi64>) -> i64
                  memref.store %h1, %elem[%c0] : memref<1xi64>
                } else {
                  memref.atomic_rmw assign %n1, %next[%h1_idx] : (i64, memref<3xi64>) -> i64
                }
              }
            }
            %inext = arith.addi %i, %c1_i64 : i64
            scf.yield %inext : i64
          }
          %ev = memref.load %elem[%c0] : memref<1xi64>
          %ev_idx = arith.index_cast %ev : i64 to index
          %ge0 = arith.cmpi sge, %ev, %c0_i64 : i64
          scf.if %ge0 {
            memref.atomic_rmw assign %c1_i64, %val[%ev_idx] : (i64, memref<3xi64>) -> i64
            %h_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %h_mem[%c0] : memref<1xi64>
            omp.atomic.read %h_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %h1 = memref.load %h_mem[%c0] : memref<1xi64>
            memref.atomic_rmw assign %h1, %next[%ev_idx] : (i64, memref<3xi64>) -> i64
            omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
            ^bb0(%hv: i64):
              %eq = arith.cmpi eq, %hv, %h1 : i64
              %sel = arith.select %eq, %ev, %hv : i64
              omp.yield (%sel : i64)
            }
            scf.while (%j = %c0_i64) : (i64) -> i64 {
              %h2_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h2_mem[%c0] : memref<1xi64>
              omp.atomic.read %h2_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h2 = memref.load %h2_mem[%c0] : memref<1xi64>
              %notdone = arith.cmpi ne, %h2, %ev : i64
              %lt8 = arith.cmpi slt, %j, %c8_i64 : i64
              %cont = arith.andi %notdone, %lt8 : i1
              scf.condition(%cont) %j : i64
            } do {
            ^bb0(%j: i64):
              %h3_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h3_mem[%c0] : memref<1xi64>
              omp.atomic.read %h3_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h3 = memref.load %h3_mem[%c0] : memref<1xi64>
              memref.atomic_rmw assign %h3, %next[%ev_idx] : (i64, memref<3xi64>) -> i64
              omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
              ^bb0(%hv: i64):
                %eq = arith.cmpi eq, %hv, %h3 : i64
                %sel = arith.select %eq, %ev, %hv : i64
                omp.yield (%sel : i64)
              }
              %jnext = arith.addi %j, %c1_i64 : i64
              scf.yield %jnext : i64
            }
            memref.atomic_rmw addi %c1_i64, %count[%c0] : (i64, memref<1xi64>) -> i64
          }
          omp.terminator
        }
        omp.section {
          // P2: pop one element, tag it, push it back
          %elem = memref.alloca() : memref<1xi64>
          memref.store %cm1_i64, %elem[%c0] : memref<1xi64>
          scf.while (%i = %c0_i64) : (i64) -> i64 {
            %ev = memref.load %elem[%c0] : memref<1xi64>
            %lt0 = arith.cmpi slt, %ev, %c0_i64 : i64
            %lt32 = arith.cmpi slt, %i, %c32_i64 : i64
            %cont = arith.andi %lt0, %lt32 : i1
            scf.condition(%cont) %i : i64
          } do {
          ^bb0(%i: i64):
            %c_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %c_mem[%c0] : memref<1xi64>
            omp.atomic.read %c_mem = %count memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %cv = memref.load %c_mem[%c0] : memref<1xi64>
            %cgt1 = arith.cmpi sgt, %cv, %c1_i64 : i64
            scf.if %cgt1 {
              %h_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h_mem[%c0] : memref<1xi64>
              omp.atomic.read %h_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h1 = memref.load %h_mem[%c0] : memref<1xi64>
              %h1_idx = arith.index_cast %h1 : i64 to index
              %n1 = memref.atomic_rmw assign %cm1_i64, %next[%h1_idx] : (i64, memref<3xi64>) -> i64
              %n1ge0 = arith.cmpi sge, %n1, %c0_i64 : i64
              scf.if %n1ge0 {
                omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
                ^bb0(%hv: i64):
                  %eq = arith.cmpi eq, %hv, %h1 : i64
                  %sel = arith.select %eq, %n1, %hv : i64
                  omp.yield (%sel : i64)
                }
                %h2_mem = memref.alloca() : memref<1xi64>
                memref.store %c0_i64, %h2_mem[%c0] : memref<1xi64>
                omp.atomic.read %h2_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
                %h2 = memref.load %h2_mem[%c0] : memref<1xi64>
                %ok = arith.cmpi eq, %h2, %n1 : i64
                scf.if %ok {
                  memref.atomic_rmw addi %cm1_i64, %count[%c0] : (i64, memref<1xi64>) -> i64
                  memref.store %h1, %elem[%c0] : memref<1xi64>
                } else {
                  memref.atomic_rmw assign %n1, %next[%h1_idx] : (i64, memref<3xi64>) -> i64
                }
              }
            }
            %inext = arith.addi %i, %c1_i64 : i64
            scf.yield %inext : i64
          }
          %ev = memref.load %elem[%c0] : memref<1xi64>
          %ev_idx = arith.index_cast %ev : i64 to index
          %ge0 = arith.cmpi sge, %ev, %c0_i64 : i64
          scf.if %ge0 {
            memref.atomic_rmw assign %c2_i64, %val[%ev_idx] : (i64, memref<3xi64>) -> i64
            %h_mem = memref.alloca() : memref<1xi64>
            memref.store %c0_i64, %h_mem[%c0] : memref<1xi64>
            omp.atomic.read %h_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
            %h1 = memref.load %h_mem[%c0] : memref<1xi64>
            memref.atomic_rmw assign %h1, %next[%ev_idx] : (i64, memref<3xi64>) -> i64
            omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
            ^bb0(%hv: i64):
              %eq = arith.cmpi eq, %hv, %h1 : i64
              %sel = arith.select %eq, %ev, %hv : i64
              omp.yield (%sel : i64)
            }
            scf.while (%j = %c0_i64) : (i64) -> i64 {
              %h2_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h2_mem[%c0] : memref<1xi64>
              omp.atomic.read %h2_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h2 = memref.load %h2_mem[%c0] : memref<1xi64>
              %notdone = arith.cmpi ne, %h2, %ev : i64
              %lt8 = arith.cmpi slt, %j, %c8_i64 : i64
              %cont = arith.andi %notdone, %lt8 : i1
              scf.condition(%cont) %j : i64
            } do {
            ^bb0(%j: i64):
              %h3_mem = memref.alloca() : memref<1xi64>
              memref.store %c0_i64, %h3_mem[%c0] : memref<1xi64>
              omp.atomic.read %h3_mem = %head memory_order(relaxed) : memref<1xi64>, memref<1xi64>, i64
              %h3 = memref.load %h3_mem[%c0] : memref<1xi64>
              memref.atomic_rmw assign %h3, %next[%ev_idx] : (i64, memref<3xi64>) -> i64
              omp.atomic.compare memory_order(relaxed) %head : memref<1xi64> {
              ^bb0(%hv: i64):
                %eq = arith.cmpi eq, %hv, %h3 : i64
                %sel = arith.select %eq, %ev, %hv : i64
                omp.yield (%sel : i64)
              }
              %jnext = arith.addi %j, %c1_i64 : i64
              scf.yield %jnext : i64
            }
            memref.atomic_rmw addi %c1_i64, %count[%c0] : (i64, memref<1xi64>) -> i64
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    %count_v = memref.load %count[%c0] : memref<1xi64>
    memref.dealloc %head : memref<1xi64>
    memref.dealloc %count : memref<1xi64>
    memref.dealloc %next : memref<3xi64>
    memref.dealloc %val : memref<3xi64>
    return %count_v : i64
  }
}
