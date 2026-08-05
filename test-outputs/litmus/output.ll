; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@flags = private global [2 x i64] [i64 10, i64 20]
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, ptr @0 }, align 8
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8

declare void @free(ptr)

declare ptr @malloc(i64)

define i64 @main() {
  %structArg = alloca { ptr }, align 8
  %.reloaded = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %1 = call ptr @malloc(i64 8)
  %2 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, i64 1, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, i64 1, 4, 0
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %8 = getelementptr inbounds nuw i64, ptr %7, i64 0
  store i64 0, ptr %8, align 4
  br label %entry

entry:                                            ; preds = %0
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %.reloaded, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_.reloaded = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @main..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %10 = getelementptr inbounds nuw i64, ptr %9, i64 0
  %11 = load i64, ptr %10, align 4
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  call void @free(ptr %12)
  ret i64 %11
}

; Function Attrs: nounwind
define internal void @main..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_.reloaded = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded, align 8
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.par.entry
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  br label %omp_section_loop.preheader

omp_section_loop.preheader:                       ; preds = %omp.par.region1
  store i32 0, ptr %p.lowerbound, align 4
  store i32 1, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @2, i32 %omp_global_thread_num, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  %3 = load i32, ptr %p.lowerbound, align 4
  %4 = load i32, ptr %p.upperbound, align 4
  %5 = sub i32 %4, %3
  %6 = add i32 %5, 1
  br label %omp_section_loop.header

omp_section_loop.header:                          ; preds = %omp_section_loop.inc, %omp_section_loop.preheader
  %omp_section_loop.iv = phi i32 [ 0, %omp_section_loop.preheader ], [ %omp_section_loop.next, %omp_section_loop.inc ]
  br label %omp_section_loop.cond

omp_section_loop.cond:                            ; preds = %omp_section_loop.header
  %omp_section_loop.cmp = icmp ult i32 %omp_section_loop.iv, %6
  br i1 %omp_section_loop.cmp, label %omp_section_loop.body, label %omp_section_loop.exit

omp_section_loop.exit:                            ; preds = %omp_section_loop.cond
  call void @__kmpc_for_static_fini(ptr @2, i32 %omp_global_thread_num)
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @3, i32 %omp_global_thread_num6)
  br label %omp_section_loop.after

omp_section_loop.after:                           ; preds = %omp_section_loop.exit
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp_section_loop.after
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_section_loop.body:                            ; preds = %omp_section_loop.cond
  %7 = add i32 %omp_section_loop.iv, %3
  %8 = mul i32 %7, 1
  %9 = add i32 %8, 0
  switch i32 %9, label %omp_section_loop.body.sections.after [
    i32 0, label %omp_section_loop.body.case
    i32 1, label %omp_section_loop.body.case3
  ]

omp_section_loop.body.case3:                      ; preds = %omp_section_loop.body
  br label %omp.section.region5

omp.section.region5:                              ; preds = %omp_section_loop.body.case3
  %10 = load i64, ptr getelementptr inbounds nuw (i8, ptr @flags, i64 8), align 4
  %11 = load i64, ptr @flags, align 4
  %12 = sub i64 %10, %11
  %13 = mul i64 %12, 3
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, 1
  %15 = getelementptr inbounds nuw i64, ptr %14, i64 0
  store i64 %13, ptr %15, align 4
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.section.region5
  br label %omp_section_loop.body.sections.after

omp_section_loop.body.case:                       ; preds = %omp_section_loop.body
  br label %omp.section.region

omp.section.region:                               ; preds = %omp_section_loop.body.case
  %16 = load i64, ptr @flags, align 4
  %17 = load i64, ptr getelementptr inbounds nuw (i8, ptr @flags, i64 8), align 4
  %18 = add i64 %16, %17
  %19 = mul i64 %18, 2
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, 1
  %21 = getelementptr inbounds nuw i64, ptr %20, i64 0
  store i64 %19, ptr %21, align 4
  br label %omp.region.cont2

omp.region.cont2:                                 ; preds = %omp.section.region
  br label %omp_section_loop.body.sections.after

omp_section_loop.body.sections.after:             ; preds = %omp.region.cont4, %omp.region.cont2, %omp_section_loop.body
  br label %omp_section_loop.inc

omp_section_loop.inc:                             ; preds = %omp_section_loop.body.sections.after
  %omp_section_loop.next = add nuw i32 %omp_section_loop.iv, 1
  br label %omp_section_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) #0

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) #0

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #0

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) #1

; Function Attrs: nounwind
declare !callback !2 void @__kmpc_fork_call(ptr, i32, ptr, ...) #0

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 8}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
