; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, ptr @0 }, align 8
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8

declare void @free(ptr)

declare ptr @malloc(i64)

define { i64, i64, i64, i64 } @main() {
  %structArg = alloca { ptr, ptr, ptr, ptr }, align 8
  %.reloaded = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.reloaded7 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.reloaded8 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.reloaded9 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %1 = call ptr @malloc(i64 8)
  %2 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, i64 1, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, i64 1, 4, 0
  %7 = call ptr @malloc(i64 8)
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %7, 0
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, ptr %7, 1
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 0, 2
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 1, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 1, 4, 0
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %14 = getelementptr i64, ptr %13, i64 0
  %15 = atomicrmw xchg ptr %14, i64 0 acq_rel, align 8
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %17 = getelementptr i64, ptr %16, i64 0
  %18 = atomicrmw xchg ptr %17, i64 0 acq_rel, align 8
  %19 = call ptr @malloc(i64 8)
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %19, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, ptr %19, 1
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 0, 2
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 1, 3, 0
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 1, 4, 0
  %25 = call ptr @malloc(i64 8)
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %25, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, ptr %25, 1
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 0, 2
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 1, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, i64 1, 4, 0
  br label %entry

entry:                                            ; preds = %0
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %.reloaded, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, ptr %.reloaded7, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, ptr %.reloaded8, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, ptr %.reloaded9, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_.reloaded7 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %.reloaded7, ptr %gep_.reloaded7, align 8
  %gep_.reloaded8 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
  store ptr %.reloaded8, ptr %gep_.reloaded8, align 8
  %gep_.reloaded9 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 3
  store ptr %.reloaded9, ptr %gep_.reloaded9, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @main..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  %31 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, 1
  %32 = getelementptr inbounds nuw i64, ptr %31, i64 0
  %33 = load i64, ptr %32, align 4
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 1
  %35 = getelementptr inbounds nuw i64, ptr %34, i64 0
  %36 = load i64, ptr %35, align 4
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %38 = getelementptr inbounds nuw i64, ptr %37, i64 0
  %39 = load i64, ptr %38, align 4
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %41 = getelementptr inbounds nuw i64, ptr %40, i64 0
  %42 = load i64, ptr %41, align 4
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  call void @free(ptr %43)
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  call void @free(ptr %44)
  %45 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, 0
  call void @free(ptr %45)
  %46 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, 0
  call void @free(ptr %46)
  call void @__kmpc_flush(ptr @1)
  %47 = insertvalue { i64, i64, i64, i64 } poison, i64 %33, 0
  %48 = insertvalue { i64, i64, i64, i64 } %47, i64 %36, 1
  %49 = insertvalue { i64, i64, i64, i64 } %48, i64 %39, 2
  %50 = insertvalue { i64, i64, i64, i64 } %49, i64 %42, 3
  ret { i64, i64, i64, i64 } %50
}

; Function Attrs: nounwind
define internal void @main..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8, !align !1
  %gep_.reloaded7 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded7 = load ptr, ptr %gep_.reloaded7, align 8, !align !1
  %gep_.reloaded8 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded8 = load ptr, ptr %gep_.reloaded8, align 8, !align !1
  %gep_.reloaded9 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded9 = load ptr, ptr %gep_.reloaded9, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded, align 8
  %3 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded7, align 8
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded8, align 8
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded9, align 8
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
  %6 = load i32, ptr %p.lowerbound, align 4
  %7 = load i32, ptr %p.upperbound, align 4
  %8 = sub i32 %7, %6
  %9 = add i32 %8, 1
  br label %omp_section_loop.header

omp_section_loop.header:                          ; preds = %omp_section_loop.inc, %omp_section_loop.preheader
  %omp_section_loop.iv = phi i32 [ 0, %omp_section_loop.preheader ], [ %omp_section_loop.next, %omp_section_loop.inc ]
  br label %omp_section_loop.cond

omp_section_loop.cond:                            ; preds = %omp_section_loop.header
  %omp_section_loop.cmp = icmp ult i32 %omp_section_loop.iv, %9
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
  %10 = add i32 %omp_section_loop.iv, %6
  %11 = mul i32 %10, 1
  %12 = add i32 %11, 0
  switch i32 %12, label %omp_section_loop.body.sections.after [
    i32 0, label %omp_section_loop.body.case
    i32 1, label %omp_section_loop.body.case3
  ]

omp_section_loop.body.case3:                      ; preds = %omp_section_loop.body
  br label %omp.section.region5

omp.section.region5:                              ; preds = %omp_section_loop.body.case3
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, 1
  %14 = getelementptr i64, ptr %13, i64 0
  %15 = atomicrmw xchg ptr %14, i64 1 acq_rel, align 8
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 1
  %17 = getelementptr i64, ptr %16, i64 0
  %18 = atomicrmw xchg ptr %17, i64 1 acq_rel, align 8
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.section.region5
  br label %omp_section_loop.body.sections.after

omp_section_loop.body.case:                       ; preds = %omp_section_loop.body
  br label %omp.section.region

omp.section.region:                               ; preds = %omp_section_loop.body.case
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %20 = getelementptr inbounds nuw i64, ptr %19, i64 0
  store i64 0, ptr %20, align 4
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %22 = getelementptr inbounds nuw i64, ptr %21, i64 0
  store i64 0, ptr %22, align 4
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 1
  %24 = getelementptr i64, ptr %23, i64 0
  %25 = atomicrmw add ptr %24, i64 0 acq_rel, align 8
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, 1
  %27 = getelementptr i64, ptr %26, i64 0
  %28 = atomicrmw add ptr %27, i64 0 acq_rel, align 8
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %30 = getelementptr inbounds nuw i64, ptr %29, i64 0
  store i64 %25, ptr %30, align 4
  %31 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %32 = getelementptr inbounds nuw i64, ptr %31, i64 0
  store i64 %28, ptr %32, align 4
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

; Function Attrs: convergent nounwind
declare void @__kmpc_flush(ptr) #1

; Function Attrs: nounwind
declare !callback !2 void @__kmpc_fork_call(ptr, i32, ptr, ...) #0

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 8}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
