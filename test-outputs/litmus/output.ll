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
  %.reloaded11 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.reloaded12 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.reloaded13 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
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
  %17 = getelementptr inbounds nuw i64, ptr %16, i64 0
  store i64 0, ptr %17, align 4
  %18 = call ptr @malloc(i64 8)
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %18, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, ptr %18, 1
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 0, 2
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 1, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 1, 4, 0
  %24 = call ptr @malloc(i64 8)
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %24, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, ptr %24, 1
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 0, 2
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 1, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 1, 4, 0
  br label %entry

entry:                                            ; preds = %0
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, ptr %.reloaded, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, ptr %.reloaded11, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %.reloaded12, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, ptr %.reloaded13, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_.reloaded11 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %.reloaded11, ptr %gep_.reloaded11, align 8
  %gep_.reloaded12 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
  store ptr %.reloaded12, ptr %gep_.reloaded12, align 8
  %gep_.reloaded13 = getelementptr { ptr, ptr, ptr, ptr }, ptr %structArg, i32 0, i32 3
  store ptr %.reloaded13, ptr %gep_.reloaded13, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @main..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %31 = getelementptr inbounds nuw i64, ptr %30, i64 0
  %32 = load i64, ptr %31, align 4
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 1
  %34 = getelementptr inbounds nuw i64, ptr %33, i64 0
  %35 = load i64, ptr %34, align 4
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %37 = getelementptr inbounds nuw i64, ptr %36, i64 0
  %38 = load i64, ptr %37, align 4
  call void @__kmpc_flush(ptr @1)
  %39 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %40 = getelementptr inbounds nuw i64, ptr %39, i64 0
  %41 = load i64, ptr %40, align 4
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 0
  call void @free(ptr %42)
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 0
  call void @free(ptr %43)
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 0
  call void @free(ptr %44)
  %45 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 0
  call void @free(ptr %45)
  %46 = insertvalue { i64, i64, i64, i64 } poison, i64 %32, 0
  %47 = insertvalue { i64, i64, i64, i64 } %46, i64 %35, 1
  %48 = insertvalue { i64, i64, i64, i64 } %47, i64 %38, 2
  %49 = insertvalue { i64, i64, i64, i64 } %48, i64 %41, 3
  ret { i64, i64, i64, i64 } %49
}

; Function Attrs: nounwind
define internal void @main..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_.reloaded = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8, !align !1
  %gep_.reloaded11 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded11 = load ptr, ptr %gep_.reloaded11, align 8, !align !1
  %gep_.reloaded12 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded12 = load ptr, ptr %gep_.reloaded12, align 8, !align !1
  %gep_.reloaded13 = getelementptr { ptr, ptr, ptr, ptr }, ptr %0, i32 0, i32 3
  %loadgep_.reloaded13 = load ptr, ptr %gep_.reloaded13, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded, align 8
  %3 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded11, align 8
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded12, align 8
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded13, align 8
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
  %omp_global_thread_num10 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @3, i32 %omp_global_thread_num10)
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
    i32 1, label %omp_section_loop.body.case5
  ]

omp_section_loop.body.case5:                      ; preds = %omp_section_loop.body
  br label %omp.section.region7

omp.section.region7:                              ; preds = %omp_section_loop.body.case5
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, 1
  %14 = getelementptr inbounds nuw i64, ptr %13, i64 0
  store i64 0, ptr %14, align 4
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 1
  %16 = getelementptr inbounds nuw i64, ptr %15, i64 0
  %17 = load i64, ptr %16, align 4
  %18 = icmp eq i64 %17, 1
  br i1 %18, label %omp.section.region8, label %omp.section.region9

omp.section.region9:                              ; preds = %omp.section.region8, %omp.section.region7
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, 1
  %20 = getelementptr inbounds nuw i64, ptr %19, i64 0
  store i64 %17, ptr %20, align 4
  br label %omp.region.cont6

omp.region.cont6:                                 ; preds = %omp.section.region9
  br label %omp_section_loop.body.sections.after

omp.section.region8:                              ; preds = %omp.section.region7
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %22 = getelementptr i64, ptr %21, i64 0
  %23 = atomicrmw xchg ptr %22, i64 1 acq_rel, align 8
  br label %omp.section.region9

omp_section_loop.body.case:                       ; preds = %omp_section_loop.body
  br label %omp.section.region

omp.section.region:                               ; preds = %omp_section_loop.body.case
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %25 = getelementptr inbounds nuw i64, ptr %24, i64 0
  store i64 0, ptr %25, align 4
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %27 = getelementptr i64, ptr %26, i64 0
  %28 = atomicrmw add ptr %27, i64 0 acq_rel, align 8
  %29 = icmp eq i64 %28, 1
  br i1 %29, label %omp.section.region3, label %omp.section.region4

omp.section.region4:                              ; preds = %omp.section.region3, %omp.section.region
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %31 = getelementptr inbounds nuw i64, ptr %30, i64 0
  store i64 %28, ptr %31, align 4
  br label %omp.region.cont2

omp.region.cont2:                                 ; preds = %omp.section.region4
  br label %omp_section_loop.body.sections.after

omp.section.region3:                              ; preds = %omp.section.region
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 1
  %33 = getelementptr inbounds nuw i64, ptr %32, i64 0
  store i64 1, ptr %33, align 4
  br label %omp.section.region4

omp_section_loop.body.sections.after:             ; preds = %omp.region.cont6, %omp.region.cont2, %omp_section_loop.body
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
