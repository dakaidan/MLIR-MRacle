module {
  func.func @main() {
    omp.parallel {
      omp.sections {
        omp.section {
          omp.critical {
            omp.terminator
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}