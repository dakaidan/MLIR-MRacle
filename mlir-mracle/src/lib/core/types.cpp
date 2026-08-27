#include "mlir-mracle/core/types.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir_mracle {

std::string outcomeRelationToString(OutcomeRelation relation) {
    switch (relation) {
    case OutcomeRelation::Equality:
        return "equality";
    case OutcomeRelation::Subset:
        return "subset";
    case OutcomeRelation::Superset:
        return "superset";
    }
    llvm_unreachable("unknown outcome relation");
}

std::string issueSeverityToString(IssueSeverity severity) {
    switch (severity) {
    case IssueSeverity::Fail:
        return "FAIL";
    case IssueSeverity::Warn:
        return "WARN";
    }
    llvm_unreachable("unknown issue severity");
}

bool outcomeRelationFromString(const std::string &s,
                               OutcomeRelation &relation) {
    if (s == "equality") {
        relation = OutcomeRelation::Equality;
        return true;
    }
    if (s == "subset") {
        relation = OutcomeRelation::Subset;
        return true;
    }
    if (s == "superset") {
        relation = OutcomeRelation::Superset;
        return true;
    }
    return false;
}

} // namespace mlir_mracle
