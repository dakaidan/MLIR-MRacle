#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>

#include "ConquerDialect.h.inc"

#define GET_OP_CLASSES
#include "ConquerOps.h.inc"