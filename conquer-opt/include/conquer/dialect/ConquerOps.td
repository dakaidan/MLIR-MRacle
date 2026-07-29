#ifndef CONQUER_IR_CONQUEROPS
#define CONQUER_IR_CONQUEROPS

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Conquer_Dialect : Dialect {
  let name = "conquer";
  let cppNamespace = "::conquer";
}

class Conquer_Op<string mnemonic, list<Trait> traits = []> :
    Op<Conquer_Dialect, mnemonic, traits>;

def Conquer_FloatTensor : RankedTensorOf<[AnyFloat]>;

def I4 : I<4>;
def I48 : I<48>;

def Conquer_QStorageElem : AnyTypeOf<[I4, I8, I16, I32, I48]>;

def Conquer_QTensor : RankedTensorOf<[Conquer_QStorageElem]>;

def Conquer_QuantiseOp : Conquer_Op<"quantise",
    [Pure, SameOperandsAndResultShape]> {
  let summary = "Marker op for float-to-integer quantisation";
  let description = [{
    Quantises a ranked floating-point tensor into a ranked signless integer tensor.

    This is a semantic compiler op, not a final TOSA op.

    The quantisation parameters are carried explicitly:
      - `scales`:
          * size 1           => per-tensor
          * size > 1         => per-channel / partitioned quantisation
      - `zero_point`:
          * scalar only, kept scalar to stay TOSA-friendly
      - `partition_kind`:
          * "per_tensor"
          * "axis"
          * "depthwise_cm"
      - `axis`:
          * required only when partition_kind == "axis"
      - `narrow_range`:
          * whether the minimum storage value is excluded during clamping
            (e.g. int8 [-127,127] instead of [-128,127])

    The result element type determines the storage width: i4/i8/i16/i32.
    Signless storage is interpreted with signed quantisation semantics.
  }];

  let arguments = (ins
    Conquer_FloatTensor:$input,
    DenseF64ArrayAttr:$scales,
    I64Attr:$zero_point,
    StrAttr:$partition_kind,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    Conquer_QTensor:$output
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Conquer_DequantiseOp : Conquer_Op<"dequantise",
    [Pure, SameOperandsAndResultShape]> {
  let summary = "Marker op for integer-to-float dequantisation";
  let description = [{
    Dequantises a ranked signless integer tensor into a ranked floating-point tensor.

    This is a semantic compiler op, not a final TOSA op.

    It uses the same qparams schema as conquer.quantise:
      - `scales`
      - `zero_point`
      - `partition_kind`
      - `axis`
      - `narrow_range`

    The input element type determines the storage width: i4/i8/i16/i32.
    Signless storage is interpreted with signed quantisation semantics.
  }];

  let arguments = (ins
    Conquer_QTensor:$input,
    DenseF64ArrayAttr:$scales,
    I64Attr:$zero_point,
    StrAttr:$partition_kind,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$narrow_range
  );

  let results = (outs
    Conquer_FloatTensor:$output
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

#endif // CONQUER_IR_CONQUEROPS
