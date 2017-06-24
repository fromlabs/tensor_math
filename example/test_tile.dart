import "dart:math" as math;

import "package:collection/collection.dart";

import "package:tensor_math_simd/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  print(new tm.NDArray.generate([4, 4], (index) => index + 1)
      .tile([2, 2]).toValue());
}
