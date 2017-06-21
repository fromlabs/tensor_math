import "dart:math" as math;

import "package:collection/collection.dart";

import "package:tensor_math_simd/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  var list = new tm.NDArray.generate([4, 4, 4, 4], (index) => index + 1,
          dataType: tm.NDDataType.float32)
      .transpose(permutationAxis: [1, 0, 2, 3]).toValue();

  print(list);

  print(new tm.NDArray(list, dataType: tm.NDDataType.float32)
      .transpose(permutationAxis: [1, 0, 2, 3]).toValue());

  print(
      new tm.NDArray(list, dataType: tm.NDDataType.float32HBlocked).toValue());

  print(
      new tm.NDArray(list, dataType: tm.NDDataType.float32VBlocked).toValue());
}
