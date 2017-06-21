import "dart:math" as math;

import "package:collection/collection.dart";

import "package:tensor_math_simd/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  var list = generateValue([2, 2, 2, 2]);

  var array2 = new tm.NDArray(list);
  var list2 = array2.transpose(permutationAxis: [1, 0, 2, 3]);

  print(list);
  print(list2);

  test1(list);
  test2(list);
  test3(list);
}

void test1(List value) {
  if (!equality.equals(
      new tm.NDArray(value, dataType: tm.NDDataType.float32).toValue(),
      value)) {
    throw new StateError("Not equals 1");
  }
}

void test2(List value) {
  if (!equality.equals(
      new tm.NDArray(value, dataType: tm.NDDataType.float32HBlocked).toValue(),
      value)) {
    throw new StateError("Not equals 2");
  }
}

void test3(List value) {
  if (!equality.equals(
      new tm.NDArray(value, dataType: tm.NDDataType.float32VBlocked).toValue(),
      value)) {
    throw new StateError("Not equals 3");
  }
}

List generateValue(List<int> shape) =>
    new tm.NDArray.generate(shape, (index) => index + 1).toValue();
