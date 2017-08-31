import "dart:math" as math;

import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest([2, 2, 4], 0);

  return;

  functionalTest([10, 10, 10], 0);
  functionalTest([10, 10, 10], 1);
  functionalTest([10, 10, 10], 2);

  functionalTest([10, 10], 0);
  functionalTest([10, 10], 1);
  functionalTest([10], 0);
}

void functionalTest(List<int> shape, int axis) {
  print("***********************************************************");

  List list = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32).reshape(newDimensions: [-1]).toValue();

  list.shuffle(new math.Random(0));

  var initialValue = new tm.NDArray(list, dataType: tm.NDDataType.float32)
      .reshape(newDimensions: shape)
      .toValue();

  var array = new tm.NDArray(initialValue, dataType: tm.NDDataType.float32);

  print(array.toValue());

  var expectedValue = array.argMax(axis: axis).toValue();

  print(expectedValue);

  var value;

  value = new tm.NDArray(initialValue, dataType: tm.NDDataType.float32VBlocked)
      .argMax(axis: axis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }
}
