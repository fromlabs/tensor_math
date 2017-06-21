import "dart:math" as math;

import "package:collection/collection.dart";

import "package:tensor_math_simd/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  test(test1);

  test(test2);

  test(test3);
}

void test(void testFunction(List value)) {
  var watch = new Stopwatch();
  watch.start();

  var minDimension = 2;
  var maxDimension = 4;
  var dimensionCount = 12;

  for (var i = minDimension; i < maxDimension; i++) {
    var combinations = math.pow(dimensionCount, i + 1);
    for (var i2 = 0; i2 < combinations; i2++) {
      var shape = new List(i + 1);

      var i3 = 0;
      var value = i2;
      var scale = combinations ~/ dimensionCount;

      while (scale > 0) {
        shape[i3] = (value ~/ scale) + 1;

        i3++;
        value = value % scale;
        scale = scale ~/ dimensionCount;
      }

      var list = generateValue(shape);

      testFunction(list);
    }
  }

  print("Elapsed in ${watch.elapsedMilliseconds} ms");
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
