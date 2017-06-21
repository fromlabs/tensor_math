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

  var list = generateValue([10, 10, 10, 100]);

  for (var i = 0; i < 1000; i++) {
    testFunction(list);
  }

  print("Elapsed in ${watch.elapsedMilliseconds} ms");
}

void test1(List value) {
  new tm.NDArray(value, dataType: tm.NDDataType.float32).toValue();
}

void test2(List value) {
  new tm.NDArray(value, dataType: tm.NDDataType.float32HBlocked).toValue();
}

void test3(List value) {
  new tm.NDArray(value, dataType: tm.NDDataType.float32VBlocked).toValue();
}

List generateValue(List<int> shape) =>
    new tm.NDArray.generate(shape, (index) => index + 1).toValue();
