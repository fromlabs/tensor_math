import "dart:typed_data";

import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

import "package:tensor_math/src/nd_array_blocked_impl.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  print(new Int32x4.bool(true, false, true, true) | new Int32x4.bool(true, false, true, false));
  print(new Int32x4.bool(true, false, true, true) & new Int32x4.bool(true, false, true, false));

  print(new Float32x4(1.0, 2.0, 3.0, 4.0).shuffleMix(new Float32x4(11.0, 12.0, 13.0, 14.0), 0));

  functionalTest([8, 5, 5]);
  functionalTest([10, 10, 10, 10]);
  functionalTest([10]);
  functionalTest([]);

  //performanceTest();
}

void functionalTest(List<int> shape) {
  var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32)
      .neg()
      .toValue();

  print(expectedValue);

  var value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32Blocked)
      .neg()
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(expectedValue);
    print(value);
    throw new StateError("not equals");
  }
}

void performanceTest() {
  test([10, 10, 10, 10], tm.NDDataType.float32, 10000);

  test([10, 10, 10, 10], tm.NDDataType.float32Blocked, 10000);

  test([10, 10, 10, 10], tm.NDDataType.float32, 100000);

  test([10, 10, 10, 10], tm.NDDataType.float32Blocked, 100000);
}

void allPerformanceTest() {}

void test(List<int> shape, tm.NDDataType type, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var array =
      new tm.NDArray.generate(shape, (index) => index + 1, dataType: type);

  for (var i = 0; i < steps; i++) {
    array.neg();
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}
