import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

import "package:tensor_math/src/nd_array_blocked_impl.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest();

  performanceTest();
}

void functionalTest() {
  var shape = [8, 5, 5];

  NDArrayBlockedImpl fromArray = new tm.NDArray.generate(
      shape, (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked);

  NDArrayBlockedImpl toArray = fromArray.cast(tm.NDDataType.float32VBlocked);

  if (!iterableEquality.equals(fromArray.toValue(), toArray.toValue())) {
    throw new StateError(
        "not equals: ${fromArray.toValue()} ${toArray.toValue()}");
  }
}

void performanceTest() {
  test([10, 10, 10, 10], tm.NDDataType.float32, tm.NDDataType.float32VBlocked,
      10000);

  test([10, 10, 10, 10], tm.NDDataType.float32, tm.NDDataType.float32VBlocked,
      100000);
}

void allPerformanceTest() {}

void test(
    List<int> shape, tm.NDDataType fromType, tm.NDDataType toType, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var array =
      new tm.NDArray.generate(shape, (index) => index + 1, dataType: fromType);

  for (var i = 0; i < steps; i++) {
    array.cast(toType);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}
