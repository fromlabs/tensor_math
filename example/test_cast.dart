import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

import "package:tensor_math/src/nd_array_blocked_impl.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.int32Blocked, tm.NDDataType.float32Blocked);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.float32Blocked, tm.NDDataType.int32Blocked);

  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.float32, tm.NDDataType.float32Blocked);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.float32, tm.NDDataType.int32Blocked);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.int32, tm.NDDataType.float32Blocked);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.int32, tm.NDDataType.int32Blocked);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.uint32, tm.NDDataType.float32Blocked);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.uint32, tm.NDDataType.int32Blocked);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.float32Blocked, tm.NDDataType.float32);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.int32Blocked, tm.NDDataType.float32);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.float32Blocked, tm.NDDataType.int32);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.int32Blocked, tm.NDDataType.int32);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.float32Blocked, tm.NDDataType.uint32);
  functionalTest(
      [2, 3, 10, 14], tm.NDDataType.int32Blocked, tm.NDDataType.uint32);

  // performanceTest();
}

void functionalTest(List<int> shape, tm.NDDataType sourceDataType,
    tm.NDDataType targetDataType) {
  tm.NDArray fromArray = sourceDataType.isBoolean
      ? new tm.NDArray.generate(shape, (index) => (index + 1) & 1 != 0,
          dataType: sourceDataType)
      : new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: sourceDataType);

  tm.NDArray toArray = fromArray.cast(targetDataType);

  tm.NDArray fromArray2 = toArray.cast(sourceDataType);

  if (!iterableEquality.equals(fromArray.toValue(), fromArray2.toValue())) {
    print("Expected: ${fromArray.toValue()}");
    print("Result: ${fromArray2.toValue()}");
    throw new StateError(
        "not equals: ${fromArray.toValue()} ${toArray.toValue()}");
  }
}

void performanceTest() {
  test([10, 10, 10, 10], tm.NDDataType.float32, tm.NDDataType.float32Blocked,
      10000);

  test([10, 10, 10, 10], tm.NDDataType.float32, tm.NDDataType.float32Blocked,
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
