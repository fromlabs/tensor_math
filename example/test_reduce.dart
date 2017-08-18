import 'package:collection/collection.dart';

import "package:tensor_math_simd/tensor_math.dart" as tm;

import "package:tensor_math_simd/src/nd_array_blocked_impl2.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest([1, 1], [0, 1]);
  functionalTest([1], [0]);
  functionalTest([1, 5, 1, 1], [2, 3]);
  functionalTest([1, 1, 1], [1, 2]);
  functionalTest([5, 5], [1]);
  functionalTest([5, 5, 5], [2]);
  functionalTest([5, 5, 5], [0, 2]);


  return;

  functionalTest([5, 5], [1]);

  functionalTest([1, 5, 5], [0, 2]);

  functionalTest([1, 5, 5], [2]);

  functionalTest([1, 1, 5], [2]);

  functionalTest([1, 1, 5], [1]);

  functionalTest([1, 1, 5], [0]);

  functionalTest([1, 5], [0]);

  functionalTest([5, 1], [1]);

  // functionalTest([10, 10, 10, 10], [2, 3]);

  // performanceTest();
}

void performanceTest() {
  List<int> shape = [10, 10, 10, 10];
  List<int> reductionAxis = [2, 3];

  // test1(shape, reductionAxis, 100000);

  test2(shape, reductionAxis, 100000);

  // test3(shape, reductionAxis, 100000);
}

void test1(List<int> shape, List<int> reductionAxis, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var array1 = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32);

  for (var i = 0; i < steps; i++) {
    array1.reduceSum(reductionAxis: reductionAxis);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}

void test2(List<int> shape, List<int> reductionAxis, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var array1 = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked);

  for (var i = 0; i < steps; i++) {
    array1.reduceSum(reductionAxis: reductionAxis);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}

void test3(List<int> shape, List<int> reductionAxis, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var array1 = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked);

  for (var i = 0; i < steps; i++) {
    array1.reduceSum(reductionAxis: reductionAxis);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}

void functionalTest(List<int> shape, List<int> reductionAxis) {
  var array = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(array.toValue());

  var expectedValue = array.reduceSum(reductionAxis: reductionAxis).toValue();

  print(expectedValue);

  var value;

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32HBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32VBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }
}
