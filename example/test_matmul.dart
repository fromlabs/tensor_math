import 'package:collection/collection.dart';

import "package:tensor_math_simd/tensor_math.dart" as tm;

import "package:tensor_math_simd/src/nd_array_blocked_impl.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  //functionalTest();

  performanceTest();
}

void functionalTest() {
  var shape1 = [2, 2, 11, 13];
  var shape2 = [2, 2, 13, 15];

  var shapeLength2 = shape2.reduce((v1, v2) => v1 * v2);
  var expectedValue = new tm.NDArray.generate(shape1, (index) => index + 1,
          dataType: tm.NDDataType.float32)
      .matMul(new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
          dataType: tm.NDDataType.float32))
      .toValue();

  var value = new tm.NDArray.generate(shape1, (index) => index + 1,
          dataType: tm.NDDataType.float32HBlocked)
      .matMul(new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
          dataType: tm.NDDataType.float32VBlocked))
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    throw new StateError("not equals: $value $expectedValue");
  }
}

void performanceTest() {
  test1([10, 10, 5, 13], [10, 10, 13, 7], 10000);

  test2([10, 10, 5, 13], [10, 10, 13, 7], 100000);
}

void test1(List<int> shape1, List<int> shape2, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var shapeLength2 = shape2.reduce((v1, v2) => v1 * v2);

  var array1 = new tm.NDArray.generate(shape1, (index) => index + 1,
      dataType: tm.NDDataType.float32);
  var array2 = new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
      dataType: tm.NDDataType.float32);

  for (var i = 0; i < steps; i++) {
    array1.matMul(array2);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}

void test2(List<int> shape1, List<int> shape2, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var shapeLength2 = shape2.reduce((v1, v2) => v1 * v2);

  var array1 = new tm.NDArray.generate(shape1, (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked);
  var array2 = new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
      dataType: tm.NDDataType.float32VBlocked);

  for (var i = 0; i < steps; i++) {
    array1.matMul(array2);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}
