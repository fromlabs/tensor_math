import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest([4, 8], [8, 8]);
  functionalTest([10, 10, 4, 8], [10, 10, 8, 8]);
  functionalTest([8, 8], [8, 8]);
  functionalTest([5, 13], [13, 7]);
  functionalTest([2, 5, 13], [2, 13, 7]);
  functionalTest([5, 7, 5, 13], [5, 7, 13, 7]);
  //functionalTest([10, 10, 5, 13], [10, 10, 13, 7]);

  //functionalTest([1, 100, 100], [1, 100, 10]);

  functionalTest([2, 2, 11, 13], [2, 2, 13, 15]);

  performanceTest();
}

void functionalTest(List<int> shape1, List<int> shape2) {
  var shapeLength2 = shape2.reduce((v1, v2) => v1 * v2);
  var expectedValue = new tm.NDArray.generate(shape1, (index) => index + 1,
          dataType: tm.NDDataType.float32)
      .matMul(new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
          dataType: tm.NDDataType.float32))
      .toValue();

  var value = new tm.NDArray.generate(shape1, (index) => index + 1,
          dataType: tm.NDDataType.float32VBlocked)
      .matMul(new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
          dataType: tm.NDDataType.float32VBlocked))
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);
    throw new StateError("not equals");
  }
}

void performanceTest() {
  test1([10, 10, 5, 13], [10, 10, 13, 7], 10000);
  test2([10, 10, 5, 13], [10, 10, 13, 7], 100000);

  test1([64, 100, 100], [64, 100, 10], 100);
  test2([64, 100, 100], [64, 100, 10], 1000);
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
      dataType: tm.NDDataType.float32VBlocked);
  var array2 = new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
      dataType: tm.NDDataType.float32VBlocked);

  for (var i = 0; i < steps; i++) {
    array1.matMul(array2);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}
