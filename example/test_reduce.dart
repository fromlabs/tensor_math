import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  print(new tm.NDArray.generate([10], (index) => index + 1,
          dataType: tm.NDDataType.float32Blocked)
      .reduceSum(reductionAxis: null));

  print(new tm.NDArray.generate([10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked)
      .reduceSum(reductionAxis: []));

  print(new tm.NDArray.generate([], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked)
      .reduceSum(reductionAxis: null));

  print(new tm.NDArray.generate([], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked)
      .reduceSum(reductionAxis: []));

  var array = new tm.NDArray([
    [true, false],
    [false, false],
    [true, false]
  ], dataType: tm.NDDataType.booleanBlocked);

  print(array.not());
  print(array.reduceAny(reductionAxis: [0]));
  print(array.reduceAny(reductionAxis: [1]));
  print(array.reduceAny(reductionAxis: [0, 1]));

  array = new tm.NDArray([
    [true, false],
    [false, false],
    [true, false]
  ]);

  print(array.not());
  print(array.reduceAny(reductionAxis: [0]));
  print(array.reduceAny(reductionAxis: [1]));
  print(array.reduceAny(reductionAxis: [0, 1]));

  //functionalTest([12, 128, 200], [0]);
  //functionalTest([12, 128, 200], [1]);
  //functionalTest([12, 128, 200], [2]);
  //functionalTest([128, 200], [0]);
  //functionalTest([128, 200], [1]);

  return;

  functionalTest([5, 1], [1]);

  functionalTest([2, 8, 8], [2]);
  functionalTest([3, 4, 10, 10], [0, 1]);
  functionalTest([3, 4, 10, 10], [2]);
  functionalTest([3, 4, 10, 10], [3]);
  functionalTest([3, 4, 10, 10], [2, 3]);
  functionalTest([3, 4, 10, 10], [0, 1, 2, 3]);

  return;

  functionalTest([3, 4, 10, 10], [0, 1]);

  //functionalTest([10, 10, 10, 10], [3]);
  //functionalTest([10, 10, 10, 10], [2, 3]);

  // performanceTest();
}

void performanceTest() {
  List<int> shape = [10, 10, 10, 10];
  List<int> reductionAxis = [2, 3];

  test1(shape, reductionAxis, 100000);

  test2(shape, reductionAxis, 100000);
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
      dataType: tm.NDDataType.float32Blocked);

  for (var i = 0; i < steps; i++) {
    array1.reduceSum(reductionAxis: reductionAxis);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}

void functionalTest(List<int> shape, List<int> reductionAxis) {
  print("shape: $shape");
  print("reductionAxis: $reductionAxis");

  var array = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32);

  // print(array.toValue());

  var expectedValue = array.reduceSum(reductionAxis: reductionAxis).toValue();

  // print(expectedValue);

  var value;

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32Blocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }
}

void functionalTest2(List<int> shape, List<int> reductionAxis) {
  var array = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(array.toValue());

  var expectedValue = array.reduceMean(reductionAxis: reductionAxis).toValue();

  print(expectedValue);

  var value;

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32Blocked)
      .reduceMean(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }
}

void functionalTest3(List<int> shape, List<int> reductionAxis) {
  var array = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(array.toValue());

  var expectedValue = array.reduceMax(reductionAxis: reductionAxis).toValue();

  print(expectedValue);

  var value;

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32Blocked)
      .reduceMax(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }
}
