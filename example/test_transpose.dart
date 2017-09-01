import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest([2, 2, 8, 8], [1, 0, 2, 3]);
  functionalTest([10, 10, 10, 10], [1, 0, 2, 3]);
  functionalTest([10, 10, 10, 10], [1, 0, 3, 2]);

  //performanceTest();
}

void functionalTest(List<int> shape, List<int> permutationAxis) {
  var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32)
      .transpose(permutationAxis: permutationAxis)
      .toValue();

  print(expectedValue);

  var value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32Blocked)
      .transpose(permutationAxis: permutationAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(expectedValue);
    print(value);
    throw new StateError("not equals");
  }
}

void performanceTest() {
  test([10, 10, 10, 10], [0, 1, 3, 2], tm.NDDataType.float32, 10000000);

  test([10, 10, 10, 10], [0, 1, 3, 2], tm.NDDataType.float32Blocked, 100000);
}

void test(List<int> shape, List<int> permutationAxis, tm.NDDataType dataType,
    int steps) {
  var watch = new Stopwatch();
  watch.start();

  var array =
      new tm.NDArray.generate(shape, (index) => index + 1, dataType: dataType);

  for (var i = 0; i < steps; i++) {
    array.transpose(permutationAxis: permutationAxis);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}
