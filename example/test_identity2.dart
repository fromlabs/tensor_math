import 'package:collection/collection.dart';

import "package:tensor_math_simd/tensor_math.dart" as tm;

import "package:tensor_math_simd/src/nd_array_blocked_impl2.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  performanceTest();
}

void performanceTest() {
  test([10, 10, 5, 13], 1000000);
}

void test(List<int> shape, int steps) {
  var watch = new Stopwatch();
  watch.start();

  NDArrayBlockedImpl array1 = new tm.NDArray.generate(
      shape, (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked);

  for (var i = 0; i < steps; i++) {
    array1.identity();
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}
