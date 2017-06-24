import "package:tensor_math_simd/tensor_math.dart" as tm;

import "package:tensor_math_simd/src/nd_array_blocked_impl.dart";

void main() {
  functionalTest();
}

void functionalTest() {
  NDArrayBlockedImpl array = new tm.NDArray.generate(
      [6, 6], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked);

  print(array.logData());

  print((array.cast(tm.NDDataType.float32).transpose(
              permutationAxis: [1, 0]).cast(tm.NDDataType.float32HBlocked)
          as NDArrayBlockedImpl)
      .logData());

  print(array.transpose(permutationAxis: [1, 0]).toValue());
}

void performanceTest() {
  test([4, 4], [1, 0], 10000000);
}

void allPerformanceTest() {
  test([10, 10, 10, 100], [0, 1, 2, 3], 100000);

  test([10, 10, 10, 100], [1, 0, 2, 3], 100000);

  test([10, 10, 10, 100], [0, 1, 3, 2], 100);

  test([10, 10, 10, 100], null, 100);
}

void test(List<int> shape, List<int> permutationAxis, int steps) {
  var watch = new Stopwatch();
  watch.start();

  var array = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked);

  for (var i = 0; i < steps; i++) {
    array.transpose(permutationAxis: permutationAxis);
  }

  print(
      "Elapsed in ${watch.elapsedMilliseconds} ms with a throughput ${1000 * steps / watch.elapsedMilliseconds} 1/s");
}
