// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";
import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  //testArrayCreation(10000000, createTestNDArray([10, 10]).toValue()); // 8584
  //testArrayCreation(100, createTestNDArray([100, 100, 100, 3]).toValue()); // 7387
  //testArrayCreation(1, createTestNDArray([100, 1000, 1000, 3]).toValue()); // 11804

  testArrayCreation(1, createTestNDArray([100, 100, 100, 100]).toValue());
}

void testArrayCreation(int iteration, value) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var i = 0; i < iteration; i++) {
    new NDArray(value).neg();
  }

  watcher.stop();

  var throughput = iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}
