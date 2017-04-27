// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:collection";

import "package:tensor_math/tensor_math.dart" as math;

void main() {
  test3();
}

void testTemplate(int iteration) {
  var watcher = new Stopwatch()..start();

  // INPUT

  for (var i = 0; i < iteration; i++) {
    // OPERAZIONE
  }

  watcher.stop();

  var throughput = iteration.toDouble() / watcher.elapsedMilliseconds;

  print(
      "testTemplate - Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}

void test1() {
  var iteration = 1000;
  var length = 1000000;
  testArrayFixed(iteration, length);
  testArrayVariable(iteration, length);

  iteration = 1000000;
  length = 100;
  testArrayFixed(iteration, length);
  testArrayVariable(iteration, length);
}

void test2() {
  var iteration = 10000000;
  var length = 10;
  testArrayDirectAccess(iteration, length);
  testArrayDelegatedAccess(iteration, length);
}

void test3() {
  var iteration = 1000;
  var length = 1000000;
  testArrayForIn(iteration, length);
  testArrayFor(iteration, length);
}

void testArrayForIn(int iteration, int length) {
  var watcher = new Stopwatch()..start();

  var array = new List<int>.filled(length, 0, growable: false);

  for (var i = 0; i < iteration; i++) {
    runTestArrayForIn(length, array);
  }

  watcher.stop();

  var throughput = iteration / watcher.elapsedMilliseconds;

  print(
      "testArrayForIn - Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}

void testArrayFor(int iteration, int length) {
  var watcher = new Stopwatch()..start();

  var array = new List<int>.filled(length, 0, growable: false);

  for (var i = 0; i < iteration; i++) {
    runTestArrayFor(length, array);
  }

  watcher.stop();

  var throughput = iteration / watcher.elapsedMilliseconds;

  print(
      "testArrayFor - Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}

void testArrayDirectAccess(int iteration, int length) {
  var watcher = new Stopwatch()..start();

  var array = new List<int>.filled(length, 0, growable: false);

  for (var i = 0; i < iteration; i++) {
    runTestArrayDirectAccess(length, array);
  }

  watcher.stop();

  var throughput = iteration / watcher.elapsedMilliseconds;

  print(
      "testArrayDirectAccess - Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}

void testArrayDelegatedAccess(int iteration, int length) {
  var watcher = new Stopwatch()..start();

  var array = new List<int>.filled(length, 0, growable: false);

  var getArray = () => array;

  for (var i = 0; i < iteration; i++) {
    runTestArrayDelegatedAccess(length, getArray);
  }

  watcher.stop();

  var throughput = iteration / watcher.elapsedMilliseconds;

  print(
      "testArrayDelegatedAccess - Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}

void testArrayFixed(int iteration, int length) {
  var watcher = new Stopwatch()..start();

  for (var i = 0; i < iteration; i++) {
    runTestArrayFixed(length);
  }

  watcher.stop();

  var throughput = iteration / watcher.elapsedMilliseconds;

  print(
      "testArrayFixed - Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}

void testArrayVariable(int iteration, int length) {
  var watcher = new Stopwatch()..start();

  for (var i = 0; i < iteration; i++) {
    runTestArrayVariable(length);
  }

  watcher.stop();

  var throughput = iteration / watcher.elapsedMilliseconds;

  print(
      "testArrayVariable - Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/ms]");
}

void runTestArrayForIn(int length, List<int> array) {
  for (var value in array) {}
}

void runTestArrayFor(int length, List<int> array) {
  for (var i = 0; i < length; i++) {
    var value = array[i];
  }
}

void runTestArrayDirectAccess(int length, List<int> array) {
  for (var i = 0; i < length; i++) {
    array[i];
  }
}

void runTestArrayDelegatedAccess(int length, List<int> getArray()) {
  for (var i = 0; i < length; i++) {
    getArray()[i];
  }
}

void runTestArrayFixed(int length) {
  var array = new List<int>(length);

  for (var i = 0; i < length; i++) {
    array[0] = 0;
  }
}

void runTestArrayVariable(int length) {
  var array = new List<int>();

  for (var i = 0; i < length; i++) {
    array.add(0);
  }
}

int _calculateLength(List<int> shape) {
  switch (shape.length) {
    case 0:
      return 1;
    case 1:
      return shape[0];
    case 2:
      return shape[0] * shape[1];
    case 3:
      return shape[0] * shape[1] * shape[2];
    default:
      var total = shape[0] * shape[1] * shape[2];
      for (var i = 3; i < shape.length; i++) {
        total *= shape[i];
      }
      return total;
  }
}
