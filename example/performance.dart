// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:typed_data";

import "package:tensor_math/tensor_math.dart";
import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  var length = 100 * 100 * 100;
  var iteration = 1000;

  testListCreation(iteration, length);
  testFloat32ListCreation(iteration, length);
  testFloat64ListCreation(iteration, length);

  testListWrite(iteration, length);
  testFloat32ListWrite(iteration, length);
  testFloat64ListWrite(iteration, length);

  testListRead(iteration, length);
  testFloat32ListRead(iteration, length);
  testFloat64ListRead(iteration, length);

/*

Start...
Elapsed: 1817 ms [1000 X 550.357732526142 1/s]
Start...
Elapsed: 687 ms [1000 X 1455.604075691412 1/s]
Start...
Elapsed: 1672 ms [1000 X 598.0861244019138 1/s]

Start...
Elapsed: 1129 ms [1000 X 885.7395925597874 1/s]
Start...
Elapsed: 706 ms [1000 X 1416.4305949008499 1/s]
Start...
Elapsed: 661 ms [1000 X 1512.8593040847202 1/s]

Start...
Elapsed: 4952 ms [1000 X 201.93861066235866 1/s]
Start...
Elapsed: 7445 ms [1000 X 134.31833445265278 1/s]
Start...
Elapsed: 7742 ms [1000 X 129.1655902867476 1/s]

*/
}

void testListRead(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  var list = new List(length);
  var value = 0.0;
  for (var x = 0; x < length; x++) {
    list[x] = value;
  }

  for (var i = 0; i < iteration; i++) {
    for (var x = 0; x < length; x++) {
      list[x];
    }
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testFloat32ListRead(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  var list = new Float32List(length);
  var value = 0.0;
  for (var x = 0; x < length; x++) {
    list[x] = value;
  }

  for (var i = 0; i < iteration; i++) {
    for (var x = 0; x < length; x++) {
      list[x];
    }
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testFloat64ListRead(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  var list = new Float64List(length);
  var value = 0.0;
  for (var x = 0; x < length; x++) {
    list[x] = value;
  }

  for (var i = 0; i < iteration; i++) {
    for (var x = 0; x < length; x++) {
      list[x];
    }
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testListWrite(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  var list = new List(length);
  var value = 0.0;

  for (var i = 0; i < iteration; i++) {
    for (var x = 0; x < length; x++) {
      list[x] = value;
    }
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testFloat32ListWrite(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  var list = new Float64List(length);
  var value = 0.0;

  for (var i = 0; i < iteration; i++) {
    for (var x = 0; x < length; x++) {
      list[x] = value;
    }
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testFloat64ListWrite(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  var list = new Float64List(length);
  var value = 0.0;

  for (var i = 0; i < iteration; i++) {
    for (var x = 0; x < length; x++) {
      list[x] = value;
    }
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testListCreation(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var i = 0; i < iteration; i++) {
    new List(length);
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testFloat32ListCreation(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var i = 0; i < iteration; i++) {
    new Float32List(length);
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}

void testFloat64ListCreation(int iteration, int length) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var i = 0; i < iteration; i++) {
    new Float64List(length);
  }

  watcher.stop();

  var throughput = 1000 * iteration / watcher.elapsedMilliseconds;

  print(
      "Elapsed: ${watcher.elapsedMilliseconds} ms [$iteration X $throughput 1/s]");
}
