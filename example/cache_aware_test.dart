// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:typed_data";

import "package:tensor_math/tensor_math.dart";
import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  int iteration = 1000000;
  int shape1 = 1000;
  int shape2 = 10;
  int stride1 = shape2;
  int stride2 = 1;
  var length = shape1 * shape2;

  var list = new Float32List(length);
  var value = 0.0;
  for (var x = 0; x < length; x++) {
    list[x] = value;
  }

  testFloat32ListRead2(iteration, list, shape1, shape2, stride1, stride2);
  testFloat32ListRead(iteration, list, shape1, shape2, stride1, stride2);
}

void testFloat32ListRead(int iteration, Float32List data, int shape1,
    int shape2, int stride1, int stride2) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var p = 0; p < iteration; p++) {
    var dataIndex = 0;
    for (var i = 0; i < shape1; ++i) {
      for (var j = 0; j < shape2; ++j) {
        data[dataIndex];

        dataIndex += stride2;
      }
      dataIndex += stride1 - stride2 * shape2;
    }
  }

  watcher.stop();

  print("Elapsed: ${watcher.elapsedMilliseconds} ms");
}

void testFloat32ListRead2(int iteration, Float32List data, int shape1,
    int shape2, int stride1, int stride2) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var p = 0; p < iteration; p++) {
    for (var i = 0; i < shape1; ++i) {
      for (var j = 0; j < shape2; ++j) {
        data[i * stride1 + j * stride2];
      }
    }
  }

  watcher.stop();

  print("Elapsed: ${watcher.elapsedMilliseconds} ms");
}

void testListRead(
    List<int> data, int shape1, int shape2, int stride1, int stride2) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  var dataIndex = 0;

  for (var i = 0; i < shape1; ++i) {
    for (var j = 0; j < shape2; ++j) {
      data[dataIndex];

      dataIndex += stride2;
    }
    dataIndex += stride1 - stride2 * shape2;
  }

  watcher.stop();

  print("Elapsed: ${watcher.elapsedMilliseconds} ms");
}
