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
  var length = shape1 * (shape2 / 4).ceilToDouble().toInt();

  var list = new Float32x4List(length);
  var value = new Float32x4.zero();
  for (var x = 0; x < length; x++) {
    list[x] = value;
  }

  testFloat32x4ListRead(iteration, list, shape1, shape2, stride1, stride2);
}

void testFloat32x4ListRead(int iteration, Float32x4List data, int shape1,
    int shape2, int stride1, int stride2) {
  print("Start...");

  var newStride1 =
      ((stride1 / 4).ceilToDouble().toInt() * 4).ceilToDouble().toInt();

  var watcher = new Stopwatch()..start();

  for (var p = 0; p < iteration; p++) {
    for (var i = 0; i < shape1; i++) {
      for (var j = 0; j < shape2; j++) {
        var index1 = i * newStride1;
        var index2 = j * stride2;

        var dataIndex = (index1 + index2) ~/ 4;

        var valueX4 = data[dataIndex];
        switch ((index1 + index2) % 4) {
          case 0:
            valueX4.x;
            break;
          case 1:
            valueX4.y;
            break;
          case 2:
            valueX4.z;
            break;
          case 3:
            valueX4.w;
            break;
        }
      }
    }
  }

  watcher.stop();

  print("Elapsed: ${watcher.elapsedMilliseconds} ms");
}

void testFloat32x4ListRead2(int iteration, Float32x4List data, int shape1,
    int shape2, int stride1, int stride2) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var p = 0; p < iteration; p++) {
    for (var i = 0; i < shape1; i++) {
      for (var j = 0; j < shape2; j += 4) {
        var index1 = i * stride1;
        var index2 = j * stride2;

        var dataX2 = data[index1 + index2];
        dataX2.x;
        dataX2.y;
        dataX2.z;
        dataX2.w;
      }
    }
  }

  watcher.stop();

  print("Elapsed: ${watcher.elapsedMilliseconds} ms");
}
