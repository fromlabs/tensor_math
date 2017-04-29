// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:typed_data";

import "package:tensor_math/tensor_math.dart";
import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  var length = 4 * 1000 * 1000 * 250;

  var list = new Float32x4List(length);
  var value = new Float32x4.zero();
  for (var x = 0; x < length; x++) {
    list[x] = value;
  }

  testFloat32x4ListRead(list);
}

void testFloat32x4ListRead(Float32x4List data) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var i = 0; i < data.length; i++) {
    var valuex4 = data[i];
    valuex4.x;
    valuex4.y;
    valuex4.z;
    valuex4.w;
  }

  watcher.stop();

  print("Elapsed: ${watcher.elapsedMilliseconds} ms");
}
