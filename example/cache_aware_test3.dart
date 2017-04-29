// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:typed_data";

import "package:tensor_math/tensor_math.dart";
import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  var length = 4 * 1000 * 1000 * 1000;

  var list = new Float32List(length);
  var value = 0.0;
  for (var x = 0; x < length; x++) {
    list[x] = value;
  }

  testFloat32ListRead(list);
}

void testFloat32ListRead(Float32List data) {
  print("Start...");

  var watcher = new Stopwatch()..start();

  for (var i = 0; i < data.length; i++) {
    data[i];
  }

  watcher.stop();

  print("Elapsed: ${watcher.elapsedMilliseconds} ms");
}
