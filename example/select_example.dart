// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  print(new NDArray([
    [1, 2],
    [3, 4]
  ]).toValue());
}