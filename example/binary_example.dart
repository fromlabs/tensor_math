// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  print(new NDArray([1, 2]).add(new NDArray(2)));

  print(new NDArray(2).add(new NDArray([1, 2])));

  print(new NDArray([1, 2]).add(new NDArray([2])));

  print(new NDArray([
    [1, 2],
    [3, 4]
  ]).add(new NDArray([2])));

  print(createTestNDArray([]));
  print(createTestNDArray([10]));
  print(createTestNDArray([10, 3]));
  print(createTestNDArray([2, 3, 4]));
}
