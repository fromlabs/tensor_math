// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  var array = createTestNDArray([2, 2, 2]);
  print(array);

  print(array.shape.reduce());
  print(array.shape.reduce(reductionAxis: [0]));
  print(array.shape.reduce(reductionAxis: [0, 1]));
  print(array.shape.reduce(reductionAxis: [2, 1]));

  print(array.reduceSum());
  print(array.reduceSum(reductionAxis: [0]));
  print(array.reduceSum(reductionAxis: [0, 1]));
  print(array.reduceSum(reductionAxis: [2, 1]));

  print(array.reduceMean());
  print(array.reduceMean(reductionAxis: [0]));
  print(array.reduceMean(reductionAxis: [0, 1]));
  print(array.reduceMean(reductionAxis: [2, 1]));
}
