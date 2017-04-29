// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  var array1 = createTestNDArray([2, 2]);

  var array2 = array1.transpose();

  print(array1);
  print(array2);
  print(array2.neg());

  array1 = createTestNDArray([2, 2, 2]);
  print(array1);


}
