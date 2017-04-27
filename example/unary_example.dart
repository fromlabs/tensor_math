// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

void main() {
  var array = new NDArray([
    [0, 1, 2],
    [3, 4, 5]
  ]).transpose();

  print(array);

  array.neg();
}
