// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

void main() {
  print(
      new NDArray([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(newDimensions: [3, 3]));

/*
[[1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]]
*/

  print(new NDArray([
    [
      [1, 1],
      [2, 2]
    ],
    [
      [3, 3],
      [4, 4]
    ]
  ]).reshape(newDimensions: [2, 4]));

/*
[[1, 1, 2, 2],
  [3, 3, 4, 4]]
*/

  print(new NDArray([
    [
      [1, 1, 1],
      [2, 2, 2]
    ],
    [
      [3, 3, 3],
      [4, 4, 4]
    ],
    [
      [5, 5, 5],
      [6, 6, 6]
    ]
  ]).reshape(newDimensions: [-1]));

// [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

  print(new NDArray([
    [
      [1, 1, 1],
      [2, 2, 2]
    ],
    [
      [3, 3, 3],
      [4, 4, 4]
    ],
    [
      [5, 5, 5],
      [6, 6, 6]
    ]
  ]).reshape(newDimensions: [2, -1]));

/*
  [[1, 1, 1, 2, 2, 2, 3, 3, 3],
  [4, 4, 4, 5, 5, 5, 6, 6, 6]]
*/

  print(new NDArray([
    [
      [1, 1, 1],
      [2, 2, 2]
    ],
    [
      [3, 3, 3],
      [4, 4, 4]
    ],
    [
      [5, 5, 5],
      [6, 6, 6]
    ]
  ]).reshape(newDimensions: [-1, 9]));

/*
  [[1, 1, 1, 2, 2, 2, 3, 3, 3],
  [4, 4, 4, 5, 5, 5, 6, 6, 6]]
*/

  print(new NDArray([
    [
      [1, 1, 1],
      [2, 2, 2]
    ],
    [
      [3, 3, 3],
      [4, 4, 4]
    ],
    [
      [5, 5, 5],
      [6, 6, 6]
    ]
  ]).reshape(newDimensions: [2, -1, 3]));

/*
  [[[1, 1, 1],
  [2, 2, 2],
  [3, 3, 3]],
  [[4, 4, 4],
  [5, 5, 5],
  [6, 6, 6]]]
*/

  print(new NDArray([7]).reshape(newDimensions: []));

  // 7
}
