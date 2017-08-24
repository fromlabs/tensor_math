// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'package:test/test.dart';

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_array_base.dart" show createTestNDArray;

void main() {
  group('Array tests', () {
    setUp(() {});

    test('Array toValue tests', () {
      expect(new NDArray(1).toValue(), equals(1));

      expect(new NDArray([1, 2]).toValue(), equals([1, 2]));

      expect(
          new NDArray([
            [1, 2],
            [3, 4]
          ]).toValue(),
          equals([
            [1, 2],
            [3, 4]
          ]));
    });

    test('Array neg tests', () {
      expect(new NDArray(1).neg().toValue(), equals(new NDArray(-1).toValue()));

      expect(new NDArray([1, 2]).neg().toValue(),
          equals(new NDArray([-1, -2]).toValue()));

      expect(
          new NDArray([
            [1, 2],
            [3, 4]
          ]).neg().toValue(),
          equals(new NDArray([
            [-1, -2],
            [-3, -4]
          ]).toValue()));

      expect(new NDArray(1).transpose().neg().toValue(),
          equals(new NDArray(-1).transpose().toValue()));

      expect(new NDArray([1, 2]).transpose().neg().toValue(),
          equals(new NDArray([-1, -2]).transpose().toValue()));

      expect(
          new NDArray([
            [1, 2],
            [3, 4]
          ]).transpose().neg().toValue(),
          equals(new NDArray([
            [-1, -2],
            [-3, -4]
          ]).transpose().toValue()));
    });

    test('Array add tests', () {
      expect(new NDArray(1).add(new NDArray(1)).toValue(),
          equals(new NDArray(2).toValue()));

      expect(new NDArray([1, 2]).add(new NDArray([1, 2])).toValue(),
          equals(new NDArray([2, 4]).toValue()));

      expect(
          new NDArray([
            [1, 2],
            [3, 4]
          ])
              .add(new NDArray([
                [1, 2],
                [3, 4]
              ]))
              .toValue(),
          equals(new NDArray([
            [2, 4],
            [6, 8]
          ]).toValue()));

      expect(
          new NDArray(1).transpose().add(new NDArray(1).transpose()).toValue(),
          equals(new NDArray(2).transpose().toValue()));

      expect(
          new NDArray([1, 2])
              .transpose()
              .add(new NDArray([1, 2]).transpose())
              .toValue(),
          equals(new NDArray([2, 4]).transpose().toValue()));

      expect(
          new NDArray([
            [1, 2],
            [3, 4]
          ])
              .transpose()
              .add(new NDArray([
                [1, 2],
                [3, 4]
              ]).transpose())
              .toValue(),
          equals(new NDArray([
            [2, 4],
            [6, 8]
          ]).transpose().toValue()));

      expect(new NDArray([1, 2]).add(new NDArray(2)).toValue(),
          equals(new NDArray([3, 4]).toValue()));

      expect(new NDArray([1, 2]).add(new NDArray([2])).toValue(),
          equals(new NDArray([3, 4]).toValue()));

      expect(new NDArray([2]).add(new NDArray([1, 2])).toValue(),
          equals(new NDArray([3, 4]).toValue()));
    });

    test('Array reduce tests', () {
      expect(createTestNDArray([2, 2, 2]).reduceSum().toValue(), equals(28));
      expect(
          createTestNDArray([2, 2, 2]).reduceSum(reductionAxis: [0]).toValue(),
          equals([
            [4, 6],
            [8, 10]
          ]));
      expect(
          createTestNDArray([2, 2, 2])
              .reduceSum(reductionAxis: [0, 1]).toValue(),
          equals([12, 16]));
      expect(
          createTestNDArray([2, 2, 2])
              .reduceSum(reductionAxis: [2, 1]).toValue(),
          equals([6, 22]));

      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32)
              .reduceMean()
              .toValue(),
          equals(3.5));
      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32)
              .reduceMean(reductionAxis: [0]).toValue(),
          equals([
            [2.0, 3.0],
            [4.0, 5.0]
          ]));
      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32)
              .reduceMean(reductionAxis: [0, 1]).toValue(),
          equals([3.0, 4.0]));
      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32)
              .reduceMean(reductionAxis: [2, 1]).toValue(),
          equals([1.5, 5.5]));
    });

    test('Array reduce keep dimensions tests', () {
      expect(
          createTestNDArray([2, 2, 2])
              .reduceSum(keepDimensions: true)
              .toValue(),
          equals([
            [
              [28]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2])
              .reduceSum(reductionAxis: [0], keepDimensions: true).toValue(),
          equals([
            [
              [4, 6],
              [8, 10]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2])
              .reduceSum(reductionAxis: [0, 1], keepDimensions: true).toValue(),
          equals([
            [
              [12, 16]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2])
              .reduceSum(reductionAxis: [2, 1], keepDimensions: true).toValue(),
          equals([
            [
              [6]
            ],
            [
              [22]
            ]
          ]));

      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32)
              .reduceMean(keepDimensions: true)
              .toValue(),
          equals([
            [
              [3.5]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32)
              .reduceMean(reductionAxis: [0], keepDimensions: true).toValue(),
          equals([
            [
              [2.0, 3.0],
              [4.0, 5.0]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32).reduceMean(
              reductionAxis: [0, 1], keepDimensions: true).toValue(),
          equals([
            [
              [3.0, 4.0]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2], dataType: NDDataType.float32).reduceMean(
              reductionAxis: [2, 1], keepDimensions: true).toValue(),
          equals([
            [
              [1.5]
            ],
            [
              [5.5]
            ]
          ]));
    });

    test('Array reshape tests', () {
      expect(
          new NDArray([1, 2, 3, 4, 5, 6, 7, 8, 9])
              .reshape(newDimensions: [3, 3]).toValue(),
          equals([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]));

      expect(
          new NDArray([
            [
              [1, 1],
              [2, 2]
            ],
            [
              [3, 3],
              [4, 4]
            ]
          ]).reshape(newDimensions: [2, 4]).toValue(),
          equals([
            [1, 1, 2, 2],
            [3, 3, 4, 4]
          ]));

      expect(
          new NDArray([
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
          ]).reshape(newDimensions: [-1]).toValue(),
          equals([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]));

      expect(
          new NDArray([
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
          ]).reshape(newDimensions: [2, -1]).toValue(),
          equals([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5, 6, 6, 6]
          ]));

      expect(
          new NDArray([
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
          ]).reshape(newDimensions: [-1, 9]).toValue(),
          equals([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5, 6, 6, 6]
          ]));

      expect(
          new NDArray([
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
          ]).reshape(newDimensions: [2, -1, 3]).toValue(),
          equals([
            [
              [1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]
            ],
            [
              [4, 4, 4],
              [5, 5, 5],
              [6, 6, 6]
            ]
          ]));
    });

    test('Array select tests', () {});
  });
}
