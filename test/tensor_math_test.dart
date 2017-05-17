// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'package:test/test.dart';

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_array_impl.dart" show createTestNDArray;

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

      expect(createTestNDArray([2, 2, 2]).reduceMean().toValue(), equals(3.5));
      expect(
          createTestNDArray([2, 2, 2]).reduceMean(reductionAxis: [0]).toValue(),
          equals([
            [2.0, 3.0],
            [4.0, 5.0]
          ]));
      expect(
          createTestNDArray([2, 2, 2])
              .reduceMean(reductionAxis: [0, 1]).toValue(),
          equals([3.0, 4.0]));
      expect(
          createTestNDArray([2, 2, 2])
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
          createTestNDArray([2, 2, 2])
              .reduceMean(keepDimensions: true)
              .toValue(),
          equals([
            [
              [3.5]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2])
              .reduceMean(reductionAxis: [0], keepDimensions: true).toValue(),
          equals([
            [
              [2.0, 3.0],
              [4.0, 5.0]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2]).reduceMean(
              reductionAxis: [0, 1], keepDimensions: true).toValue(),
          equals([
            [
              [3.0, 4.0]
            ]
          ]));
      expect(
          createTestNDArray([2, 2, 2]).reduceMean(
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

  group('Shape tests', () {
    setUp(() {});

    test('Shape dimension test', () {
      var shape = new NDShape();
      expect(shape.dimension, isNull);
      expect(shape.length, isNull);
      expect(shape.isUnknownDimension, isTrue);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([]);
      expect(shape.dimension, 0);
      expect(shape.length, 1);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isTrue);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([null]);
      expect(shape.dimension, 1);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isTrue);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([1]);
      expect(shape.dimension, 1);
      expect(shape.length, 1);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isTrue);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([10]);
      expect(shape.dimension, 1);
      expect(shape.length, 10);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isTrue);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([null, null]);
      expect(shape.dimension, 2);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = new NDShape([1, null]);
      expect(shape.dimension, 2);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = new NDShape([null, 1]);
      expect(shape.dimension, 2);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = new NDShape([1, 1]);
      expect(shape.dimension, 2);
      expect(shape.length, 1);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = new NDShape([10, 10]);
      expect(shape.dimension, 2);
      expect(shape.length, 100);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = new NDShape([null, null, null]);
      expect(shape.dimension, 3);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([10, 10, 10]);
      expect(shape.dimension, 3);
      expect(shape.length, 1000);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);
    });

    test('Shape transpose test', () {
      var shape = new NDShape(null).transpose();
      expect(shape.dimension, isNull);
      expect(shape.length, isNull);
      expect(shape.isUnknownDimension, isTrue);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([null, null, null]).transpose();
      expect(shape.dimension, 3);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = new NDShape([1, 2, 3]).transpose();
      expect(shape.dimension, 3);
      expect(shape.length, 6);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);
      expect(shape.dimensions, orderedEquals([3, 2, 1]));
      expect(shape.get(0), 3);
      expect(shape.get(1), 2);
      expect(shape.get(2), 1);
      expect(shape[0], 3);
      expect(shape[1], 2);
      expect(shape[2], 1);

      expect(() => new NDShape([1, 2, 3]).transpose(permutationAxis: []),
          throwsArgumentError);

      expect(() => new NDShape([1, 2, 3]).transpose(permutationAxis: [0]),
          throwsArgumentError);

      expect(
          () => new NDShape([1, 2, 3]).transpose(permutationAxis: [0, 1, null]),
          throwsArgumentError);

      shape = new NDShape([1, 2, 3]).transpose(permutationAxis: [0, 1, 2]);
      expect(shape.dimensions, orderedEquals([1, 2, 3]));

      expect(() => new NDShape([1, 2, 3]).transpose(permutationAxis: [1, 2, 3]),
          throwsRangeError);

      expect(() => new NDShape([1, 2, 3]).transpose(permutationAxis: [0, 1, 1]),
          throwsArgumentError);
    });

    test('Shape merge test', () {
      expect(() => new NDShape([]).merge(new NDShape([null])),
          throwsArgumentError);

      expect(() => new NDShape([null]).merge(new NDShape([])),
          throwsArgumentError);

      expect(
          () => new NDShape([1]).merge(new NDShape([2])), throwsArgumentError);

      expect(
          () => new NDShape([2]).merge(new NDShape([1])), throwsArgumentError);

      expect(() => new NDShape([1, 1]).merge(new NDShape([1, 2])),
          throwsArgumentError);

      expect(() => new NDShape([1, 2]).merge(new NDShape([1, 1])),
          throwsArgumentError);

      expect(new NDShape().merge(new NDShape()).dimensions, isNull);

      expect(
          new NDShape([]).merge(new NDShape([])).dimensions, orderedEquals([]));

      expect(
          new NDShape().merge(new NDShape([])).dimensions, orderedEquals([]));

      expect(
          new NDShape([]).merge(new NDShape()).dimensions, orderedEquals([]));

      expect(new NDShape([1]).merge(new NDShape([1])).dimensions,
          orderedEquals([1]));

      expect(new NDShape([null]).merge(new NDShape([null])).dimensions,
          orderedEquals([null]));

      expect(new NDShape([1]).merge(new NDShape([null])).dimensions,
          orderedEquals([1]));

      expect(new NDShape([null]).merge(new NDShape([1])).dimensions,
          orderedEquals([1]));

      expect(new NDShape([1, 2]).merge(new NDShape([1, null])).dimensions,
          orderedEquals([1, 2]));

      expect(new NDShape([1, null]).merge(new NDShape([1, 2])).dimensions,
          orderedEquals([1, 2]));
    });

    test('Shape broadcast test', () {
      expect(() => new NDShape([2]).add(new NDShape([3])), throwsArgumentError);

      expect(() => new NDShape([2, 2]).add(new NDShape([2, 3])),
          throwsArgumentError);

      expect(
          () => new NDShape([2, 2]).add(new NDShape([3])), throwsArgumentError);

      expect(
          () => new NDShape([3]).add(new NDShape([2, 2])), throwsArgumentError);

      expect(new NDShape().add(new NDShape()).dimensions, isNull);

      expect(
          new NDShape([]).add(new NDShape([])).dimensions, orderedEquals([]));

      expect(new NDShape([2]).add(new NDShape([2])).dimensions,
          orderedEquals([2]));

      expect(new NDShape([2, 3]).add(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape().add(new NDShape([])).dimensions, isNull);

      expect(new NDShape([]).add(new NDShape()).dimensions, isNull);

      expect(new NDShape([3, 2]).add(new NDShape()).dimensions, isNull);

      expect(new NDShape([2, 3]).add(new NDShape([null, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([null, 3]).add(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([2, 3]).add(new NDShape([1, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([1, 3]).add(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([2, 3]).add(new NDShape([])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([]).add(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([2, 3]).add(new NDShape([3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([3]).add(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));
    });

    test('Shape matrix multiplication test', () {
      expect(
          () => new NDShape([]).matMul(new NDShape([])), throwsArgumentError);

      expect(() => new NDShape([]).matMul(new NDShape()), throwsArgumentError);

      expect(() => new NDShape().matMul(new NDShape([])), throwsArgumentError);

      expect(new NDShape().matMul(new NDShape()).dimensions, isNull);

      expect(new NDShape([1, 2]).matMul(new NDShape([2, 3])).dimensions,
          orderedEquals([1, 3]));

      expect(
          new NDShape([1, 2, 3]).matMul(new NDShape([null, 3, 2])).dimensions,
          orderedEquals([1, 2, 2]));
    });

    test('Shape reduce test', () {
      expect(() => new NDShape([1, 1, null]).reduceMax(reductionAxis: [0, 0]),
          throwsArgumentError);

      expect(
          () =>
              new NDShape([1, 1, null]).reduceMax(reductionAxis: [0, 1, 2, 3]),
          throwsArgumentError);

      expect(new NDShape([]).reduceMax().dimensions, isEmpty);

      expect(new NDShape([2, 2, 2]).reduceMax().dimensions, orderedEquals([]));

      expect(new NDShape([2, 2, 2]).reduceMax(reductionAxis: [0]).dimensions,
          orderedEquals([2, 2]));

      expect(new NDShape([2, 2, 2]).reduceMax(reductionAxis: [0, 1]).dimensions,
          orderedEquals([2]));

      expect(
          new NDShape([2, 2, 2]).reduceMax(reductionAxis: [0, 1, 2]).dimensions,
          orderedEquals([]));
    });

    test('Shape reduce keep dimensions test', () {
      expect(
          () => new NDShape([1, 1, null])
              .reduceMax(reductionAxis: [0, 0], keepDimensions: true),
          throwsArgumentError);

      expect(
          () => new NDShape([1, 1, null])
              .reduceMax(reductionAxis: [0, 1, 2, 3], keepDimensions: true),
          throwsArgumentError);

      expect(
          new NDShape([]).reduceMax(keepDimensions: true).dimensions, isEmpty);

      expect(new NDShape([2, 2, 2]).reduceMax(keepDimensions: true).dimensions,
          orderedEquals([1, 1, 1]));

      expect(
          new NDShape([2, 2, 2])
              .reduceMax(reductionAxis: [0], keepDimensions: true).dimensions,
          orderedEquals([1, 2, 2]));

      expect(
          new NDShape([2, 2, 2]).reduceMax(
              reductionAxis: [0, 1], keepDimensions: true).dimensions,
          orderedEquals([1, 1, 2]));

      expect(
          new NDShape([2, 2, 2]).reduceMax(
              reductionAxis: [0, 1, 2], keepDimensions: true).dimensions,
          orderedEquals([1, 1, 1]));
    });

    test('Shape reshape test', () {
      expect(() => new NDShape([2]).reshape(newDimensions: []),
          throwsArgumentError);

      expect(new NDShape([2]).reshape(newDimensions: null).dimensions, isNull);

      expect(new NDShape().reshape(newDimensions: [2, -1]).dimensions,
          orderedEquals([2, null]));

      expect(new NDShape().reshape(newDimensions: [2, 4]).dimensions,
          orderedEquals([2, 4]));

      expect(new NDShape([9]).reshape(newDimensions: [3, 3]).dimensions,
          orderedEquals([3, 3]));

      expect(new NDShape([2, 2, 2]).reshape(newDimensions: [2, 4]).dimensions,
          orderedEquals([2, 4]));

      expect(new NDShape([3, 2, 3]).reshape(newDimensions: [-1]).dimensions,
          orderedEquals([18]));

      expect(new NDShape([3, 2, 3]).reshape(newDimensions: [2, -1]).dimensions,
          orderedEquals([2, 9]));

      expect(new NDShape([3, 2, 3]).reshape(newDimensions: [-1, 9]).dimensions,
          orderedEquals([2, 9]));

      expect(
          new NDShape([3, 2, 3]).reshape(newDimensions: [2, -1, 3]).dimensions,
          orderedEquals([2, 3, 3]));

      expect(new NDShape([1]).reshape(newDimensions: []).dimensions,
          orderedEquals([]));
    });
  });
}
