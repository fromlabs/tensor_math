// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'package:test/test.dart';

import "package:tensor_math/tensor_math.dart";

import "package:tensor_math/src/nd_shape_impl.dart" show createTestNDShape;
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

    test('Array select tests', () {});
  });

  group('Shape tests', () {
    setUp(() {});

    test('Shape dimension test', () {
      var shape = createTestNDShape();
      expect(shape.dimension, isNull);
      expect(shape.length, isNull);
      expect(shape.isUnknownDimension, isTrue);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([]);
      expect(shape.dimension, 0);
      expect(shape.length, 1);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isTrue);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([null]);
      expect(shape.dimension, 1);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isTrue);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([1]);
      expect(shape.dimension, 1);
      expect(shape.length, 1);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isTrue);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([10]);
      expect(shape.dimension, 1);
      expect(shape.length, 10);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isTrue);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([null, null]);
      expect(shape.dimension, 2);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = createTestNDShape([1, null]);
      expect(shape.dimension, 2);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = createTestNDShape([null, 1]);
      expect(shape.dimension, 2);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = createTestNDShape([1, 1]);
      expect(shape.dimension, 2);
      expect(shape.length, 1);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = createTestNDShape([10, 10]);
      expect(shape.dimension, 2);
      expect(shape.length, 100);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isTrue);

      shape = createTestNDShape([null, null, null]);
      expect(shape.dimension, 3);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([10, 10, 10]);
      expect(shape.dimension, 3);
      expect(shape.length, 1000);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isFalse);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);
    });

    test('Shape transpose test', () {
      var shape = createTestNDShape(null).transpose();
      expect(shape.dimension, isNull);
      expect(shape.length, isNull);
      expect(shape.isUnknownDimension, isTrue);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([null, null, null]).transpose();
      expect(shape.dimension, 3);
      expect(shape.length, null);
      expect(shape.isUnknownDimension, isFalse);
      expect(shape.isUnknownLength, isTrue);
      expect(shape.isScalar, isFalse);
      expect(shape.isVector, isFalse);
      expect(shape.isMatrix, isFalse);

      shape = createTestNDShape([1, 2, 3]).transpose();
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

      expect(() => createTestNDShape([1, 2, 3]).transpose(permutationAxis: []),
          throwsArgumentError);

      expect(() => createTestNDShape([1, 2, 3]).transpose(permutationAxis: [0]),
          throwsArgumentError);

      expect(
          () => createTestNDShape([1, 2, 3])
              .transpose(permutationAxis: [0, 1, null]),
          throwsArgumentError);

      shape =
          createTestNDShape([1, 2, 3]).transpose(permutationAxis: [0, 1, 2]);
      expect(shape.dimensions, orderedEquals([1, 2, 3]));

      expect(
          () => createTestNDShape([1, 2, 3])
              .transpose(permutationAxis: [1, 2, 3]),
          throwsRangeError);

      expect(
          () => createTestNDShape([1, 2, 3])
              .transpose(permutationAxis: [0, 1, 1]),
          throwsArgumentError);
    });

    test('Shape merge test', () {
      expect(() => createTestNDShape([]).merge(createTestNDShape([null])),
          throwsArgumentError);

      expect(() => createTestNDShape([null]).merge(createTestNDShape([])),
          throwsArgumentError);

      expect(() => createTestNDShape([1]).merge(createTestNDShape([2])),
          throwsArgumentError);

      expect(() => createTestNDShape([2]).merge(createTestNDShape([1])),
          throwsArgumentError);

      expect(() => createTestNDShape([1, 1]).merge(createTestNDShape([1, 2])),
          throwsArgumentError);

      expect(() => createTestNDShape([1, 2]).merge(createTestNDShape([1, 1])),
          throwsArgumentError);

      expect(createTestNDShape().merge(createTestNDShape()).dimensions, isNull);

      expect(createTestNDShape([]).merge(createTestNDShape([])).dimensions,
          orderedEquals([]));

      expect(createTestNDShape().merge(createTestNDShape([])).dimensions,
          orderedEquals([]));

      expect(createTestNDShape([]).merge(createTestNDShape()).dimensions,
          orderedEquals([]));

      expect(createTestNDShape([1]).merge(createTestNDShape([1])).dimensions,
          orderedEquals([1]));

      expect(
          createTestNDShape([null]).merge(createTestNDShape([null])).dimensions,
          orderedEquals([null]));

      expect(createTestNDShape([1]).merge(createTestNDShape([null])).dimensions,
          orderedEquals([1]));

      expect(createTestNDShape([null]).merge(createTestNDShape([1])).dimensions,
          orderedEquals([1]));

      expect(
          createTestNDShape([1, 2])
              .merge(createTestNDShape([1, null]))
              .dimensions,
          orderedEquals([1, 2]));

      expect(
          createTestNDShape([1, null])
              .merge(createTestNDShape([1, 2]))
              .dimensions,
          orderedEquals([1, 2]));
    });

    test('Shape broadcast test', () {
      expect(() => createTestNDShape([2]).broadcast(createTestNDShape([3])),
          throwsArgumentError);

      expect(
          () => createTestNDShape([2, 2]).broadcast(createTestNDShape([2, 3])),
          throwsArgumentError);

      expect(() => createTestNDShape([2, 2]).broadcast(createTestNDShape([3])),
          throwsArgumentError);

      expect(() => createTestNDShape([3]).broadcast(createTestNDShape([2, 2])),
          throwsArgumentError);

      expect(createTestNDShape().broadcast(createTestNDShape()).dimensions,
          isNull);

      expect(createTestNDShape([]).broadcast(createTestNDShape([])).dimensions,
          orderedEquals([]));

      expect(
          createTestNDShape([2]).broadcast(createTestNDShape([2])).dimensions,
          orderedEquals([2]));

      expect(
          createTestNDShape([2, 3])
              .broadcast(createTestNDShape([2, 3]))
              .dimensions,
          orderedEquals([2, 3]));

      expect(createTestNDShape().broadcast(createTestNDShape([])).dimensions,
          orderedEquals([]));

      expect(createTestNDShape([]).broadcast(createTestNDShape()).dimensions,
          orderedEquals([]));

      expect(
          createTestNDShape([2, 3])
              .broadcast(createTestNDShape([null, 3]))
              .dimensions,
          orderedEquals([2, 3]));

      expect(
          createTestNDShape([null, 3])
              .broadcast(createTestNDShape([2, 3]))
              .dimensions,
          orderedEquals([2, 3]));

      expect(
          createTestNDShape([2, 3])
              .broadcast(createTestNDShape([1, 3]))
              .dimensions,
          orderedEquals([2, 3]));

      expect(
          createTestNDShape([1, 3])
              .broadcast(createTestNDShape([2, 3]))
              .dimensions,
          orderedEquals([2, 3]));

      expect(
          createTestNDShape([2, 3]).broadcast(createTestNDShape([])).dimensions,
          orderedEquals([2, 3]));

      expect(
          createTestNDShape([]).broadcast(createTestNDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(
          createTestNDShape([2, 3])
              .broadcast(createTestNDShape([3]))
              .dimensions,
          orderedEquals([2, 3]));

      expect(
          createTestNDShape([3])
              .broadcast(createTestNDShape([2, 3]))
              .dimensions,
          orderedEquals([2, 3]));
    });

    test('Shape matrix multiplication test', () {
      expect(() => createTestNDShape([]).matMul(createTestNDShape([])),
          throwsArgumentError);

      expect(() => createTestNDShape([]).matMul(createTestNDShape()),
          throwsArgumentError);

      expect(() => createTestNDShape().matMul(createTestNDShape([])),
          throwsArgumentError);

      expect(
          createTestNDShape().matMul(createTestNDShape()).dimensions, isNull);

      expect(
          createTestNDShape([1, 2])
              .matMul(createTestNDShape([2, 3]))
              .dimensions,
          orderedEquals([1, 3]));

      expect(
          createTestNDShape([1, 2, 3])
              .matMul(createTestNDShape([null, 3, 2]))
              .dimensions,
          orderedEquals([1, 2, 2]));
    });

    test('Shape reduce test', () {
      expect(() => createTestNDShape([]).reduce(), throwsStateError);

      expect(
          () => createTestNDShape([1, 1, null]).reduce(reductionAxis: [0, 0]),
          throwsArgumentError);

      expect(
          () => createTestNDShape([1, 1, null])
              .reduce(reductionAxis: [0, 1, 2, 3]),
          throwsArgumentError);

      expect(
          createTestNDShape([2, 2, 2]).reduce().dimensions, orderedEquals([]));

      expect(createTestNDShape([2, 2, 2]).reduce(reductionAxis: [0]).dimensions,
          orderedEquals([2, 2]));

      expect(
          createTestNDShape([2, 2, 2]).reduce(reductionAxis: [0, 1]).dimensions,
          orderedEquals([2]));

      expect(
          createTestNDShape([2, 2, 2])
              .reduce(reductionAxis: [0, 1, 2]).dimensions,
          orderedEquals([]));
    });
  });
}
