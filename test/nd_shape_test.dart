// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'package:test/test.dart';

import "package:tensor_math/tensor_math.dart";

void main() {
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
      expect(() => new NDShape([]).mergeWith(new NDShape([null])),
          throwsArgumentError);

      expect(() => new NDShape([null]).mergeWith(new NDShape([])),
          throwsArgumentError);

      expect(() => new NDShape([1]).mergeWith(new NDShape([2])),
          throwsArgumentError);

      expect(() => new NDShape([2]).mergeWith(new NDShape([1])),
          throwsArgumentError);

      expect(() => new NDShape([1, 1]).mergeWith(new NDShape([1, 2])),
          throwsArgumentError);

      expect(() => new NDShape([1, 2]).mergeWith(new NDShape([1, 1])),
          throwsArgumentError);

      expect(new NDShape().mergeWith(new NDShape()).dimensions, isNull);

      expect(new NDShape([]).mergeWith(new NDShape([])).dimensions,
          orderedEquals([]));

      expect(new NDShape().mergeWith(new NDShape([])).dimensions,
          orderedEquals([]));

      expect(new NDShape([]).mergeWith(new NDShape()).dimensions,
          orderedEquals([]));

      expect(new NDShape([1]).mergeWith(new NDShape([1])).dimensions,
          orderedEquals([1]));

      expect(new NDShape([null]).mergeWith(new NDShape([null])).dimensions,
          orderedEquals([null]));

      expect(new NDShape([1]).mergeWith(new NDShape([null])).dimensions,
          orderedEquals([1]));

      expect(new NDShape([null]).mergeWith(new NDShape([1])).dimensions,
          orderedEquals([1]));

      expect(new NDShape([1, 2]).mergeWith(new NDShape([1, null])).dimensions,
          orderedEquals([1, 2]));

      expect(new NDShape([1, null]).mergeWith(new NDShape([1, 2])).dimensions,
          orderedEquals([1, 2]));
    });

    test('Shape broadcast test', () {
      expect(() => new NDShape([2]).broadcast(new NDShape([3])),
          throwsArgumentError);

      expect(() => new NDShape([2, 2]).broadcast(new NDShape([2, 3])),
          throwsArgumentError);

      expect(() => new NDShape([2, 2]).broadcast(new NDShape([3])),
          throwsArgumentError);

      expect(() => new NDShape([3]).broadcast(new NDShape([2, 2])),
          throwsArgumentError);

      expect(new NDShape().broadcast(new NDShape()).dimensions, isNull);

      expect(new NDShape([]).broadcast(new NDShape([])).dimensions,
          orderedEquals([]));

      expect(new NDShape([2]).broadcast(new NDShape([2])).dimensions,
          orderedEquals([2]));

      expect(new NDShape([2, 3]).broadcast(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape().broadcast(new NDShape([])).dimensions, isNull);

      expect(new NDShape([]).broadcast(new NDShape()).dimensions, isNull);

      expect(new NDShape([3, 2]).broadcast(new NDShape()).dimensions, isNull);

      expect(new NDShape([2, 3]).broadcast(new NDShape([null, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([null, 3]).broadcast(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([2, 3]).broadcast(new NDShape([1, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([1, 3]).broadcast(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([2, 3]).broadcast(new NDShape([])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([]).broadcast(new NDShape([2, 3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([2, 3]).broadcast(new NDShape([3])).dimensions,
          orderedEquals([2, 3]));

      expect(new NDShape([3]).broadcast(new NDShape([2, 3])).dimensions,
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
      expect(() => new NDShape([1, 1, null]).reduce(reductionAxis: [0, 0]),
          throwsArgumentError);

      expect(
          () => new NDShape([1, 1, null]).reduce(reductionAxis: [0, 1, 2, 3]),
          throwsArgumentError);

      expect(new NDShape([]).reduce().dimensions, isEmpty);

      expect(new NDShape([2, 2, 2]).reduce().dimensions, orderedEquals([]));

      expect(new NDShape([2, 2, 2]).reduce(reductionAxis: [0]).dimensions,
          orderedEquals([2, 2]));

      expect(new NDShape([2, 2, 2]).reduce(reductionAxis: [0, 1]).dimensions,
          orderedEquals([2]));

      expect(new NDShape([2, 2, 2]).reduce(reductionAxis: [0, 1, 2]).dimensions,
          orderedEquals([]));
    });

    test('Shape reduce keep dimensions test', () {
      expect(
          () => new NDShape([1, 1, null])
              .reduce(reductionAxis: [0, 0], keepDimensions: true),
          throwsArgumentError);

      expect(
          () => new NDShape([1, 1, null])
              .reduce(reductionAxis: [0, 1, 2, 3], keepDimensions: true),
          throwsArgumentError);

      expect(new NDShape([]).reduce(keepDimensions: true).dimensions, isEmpty);

      expect(new NDShape([2, 2, 2]).reduce(keepDimensions: true).dimensions,
          orderedEquals([1, 1, 1]));

      expect(
          new NDShape([2, 2, 2])
              .reduce(reductionAxis: [0], keepDimensions: true).dimensions,
          orderedEquals([1, 2, 2]));

      expect(
          new NDShape([2, 2, 2])
              .reduce(reductionAxis: [0, 1], keepDimensions: true).dimensions,
          orderedEquals([1, 1, 2]));

      expect(
          new NDShape([2, 2, 2]).reduce(
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
