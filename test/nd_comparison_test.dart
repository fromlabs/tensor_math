// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as math;

import 'package:test/test.dart';

import "package:tensor_math_simd/tensor_math.dart" as tm;

void main() {
  group('Array tests', () {
    test('Loading data tests', () {
      var test = (List<int> shape) {
        var value =
            new tm.NDArray.generate(shape, (index) => index + 1).toValue();

        expect(
            value,
            equals(new tm.NDArray(value, dataType: tm.NDDataType.float32)
                .toValue()));
        expect(
            value,
            equals(
                new tm.NDArray(value, dataType: tm.NDDataType.float32HBlocked)
                    .toValue()));
        expect(
            value,
            equals(
                new tm.NDArray(value, dataType: tm.NDDataType.float32VBlocked)
                    .toValue()));
      };

      var minDimension = 2;
      var maxDimension = 4;
      var dimensionCount = 12;

      for (var i = minDimension; i < maxDimension; i++) {
        var combinations = math.pow(dimensionCount, i + 1);
        for (var i2 = 0; i2 < combinations; i2++) {
          var shape = new List(i + 1);

          var i3 = 0;
          var index = i2;
          var scale = combinations ~/ dimensionCount;

          while (scale > 0) {
            shape[i3] = (index ~/ scale) + 1;

            i3++;
            index = index % scale;
            scale = scale ~/ dimensionCount;
          }

          test(shape);
        }
      }
    });
  });

  test('Reshape tests', () {
    var test = (List<int> shape, List<int> newShape) {
      var value =
          new tm.NDArray.generate(shape, (index) => index + 1).toValue();

      var expectedValue = new tm.NDArray(value, dataType: tm.NDDataType.float32)
          .reshape(newDimensions: newShape)
          .toValue();
      expect(
          expectedValue,
          equals(new tm.NDArray(value, dataType: tm.NDDataType.float32HBlocked)
              .reshape(newDimensions: newShape)
              .toValue()));
      expect(
          expectedValue,
          equals(new tm.NDArray(value, dataType: tm.NDDataType.float32VBlocked)
              .reshape(newDimensions: newShape)
              .toValue()));
    };

    test([2, 2, 4, 4], [2, 2, 4, 4]);

    test([2, 2, 4, 4], [1, 4, 4, 4]);
    test([2, 2, 4, 4], [4, 1, 4, 4]);

    test([2, 2, 11, 13], [1, 4, 11, 13]);
    test([2, 2, 11, 13], [4, 1, 11, 13]);

    test([2, 2, 2, 2], [2, 2, 1, 4]);
    test([2, 2, 2, 2], [2, 2, 4, 1]);

    test([4, 4, 11, 13], [1, 1, 52, 44]);
  });

  test('Transpose tests', () {
    var test = (List<int> shape, List<int> permutationAxis) {
      var value =
          new tm.NDArray.generate(shape, (index) => index + 1).toValue();

      var expectedArray =
          new tm.NDArray(value, dataType: tm.NDDataType.float32);

      var hArray =
          new tm.NDArray(value, dataType: tm.NDDataType.float32HBlocked);

      var vArray =
          new tm.NDArray(value, dataType: tm.NDDataType.float32HBlocked);

      var expectedValue =
          expectedArray.transpose(permutationAxis: permutationAxis).toValue();

      expect(expectedValue,
          equals(hArray.transpose(permutationAxis: permutationAxis).toValue()));

      expect(expectedValue,
          equals(vArray.transpose(permutationAxis: permutationAxis).toValue()));
    };

    test([4, 4, 11, 13], [0, 1, 2, 3]);

    test([4, 4, 11, 13], null);
  });
}
