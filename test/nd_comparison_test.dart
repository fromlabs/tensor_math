// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as math;

import 'package:test/test.dart';

import "package:tensor_math/tensor_math.dart" as tm;

Iterable<List<int>> generateShapeCombinations(
    int dimension, int dimensionCount) sync* {
  var combinations = math.pow(dimensionCount, dimension + 1);
  for (var i2 = 0; i2 < combinations; i2++) {
    var shape = new List(dimension + 1);

    var i3 = 0;
    var index = i2;
    var scale = combinations ~/ dimensionCount;

    while (scale > 0) {
      shape[i3] = (index ~/ scale) + 1;

      i3++;
      index = index % scale;
      scale = scale ~/ dimensionCount;
    }

    yield shape;
  }
}

Iterable<List<int>> generateReductionAxisCombinations(int dimension) sync* {
  for (var d = 1; d <= dimension; d++) {
    var i = 0;
    var indexes = new List(d);
    indexes[0] = 0;

    for (;;) {
      if (indexes[i] > dimension - 1) {
        i--;

        if (i >= 0) {
          indexes[i]++;
        } else {
          break;
        }
      } else if (i < d - 1) {
        // inner
        i++;

        indexes[i] = indexes[i - 1] + 1;
      } else {
        // last
        yield new List.from(indexes);

        indexes[i]++;
      }
    }
  }
}

Iterable<List<int>> generatePermutationAxisCombinations(int dimension) sync* {
  var i = 0;
  var indexes = new List(dimension);
  indexes[0] = 0;

  for (;;) {
    if (indexes[i] > dimension - 1) {
      i--;

      if (i >= 0) {
        indexes[i]++;
        while (indexes.indexOf(indexes[i]) != i) {
          indexes[i]++;
        }
      } else {
        break;
      }
    } else if (i < dimension - 1) {
      // inner
      i++;

      indexes[i] = 0;
      while (indexes.indexOf(indexes[i]) != i) {
        indexes[i]++;
      }
    } else {
      // last
      yield new List.from(indexes);

      indexes[i]++;
      while (indexes.indexOf(indexes[i]) != i) {
        indexes[i]++;
      }
    }
  }
}

void main() {
  group('Array tests', () {
    test('Create tests', () {
      var test = (List<int> shape) {
        var value = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .toValue();

        expect(
          new tm.NDArray(value, dataType: tm.NDDataType.float32).toValue(),
          equals(value),
        );
        expect(
          new tm.NDArray(value, dataType: tm.NDDataType.float32Blocked)
              .toValue(),
          equals(value),
        );
      };

      test([]);

      var maxDimension = 4;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          test(shape);
        }
      }
    });

    test('Generate tests', () {
      var test = (List<int> shape) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .toValue();
        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .toValue(),
            equals(expectedValue));
      };

      test([]);

      var maxDimension = 4;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          test(shape);
        }
      }
    });

    test('Cast tests', () {
      var test = (List<int> shape) {
        var fArray = new tm.NDArray.generate(shape, (index) => index + 1,
            dataType: tm.NDDataType.float32);
        var iArray = new tm.NDArray.generate(shape, (index) => index + 1,
            dataType: tm.NDDataType.int64);
        var vArray = new tm.NDArray.generate(shape, (index) => index + 1,
            dataType: tm.NDDataType.float32Blocked);

        var expectedFValue = fArray.toValue();
        var expectedIValue = iArray.toValue();

        expect(fArray.cast(tm.NDDataType.float32Blocked).toValue(),
            equals(expectedFValue));
        expect(fArray.cast(tm.NDDataType.float32Blocked).toValue(),
            equals(expectedFValue));

        expect(vArray.cast(tm.NDDataType.float32).toValue(),
            equals(expectedFValue));
        expect(vArray.cast(tm.NDDataType.float32Blocked).toValue(),
            equals(expectedFValue));
        expect(
            vArray.cast(tm.NDDataType.int64).toValue(), equals(expectedIValue));
      };

      test([]);

      var maxDimension = 4;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          test(shape);
        }
      }
    });

    test('Reduce sum tests', () {
      var test = (List<int> shape, List<int> reductionAxis) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .reduceSum(reductionAxis: reductionAxis)
            .toValue();

        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .reduceSum(reductionAxis: reductionAxis)
                .toValue(),
            equals(expectedValue));
      };

      var maxDimension = 5;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          print("shape: $shape");

          for (var reductionAxis in generateReductionAxisCombinations(i + 1)) {
            try {
              test(shape, reductionAxis);
            } catch (e) {
              print("reductionAxis: $reductionAxis");

              rethrow;
            }
          }
        }
      }
    });

    test('Reduce mean tests', () {
      var test = (List<int> shape, List<int> reductionAxis) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .reduceMean(reductionAxis: reductionAxis)
            .toValue();

        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .reduceMean(reductionAxis: reductionAxis)
                .toValue(),
            equals(expectedValue));
      };

      var maxDimension = 5;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          print("shape: $shape");

          for (var reductionAxis in generateReductionAxisCombinations(i + 1)) {
            try {
              test(shape, reductionAxis);
            } catch (e) {
              print("reductionAxis: $reductionAxis");

              rethrow;
            }
          }
        }
      }
    });

    test('Reduce max tests', () {
      var test = (List<int> shape, List<int> reductionAxis) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .reduceMax(reductionAxis: reductionAxis)
            .toValue();

        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .reduceMax(reductionAxis: reductionAxis)
                .toValue(),
            equals(expectedValue));
      };

      var maxDimension = 5;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          print("shape: $shape");

          for (var reductionAxis in generateReductionAxisCombinations(i + 1)) {
            try {
              test(shape, reductionAxis);
            } catch (e) {
              print("reductionAxis: $reductionAxis");

              rethrow;
            }
          }
        }
      }
    });

    test('Reduce sum keeping dimensions tests', () {
      var test = (List<int> shape, List<int> reductionAxis) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .reduceSum(reductionAxis: reductionAxis, keepDimensions: true)
            .toValue();

        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .reduceSum(reductionAxis: reductionAxis, keepDimensions: true)
                .toValue(),
            equals(expectedValue));
      };

      var maxDimension = 5;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          print("shape: $shape");

          for (var reductionAxis in generateReductionAxisCombinations(i + 1)) {
            try {
              test(shape, reductionAxis);
            } catch (e) {
              print("reductionAxis: $reductionAxis");

              rethrow;
            }
          }
        }
      }
    });

    test('Arg max tests', () {
      var test = (List<int> shape, int axis) {
        List list = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .reshape(newDimensions: [-1]).toValue();

        list.shuffle();

        var initialValue = new tm.NDArray(list, dataType: tm.NDDataType.float32)
            .reshape(newDimensions: shape)
            .toValue();

        var expectedValue =
            new tm.NDArray(initialValue, dataType: tm.NDDataType.float32)
                .argMax(axis: axis)
                .toValue();

        expect(
            new tm.NDArray(initialValue,
                    dataType: tm.NDDataType.float32Blocked)
                .argMax(axis: axis)
                .toValue(),
            equals(expectedValue));
      };

      var maxDimension = 5;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          print("shape: $shape");

          for (var axis = 0; axis < shape.length; axis++) {
            try {
              for (var l = 0; l < 10; l++) {
                test(shape, axis);
              }
            } catch (e) {
              print("axis: $axis");

              rethrow;
            }
          }
        }
      }
    });

    test('Matmul tests', () {
      var test = (List<int> shape1, List<int> shape2) {
        var shapeLength2 = shape2.reduce((v1, v2) => v1 * v2);
        var expectedValue = new tm.NDArray.generate(
                shape1, (index) => index + 1, dataType: tm.NDDataType.float32)
            .matMul(new tm.NDArray.generate(
                shape2, (index) => shapeLength2 - index,
                dataType: tm.NDDataType.float32))
            .toValue();

        expect(
            new tm.NDArray.generate(shape1, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .matMul(new tm.NDArray.generate(
                    shape2, (index) => shapeLength2 - index,
                    dataType: tm.NDDataType.float32Blocked))
                .toValue(),
            equals(expectedValue));
      };

      var minDimension = 2;
      var maxDimension = 4;
      var dimensionCount = 8;

      for (var i = minDimension; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          for (var i3 = 1; i3 < dimensionCount; i3++) {
            var shape2 = shape.sublist(0, shape.length - 2);
            shape2.add(shape[shape.length - 1]);
            shape2.add(i3);
            test(shape, shape2);
          }
        }
      }
    });

    test('Transpose tests', () {
      var test = (List<int> shape, List<int> permutationAxis) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .transpose(permutationAxis: permutationAxis)
            .toValue();

        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .transpose(permutationAxis: permutationAxis)
                .toValue(),
            equals(expectedValue));
      };

      var maxDimension = 5;
      var dimensionCount = 11;

      for (var i = 1; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          print("shape: $shape");

          for (var permutationAxis
              in generatePermutationAxisCombinations(i + 1)) {
            try {
              test(shape, permutationAxis);
            } catch (e) {
              print("permutationAxis: $permutationAxis");

              rethrow;
            }
          }
        }
      }
    });

    test('Tile tests', () {
      var test = (List<int> shape, List<int> multiplies) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .tile(multiplies)
            .toValue();

        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .tile(multiplies)
                .toValue(),
            equals(expectedValue));
      };

      var maxDimension = 5;
      var dimensionCount = 11;

      for (var i = 0; i < maxDimension; i++) {
        for (var shape in generateShapeCombinations(i, dimensionCount)) {
          print("shape: $shape");

          for (var reductionAxis in generateReductionAxisCombinations(i + 1)) {
            for (var m = 2; m < 5; m++) {
              var multiplies = new List.filled(shape.length, 1);
              for (var axis in reductionAxis) {
                multiplies[axis] = m;
              }

              try {
                test(shape, multiplies);
              } catch (e) {
                print("multiplies: $multiplies");

                rethrow;
              }
            }
          }
        }
      }
    });

    test('Neg data tests', () {
      var test = (List<int> shape) {
        var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
                dataType: tm.NDDataType.float32)
            .neg()
            .toValue();
        expect(
            new tm.NDArray.generate(shape, (index) => index + 1,
                    dataType: tm.NDDataType.float32Blocked)
                .neg()
                .toValue(),
            equals(expectedValue));
      };

      var minDimension = 2;
      var maxDimension = 4;
      var dimensionCount = 11;

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

    test('Add tests', () {
      var test = (List<int> shape1, List<int> shape2) {
        var array1 = new tm.NDArray.generate(shape1, (index) => index + 1,
            dataType: tm.NDDataType.float32);
        var shapeLength2 = shape2.reduce((v1, v2) => v1 * v2);
        var array2 = new tm.NDArray.generate(
            shape2, (index) => shapeLength2 - index,
            dataType: tm.NDDataType.float32);
        var expectedArray = array1.add(array2);

        var value1 = array1.toValue();
        var value2 = array2.toValue();
        var expectedValue = expectedArray.toValue();

        var vArray1 =
            new tm.NDArray(value1, dataType: tm.NDDataType.float32Blocked);

        var vArray2 =
            new tm.NDArray(value2, dataType: tm.NDDataType.float32Blocked);

        expect(vArray1.add(vArray2).toValue(), equals(expectedValue));
      };

      // head broadcast
      test([4, 4], [4, 4]);
      test([1, 4, 4], [3, 4, 4]);
      test([4, 4], [2, 3, 4, 4]);
      test([11, 13], [2, 3, 11, 13]);

      // full broadcast
      test([1, 1], [2, 3, 2, 2]);
      test([1, 1], [2, 3, 3, 3]);
      test([1, 1], [2, 3, 4, 4]);
    });

    test('Reshape tests', () {
      var test = (List<int> shape, List<int> newShape) {
        var value =
            new tm.NDArray.generate(shape, (index) => index + 1).toValue();

        var expectedValue =
            new tm.NDArray(value, dataType: tm.NDDataType.float32)
                .reshape(newDimensions: newShape)
                .toValue();
        expect(
            new tm.NDArray(value, dataType: tm.NDDataType.float32Blocked)
                .reshape(newDimensions: newShape)
                .toValue(),
            equals(expectedValue));
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
  });
}
