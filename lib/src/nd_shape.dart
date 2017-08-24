// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math";

import 'package:collection/collection.dart';

import "nd_data_type.dart";

final _iterableEquality = new IterableEquality<int>();

class NDShape {
  final List<int> dimensions;

  final int _length;

  NDShape([List<int> dimensions])
      : this.dimensions =
            dimensions != null ? new List.unmodifiable(dimensions) : null,
        this._length = _calculateLength(dimensions);

  int get dimensionCount => dimensions?.length;

  int get length => _length;

  bool get hasUnknownDimensions => dimensions == null;

  bool get hasUnknownLength => _length == null;

  bool get isScalar => dimensionCount == 0;

  bool get isVector => dimensionCount == 1;

  bool get isMatrix => dimensionCount == 2;

  bool get isTensor3D => dimensionCount == 3;

  bool get isTensor4D => dimensionCount == 4;

  int get(int axe) => dimensions != null ? dimensions[axe] : null;

  int operator [](int axe) => get(axe);

  bool isCompatibleWith(NDShape shape2) {
    if (hasUnknownDimensions || shape2.hasUnknownDimensions) {
      return true;
    } else if (dimensionCount != shape2.dimensionCount) {
      return false;
    } else {
      for (var i = 0; i < dimensionCount; i++) {
        var axeDimension = dimensions[i];
        var axeDimension2 = shape2.dimensions[i];
        if (axeDimension == null || axeDimension2 == null) {
          // continue
        } else if (axeDimension != null && axeDimension2 != null) {
          if (axeDimension != axeDimension2) {
            return false;
          }
        } else {
          return false;
        }
      }
    }
    return true;
  }

  @override
  // ignore: hash_and_equals
  bool operator ==(other) => other is NDShape
      ? _iterableEquality.equals(dimensions, other.dimensions)
      : false;

  NDShape matMul(covariant NDShape shape2) => _matMul(shape2);

  NDShape reshape({List<int> newDimensions}) =>
      _reshape(newDimensions: newDimensions);

  NDShape cast(NDDataType dataType) => this;

  NDShape mergeWith(covariant NDShape shape2) => _merge(shape2);

  NDShape broadcast(covariant NDShape shape2) => _broadcast(shape2);

  NDShape reduce({List<int> reductionAxis, bool keepDimensions = false}) =>
      _reduce(reductionAxis: reductionAxis, keepDimensions: keepDimensions);

  NDShape arg({int axis}) {
    if (axis != null) {
      return _reduce(reductionAxis: [axis], keepDimensions: false);
    } else {
      return reshape(newDimensions: [-1]).arg(axis: 0);
    }
  }

  NDShape tile(List<int> multiplies) => _tile(multiplies);

  NDShape transpose({List<int> permutationAxis}) =>
      _transpose(permutationAxis: permutationAxis);

  NDShape _merge(NDShape shape2) {
    if (dimensionCount != null && shape2.dimensionCount != null) {
      if (dimensionCount == shape2.dimensionCount) {
        var resultDimensions = new List(dimensionCount);

        for (var i = 0; i < resultDimensions.length; i++) {
          var dimension1 = this.dimensions[i];
          var dimension2 = shape2.dimensions[i];
          if (dimension1 != null && dimension2 != null) {
            if (dimension1 == dimension2) {
              resultDimensions[i] = dimension1;
            } else {
              throw new ArgumentError(
                  "Shape dimensions $i must be equal: $this != $shape2");
            }
          } else if (dimension1 != null) {
            resultDimensions[i] = dimension1;
          } else {
            resultDimensions[i] = dimension2;
          }
        }

        return new NDShape(resultDimensions);
      } else {
        throw new ArgumentError(
            "Shape dimensions must be equal: $this != $shape2");
      }
    } else {
      return dimensionCount != null ? this : shape2;
    }
  }

  NDShape _reduce({List<int> reductionAxis, bool keepDimensions = false}) {
    if (hasUnknownDimensions) {
      return this;
    } else if (dimensionCount == 0) {
      throw new StateError("Can't reduce a scalar");
    } else {
      var newReductionAxis =
          convertToValidReductionAxis(reductionAxis, dimensionCount);

      if (newReductionAxis.length > dimensionCount) {
        throw new ArgumentError.value(newReductionAxis, "reduction axis",
            "Max dimension is $dimensionCount");
      } else if (newReductionAxis.length != newReductionAxis.toSet().length) {
        throw new ArgumentError.value(newReductionAxis, "reduction axis",
            "Must be unique indexes $newReductionAxis");
      }

      var resultDimensions = new List(keepDimensions
          ? dimensionCount
          : dimensionCount - newReductionAxis.length);
      var axis = newReductionAxis.toSet();
      var resultIndex = 0;
      for (var i = 0; i < dimensionCount; i++) {
        if (keepDimensions) {
          resultDimensions[resultIndex++] =
              !axis.contains(i) ? dimensions[i] : 1;
        } else {
          if (!axis.contains(i)) {
            resultDimensions[resultIndex++] = dimensions[i];
          }
        }
      }

      return new NDShape(resultDimensions);
    }
  }

  NDShape _transpose({List<int> permutationAxis}) {
    if (hasUnknownDimensions) {
      return this;
    } else {
      var newPermutationAxis = permutationAxis;

      if (dimensionCount < 2) {
        throw new ArgumentError("Shape dimension must be almost 2: $this");
      } else if (newPermutationAxis == null) {
        newPermutationAxis = new List.generate(
            dimensionCount, (index) => dimensionCount - index - 1);
      } else if (permutationAxis.length != dimensionCount) {
        throw new ArgumentError.value(permutationAxis, "permutation axis",
            "Dimension is $dimensionCount");
      } else if (permutationAxis.length != permutationAxis.toSet().length) {
        throw new ArgumentError.value(permutationAxis, "permutation axis",
            "Must be unique indexes $permutationAxis");
      }

      var resultDimensions = new List(dimensionCount);

      for (var i = 0; i < resultDimensions.length; i++) {
        resultDimensions[i] = dimensions[newPermutationAxis[i]];
      }

      return new NDShape(resultDimensions);
    }
  }

  NDShape _broadcast(NDShape shape2) {
    if (dimensionCount != null && shape2.dimensionCount != null) {
      var resultDimensions =
          new List(max(dimensionCount, shape2.dimensionCount));

      var resultIndex = resultDimensions.length - 1;
      var index1 = dimensionCount - 1;
      var index2 = shape2.dimensionCount - 1;
      while (index1 >= 0 || index2 >= 0) {
        var d1 = index1 >= 0 ? dimensions[index1] : 1;
        var d2 = index2 >= 0 ? shape2.dimensions[index2] : 1;
        if (d1 == 1 || d1 == d2 || (d2 != 1 && d1 == null)) {
          resultDimensions[resultIndex--] = d2;
        } else if (d2 == 1 || d1 != 1 && d2 == null) {
          resultDimensions[resultIndex--] = d1;
        } else {
          throw new ArgumentError(
              "Shapes must be broadcastable: $this != $shape2");
        }

        index1--;
        index2--;
      }

      return new NDShape(resultDimensions);
    } else {
      return hasUnknownDimensions ? this : shape2;
    }
  }

  NDShape _matMul(NDShape shape2) {
    if (hasUnknownDimensions) {
      if (shape2.hasUnknownDimensions) {
        return this;
      } else if (shape2.dimensionCount < 2) {
        throw new ArgumentError("Shape dimension must be almost 2: $shape2");
      } else {
        var resultDimensions = new List(shape2.dimensionCount);

        resultDimensions[shape2.dimensionCount - 2] = null;
        resultDimensions[shape2.dimensionCount - 1] =
            shape2.dimensions[shape2.dimensionCount - 1];

        return new NDShape(resultDimensions);
      }
    } else if (shape2.hasUnknownDimensions) {
      if (dimensionCount < 2) {
        throw new ArgumentError("Shape dimension must be almost 2: $this");
      } else {
        var resultDimensions = new List(dimensionCount);

        resultDimensions[dimensionCount - 2] = dimensions[dimensionCount - 2];
        resultDimensions[dimensionCount - 1] = null;

        return new NDShape(resultDimensions);
      }
    } else if (dimensionCount < 2) {
      throw new ArgumentError("Shape dimension must be almost 2: $this");
    } else if (shape2.dimensionCount < 2) {
      throw new ArgumentError("Shape dimension must be almost 2: $shape2");
    } else if (dimensionCount != shape2.dimensionCount) {
      throw new ArgumentError(
          "Shape dimensions must be equal: $this != $shape2");
    } else {
      var resultDimensions = new List(dimensionCount);

      // check head
      if (dimensionCount > 2) {
        for (var i = 0; i < dimensionCount - 2; i++) {
          var dimension1 = dimensions[i];
          var dimension2 = shape2.dimensions[i];
          if (dimension1 != null &&
              dimension2 != null &&
              dimension1 != dimension2) {
            throw new ArgumentError(
                "Shape dimensions $i must be equal: $this != $shape2");
          } else {
            resultDimensions[i] = dimension1 ?? dimension2;
          }
        }
      }

      // check tail matrix
      var cols1 = dimensions[dimensionCount - 1];
      var rows2 = shape2.dimensions[shape2.dimensionCount - 2];
      if (cols1 != null && rows2 != null && cols1 != rows2) {
        throw new ArgumentError(
            "Shape not valid for matrix multiplication: $cols1 != $rows2");
      } else {
        resultDimensions[dimensionCount - 2] = dimensions[dimensionCount - 2];
        resultDimensions[dimensionCount - 1] =
            shape2.dimensions[shape2.dimensionCount - 1];

        return new NDShape(resultDimensions);
      }
    }
  }

  NDShape _reshape({List<int> newDimensions}) {
    if (newDimensions == null) {
      if (hasUnknownDimensions) {
        return this;
      } else {
        return new NDShape();
      }
    }

    var newLength = 1;
    var wildcardDimensionIndex;
    for (var i = 0; i < newDimensions.length; i++) {
      var dimension = newDimensions[i];

      if (dimension == -1) {
        if (wildcardDimensionIndex == null) {
          wildcardDimensionIndex = i;
        } else {
          throw new ArgumentError.value(newDimensions, "reshape dimensions",
              "Just one -1 dimension allowed");
        }
      } else {
        newLength *= dimension;
      }
    }

    var newDimensions2;
    if (wildcardDimensionIndex != null) {
      newDimensions2 = new List.from(newDimensions);

      if (!hasUnknownLength) {
        if (length % newLength == 0) {
          newDimensions2[wildcardDimensionIndex] = length ~/ newLength;
        } else {
          throw new ArgumentError.value(newDimensions, "reshape dimensions",
              "Reshape not allowed: $this != $newDimensions");
        }
      } else {
        newDimensions2[wildcardDimensionIndex] = null;
      }
    } else {
      newDimensions2 = newDimensions;
    }

    var newShape = new NDShape(newDimensions2);
    if (hasUnknownLength || newShape.length == length) {
      return newShape;
    } else {
      throw new ArgumentError.value(
          newShape.length, "new shape length", "Must be $length");
    }
  }

  NDShape _tile(List<int> multiplies) {
    if (isScalar) {
      throw new StateError("Can't tile a scalar");
    } else if (multiplies == null || multiplies.length != dimensionCount) {
      throw new ArgumentError.value(
          multiplies, "tile multiplis", "Must be $dimensionCount length");
    }

    var newDimensions = new List.generate(
        dimensionCount, (index) => multiplies[index] * dimensions[index]);

    return new NDShape(newDimensions);
  }

  @override
  String toString() =>
      "<Shape: dimensions=$dimensions, dimension=$dimensionCount, length=$length>";

  static int _calculateLength(List<int> dimensions) {
    if (dimensions == null) {
      return null;
    } else if (dimensions.isNotEmpty) {
      var total = 1;
      for (var i = 0; i < dimensions.length; i++) {
        if (dimensions[i] != null) {
          total *= dimensions[i];
        } else {
          return null;
        }
      }
      return total;
    } else {
      return 1;
    }
  }
}

List<int> convertToValidReductionAxis(List<int> reductionAxis, int dimension) {
  if (reductionAxis == null) {
    if (dimension > 0) {
      return new List.generate(dimension, (index) => index);
    } else {
      return null;
    }
  } else {
    return new List.from(reductionAxis)..sort();
  }
}
