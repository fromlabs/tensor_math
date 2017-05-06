// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math";

import "nd_shape.dart";

NDShape createTestNDShape([List<int> dimensions]) =>
    new NDShapeImpl(dimensions);

NDShape broadcastIterable(Iterable<NDShape> shapes) =>
    shapes.reduce((total, element) => total.broadcast(element));

class NDShapeImpl implements NDShape {
  final List<int> internalDimensions;
  final int _length;

  NDShapeImpl(this.internalDimensions)
      : this._length = _calculateLength(internalDimensions);

  @override
  int get dimension => internalDimensions?.length;

  @override
  int get length => _length;

  @override
  bool get isUnknownDimension => internalDimensions == null;

  @override
  bool get isUnknownLength => _length == null;

  @override
  bool get isScalar => dimension == 0;

  @override
  bool get isVector => dimension == 1;

  @override
  bool get isMatrix => dimension == 2;

  @override
  bool get isTensor3D => dimension == 3;

  @override
  bool get isTensor4D => dimension == 4;

  @override
  List<int> get dimensions =>
      internalDimensions != null ? new List.from(internalDimensions) : null;

  @override
  int get(int axe) =>
      internalDimensions != null ? internalDimensions[axe] : null;

  @override
  int operator [](int axe) => get(axe);

  @override
  NDShape reduce({List<int> reductionAxis}) {
    if (isUnknownDimension) {
      return this;
    } else {
      var newReductionAxis = reductionAxis;

      if (reductionAxis == null && dimension > 0) {
        newReductionAxis = new List.generate(dimension, (index) => index);
      } else if (reductionAxis == null || reductionAxis.isEmpty) {
        return this;
      } else if (dimension == 0) {
        throw new StateError("Can't reduce a scalar");
      } else if (reductionAxis.length > dimension) {
        throw new ArgumentError.value(
            reductionAxis, "reduction axis", "Max dimension is $dimension");
      } else if (reductionAxis.length != reductionAxis.toSet().length) {
        throw new ArgumentError.value(reductionAxis, "reduction axis",
            "Must be unique indexes $reductionAxis");
      }

      var resultDimensions = new List(dimension - newReductionAxis.length);
      var axis = newReductionAxis.toSet();
      var resultIndex = 0;
      for (var i = 0; i < dimension; i++) {
        if (!axis.contains(i)) {
          resultDimensions[resultIndex++] = internalDimensions[i];
        }
      }

      return new NDShapeImpl(resultDimensions);
    }
  }

  @override
  NDShape transpose({List<int> permutationAxis}) {
    if (isUnknownDimension) {
      return this;
    } else {
      var newPermutationAxis = permutationAxis;

      if (newPermutationAxis == null) {
        newPermutationAxis =
            new List.generate(dimension, (index) => dimension - index - 1);
      } else if (permutationAxis.length != dimension) {
        throw new ArgumentError.value(
            permutationAxis, "permutation axis", "Dimension is $dimension");
      } else if (permutationAxis.length != permutationAxis.toSet().length) {
        throw new ArgumentError.value(permutationAxis, "permutation axis",
            "Must be unique indexes $permutationAxis");
      }

      var resultDimensions = new List(dimension);

      for (var i = 0; i < resultDimensions.length; i++) {
        resultDimensions[i] = internalDimensions[newPermutationAxis[i]];
      }

      return new NDShapeImpl(resultDimensions);
    }
  }

  @override
  NDShape merge(covariant NDShapeImpl shape2) {
    if (dimension != null && shape2.dimension != null) {
      if (dimension == shape2.dimension) {
        var resultDimensions = new List(dimension);

        for (var i = 0; i < resultDimensions.length; i++) {
          var dimension1 = this.internalDimensions[i];
          var dimension2 = shape2.internalDimensions[i];
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

        return new NDShapeImpl(resultDimensions);
      } else {
        throw new ArgumentError(
            "Shape dimensions must be equal: $this != $shape2");
      }
    } else {
      return dimension != null ? this : shape2;
    }
  }

  @override
  NDShape broadcast(covariant NDShapeImpl shape2) {
    if (dimension != null && shape2.dimension != null) {
      var resultDimensions = new List(max(dimension, shape2.dimension));

      var resultIndex = resultDimensions.length - 1;
      var index1 = dimension - 1;
      var index2 = shape2.dimension - 1;
      while (index1 >= 0 || index2 >= 0) {
        var d1 = index1 >= 0 ? (internalDimensions[index1] ?? 1) : 1;
        var d2 = index2 >= 0 ? (shape2.internalDimensions[index2] ?? 1) : 1;
        if (d1 == 1 || d1 == d2) {
          resultDimensions[resultIndex--] = d2;
        } else if (d2 == 1) {
          resultDimensions[resultIndex--] = d1;
        } else {
          throw new ArgumentError(
              "Shapes must be broadcastable: $this != $shape2");
        }

        index1--;
        index2--;
      }

      return new NDShapeImpl(resultDimensions);
    } else {
      return dimension != null ? this : shape2;
    }
  }

  @override
  NDShape matMul(covariant NDShapeImpl shape2) {
    if (dimension == null) {
      if (shape2.dimension == null) {
        return this;
      } else if (shape2.dimension < 2) {
        throw new ArgumentError("Shape dimension must be almost 2: $shape2");
      } else {
        var resultDimensions = new List(shape2.dimension);

        resultDimensions[shape2.dimension - 2] = null;
        resultDimensions[shape2.dimension - 1] =
            shape2.internalDimensions[shape2.dimension - 1];

        return new NDShapeImpl(resultDimensions);
      }
    } else if (shape2.dimension == null) {
      if (dimension == null) {
        return this;
      } else if (dimension < 2) {
        throw new ArgumentError("Shape dimension must be almost 2: $this");
      } else {
        var resultDimensions = new List(dimension);

        resultDimensions[dimension - 2] = internalDimensions[dimension - 2];
        resultDimensions[dimension - 1] = null;

        return new NDShapeImpl(resultDimensions);
      }
    } else if (dimension < 2) {
      throw new ArgumentError("Shape dimension must be almost 2: $this");
    } else if (shape2.dimension < 2) {
      throw new ArgumentError("Shape dimension must be almost 2: $shape2");
    } else if (dimension != shape2.dimension) {
      throw new ArgumentError(
          "Shape dimensions must be equal: $this != $shape2");
    } else {
      var resultDimensions = new List(dimension);

      // check head
      if (dimension > 2) {
        for (var i = 0; i < dimension - 2; i++) {
          var dimension1 = internalDimensions[i];
          var dimension2 = shape2.internalDimensions[i];
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
      var cols1 = internalDimensions[dimension - 1];
      var rows2 = shape2.internalDimensions[shape2.dimension - 2];
      if (cols1 != null && rows2 != null && cols1 != rows2) {
        throw new ArgumentError(
            "Shape not valid for matrix multiplication: $cols1 != $rows2");
      } else {
        resultDimensions[dimension - 2] = internalDimensions[dimension - 2];
        resultDimensions[dimension - 1] =
            shape2.internalDimensions[shape2.dimension - 1];

        return new NDShapeImpl(resultDimensions);
      }
    }
  }

  @override
  NDShape reshape({List<int> newDimensions}) {
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

      if (!isUnknownLength) {
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

    var newShape = new NDShapeImpl(newDimensions2);
    if (isUnknownLength || newShape.length == length) {
      return newShape;
    } else {
      throw new ArgumentError.value(
          newShape.length, "new shape length", "Must be $length");
    }
  }

  @override
  NDShape tile(List<int> multiplies) {
    if (isScalar) {
      throw new StateError("Can't tile a scalar");
    } else if (multiplies == null || multiplies.length != dimension) {
      throw new ArgumentError.value(
          multiplies, "tile multiplis", "Must be $dimension length");
    }

    var newDimensions = new List.generate(
        dimension, (index) => multiplies[index] * dimensions[index]);

    return new NDShapeImpl(newDimensions);
  }

  @override
  String toString() =>
      "<Shape: dimensions=$internalDimensions, dimension=$dimension, length=$length>";

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
