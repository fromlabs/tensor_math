// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "nd_shape.dart";

import "nd_shape_impl.dart";
import "nd_array_impl.dart";

export "nd_array_impl.dart" show adds;

abstract class NDArray {
  factory NDArray(value) => new NDArrayImpl(value);

  factory NDArray.zeros(List<int> dimensions) {
    var shape = new NDShapeImpl(dimensions);

    return new NDArrayImpl(new List.filled(shape.length, 0))
        .reshape(newDimensions: dimensions);
  }

  factory NDArray.ones(List<int> dimensions) {
    var shape = new NDShapeImpl(dimensions);

    return new NDArrayImpl(new List.filled(shape.length, 1))
        .reshape(newDimensions: dimensions);
  }

  factory NDArray.filled(List<int> dimensions, dynamic value) {
    var shape = new NDShapeImpl(dimensions);

    return new NDArrayImpl(new List.filled(shape.length, value))
        .reshape(newDimensions: dimensions);
  }

  NDShape get shape;

  dynamic toValue();

  E toScalar<E>();

  List<E> toVector<E>();

  List<List<E>> toMatrix<E>();

  List<List<List<E>>> toTensor3D<E>();

  List<List<List<List<E>>>> toTensor4D<E>();

  NDArray reshape({List<int> newDimensions});

  NDArray tile(List<int> multiplies);

  NDArray transpose({List<int> permutationAxis});

  NDArray matMul(value2);

  NDArray reduceSum({List<int> reductionAxis, bool keepDimensions = false});

  NDArray reduceMean({List<int> reductionAxis, bool keepDimensions = false});

  NDArray reduceMax({List<int> reductionAxis, bool keepDimensions = false});

  NDArray reduceAny({List<int> reductionAxis, bool keepDimensions = false});

  NDArray argmax({int axis});

  NDArray abs();

  NDArray exp();

  NDArray inv();

  NDArray log();

  NDArray neg();

  NDArray sign();

  NDArray add(value2);

  NDArray sub(value2);

  NDArray mul(value2);

  NDArray div(value2);

  NDArray not();

  NDArray isEquals(value2);

  NDArray isNotEquals(value2);

  NDArray isGreater(value2);

  NDArray isGreaterOrEquals(value2);

  NDArray isLess(value2);

  NDArray isLessOrEquals(value2);

  NDArray select(thenValue, elseValue);

  NDArray operator -();

  NDArray operator +(value2);

  NDArray operator -(value2);

  NDArray operator *(value2);

  NDArray operator /(value2);

  NDArray operator >(value2);

  NDArray operator >=(value2);

  NDArray operator <(value2);

  NDArray operator <=(value2);
}
