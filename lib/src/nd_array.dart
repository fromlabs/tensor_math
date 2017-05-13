// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "nd_shapeable.dart";

import "nd_shape_impl.dart";
import "nd_array_impl.dart";

export "nd_array_impl.dart" show adds;

abstract class NDArray implements NDShapeable {
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

  factory NDArray.generate(List<int> dimensions, dynamic generator(int index)) {
    var shape = new NDShapeImpl(dimensions);

    return new NDArrayImpl(new List.generate(shape.length, generator))
        .reshape(newDimensions: dimensions);
  }

  dynamic toValue();

  E toScalar<E>();

  List<E> toVector<E>();

  List<List<E>> toMatrix<E>();

  List<List<List<E>>> toTensor3D<E>();

  List<List<List<List<E>>>> toTensor4D<E>();

  @override
  NDArray reshape({List<int> newDimensions});

  @override
  NDArray tile(List<int> multiplies);

  @override
  NDArray transpose({List<int> permutationAxis});

  @override
  NDArray matMul(value2);

  @override
  NDArray reduceSum({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDArray reduceMean({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDArray reduceMax({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDArray reduceAny({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDArray argMax({int axis});

  @override
  NDArray abs();

  @override
  NDArray exp();

  @override
  NDArray inv();

  @override
  NDArray log();

  @override
  NDArray neg();

  @override
  NDArray sign();

  @override
  NDArray add(value2);

  @override
  NDArray sub(value2);

  @override
  NDArray mul(value2);

  @override
  NDArray div(value2);

  @override
  NDArray not();

  @override
  NDArray isEqual(value2);

  @override
  NDArray isNotEqual(value2);

  @override
  NDArray isGreater(value2);

  @override
  NDArray isGreaterOrEqual(value2);

  @override
  NDArray isLess(value2);

  @override
  NDArray isLessOrEqual(value2);

  @override
  NDArray select(thenValue, elseValue);

  @override
  NDArray operator -();

  @override
  NDArray operator +(value2);

  @override
  NDArray operator -(value2);

  @override
  NDArray operator *(value2);

  @override
  NDArray operator /(value2);

  @override
  NDArray operator >(value2);

  @override
  NDArray operator >=(value2);

  @override
  NDArray operator <(value2);

  @override
  NDArray operator <=(value2);
}
