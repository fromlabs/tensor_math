// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "nd_shape.dart";

import "nd_array_impl.dart";

export "nd_array_impl.dart" show adds;

abstract class NDArray {
  factory NDArray(value) => new NDArrayImpl(value);

  NDShape get shape;

  dynamic toValue();

  E toScalar<E>();

  List<E> toVector<E>();

  List<List<E>> toMatrix<E>();

  List<List<List<E>>> toTensor3D<E>();

  List<List<List<List<E>>>> toTensor4D<E>();

  NDArray reshape({List<int> newDimensions});

  NDArray transpose({List<int> permutationAxis});

  NDArray matMul(value2);

  NDArray reduceSum({List<int> reductionAxis});

  NDArray reduceMean({List<int> reductionAxis});

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

  NDArray equals(value2);

  NDArray notEquals(value2);

  NDArray greater(value2);

  NDArray greaterOrEquals(value2);

  NDArray less(value2);

  NDArray lessOrEquals(value2);

  NDArray select(thenValue, elseValue);

  bool any();

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
