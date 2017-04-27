// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "nd_shape.dart";

import "nd_array_impl.dart";

abstract class NDArray {
  factory NDArray(value) => new NDArrayImpl(value);

  NDShape get shape;

  dynamic toValue();

  E toScalar<E>();

  List<E> toVector<E>();

  List<List<E>> toMatrix<E>();

  List<List<List<E>>> toTensor3D<E>();

  List<List<List<List<E>>>> toTensor4D<E>();

  NDArray transpose({List<int> permutationAxis});

  NDArray matMul(value2);

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

  NDArray equal(value2);

  NDArray notEqual(value2);

  NDArray greater(value2);

  NDArray greaterEqual(value2);

  NDArray less(value2);

  NDArray lessEqual(value2);

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
