// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "nd_object.dart";

import "nd_shape.dart";
import "nd_data_type.dart";
import "nd_array_impl.dart";

export "nd_array_impl.dart" show adds, toNDArray;

abstract class NDArray implements NDObject {
  factory NDArray(value, {NDDataType dataType, NDArray reuse}) =>
      new NDArrayImpl(value, dataType, reuse);

  factory NDArray.zeros(List<int> dimensions,
      {NDDataType dataType, NDArray reuse}) {
    var shape = new NDShape(dimensions);

    return new NDArrayImpl(new List.filled(shape.length, 0), dataType, reuse)
        .reshape(newDimensions: dimensions);
  }

  factory NDArray.ones(List<int> dimensions,
      {NDDataType dataType, NDArray reuse}) {
    var shape = new NDShape(dimensions);

    return new NDArrayImpl(new List.filled(shape.length, 1), dataType, reuse)
        .reshape(newDimensions: dimensions);
  }

  factory NDArray.filled(List<int> dimensions, dynamic value,
      {NDDataType dataType, NDArray reuse}) {
    var shape = new NDShape(dimensions);

    return new NDArrayImpl(
            new List.filled(shape.length, value), dataType, reuse)
        .reshape(newDimensions: dimensions);
  }

  factory NDArray.generate(List<int> dimensions, dynamic generator(int index),
      {NDDataType dataType, NDArray reuse}) {
    var shape = new NDShape(dimensions);

    return new NDArrayImpl(
            new List.generate(shape.length, generator), dataType, reuse)
        .reshape(newDimensions: dimensions);
  }

  dynamic toValue();

  E toScalar<E>();

  List<E> toVector<E>();

  List<List<E>> toMatrix<E>();

  List<List<List<E>>> toTensor3D<E>();

  List<List<List<List<E>>>> toTensor4D<E>();

  @override
  NDArray cast(NDDataType toDataType, {covariant NDArray reuse});

  @override
  NDArray reshape({List<int> newDimensions, covariant NDArray reuse});

  @override
  NDArray tile(List<int> multiplies, {covariant NDArray reuse});

  @override
  NDArray transpose({List<int> permutationAxis, covariant NDArray reuse});

  @override
  NDArray matMul(value2, {covariant NDArray reuse});

  @override
  NDArray reduceSum(
      {List<int> reductionAxis,
      bool keepDimensions = false,
      covariant NDArray reuse});

  @override
  NDArray reduceMean(
      {List<int> reductionAxis,
      bool keepDimensions = false,
      covariant NDArray reuse});

  @override
  NDArray reduceMax(
      {List<int> reductionAxis,
      bool keepDimensions = false,
      covariant NDArray reuse});

  @override
  NDArray reduceAny(
      {List<int> reductionAxis,
      bool keepDimensions = false,
      covariant NDArray reuse});

  @override
  NDArray argMax({int axis, covariant NDArray reuse});

  @override
  NDArray abs({covariant NDArray reuse});

  @override
  NDArray exp({covariant NDArray reuse});

  @override
  NDArray inv({covariant NDArray reuse});

  @override
  NDArray log({covariant NDArray reuse});

  @override
  NDArray neg({covariant NDArray reuse});

  @override
  NDArray sign({covariant NDArray reuse});

  @override
  NDArray add(value2, {covariant NDArray reuse});

  @override
  NDArray sub(value2, {covariant NDArray reuse});

  @override
  NDArray mul(value2, {covariant NDArray reuse});

  @override
  NDArray div(value2, {covariant NDArray reuse});

  @override
  NDArray not({covariant NDArray reuse});

  @override
  NDArray isEqual(value2, {covariant NDArray reuse});

  @override
  NDArray isNotEqual(value2, {covariant NDArray reuse});

  @override
  NDArray isGreater(value2, {covariant NDArray reuse});

  @override
  NDArray isGreaterOrEqual(value2, {covariant NDArray reuse});

  @override
  NDArray isLess(value2, {covariant NDArray reuse});

  @override
  NDArray isLessOrEqual(value2, {covariant NDArray reuse});

  @override
  NDArray select(thenValue, elseValue,
      {NDDataType dataType, covariant NDArray reuse});

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
