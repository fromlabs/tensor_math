// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "nd_object.dart";

import "nd_data_type.dart";
import "nd_array_base.dart";

export "nd_array_base.dart" show adds, toNDArray;

abstract class NDArray implements NDObject {
  factory NDArray(value, {NDDataType dataType, NDArray reuse}) =>
      new NDArrayBase(value, dataType, reuse);

  factory NDArray.zeros(List<int> dimensions,
          {NDDataType dataType, NDArray reuse}) =>
      new NDArrayBase.filled(dimensions, 0, dataType, reuse);

  factory NDArray.ones(List<int> dimensions,
          {NDDataType dataType, NDArray reuse}) =>
      new NDArrayBase.filled(dimensions, 1, dataType, reuse);

  factory NDArray.filled(List<int> dimensions, dynamic fillValue,
          {NDDataType dataType, NDArray reuse}) =>
      new NDArrayBase.filled(dimensions, fillValue, dataType, reuse);

  factory NDArray.generate(List<int> dimensions, dynamic generator(int index),
          {NDDataType dataType, NDArray reuse}) =>
      new NDArrayBase.generate(dimensions, generator, dataType, reuse);

  bool get isNormalized;

  @override
  NDArray normalize({covariant NDArray reuse});

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
  NDArray argMax({int axis = 0, covariant NDArray reuse});

  @override
  NDArray reduceOperation(
      {List<int> reductionAxis,
      bool keepDimensions = false,
      covariant NDArray reuse,
      @required void begin(),
      @required void onValue(value, int valueCount),
      @required dynamic end()});

  @override
  NDArray argOperation(
      {int axis = 0,
      covariant NDArray reuse,
      @required void begin(),
      @required void onValue(dimensionIndex, value, int valueCount),
      @required dynamic end()});

  @override
  NDArray abs({covariant NDArray reuse});

  @override
  NDArray exp({covariant NDArray reuse});

  @override
  NDArray reciprocal({covariant NDArray reuse});

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
