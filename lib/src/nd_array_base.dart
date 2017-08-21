// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as math;

import 'dart:typed_data';
import "package:meta/meta.dart";

import 'nd_descriptor.dart';
import 'nd_shape.dart';
import 'nd_data_type.dart';
import "nd_array.dart";

import "nd_array_impl.dart";
import "nd_array_blocked_impl2.dart";

NDArray createTestNDArray(List<int> dimensions, {NDDataType dataType}) =>
    new NDArray.generate(dimensions, (index) => index, dataType: dataType);

NDArray adds(Iterable values, {NDDataType dataType, NDArray reuse}) {
  // TODO versione ottimizzata di adds

  // TODO sfruttare reuse se possibile

  var arrays = values
      .map<NDArray>((value) => toNDArray(value, dataType: dataType))
      .toList();

  if (arrays.length > 1) {
    return arrays.reduce((total, element) => total + element);
  } else {
    return arrays.first;
  }
}

NDArray toNDArray(value, {NDDataType dataType}) {
  if (value is NDArray) {
    if (dataType != null && value.dataType != dataType) {
      throw new UnsupportedError(
          "NDArray(${value.dataType}) != NDArray($dataType)");
    }

    return value;
  } else {
    return new NDArray(value, dataType: dataType);
  }
}

abstract class NDArrayBase implements NDArray {
  @override
  final NDDescriptor descriptor;

  factory NDArrayBase(value, NDDataType dataType, NDArray reuse) {
    var shape = _calculateShape(value);
    var newDataType = dataType ?? _calculateDataType(value);
    var descriptor = new NDDescriptor(shape: shape, dataType: newDataType);

    if (newDataType.isBlocked) {
      return new NDArrayBlockedImpl(value, descriptor, reuse);
    } else {
      return new NDArrayImpl(value, descriptor, reuse);
    }
  }

  factory NDArrayBase.filled(
      List<int> dimensions, fillValue, NDDataType dataType, NDArray reuse) {
    var shape = new NDShape(dimensions);
    var newDataType = dataType ?? _calculateDataType(fillValue);
    var descriptor = new NDDescriptor(shape: shape, dataType: newDataType);

    if (newDataType.isBlocked) {
      return new NDArrayBlockedImpl.filled(fillValue, descriptor, reuse);
    } else {
      return new NDArrayImpl.filled(fillValue, descriptor, reuse);
    }
  }

  factory NDArrayBase.generate(List<int> dimensions, Function generator,
      NDDataType dataType, NDArray reuse) {
    var shape = new NDShape(dimensions);
    var newDataType = dataType ?? _calculateDataType(generator(0));
    var descriptor = new NDDescriptor(shape: shape, dataType: newDataType);

    if (newDataType.isBlocked) {
      return new NDArrayBlockedImpl.generate(generator, descriptor, reuse);
    } else {
      return new NDArrayImpl.generate(generator, descriptor, reuse);
    }
  }

  factory NDArrayBase.castFrom(
      NDArrayBase fromArray, NDDataType toDataType, NDArray reuse) {
    if (fromArray.dataType == toDataType) {
      return fromArray;
    } else if (toDataType.isBlocked) {
      return new NDArrayBlockedImpl.castFrom(fromArray, toDataType, reuse);
    } else {
      return new NDArrayImpl.castFrom(fromArray, toDataType, reuse);
    }
  }

  NDArrayBase.raw(this.descriptor);

  Iterable get valueIterable;

  NDArrayBase elementWiseUnaryOperationInternal(
      NDDescriptor resultDescriptor, NDArray reuse, unaryOperation(value));

  NDArrayBase elementWiseBinaryOperationInternal(
      NDArrayBase array2,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      binaryOperation(value1, value2));

  NDArrayBase elementWiseTernaryOperationInternal(
      NDArrayBase array2,
      NDArrayBase array3,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      ternaryOperation(value1, value2, value3));

  NDArray reduceOperationInternal(List<int> reductionAxis, bool keepDimensions,
      NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(), void onValue(value, int valueCount), dynamic end()});

  NDArray argOperationInternal(
      int axis, NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(),
      void onValue(dimensionIndex, value, int valueCount),
      dynamic end()});

  @override
  NDDataType get dataType => descriptor.dataType;

  @override
  NDShape get shape => descriptor.shape;

  @override
  E toScalar<E>() {
    if (shape.isScalar) {
      return toValue();
    } else {
      throw new StateError("Not a scalar (shape: $shape)");
    }
  }

  @override
  List<E> toVector<E>() {
    if (shape.isVector) {
      return toValue();
    } else {
      throw new StateError("Not a vector (shape: $shape)");
    }
  }

  @override
  List<List<E>> toMatrix<E>() {
    if (shape.isMatrix) {
      return toValue();
    } else {
      throw new StateError("Not a matrix (shape: $shape)");
    }
  }

  @override
  List<List<List<E>>> toTensor3D<E>() {
    if (shape.isTensor3D) {
      return toValue();
    } else {
      throw new StateError("Not a 3-dimensional tensor (shape: $shape)");
    }
  }

  @override
  List<List<List<List<E>>>> toTensor4D<E>() {
    if (shape.isTensor4D) {
      return toValue();
    } else {
      throw new StateError("Not a 4-dimensional tensor (shape: $shape)");
    }
  }

  @override
  NDArray abs({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.abs(), reuse, (value) => value.abs());

  @override
  NDArray exp({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.exp(), reuse, (value) => math.exp(value));

  @override
  NDArray inv({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.inv(), reuse, (value) => 1 / value);

  @override
  NDArray log({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.log(), reuse, (value) => math.log(value));

  @override
  NDArray neg({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.neg(), reuse, (value) => -value);

  @override
  NDArray sign({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.sign(), reuse, (value) => value.sign());

  @override
  NDArray not({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.not(), reuse, (value) => !value);

  @override
  NDArray add(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.add(array2.descriptor),
        reuse,
        (value1, value2) => value1 + value2);
  }

  @override
  NDArray div(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.div(array2.descriptor),
        reuse,
        (value1, value2) => value1 / value2);
  }

  @override
  NDArray isGreater(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.isGreater(array2.descriptor),
        reuse,
        (value1, value2) => value1 > value2);
  }

  @override
  NDArray isGreaterOrEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.isGreaterOrEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 >= value2);
  }

  @override
  NDArray isLess(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.isLess(array2.descriptor),
        reuse,
        (value1, value2) => value1 < value2);
  }

  @override
  NDArray isLessOrEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.isLessOrEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 <= value2);
  }

  @override
  NDArray mul(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.mul(array2.descriptor),
        reuse,
        (value1, value2) => value1 * value2);
  }

  @override
  NDArray isEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.isEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 == value2);
  }

  @override
  NDArray isNotEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.isNotEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 != value2);
  }

  @override
  NDArray sub(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.sub(array2.descriptor),
        reuse,
        (value1, value2) => value1 - value2);
  }

  @override
  NDArray operator *(value2) => mul(value2);

  @override
  NDArray operator +(value2) => add(value2);

  @override
  NDArray operator -() => neg();

  @override
  NDArray operator -(value2) => sub(value2);

  @override
  NDArray operator /(value2) => div(value2);

  @override
  NDArray operator <(value2) => isLess(value2);

  @override
  NDArray operator <=(value2) => isLessOrEqual(value2);

  @override
  NDArray operator >(value2) => isGreater(value2);

  @override
  NDArray operator >=(value2) => isGreaterOrEqual(value2);

  @override
  NDArray select(thenValue, elseValue, {NDDataType dataType, NDArray reuse}) {
    var thenArray = toNDArray(thenValue, dataType: dataType);
    var elseArray = toNDArray(elseValue, dataType: dataType);

    var resultDescriptor =
        descriptor.select(thenArray.descriptor, elseArray.descriptor);

    return elementWiseTernaryOperationInternal(
        thenArray,
        elseArray,
        resultDescriptor,
        reuse,
        (value1, value2, value3) => value1 ? value2 : value3);
  }

  @override
  NDArray cast(NDDataType toDataType, {NDArray reuse}) =>
      new NDArrayBase.castFrom(this, toDataType, reuse);

  @override
  NDArray elementWiseUnaryOperation(
          {NDDataType resultDataType,
          covariant NDArray reuse,
          unaryOperation(value)}) =>
      elementWiseUnaryOperationInternal(
          descriptor.elementWiseUnaryOperation(resultDataType: resultDataType),
          reuse,
          unaryOperation);

  @override
  NDArray elementWiseBinaryOperation(covariant NDArray value2,
      {NDDataType dataType2,
      @required NDDataType resultDataType,
      covariant NDArray reuse,
      binaryOperation(value1, value2)}) {
    var array2 = toNDArray(value2, dataType: dataType2);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.elementWiseBinaryOperation(array2.descriptor,
            resultDataType: resultDataType),
        reuse,
        binaryOperation);
  }

  @override
  NDArray elementWiseTernaryOperation(
      covariant NDArray value2, covariant NDArray value3,
      {NDDataType dataType2,
      NDDataType dataType3,
      NDDataType resultDataType,
      covariant NDArray reuse,
      ternaryOperation(value1, value2, value3)}) {
    var array2 = toNDArray(value2, dataType: dataType2);
    var array3 = toNDArray(value3, dataType: dataType3);

    return elementWiseTernaryOperationInternal(
        array2,
        array3,
        descriptor.elementWiseTernaryOperation(
            array2.descriptor, array3.descriptor,
            resultDataType: resultDataType),
        reuse,
        ternaryOperation);
  }

  @override
  NDArray reduceOperation(
      {List<int> reductionAxis,
      bool keepDimensions = false,
      covariant NDArray reuse,
      @required void begin(),
      @required void onValue(value, int valueCount),
      @required dynamic end()}) {
    var resultDescriptor = descriptor.reduceOperation(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: begin, onValue: onValue, end: end);
  }

  @override
  NDArray argOperation(
      {int axis = 0,
      covariant NDArray reuse,
      @required void begin(),
      @required void onValue(dimensionIndex, value, int valueCount),
      @required dynamic end()}) {
    var resultDescriptor = descriptor.argOperation(axis: axis);

    return argOperationInternal(axis, resultDescriptor, reuse,
        begin: begin, onValue: onValue, end: end);
  }
}

NDShape _calculateShape(value) {
  var dimensions = [];
  dynamic element = value;
  while (element is List) {
    dimensions.add(element.length);
    element = element[0];
  }
  return new NDShape(dimensions);
}

NDDataType _calculateDataType(value) {
  dynamic firstValue = value;
  while (firstValue is List) {
    firstValue = firstValue[0];
  }

  if (firstValue is double) {
    return NDDataType.float32;
  } else if (firstValue is int) {
    return NDDataType.int32;
  } else if (firstValue is bool) {
    return NDDataType.boolean;
  } else if (firstValue is String) {
    return NDDataType.string;
  } else {
    return NDDataType.generic;
  }
}
