// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "nd_data_type.dart";
import 'nd_shape.dart';
import "nd_object.dart";

NDDescriptor addDescriptors(Iterable<NDDescriptor> descriptors) =>
    descriptors.reduce((total, element) => total + element);

class NDDescriptor implements NDObject {
  @override
  final NDShape shape;

  @override
  final NDDataType dataType;

  NDDescriptor({NDShape shape, NDDataType dataType})
      : this.shape = shape ?? new NDShape(),
        this.dataType = dataType ?? NDDataType.unknown;

  @override
  NDDescriptor get descriptor => this;

  bool isCompatibleWith(NDDescriptor descriptor2) =>
      dataType.isCompatibleWith(descriptor2.dataType) &&
      shape.isCompatibleWith(descriptor2.shape);

  @override
  // ignore: hash_and_equals
  bool operator ==(other) => other is NDDescriptor
      ? dataType == other.dataType && shape == other.shape
      : false;

  NDDescriptor mergeWith(NDDescriptor descriptor2) => new NDDescriptor(
      shape: shape.mergeWith(descriptor2.shape),
      dataType: dataType.mergeWith(descriptor2.dataType));

  @override
  NDDescriptor normalize({covariant NDDescriptor reuse}) => descriptor;

  @override
  NDDescriptor cast(NDDataType toDataType, {covariant NDDescriptor reuse}) {
    if (!dataType.isCastableTo(toDataType)) {
      throw new UnsupportedError(
          "Cast from NDArray($dataType) to NDArray($toDataType)");
    }

    return new NDDescriptor(shape: shape, dataType: toDataType);
  }

  @override
  NDDescriptor operator *(covariant NDDescriptor descriptor2) =>
      mul(descriptor2);

  @override
  NDDescriptor operator +(covariant NDDescriptor descriptor2) =>
      add(descriptor2);

  @override
  NDDescriptor operator -() => neg();

  @override
  NDDescriptor operator -(covariant NDDescriptor descriptor2) =>
      sub(descriptor2);

  @override
  NDDescriptor operator /(covariant NDDescriptor descriptor2) =>
      div(descriptor2);

  @override
  NDDescriptor operator <(covariant NDDescriptor descriptor2) =>
      isLess(descriptor2);

  @override
  NDDescriptor operator <=(covariant NDDescriptor descriptor2) =>
      isLessOrEqual(descriptor2);

  @override
  NDDescriptor operator >(covariant NDDescriptor descriptor2) =>
      isGreater(descriptor2);

  @override
  NDDescriptor operator >=(covariant NDDescriptor descriptor2) =>
      isGreaterOrEqual(descriptor2);

  @override
  NDDescriptor argMax({int axis = 0, covariant NDDescriptor reuse}) =>
      arg(axis: axis);

  NDDescriptor broadcast(covariant NDDescriptor descriptor2,
          {@required NDDataType resultDataType}) =>
      new NDDescriptor(
          shape: shape.broadcast(descriptor2.shape), dataType: resultDataType);

  @override
  NDDescriptor tile(List<int> multiplies, {covariant NDDescriptor reuse}) =>
      new NDDescriptor(shape: shape.tile(multiplies), dataType: dataType);

  @override
  NDDescriptor transpose(
          {List<int> permutationAxis, covariant NDDescriptor reuse}) =>
      new NDDescriptor(
          shape: shape.transpose(permutationAxis: permutationAxis),
          dataType: dataType);

  @override
  NDDescriptor matMul(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    } else if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return new NDDescriptor(
        shape: shape.matMul(descriptor2.shape), dataType: dataType);
  }

  @override
  NDDescriptor not({covariant NDDescriptor reuse}) {
    if (!dataType.isBoolean) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only boolean types");
    }

    return this;
  }

  @override
  NDDescriptor isEqual(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    }

    return new NDDescriptor(
        shape: shape.broadcast(descriptor2.shape),
        dataType: dataType.isBlocked
            ? NDDataType.booleanBlocked
            : NDDataType.boolean);
  }

  @override
  NDDescriptor isGreater(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    }

    return new NDDescriptor(
        shape: shape.broadcast(descriptor2.shape),
        dataType: dataType.isBlocked
            ? NDDataType.booleanBlocked
            : NDDataType.boolean);
  }

  @override
  NDDescriptor isGreaterOrEqual(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    }

    return new NDDescriptor(
        shape: shape.broadcast(descriptor2.shape),
        dataType: dataType.isBlocked
            ? NDDataType.booleanBlocked
            : NDDataType.boolean);
  }

  @override
  NDDescriptor isLess(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    }

    return new NDDescriptor(
        shape: shape.broadcast(descriptor2.shape),
        dataType: dataType.isBlocked
            ? NDDataType.booleanBlocked
            : NDDataType.boolean);
  }

  @override
  NDDescriptor isLessOrEqual(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    }

    return new NDDescriptor(
        shape: shape.broadcast(descriptor2.shape),
        dataType: dataType.isBlocked
            ? NDDataType.booleanBlocked
            : NDDataType.boolean);
  }

  @override
  NDDescriptor isNotEqual(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    }

    return new NDDescriptor(
        shape: shape.broadcast(descriptor2.shape),
        dataType: dataType.isBlocked
            ? NDDataType.booleanBlocked
            : NDDataType.boolean);
  }

  @override
  NDDescriptor select(covariant NDDescriptor thenDescriptor,
      covariant NDDescriptor elseDescriptor,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isBoolean) {
      throw new UnsupportedError(
          "Select on NDArray(${this.dataType}) condition: supported only boolean types");
    } else if (!thenDescriptor.dataType
        .isCompatibleWith(elseDescriptor.dataType)) {
      throw new UnsupportedError(
          "NDArray(${thenDescriptor.dataType}) is not compatible with NDArray(${elseDescriptor.dataType})");
    }

    return new NDDescriptor(
        shape: shape
            .broadcast(thenDescriptor.shape)
            .broadcast(elseDescriptor.shape),
        dataType: thenDescriptor.dataType);
  }

  @override
  NDDescriptor reshape(
          {List<int> newDimensions, covariant NDDescriptor reuse}) =>
      new NDDescriptor(
          shape: shape.reshape(newDimensions: newDimensions),
          dataType: dataType);

  NDDescriptor reduce(
          {@required NDDataType resultDataType,
          List<int> reductionAxis,
          bool keepDimensions: false}) =>
      new NDDescriptor(
          shape: shape.reduce(
              reductionAxis: reductionAxis, keepDimensions: keepDimensions),
          dataType: resultDataType);

  NDDescriptor arg({int axis = 0}) => new NDDescriptor(
      shape: shape.arg(axis: axis),
      dataType:
          dataType.isBlocked ? NDDataType.int32Blocked : NDDataType.uint32);

  @override
  NDDescriptor reduceAny(
      {List<int> reductionAxis,
      bool keepDimensions: false,
      covariant NDDescriptor reuse}) {
    if (!dataType.isBoolean) {
      throw new UnsupportedError(
          "Reduce any on NDArray($dataType) condition: supported only boolean types");
    }

    return reduce(
        reductionAxis: reductionAxis,
        keepDimensions: keepDimensions,
        resultDataType: dataType.isBlocked
            ? NDDataType.booleanBlocked
            : NDDataType.boolean);
  }

  @override
  NDDescriptor reduceMax(
      {List<int> reductionAxis,
      bool keepDimensions: false,
      covariant NDDescriptor reuse}) {
    if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return reduce(
        reductionAxis: reductionAxis,
        keepDimensions: keepDimensions,
        resultDataType: dataType);
  }

  @override
  NDDescriptor reduceMean(
      {List<int> reductionAxis,
      bool keepDimensions: false,
      covariant NDDescriptor reuse}) {
    if (!dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only float data type");
    }

    return reduce(
        reductionAxis: reductionAxis,
        keepDimensions: keepDimensions,
        resultDataType: dataType);
  }

  @override
  NDDescriptor reduceSum(
      {List<int> reductionAxis,
      bool keepDimensions: false,
      covariant NDDescriptor reuse}) {
    if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return reduce(
        reductionAxis: reductionAxis,
        keepDimensions: keepDimensions,
        resultDataType: dataType);
  }

  @override
  NDDescriptor abs({covariant NDDescriptor reuse}) {
    if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return this;
  }

  @override
  NDDescriptor exp({covariant NDDescriptor reuse}) {
    if (!dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only float data type");
    }

    return this;
  }

  @override
  NDDescriptor reciprocal({covariant NDDescriptor reuse}) {
    if (!dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only float data type");
    }

    return this;
  }

  @override
  NDDescriptor log({covariant NDDescriptor reuse}) {
    if (!dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only float data type");
    }

    return this;
  }

  @override
  NDDescriptor neg({covariant NDDescriptor reuse}) {
    if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return this;
  }

  @override
  NDDescriptor sign({covariant NDDescriptor reuse}) {
    if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return this;
  }

  @override
  NDDescriptor sub(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    } else if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return broadcast(descriptor2, resultDataType: dataType);
  }

  @override
  NDDescriptor mul(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    } else if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return broadcast(descriptor2, resultDataType: dataType);
  }

  @override
  NDDescriptor add(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    } else if (!dataType.isNumeric) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only numeric data type");
    }

    return broadcast(descriptor2, resultDataType: dataType);
  }

  @override
  NDDescriptor div(covariant NDDescriptor descriptor2,
      {covariant NDDescriptor reuse}) {
    if (!this.dataType.isCompatibleWith(descriptor2.dataType)) {
      throw new UnsupportedError(
          "NDArray($dataType) is not compatible with NDArray(${descriptor2.dataType})");
    } else if (!dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only float data type");
    }

    return broadcast(descriptor2, resultDataType: dataType);
  }

  @override
  NDDescriptor elementWiseUnaryOperation(
          {@required NDDataType resultDataType,
          covariant NDDescriptor reuse,
          unaryOperation(value, int valueCount)}) =>
      new NDDescriptor(shape: shape, dataType: resultDataType);

  @override
  NDDescriptor elementWiseBinaryOperation(covariant NDDescriptor descriptor2,
          {NDDataType dataType2,
          @required NDDataType resultDataType,
          covariant NDDescriptor reuse,
          binaryOperation(value1, value2, int valueCount)}) =>
      broadcast(descriptor2, resultDataType: resultDataType);

  @override
  NDDescriptor elementWiseTernaryOperation(covariant NDDescriptor descriptor2,
          covariant NDDescriptor descriptor3,
          {NDDataType dataType2,
          NDDataType dataType3,
          @required NDDataType resultDataType,
          covariant NDDescriptor reuse,
          ternaryOperation(value1, value2, value3, int valueCount)}) =>
      broadcast(descriptor2, resultDataType: resultDataType)
          .broadcast(descriptor3, resultDataType: resultDataType);

  @override
  NDDescriptor reduceOperation(
          {List<int> reductionAxis,
          bool keepDimensions = false,
          @required NDDataType resultDataType,
          covariant NDDescriptor reuse,
          void begin(),
          void onValue(value, int valueCount),
          dynamic end()}) =>
      this.reduce(
          reductionAxis: reductionAxis,
          keepDimensions: keepDimensions,
          resultDataType: resultDataType);

  @override
  NDDescriptor argOperation(
          {int axis = 0,
          covariant NDDescriptor reuse,
          void begin(),
          void onValue(dimensionIndex, value, int valueCount),
          dynamic end()}) =>
      this.arg(axis: axis);

  /*
  @override
  NDDescriptor conv2d(
      {covariant NDDescriptor kernel,
      covariant NDDescriptor bias,
      List<int> strides = const [1, 1],
      covariant NDDescriptor reuse}) {
    if (!dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only float data type");
    }

    if (!kernel.dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray(${kernel.dataType})): supported only float data type");
    }

    if (bias != null && !bias.dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray(${bias.dataType}) is not compatible with NDArray(${bias.dataType})");
    }

    return new NDDescriptor(
        shape: shape.conv2d(
            kernel: kernel.shape, bias: bias?.shape, strides: strides),
        dataType: dataType);
  }

  @override
  NDDescriptor maxPool({List<int> kernelShape, covariant NDDescriptor reuse}) {
    if (!dataType.isFloat) {
      throw new UnsupportedError(
          "NDArray($dataType)): supported only float data type");
    }

    return new NDDescriptor(
        shape: shape.maxPool(kernelShape: kernelShape), dataType: dataType);
  }
*/
  @override
  NDDescriptor oneHot(
      {int axis = 0,
      @required int dimensionCount,
      @required NDDataType resultDataType,
      covariant NDDescriptor reuse}) {
    // TODO controlli

    return new NDDescriptor(
        shape: shape.oneHot(axis: axis, dimensionCount: dimensionCount),
        dataType: resultDataType);
  }

  @override
  NDDescriptor im2col(
      {int blockHeight,
      int blockWidth,
      int vStride = 1,
      int hStride = 1,
      bool keepInputDepth = false,
      covariant NDDescriptor reuse}) {
    // TODO controlli

    return new NDDescriptor(
        shape: shape.im2col(
            blockHeight: blockHeight,
            blockWidth: blockWidth,
            vStride: vStride,
            hStride: hStride,
            keepInputDepth: keepInputDepth),
        dataType: dataType);
  }

  @override
  NDDescriptor col2im(
      {List<int> inputDimensions,
      int blockHeight,
      int blockWidth,
      int vStride = 1,
      int hStride = 1,
      covariant NDDescriptor reuse}) {
    // TODO controlli

    return new NDDescriptor(
        shape: shape.col2im(
            inputDimensions: inputDimensions,
            blockHeight: blockHeight,
            blockWidth: blockWidth,
            vStride: vStride,
            hStride: hStride),
        dataType: dataType);
  }

  @override
  String toString() =>
      "[Descriptor: shape: ${shape.dimensions}, dataType: $dataType]";
}
