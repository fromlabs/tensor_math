// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "nd_data_type.dart";
import "nd_shape.dart";
import "nd_descriptor.dart";

abstract class NDObject {
  NDDescriptor get descriptor;

  NDDataType get dataType;

  NDShape get shape;

  NDObject normalize({NDObject reuse});

  NDObject cast(NDDataType toDataType, {NDObject reuse});

  NDObject reshape({List<int> newDimensions, NDObject reuse});

  NDObject tile(List<int> multiplies, {NDObject reuse});

  NDObject transpose({List<int> permutationAxis, NDObject reuse});

  NDObject matMul(value2, {NDObject reuse});

  NDObject reduceSum(
      {List<int> reductionAxis, bool keepDimensions = false, NDObject reuse});

  NDObject reduceMean(
      {List<int> reductionAxis, bool keepDimensions = false, NDObject reuse});

  NDObject reduceMax(
      {List<int> reductionAxis, bool keepDimensions = false, NDObject reuse});

  NDObject reduceAny(
      {List<int> reductionAxis, bool keepDimensions = false, NDObject reuse});

  NDObject argMax({int axis = 0, NDObject reuse});

  NDObject abs({NDObject reuse});

  NDObject exp({NDObject reuse});

  NDObject sqrt({NDObject reuse});

  NDObject pow(num exponent, {NDObject reuse});

  NDObject reciprocal({NDObject reuse});

  NDObject log({NDObject reuse});

  NDObject neg({NDObject reuse});

  NDObject sign({NDObject reuse});

  NDObject add(value2, {NDObject reuse});

  NDObject sub(value2, {NDObject reuse});

  NDObject mul(value2, {NDObject reuse});

  NDObject div(value2, {NDObject reuse});

  NDObject not({NDObject reuse});

  NDObject isEqual(value2, {NDObject reuse});

  NDObject isNotEqual(value2, {NDObject reuse});

  NDObject isGreater(value2, {NDObject reuse});

  NDObject isGreaterOrEqual(value2, {NDObject reuse});

  NDObject isLess(value2, {NDObject reuse});

  NDObject isLessOrEqual(value2, {NDObject reuse});

  NDObject select(thenValue, elseValue, {NDObject reuse});

  NDObject operator -();

  NDObject operator +(value2);

  NDObject operator -(value2);

  NDObject operator *(value2);

  NDObject operator /(value2);

  NDObject operator >(value2);

  NDObject operator >=(value2);

  NDObject operator <(value2);

  NDObject operator <=(value2);

  NDObject elementWiseUnaryOperation(
      {@required NDDataType resultDataType,
      NDObject reuse,
      @required unaryOperation(value, int valueCount)});

  NDObject elementWiseBinaryOperation(value2,
      {NDDataType dataType2,
      @required NDDataType resultDataType,
      NDObject reuse,
      @required binaryOperation(value1, value2, int valueCount)});

  NDObject elementWiseTernaryOperation(value2, value3,
      {NDDataType dataType2,
      NDDataType dataType3,
      @required NDDataType resultDataType,
      NDObject reuse,
      @required ternaryOperation(value1, value2, value3, int valueCount)});

  NDObject reduceOperation(
      {List<int> reductionAxis,
      bool keepDimensions = false,
      NDObject reuse,
      @required void begin(),
      @required void onValue(value, int valueCount),
      @required dynamic end()});

  NDObject argOperation(
      {int axis = 0,
      NDObject reuse,
      @required void begin(),
      @required void onValue(dimensionIndex, value, int valueCount),
      @required dynamic end()});

  NDObject oneHot(
      {int axis = 0,
      @required int dimensionCount,
      @required NDDataType resultDataType,
      NDObject reuse});

  NDObject im2col(
      {int blockHeight,
      int blockWidth,
      int heightStride = 1,
      int widthStride = 1,
      bool keepInputDepth = false,
      NDObject reuse});

  NDObject col2im(
      {List<int> imageDimensions,
      int blockHeight,
      int blockWidth,
      int heightStride = 1,
      int widthStride = 1,
      NDObject reuse});
}
