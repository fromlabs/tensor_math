// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:typed_data";
import "dart:math" as math;

import "package:collection/collection.dart";

import 'nd_descriptor.dart';
import 'nd_shape.dart';
import 'nd_data_type.dart';
import "nd_array.dart";

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

final _iterableEquality = new IterableEquality();

class NDArrayImpl implements NDArray {
  @override
  final NDDescriptor descriptor;

  final List _data;

  final List<int> _stride;

  final int _offset;

  factory NDArrayImpl(value, NDDataType dataType, NDArray reuse) {
    var shape = _calculateShape(value);
    var newDataType = dataType ?? _calculateDataType(value);
    var descriptor = new NDDescriptor(shape: shape, dataType: newDataType);
    var stride = _calculateDefaultStride(shape);

    var data = _createData(descriptor, reuse);

    _loadData(value, data, descriptor);

    return new NDArrayImpl._(data, descriptor, stride, 0);
  }

  NDArrayImpl._(this._data, this.descriptor, this._stride, this._offset);

  @override
  NDDataType get dataType => descriptor.dataType;

  @override
  NDShape get shape => descriptor.shape;

  @override
  dynamic toValue() => _toValue();

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
  NDArray cast(NDDataType toDataType, {NDArray reuse}) {
    var resultDescriptor = descriptor.cast(toDataType);

    if (dataType == toDataType) {
      return this;
    } else if ((dataType.isFloat && toDataType.isFloat) ||
        (dataType.isInteger && toDataType.isInteger)) {
      return _elementWiseUnaryOperation(
          resultDescriptor, reuse, (num value) => value);
    } else if (dataType.isFloat && toDataType.isInteger) {
      return _elementWiseUnaryOperation(
          resultDescriptor, reuse, (double value) => value.toInt());
    } else if (dataType.isInteger && toDataType.isFloat) {
      return _elementWiseUnaryOperation(
          resultDescriptor, reuse, (int value) => value.toDouble());
    } else if (dataType.isNumeric && toDataType.isBoolean) {
      return _elementWiseUnaryOperation(
          resultDescriptor, reuse, (num value) => value != 0);
    } else if (dataType.isBoolean && dataType.isFloat) {
      return _elementWiseUnaryOperation(
          resultDescriptor, reuse, (bool value) => value ? 1.0 : 0.0);
    } else if (dataType.isBoolean && toDataType.isInteger) {
      return _elementWiseUnaryOperation(
          resultDescriptor, reuse, (bool value) => value ? 1 : 0);
    } else {
      throw new StateError("DEAD CODE");
    }
  }

  @override
  NDArray abs({NDArray reuse}) => _elementWiseUnaryOperation(
      descriptor.abs(), reuse, (value) => value.abs());

  @override
  NDArray exp({NDArray reuse}) => _elementWiseUnaryOperation(
      descriptor.exp(), reuse, (value) => math.exp(value));

  @override
  NDArray inv({NDArray reuse}) =>
      _elementWiseUnaryOperation(descriptor.inv(), reuse, (value) => 1 / value);

  @override
  NDArray log({NDArray reuse}) => _elementWiseUnaryOperation(
      descriptor.log(), reuse, (value) => math.log(value));

  @override
  NDArray neg({NDArray reuse}) =>
      _elementWiseUnaryOperation(descriptor.neg(), reuse, (value) => -value);

  @override
  NDArray sign({NDArray reuse}) => _elementWiseUnaryOperation(
      descriptor.sign(), reuse, (value) => value.sign());

  @override
  NDArray not({NDArray reuse}) =>
      _elementWiseUnaryOperation(descriptor.not(), reuse, (value) => !value);

  @override
  NDArray reshape({List<int> newDimensions, NDArray reuse}) {
    if (_iterableEquality.equals(shape.dimensions, newDimensions)) {
      return this;
    } else {
      var resultDescriptor = descriptor.reshape(newDimensions: newDimensions);

      var identity =
      _elementWiseUnaryOperation(resultDescriptor, reuse, (value) => value);

      var resultStride = _calculateDefaultStride(resultDescriptor.shape);

      return new NDArrayImpl._(identity._data, resultDescriptor, resultStride, 0);
    }
  }

  @override
  NDArray tile(List<int> multiplies, {NDArray reuse}) {
    var resultDescriptor = descriptor.tile(multiplies);
    var resultShape = resultDescriptor.shape;
    var resultData = _createData(resultDescriptor, reuse);
    var resultStride = _calculateDefaultStride(resultShape);

    var shapeIndex = 0;
    var dimensionIndexes = new List(resultShape.dimension);
    var dimensionSourceIndexes = new List(resultShape.dimension);
    var data1Indexes = new List(resultShape.dimension);
    var startData1Indexes = new List(resultShape.dimension);
    var stride1 = _stride;
    var data1Index = data1Indexes[shapeIndex] = _offset;
    var startData1Index = startData1Indexes[shapeIndex] = data1Index;
    var resultDataIndex = 0;
    var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
    var dimensionSourceIndex = dimensionSourceIndexes[shapeIndex] = 0;
    while (resultDataIndex < resultData.length) {
      if (dimensionIndex < resultShape[shapeIndex]) {
        if (shapeIndex == resultShape.dimension - 1) {
          resultData[resultDataIndex++] = _data[data1Index];
          dimensionIndex++;
          if (dimensionSourceIndex < shape[shapeIndex] - 1) {
            data1Index += stride1[shapeIndex];
            dimensionSourceIndex++;
          } else {
            data1Index = startData1Index;
            dimensionSourceIndex = dimensionSourceIndexes[shapeIndex] = 0;
          }
        } else {
          shapeIndex++;
          data1Indexes[shapeIndex] = data1Index;
          startData1Index = startData1Indexes[shapeIndex] = data1Index;
          dimensionIndex = dimensionIndexes[shapeIndex] = 0;
          dimensionSourceIndex = dimensionSourceIndexes[shapeIndex] = 0;
        }
      } else {
        shapeIndex--;
        dimensionIndex =
            dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        if (dimensionSourceIndexes[shapeIndex] < shape[shapeIndex] - 1) {
          data1Index = data1Indexes[shapeIndex] =
              data1Indexes[shapeIndex] + stride1[shapeIndex];
          dimensionSourceIndex = dimensionSourceIndexes[shapeIndex] =
              dimensionSourceIndexes[shapeIndex] + 1;
        } else {
          data1Index = data1Indexes[shapeIndex] = startData1Indexes[shapeIndex];
          dimensionSourceIndex = dimensionSourceIndexes[shapeIndex] = 0;
        }
      }
    }

    return new NDArrayImpl._(resultData, resultDescriptor, resultStride, 0);
  }

  @override
  NDArray transpose({List<int> permutationAxis, NDArray reuse}) {
    var resultDescriptor =
        descriptor.transpose(permutationAxis: permutationAxis);

    var newPermutationAxis = permutationAxis ??
        new List.generate(
            shape.dimension, (index) => shape.dimension - index - 1);

    var resultStride = new List(shape.dimension);

    for (var i = 0; i < newPermutationAxis.length; i++) {
      var permutationAxe = newPermutationAxis[i];
      resultStride[i] = _stride[permutationAxe];
    }

    return new NDArrayImpl._(_data, resultDescriptor, resultStride, _offset);
  }

  @override
  NDArray matMul(value2, {NDArray reuse}) {
    NDArrayImpl array2 = toNDArray(value2, dataType: dataType);

    var resultDescriptor = descriptor.matMul(array2.descriptor);
    var resultShape = resultDescriptor.shape;
    var resultData = _createData(resultDescriptor, reuse);
    var resultStride = _calculateDefaultStride(resultShape);

    var shapeIndex = 0;
    var dimensionIndexes = new List(resultShape.dimension);
    var data1Indexes = new List(resultShape.dimension);
    var data1Index = data1Indexes[shapeIndex] = _offset;
    var data2Indexes = new List(resultShape.dimension);
    var data2Index = data2Indexes[shapeIndex] = array2._offset;
    var resultDataIndex = 0;
    var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
    while (resultDataIndex < resultData.length) {
      if (dimensionIndex < resultShape[shapeIndex]) {
        if (shapeIndex == resultShape.dimension - 2) {
          var shapeDimension1 = shape[shapeIndex];
          var shapeDimensionInternal = shape[shapeIndex + 1];
          var shapeDimension2 = array2.shape[shapeIndex + 1];
          var stride1 = _stride[shapeIndex];
          var stride1Internal = _stride[shapeIndex + 1];
          var stride2Internal = array2._stride[shapeIndex];
          var stride2 = array2._stride[shapeIndex + 1];

          for (var row1Index = 0, rowData1Index = data1Index;
              row1Index < shapeDimension1;
              row1Index++, rowData1Index += stride1) {
            for (var column2Index = 0, columnData2Index = data2Index;
                column2Index < shapeDimension2;
                column2Index++, columnData2Index += stride2) {
              var sumValue = 0;
              var innerIndex = 0;
              var data1Index = rowData1Index;
              var data2Index = columnData2Index;

              while (innerIndex++ < shapeDimensionInternal) {
                sumValue += _data[data1Index] * array2._data[data2Index];

                data1Index += stride1Internal;
                data2Index += stride2Internal;
              }

              resultData[resultDataIndex++] = sumValue;
            }
          }

          dimensionIndex = shapeDimension1;
        } else {
          shapeIndex++;
          data1Indexes[shapeIndex] = data1Index;
          data2Indexes[shapeIndex] = data2Index;
          dimensionIndex = dimensionIndexes[shapeIndex] = 0;
        }
      } else {
        shapeIndex--;
        data1Index = data1Indexes[shapeIndex] =
            data1Indexes[shapeIndex] + _stride[shapeIndex];
        data2Index = data2Indexes[shapeIndex] =
            data2Indexes[shapeIndex] + array2._stride[shapeIndex];
        dimensionIndex =
            dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
      }
    }

    return new NDArrayImpl._(resultData, resultDescriptor, resultStride, 0);
  }

  @override
  NDArray reduceSum(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceSum(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var total;

    return _reduceOperation(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        initReduction: () {
          total = 0;
        },
        onValueToReduce: (int valueIndex, value) {
          total += value;
        },
        reduce: () => total);
  }

  @override
  NDArray reduceMean(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceMean(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var total;
    var count;

    return _reduceOperation(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        initReduction: () {
          total = 0;
          count = 0;
        },
        onValueToReduce: (int valueIndex, value) {
          total += value;
          count++;
        },
        reduce: () => total / count);
  }

  @override
  NDArray reduceMax(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceMax(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var maxValue;

    return _reduceOperation(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        initReduction: () {
          maxValue = null;
        },
        onValueToReduce: (int valueIndex, value) {
          if (maxValue == null || value > maxValue) {
            maxValue = value;
          }
        },
        reduce: () => maxValue);
  }

  @override
  NDArray reduceAny(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceAny(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    bool total;

    return _reduceOperation(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        initReduction: () {
          total = false;
        },
        onValueToReduce: (int valueIndex, bool value) {
          total = total || value;
        },
        reduce: () => total);
  }

  @override
  NDArray argMax({int axis, NDArray reuse}) {
    if (axis != null) {
      var resultDescriptor = descriptor.argMax(axis: axis);

      var maxValueIndex;
      var maxValue;

      return _reduceOperation([axis], false, resultDescriptor, reuse,
          initReduction: () {
            maxValueIndex = null;
            maxValue = null;
          },
          onValueToReduce: (int valueIndex, value) {
            if (maxValue == null || value > maxValue) {
              maxValueIndex = valueIndex;
              maxValue = value;
            }
          },
          reduce: () => maxValueIndex);
    } else {
      return reshape(newDimensions: [-1]).argMax(axis: 0, reuse: reuse);
    }
  }

  @override
  NDArray add(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.add(array2.descriptor),
        reuse,
        (value1, value2) => value1 + value2);
  }

  @override
  NDArray div(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.div(array2.descriptor),
        reuse,
        (value1, value2) => value1 / value2);
  }

  @override
  NDArray isGreater(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.isGreater(array2.descriptor),
        reuse,
        (value1, value2) => value1 > value2);
  }

  @override
  NDArray isGreaterOrEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.isGreaterOrEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 >= value2);
  }

  @override
  NDArray isLess(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.isLess(array2.descriptor),
        reuse,
        (value1, value2) => value1 < value2);
  }

  @override
  NDArray isLessOrEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.isLessOrEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 <= value2);
  }

  @override
  NDArray mul(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.mul(array2.descriptor),
        reuse,
        (value1, value2) => value1 * value2);
  }

  @override
  NDArray isEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.isEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 == value2);
  }

  @override
  NDArray isNotEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
        array2,
        descriptor.isNotEqual(array2.descriptor),
        reuse,
        (value1, value2) => value1 != value2);
  }

  @override
  NDArray sub(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return _elementWiseBinaryOperation(
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

    return _elementWiseTernaryOperation(
        thenArray,
        elseArray,
        descriptor.select(thenArray.descriptor, elseArray.descriptor),
        reuse,
        (value1, value2, value3) => value1 ? value2 : value3);
  }

  @override
  String toString() =>
      "<value: ${_toValue()}, shape: $shape, dataType: $dataType, stride: $_stride, offset: $_offset>";

  dynamic _toValue() {
    if (shape.isScalar) {
      return _data[_offset];
    } else {
      var shapeIndex = 0;
      var dimensionValues = new List(shape.dimension);
      var dimensionIndexes = new List(shape.dimension);
      var dataIndexes = new List(shape.dimension);
      var dataIndex = dataIndexes[shapeIndex] = _offset;
      var resultValues =
          dimensionValues[shapeIndex] = new List(shape[shapeIndex]);
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      var i = 0;
      while (i < shape.length) {
        if (dimensionIndex < shape[shapeIndex]) {
          if (shapeIndex == shape.dimension - 1) {
            resultValues[dimensionIndex] = _data[dataIndex];
            dataIndex += _stride[shapeIndex];
            dimensionIndex++;
            i++;
          } else {
            shapeIndex++;
            dimensionValues[shapeIndex] = new List(shape[shapeIndex]);
            resultValues =
                resultValues[dimensionIndex] = dimensionValues[shapeIndex];
            dataIndexes[shapeIndex] = dataIndex;
            dimensionIndex = dimensionIndexes[shapeIndex] = 0;
          }
        } else {
          shapeIndex--;
          dataIndex = dataIndexes[shapeIndex] =
              dataIndexes[shapeIndex] + _stride[shapeIndex];
          resultValues = dimensionValues[shapeIndex];
          dimensionIndex =
              dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        }
      }
      return dimensionValues[0];
    }
  }

  NDArrayImpl _elementWiseUnaryOperation(
      NDDescriptor resultDescriptor, NDArray reuse, unaryOperation(value)) {
    var resultData = _createData(resultDescriptor, reuse);
    var resultStride;

    if (shape.isScalar) {
      resultStride = _stride;

      resultData[0] = unaryOperation(_data[_offset]);
    } else {
      resultStride = _calculateDefaultStride(shape);

      var shapeIndex = 0;
      var dimensionIndexes = new List(shape.dimension);
      var dataIndexes = new List(shape.dimension);
      var dataIndex = dataIndexes[shapeIndex] = _offset;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < shape[shapeIndex]) {
          if (shapeIndex == shape.dimension - 1) {
            var axeDimension = shape[shapeIndex];
            var axeStride = _stride[shapeIndex];
            while (dimensionIndex++ < axeDimension) {
              resultData[resultDataIndex++] = unaryOperation(_data[dataIndex]);
              dataIndex += axeStride;
            }
          } else {
            shapeIndex++;
            dataIndexes[shapeIndex] = dataIndex;
            dimensionIndex = dimensionIndexes[shapeIndex] = 0;
          }
        } else {
          shapeIndex--;
          dataIndex = dataIndexes[shapeIndex] =
              dataIndexes[shapeIndex] + _stride[shapeIndex];
          dimensionIndex =
              dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        }
      }
    }

    return new NDArrayImpl._(resultData, resultDescriptor, resultStride, 0);
  }

  NDArrayImpl _elementWiseBinaryOperation(
      NDArrayImpl array2,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      binaryOperation(value1, value2)) {
    var resultShape = resultDescriptor.shape;
    var resultData = _createData(resultDescriptor, reuse);
    var resultStride;

    if (resultShape.isScalar) {
      resultStride = _stride;

      resultData[0] =
          binaryOperation(_data[_offset], array2._data[array2._offset]);
    } else {
      resultStride = _calculateDefaultStride(resultShape);

      var shapeIndex = 0;
      var dimensionIndexes = new List(resultShape.dimension);
      var data1Indexes = new List(resultShape.dimension);
      var stride1 = _calculateBroadcastedStride(resultShape, this);
      var data1Index = data1Indexes[shapeIndex] = _offset;
      var data2Indexes = new List(resultShape.dimension);
      var stride2 = _calculateBroadcastedStride(resultShape, array2);
      var data2Index = data2Indexes[shapeIndex] = array2._offset;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < resultShape[shapeIndex]) {
          if (shapeIndex == resultShape.dimension - 1) {
            resultData[resultDataIndex++] =
                binaryOperation(_data[data1Index], array2._data[data2Index]);
            data1Index += stride1[shapeIndex];
            data2Index += stride2[shapeIndex];
            dimensionIndex++;
          } else {
            shapeIndex++;
            data1Indexes[shapeIndex] = data1Index;
            data2Indexes[shapeIndex] = data2Index;
            dimensionIndex = dimensionIndexes[shapeIndex] = 0;
          }
        } else {
          shapeIndex--;
          data1Index = data1Indexes[shapeIndex] =
              data1Indexes[shapeIndex] + stride1[shapeIndex];
          data2Index = data2Indexes[shapeIndex] =
              data2Indexes[shapeIndex] + stride2[shapeIndex];
          dimensionIndex =
              dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        }
      }
    }

    return new NDArrayImpl._(resultData, resultDescriptor, resultStride, 0);
  }

  NDArrayImpl _elementWiseTernaryOperation(
      NDArrayImpl array2,
      NDArrayImpl array3,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      ternaryOperation(value1, value2, value3)) {
    var resultShape = resultDescriptor.shape;
    var resultData = _createData(resultDescriptor, reuse);
    var resultStride;
    var resultOffset = 0;

    if (resultShape.isScalar) {
      resultStride = _stride;

      resultData[0] = ternaryOperation(_data[_offset],
          array2._data[array2._offset], array3._data[array3._offset]);
    } else {
      resultStride = _calculateDefaultStride(resultShape);

      var shapeIndex = 0;
      var dimensionIndexes = new List(resultShape.dimension);
      var data1Indexes = new List(resultShape.dimension);
      var stride1 = _calculateBroadcastedStride(resultShape, this);
      var data1Delta = stride1[shapeIndex];
      var data1Index = data1Indexes[shapeIndex] = _offset;
      var data2Indexes = new List(resultShape.dimension);
      var stride2 = _calculateBroadcastedStride(resultShape, array2);
      var data2Delta = stride2[shapeIndex];
      var data2Index = data2Indexes[shapeIndex] = array2._offset;
      var data3Indexes = new List(resultShape.dimension);
      var stride3 = _calculateBroadcastedStride(resultShape, array3);
      var data3Delta = stride3[shapeIndex];
      var data3Index = data3Indexes[shapeIndex] = array3._offset;
      var resultDataIndex = resultOffset;
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < resultShape[shapeIndex]) {
          if (shapeIndex == resultShape.dimension - 1) {
            resultData[resultDataIndex++] = ternaryOperation(_data[data1Index],
                array2._data[data2Index], array3._data[data3Index]);
            data1Index += data1Delta;
            data2Index += data2Delta;
            data3Index += data3Delta;
            dimensionIndex++;
          } else {
            shapeIndex++;
            data1Delta = stride1[shapeIndex];
            data1Indexes[shapeIndex] = data1Index;
            data2Delta = stride2[shapeIndex];
            data2Indexes[shapeIndex] = data2Index;
            data3Delta = stride3[shapeIndex];
            data3Indexes[shapeIndex] = data3Index;
            dimensionIndex = dimensionIndexes[shapeIndex] = 0;
          }
        } else {
          shapeIndex--;
          data1Delta = stride1[shapeIndex];
          data1Index =
              data1Indexes[shapeIndex] = data1Indexes[shapeIndex] + data1Delta;
          data2Delta = stride2[shapeIndex];
          data2Index =
              data2Indexes[shapeIndex] = data2Indexes[shapeIndex] + data2Delta;
          data3Delta = stride3[shapeIndex];
          data3Index =
              data3Indexes[shapeIndex] = data3Indexes[shapeIndex] + data3Delta;
          dimensionIndex =
              dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        }
      }
    }

    return new NDArrayImpl._(
        resultData, resultDescriptor, resultStride, resultOffset);
  }

  NDArray _reduceOperation(List<int> reductionAxis, bool keepDimensions,
      NDDescriptor resultDescriptor, NDArray reuse,
      {void initReduction(),
      void onValueToReduce(int valueIndex, value),
      dynamic reduce()}) {
    var newReductionAxis =
        reductionAxis ?? new List.generate(shape.dimension, (index) => index);

    if (newReductionAxis.isNotEmpty) {
      var resultShape = resultDescriptor.shape;
      var resultData = _createData(resultDescriptor, reuse);
      var resultStride = _calculateDefaultStride(resultShape);

      var shapeIndex = 0;
      var permutedIndexes = new List(shape.dimension);
      var axis = new Set.from(newReductionAxis);
      var resultIndex = 0;
      for (var i = 0; i < shape.dimension; i++) {
        if (!axis.contains(i)) {
          permutedIndexes[resultIndex++] = i;
        }
      }
      for (var i = 0; i < newReductionAxis.length; i++) {
        permutedIndexes[resultIndex++] = newReductionAxis[i];
      }
      var dimensionIndexes = new List(shape.dimension);
      var dataIndexes = new List(shape.dimension);
      var dataIndex = dataIndexes[permutedIndexes[shapeIndex]] = _offset;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[permutedIndexes[shapeIndex]] = 0;

      initReduction();

      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < shape[permutedIndexes[shapeIndex]]) {
          if (shapeIndex == shape.dimension - newReductionAxis.length - 1) {
            initReduction();
          }

          if (shapeIndex == shape.dimension - 1) {
            onValueToReduce(dimensionIndex, _data[dataIndex]);

            dataIndex += _stride[permutedIndexes[shapeIndex]];
            dimensionIndex++;
          } else {
            shapeIndex++;
            dataIndexes[permutedIndexes[shapeIndex]] = dataIndex;
            dimensionIndex = dimensionIndexes[permutedIndexes[shapeIndex]] = 0;
          }
        } else {
          shapeIndex--;

          if (shapeIndex == shape.dimension - newReductionAxis.length - 1) {
            resultData[resultDataIndex++] = reduce();
          }

          if (shapeIndex >= 0) {
            dataIndex = dataIndexes[permutedIndexes[shapeIndex]] =
                dataIndexes[permutedIndexes[shapeIndex]] +
                    _stride[permutedIndexes[shapeIndex]];
            dimensionIndex = dimensionIndexes[permutedIndexes[shapeIndex]] =
                dimensionIndexes[permutedIndexes[shapeIndex]] + 1;
          }
        }
      }

      return new NDArrayImpl._(resultData, resultDescriptor, resultStride, 0);
    } else {
      return this;
    }
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

List<int> _calculateDefaultStride(NDShape shape) {
  List<int> stride = new List(shape.dimension);
  var factor = 1;
  for (var i = shape.dimension - 1; i >= 0; i--) {
    stride[i] = factor;
    factor *= shape[i];
  }
  return stride;
}

void _loadData(value, List data, NDDescriptor descriptor) {
  if (descriptor.dataType.isFloat) {
    _loadConvertedData(
        value, data, descriptor, (num value) => value.toDouble());
  } else if (descriptor.dataType.isInteger) {
    _loadConvertedData(value, data, descriptor, (num value) => value.toInt());
  } else {
    _loadConvertedData(value, data, descriptor, (value) => value);
  }
}

void _loadConvertedData(
    value, List data, NDDescriptor descriptor, dynamic converter(value)) {
  var shape = descriptor.shape;
  if (shape.isScalar) {
    data[0] = converter(value);
  } else {
    var dimensionValues = new List(shape.dimension);
    var dimensionIndexes = new List(shape.dimension);
    var shapeIndex = 0;
    var dataIndex = 0;
    var dimensionValue = dimensionValues[0] = value;
    var dimensionIndex = dimensionIndexes[0] = 0;
    while (dataIndex < data.length) {
      if (dimensionIndex < shape[shapeIndex]) {
        if (shapeIndex == shape.dimension - 1) {
          data[dataIndex++] = converter(dimensionValue[dimensionIndex++]);
        } else {
          shapeIndex++;
          dimensionValue =
              dimensionValues[shapeIndex] = dimensionValue[dimensionIndex];
          dimensionIndex = dimensionIndexes[shapeIndex] = 0;
        }
      } else {
        shapeIndex--;
        dimensionValue = dimensionValues[shapeIndex];
        dimensionIndex =
            dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
      }
    }
  }
}

List<int> _calculateBroadcastedStride(
        NDShape broadcastedShape, NDArrayImpl array) =>
    new List.generate(broadcastedShape.dimension, (index) {
      var dimensionDelta = broadcastedShape.dimension - array.shape.dimension;
      if (index < dimensionDelta || array.shape[index - dimensionDelta] == 1) {
        return 0;
      } else {
        return array._stride[index - dimensionDelta];
      }
    }, growable: false);

List _createData(NDDescriptor descriptor, NDArrayImpl reuse) {
  if (reuse != null &&
      reuse.dataType == descriptor.dataType &&
      reuse._data.length == descriptor.shape.length) {
    return reuse._data;
  } else {
    switch (descriptor.dataType) {
      case NDDataType.float32:
        return new Float32List(descriptor.shape.length);
      case NDDataType.float64:
        return new Float64List(descriptor.shape.length);
      case NDDataType.int8:
        return new Int8List(descriptor.shape.length);
      case NDDataType.uint8:
        return new Uint8List(descriptor.shape.length);
      case NDDataType.uint8Clamped:
        return new Uint8ClampedList(descriptor.shape.length);
      case NDDataType.int16:
        return new Int16List(descriptor.shape.length);
      case NDDataType.uint16:
        return new Uint16List(descriptor.shape.length);
      case NDDataType.int32:
        return new Int32List(descriptor.shape.length);
      case NDDataType.uint32:
        return new Uint32List(descriptor.shape.length);
      case NDDataType.int64:
        return new Int64List(descriptor.shape.length);
      case NDDataType.uint64:
        return new Uint64List(descriptor.shape.length);
      case NDDataType.boolean:
        // TODO rivedere con typed data
        return new List<bool>(descriptor.shape.length);
      case NDDataType.string:
        return new List<String>(descriptor.shape.length);
      case NDDataType.generic:
        return new List(descriptor.shape.length);
      default:
        throw new StateError("DEAD CODE");
    }
  }
}
