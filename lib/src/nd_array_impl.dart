// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'nd_shape.dart';
import "nd_array.dart";

import 'nd_shape_impl.dart';

NDArray createTestNDArray(List<int> dimensions, [generator(index)]) {
  var shape = new NDShapeImpl(dimensions);
  var data = new List.generate(shape.length, generator ?? (index) => index);
  var stride = _calculateDefaultStride(shape);

  return new NDArrayImpl._(data, shape, stride, 0);
}

class NDArrayImpl implements NDArray {
  @override
  final NDShape shape;

  final List _data;

  final List<int> _stride;

  final int _offset;

  factory NDArrayImpl(value) {
    var shape = _calculateShape(value);
    var data = _calculateFlatData(value, shape);
    var stride = _calculateDefaultStride(shape);

    return new NDArrayImpl._(data, shape, stride, 0);
  }

  NDArrayImpl._(this._data, this.shape, this._stride, this._offset);

  @override
  NDArray abs() => _elementWiseUnaryOperation((value) => value.abs());

  @override
  NDArray exp() => _elementWiseUnaryOperation((value) => value.exp());

  @override
  NDArray inv() => _elementWiseUnaryOperation((value) => 1 / value);

  @override
  NDArray log() => _elementWiseUnaryOperation((value) => value.log());

  @override
  NDArray neg() => _elementWiseUnaryOperation((value) => -value);

  @override
  NDArray sign() => _elementWiseUnaryOperation((value) => value.sign());

  @override
  NDArray not() => _elementWiseUnaryOperation((value) => !value);

  @override
  List<List<E>> toMatrix<E>() {
    if (shape.isMatrix) {
      return toValue();
    } else {
      throw new StateError("Not a matrix (shape: $shape)");
    }
  }

  @override
  E toScalar<E>() {
    if (shape.isScalar) {
      return toValue();
    } else {
      throw new StateError("Not a scalar (shape: $shape)");
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
  dynamic toValue() => _toValue();

  @override
  List<E> toVector<E>() {
    if (shape.isVector) {
      return toValue();
    } else {
      throw new StateError("Not a vector (shape: $shape)");
    }
  }

  @override
  NDArray transpose({List<int> permutationAxis}) {
    var newPermutationAxis = permutationAxis;

    if (newPermutationAxis == null) {
      newPermutationAxis = new List.generate(
          shape.dimension, (index) => shape.dimension - index - 1);
    } else if (permutationAxis.length != shape.dimension) {
      throw new ArgumentError.value(permutationAxis, "permutation axis",
          "Dimension is ${shape.dimension}");
    } else if (permutationAxis.length != new Set.from(permutationAxis).length) {
      throw new ArgumentError.value(permutationAxis, "permutation axis",
          "Must be unique $permutationAxis");
    }

    var permutedDimensions = new List(shape.dimension);
    var permutedStride = new List(shape.dimension);

    for (var i = 0; i < newPermutationAxis.length; i++) {
      var permutationAxe = newPermutationAxis[i];
      permutedDimensions[i] = shape[permutationAxe];
      permutedStride[i] = _stride[permutationAxe];
    }

    var permutedShape = new NDShapeImpl(permutedDimensions);

    return new NDArrayImpl._(_data, permutedShape, permutedStride, _offset);
  }

  @override
  NDArray add(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 + value2);

  @override
  NDArray div(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 / value2);

  @override
  NDArray equal(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 == value2);

  @override
  NDArray greater(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 > value2);

  @override
  NDArray greaterEqual(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 >= value2);

  @override
  NDArray less(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 < value2);

  @override
  NDArray lessEqual(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 <= value2);

  @override
  NDArray matMul(value2) {
    var array2 = _toNDArray(value2);

    var resultShape = shape.matMul(array2.shape);
    var resultData = new List(resultShape.length);
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
          for (var row1Index = 0, rowData1Index = data1Index;
              row1Index < shape[shapeIndex];
              row1Index++, rowData1Index += _stride[shapeIndex]) {
            for (var column2Index = 0, columnData2Index = data2Index;
                column2Index < array2.shape[shapeIndex + 1];
                column2Index++,
                columnData2Index += array2._stride[shapeIndex + 1]) {
              var sumValue = 0;
              for (var innerIndex = 0,
                      data1Index = rowData1Index,
                      data2Index = columnData2Index;
                  innerIndex < shape[shapeIndex + 1];
                  innerIndex++,
                  data1Index += _stride[shapeIndex + 1],
                  data2Index += array2._stride[shapeIndex]) {
                sumValue += _data[data1Index] * array2._data[data2Index];
              }
              resultData[resultDataIndex++] = sumValue;
            }
          }

          dimensionIndex = shape[shapeIndex];
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

    return new NDArrayImpl._(resultData, resultShape, resultStride, 0);
  }

  dynamic _matMul2d(value1, value2, List<int> shape1, List<int> shape2) {
    var result = [];
    for (int row1 = 0; row1 < shape1.first; row1++) {
      var row = [];
      for (int column2 = 0; column2 < shape2.last; column2++) {
        var sumValue = 0;
        for (int index = 0; index < shape1.last; index++) {
          sumValue += value1[row1][index] * value2[index][column2];
        }
        row.add(sumValue);
      }
      result.add(row);
    }
    return result;
  }

  @override
  NDArray mul(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 * value2);

  @override
  NDArray notEqual(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 != value2);

  @override
  NDArray sub(value2) =>
      _elementWiseBinaryOperation(value2, (value1, value2) => value1 - value2);

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
  NDArray operator <(value2) => less(value2);

  @override
  NDArray operator <=(value2) => lessEqual(value2);

  @override
  NDArray operator >(value2) => greater(value2);

  @override
  NDArray operator >=(value2) => greaterEqual(value2);

  @override
  NDArray select(thenValue, elseValue) => _elementWiseTernaryOperation(
      thenValue,
      elseValue,
      (value1, value2, value3) => value1 ? value2 : value3);

  @override
  String toString() =>
      "<value: ${toValue()}, shape: $shape, stride: $_stride, offset: $_offset>";

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

  NDArrayImpl _toNDArray(value) =>
      value is NDArray ? value : new NDArrayImpl(value);

  NDArrayImpl _elementWiseUnaryOperation(unaryOperation(value)) {
    var resultData = new List(shape.length);
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
            resultData[resultDataIndex++] = unaryOperation(_data[dataIndex]);
            dataIndex += _stride[shapeIndex];
            dimensionIndex++;
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

    return new NDArrayImpl._(resultData, shape, resultStride, 0);
  }

  NDArrayImpl _elementWiseBinaryOperation(
      value2, binaryOperation(value1, value2)) {
    var array2 = _toNDArray(value2);

    var resultShape = shape.broadcast(array2.shape);
    var resultData = new List(resultShape.length);
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

    return new NDArrayImpl._(resultData, resultShape, resultStride, 0);
  }

  NDArrayImpl _elementWiseTernaryOperation(
      value2, value3, ternaryOperation(value1, value2, value3)) {
    var array2 = _toNDArray(value2);
    var array3 = _toNDArray(value3);

    var resultShape = shape.broadcast(array2.shape).broadcast(array3.shape);
    var resultData = new List(resultShape.length);
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
              data3Indexes[shapeIndex] = data2Indexes[shapeIndex] + data2Delta;
          dimensionIndex =
              dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        }
      }
    }

    return new NDArrayImpl._(
        resultData, resultShape, resultStride, resultOffset);
  }
}

NDShape _calculateShape(value) {
  var dimensions = [];
  dynamic element = value;
  while (element is List) {
    dimensions.add(element.length);
    element = element[0];
  }
  return new NDShapeImpl(dimensions);
}

List<int> _calculateDefaultStride(NDShapeImpl shape) {
  List<int> stride = new List(shape.dimension);
  var factor = 1;
  for (var i = shape.dimension - 1; i >= 0; i--) {
    stride[i] = factor;
    factor *= shape[i];
  }
  return stride;
}

List _calculateFlatData(value, NDShapeImpl shape) {
  var data = new List(shape.length);
  if (shape.isScalar) {
    data[0] = value;
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
          data[dataIndex++] = dimensionValue[dimensionIndex++];
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
  return data;
}

List<int> _calculateBroadcastedStride(
        NDShapeImpl broadcastedShape, NDArrayImpl array) =>
    new List.generate(broadcastedShape.dimension, (index) {
      var dimensionDelta = broadcastedShape.dimension - array.shape.dimension;
      if (index < dimensionDelta || array.shape[index - dimensionDelta] == 1) {
        return 0;
      } else {
        return array._stride[index - dimensionDelta];
      }
    }, growable: false);