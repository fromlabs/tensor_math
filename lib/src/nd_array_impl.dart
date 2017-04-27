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
  dynamic toValue() => _toValue(_offset, 0);

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
  NDArray matrixMul(value2) {
    // TODO to implement NDArrayImpl.matrixMul
    throw new UnimplementedError("to implement NDArrayImpl.matrixMul: $this");
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

  dynamic _toValue(int offset, int shapeIndex) {
    var shapeLength = shape.dimension - shapeIndex;
    switch (shapeLength) {
      case 0:
        return _data[offset];
      case 1:
        var length = shape[shapeIndex];
        var delta = _stride[shapeIndex];
        var values = new List(length);
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          values[i] = _data[index];
        }
        return values;
      case 2:
        var length = shape[shapeIndex];
        var delta = _stride[shapeIndex];
        var values = new List<List>(length);
        var shapeIndex2 = shapeIndex + 1;
        var length2 = shape[shapeIndex2];
        var delta2 = _stride[shapeIndex2];
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          var values2 = new List(length2);
          for (var i2 = 0, index2 = index;
              i2 < length2;
              i2++, index2 += delta2) {
            values2[i2] = _data[index2];
          }
          values[i] = values2;
        }
        return values;
      case 3:
        var length = shape[shapeIndex];
        var delta = _stride[shapeIndex];
        var values = new List<List<List>>(length);
        var shapeIndex2 = shapeIndex + 1;
        var length2 = shape[shapeIndex2];
        var delta2 = _stride[shapeIndex2];
        var shapeIndex3 = shapeIndex2 + 1;
        var length3 = shape[shapeIndex3];
        var delta3 = _stride[shapeIndex3];
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          var values2 = new List<List>(length2);
          for (var i2 = 0, index2 = index;
              i2 < length2;
              i2++, index2 += delta2) {
            var values3 = new List(length3);
            for (var i3 = 0, index3 = index2;
                i3 < length3;
                i3++, index3 += delta3) {
              values3[i3] = _data[index3];
            }
            values2[i2] = values3;
          }
          values[i] = values2;
        }
        return values;
      case 4:
        var length = shape[shapeIndex];
        var delta = _stride[shapeIndex];
        var values = new List<List<List<List>>>(length);
        var shapeIndex2 = shapeIndex + 1;
        var length2 = shape[shapeIndex2];
        var delta2 = _stride[shapeIndex2];
        var shapeIndex3 = shapeIndex2 + 1;
        var length3 = shape[shapeIndex3];
        var delta3 = _stride[shapeIndex3];
        var shapeIndex4 = shapeIndex3 + 1;
        var length4 = shape[shapeIndex4];
        var delta4 = _stride[shapeIndex4];
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          var values2 = new List<List<List>>(length2);
          for (var i2 = 0, index2 = index;
              i2 < length2;
              i2++, index2 += delta2) {
            var values3 = new List<List>(length3);
            for (var i3 = 0, index3 = index2;
                i3 < length3;
                i3++, index3 += delta3) {
              var values4 = new List(length3);
              for (var i4 = 0, index4 = index3;
                  i4 < length4;
                  i4++, index4 += delta4) {
                values4[i4] = _data[index4];
              }
              values3[i3] = values4;
            }
            values2[i2] = values3;
          }
          values[i] = values2;
        }
        return values;
      default:
        // TODO capire se possibile non utilizzare la ricorsione
        var length = shape[shapeIndex];
        var delta = _stride[shapeIndex];
        var values = new List(length);
        var nextShapeIndex = shapeIndex + 1;
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          values[i] = _toValue(index, nextShapeIndex);
        }
        return values;
    }
  }

  NDArrayImpl _toNDArray(value) =>
      value is NDArray ? value : new NDArrayImpl(value);

  NDArrayImpl _elementWiseUnaryOperation(unaryOperation(value)) {
    var resultData = new List(shape.length);
    var resultShape = shape;
    var resultStride;
    var resultOffset = 0;

    switch (shape.dimension) {
      case 0:
        resultStride = _stride;

        resultData[0] = unaryOperation(_data[_offset]);

        break;
      default:
        resultStride = _calculateDefaultStride(resultShape);

        var shapeIndex = 0;
        var dimensionIndexes = new List(shape.dimension);
        var dataIndexes = new List(shape.dimension);
        var dataDelta = _stride[shapeIndex];
        var dataIndex = dataIndexes[shapeIndex] = _offset;
        var resultDataIndex = resultOffset;
        var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
        while (resultDataIndex < resultData.length) {
          if (dimensionIndex < shape[shapeIndex]) {
            if (shapeIndex == shape.dimension - 1) {
              resultData[resultDataIndex++] = unaryOperation(_data[dataIndex]);
              dataIndex += dataDelta;
              dimensionIndex++;
            } else {
              shapeIndex++;
              dataDelta = _stride[shapeIndex];
              dataIndexes[shapeIndex] = dataIndex;
              dimensionIndex = dimensionIndexes[shapeIndex] = 0;
            }
          } else {
            shapeIndex--;
            dataDelta = _stride[shapeIndex];
            dataIndex =
                dataIndexes[shapeIndex] = dataIndexes[shapeIndex] + dataDelta;
            dimensionIndex =
                dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
          }
        }

        break;
    }

    return new NDArrayImpl._(
        resultData, resultShape, resultStride, resultOffset);
  }

  NDArrayImpl _elementWiseBinaryOperation(
      value2, binaryOperation(value1, value2)) {
    var array2 = _toNDArray(value2);

    var resultShape = shape.broadcast(array2.shape);
    var resultData = new List(resultShape.length);
    var resultStride;
    var resultOffset = 0;

    switch (resultShape.dimension) {
      case 0:
        resultStride = _stride;

        resultData[0] =
            binaryOperation(_data[_offset], array2._data[array2._offset]);

        break;
      default:
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
        var resultDataIndex = resultOffset;
        var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
        while (resultDataIndex < resultData.length) {
          if (dimensionIndex < resultShape[shapeIndex]) {
            if (shapeIndex == resultShape.dimension - 1) {
              resultData[resultDataIndex++] =
                  binaryOperation(_data[data1Index], array2._data[data2Index]);
              data1Index += data1Delta;
              data2Index += data2Delta;
              dimensionIndex++;
            } else {
              shapeIndex++;
              data1Delta = stride1[shapeIndex];
              data1Indexes[shapeIndex] = data1Index;
              data2Delta = stride2[shapeIndex];
              data2Indexes[shapeIndex] = data2Index;
              dimensionIndex = dimensionIndexes[shapeIndex] = 0;
            }
          } else {
            shapeIndex--;
            data1Delta = stride1[shapeIndex];
            data1Index = data1Indexes[shapeIndex] =
                data1Indexes[shapeIndex] + data1Delta;
            data2Delta = stride2[shapeIndex];
            data2Index = data2Indexes[shapeIndex] =
                data2Indexes[shapeIndex] + data2Delta;
            dimensionIndex =
                dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
          }
        }

        break;
    }

    return new NDArrayImpl._(
        resultData, resultShape, resultStride, resultOffset);
  }

  NDArrayImpl _elementWiseUnaryOperation2(unaryOperation(value)) {
    var data = new List(shape.length);
    var resultStride;

    if (shape.isScalar) {
      resultStride = _stride;

      data[0] = unaryOperation(_data[_offset]);
    } else if (shape.isVector) {
      resultStride = _calculateDefaultStride(shape);

      var length = shape[0];
      var delta = _stride[0];
      for (var i = 0, index = _offset; i < length; i++, index += delta) {
        data[i] = unaryOperation(_data[index]);
      }
    } else if (shape.isMatrix) {
      resultStride = _calculateDefaultStride(shape);

      var length = shape[0];
      var delta = _stride[0];
      var resultDelta = 1;
      for (var i = 0, resultIndex = 0, index = _offset;
          i < length;
          i++, resultIndex += resultDelta, index += delta) {
        data[resultIndex] = unaryOperation(_data[index]);
      }
    } else if (shape.isTensor3D) {
      // TODO to implement NDArrayImpl._elementWiseUnaryOperation
      throw new UnimplementedError(
          "to implement NDArrayImpl._elementWiseUnaryOperation: $this");
    } else if (shape.isTensor4D) {
      // TODO to implement NDArrayImpl._elementWiseUnaryOperation
      throw new UnimplementedError(
          "to implement NDArrayImpl._elementWiseUnaryOperation: $this");
    } else {
      // TODO to implement NDArrayImpl._elementWiseUnaryOperation
      throw new UnimplementedError(
          "to implement NDArrayImpl._elementWiseUnaryOperation: $this");
    }

    return new NDArrayImpl._(data, shape, resultStride, 0);
  }

  NDArrayImpl _elementWiseTernaryOperation(
      value2, value3, ternaryOperation(value1, value2, value3)) {
    var array2 = _toNDArray(value2);
    var array3 = _toNDArray(value3);

    var resultShape = shape.broadcast(array2.shape).broadcast(array3.shape);
    var resultData = new List(resultShape.length);
    var resultStride;
    var resultOffset = 0;

    switch (resultShape.dimension) {
      case 0:
        resultStride = _stride;

        resultData[0] = ternaryOperation(_data[_offset],
            array2._data[array2._offset], array3._data[array3._offset]);

        break;
      default:
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
              resultData[resultDataIndex++] = ternaryOperation(
                  _data[data1Index],
                  array2._data[data2Index],
                  array3._data[data3Index]);
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
            data1Index = data1Indexes[shapeIndex] =
                data1Indexes[shapeIndex] + data1Delta;
            data2Delta = stride2[shapeIndex];
            data2Index = data2Indexes[shapeIndex] =
                data2Indexes[shapeIndex] + data2Delta;
            data3Delta = stride3[shapeIndex];
            data3Index = data3Indexes[shapeIndex] =
                data2Indexes[shapeIndex] + data2Delta;
            dimensionIndex =
                dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
          }
        }

        break;
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
  switch (shape.dimension) {
    case 0:
      data[0] = value;
      break;
    case 1:
      List values = value;
      data.setAll(0, values);
      break;
    case 2:
      List<List> values = value;
      var dataIndex = 0;
      for (var i = 0; i < values.length; i++) {
        var values2 = values[i];
        data.setAll(dataIndex, values2);
        dataIndex += values2.length;
      }

      break;
    case 3:
      List<List<List>> values = value;
      var dataIndex = 0;
      for (var i = 0; i < values.length; i++) {
        var values2 = values[i];
        for (var i2 = 0; i2 < values2.length; i2++) {
          var values3 = values2[i2];
          data.setAll(dataIndex, values3);
          dataIndex += values3.length;
        }
      }
      break;
    case 4:
      List<List<List<List>>> values = value;
      var dataIndex = 0;
      for (var i = 0; i < values.length; i++) {
        var values2 = values[i];
        for (var i2 = 0; i2 < values2.length; i2++) {
          var values3 = values2[i2];
          for (var i3 = 0; i3 < values3.length; i3++) {
            var values4 = values3[i3];
            data.setAll(dataIndex, values4);
            dataIndex += values4.length;
          }
        }
      }
      break;
    default:
      var dimensionValues = new List(shape.dimension - 4);
      var dimensionIndexes = new List(shape.dimension - 4);
      var shapeIndex = 0;
      var dataIndex = 0;
      var dimensionValue = dimensionValues[0] = value;
      var dimensionIndex = dimensionIndexes[0] = 0;
      while (dataIndex < data.length) {
        if (dimensionIndex < shape[shapeIndex]) {
          if (shapeIndex == shape.dimension - 5) {
            List<List<List<List>>> values = dimensionValue[dimensionIndex++];
            for (var i = 0; i < values.length; i++) {
              var values2 = values[i];
              for (var i2 = 0; i2 < values2.length; i2++) {
                var values3 = values2[i2];
                for (var i3 = 0; i3 < values3.length; i3++) {
                  var values4 = values3[i3];
                  data.setAll(dataIndex, values4);
                  dataIndex += values4.length;
                }
              }
            }
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
