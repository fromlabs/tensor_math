// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math";

// TODO component wise (unary, binary, ternary)
// TODO mat mul
// TODO select
// TODO reduce
// TODO reshape
// TODO canonical
// TODO typed data

class MultiDimensionalArray<E> {
  final List<E> _data;
  final List<int> shape;
  final List<int> _stride;
  final int _offset;
  int _length;

  factory MultiDimensionalArray(value) {
    var shape = _calculateShape(value);
    var length = _calculateLength(shape);
    var data = _calculateFlatData(value, shape, length);
    var stride = _calculateDefaultStride(shape);

    return new MultiDimensionalArray._(data, shape, stride, 0);
  }

  MultiDimensionalArray._(
      this._data, List<int> shape, List<int> stride, this._offset)
      : this.shape = new List.unmodifiable(shape),
        this._stride = new List.unmodifiable(stride);

  int get rank => shape.length;

  int get length {
    if (_length == null) {
      _length = _calculateLength(shape);
    }
    return _length;
  }

  dynamic get value => _toValue(_offset, 0);

  E get scalarValue {
    if (shape.isEmpty) {
      return value;
    } else {
      throw new StateError("Not a scalar (shape: $shape)");
    }
  }

  List<E> get vectorValue {
    if (rank == 1) {
      return value;
    } else {
      throw new StateError("Not a vector (shape: $shape)");
    }
  }

  List<List<E>> get matrixValue {
    if (rank == 2) {
      return value;
    } else {
      throw new StateError("Not a matrix (shape: $shape)");
    }
  }

  MultiDimensionalArray<E> transpose({List<int> permutationAxis}) {
    var newPermutationAxis = permutationAxis;

    if (newPermutationAxis == null) {
      newPermutationAxis = new List.generate(rank, (index) => rank - index - 1);
    } else if (permutationAxis.length != rank) {
      throw new ArgumentError.value(
          permutationAxis, "permutation axis", "Rank is $rank");
    }

    var permutedShape = new List(rank);
    var permutedStride = new List(rank);

    for (var i = 0; i < newPermutationAxis.length; i++) {
      var permutationAxe = newPermutationAxis[i];
      permutedShape[i] = shape[permutationAxe];
      permutedStride[i] = _stride[permutationAxe];
    }

    return new MultiDimensionalArray._(
        _data, permutedShape, permutedStride, _offset);
  }

  MultiDimensionalArray<E> abs() =>
      _elementWiseUnaryOperation<E>((dynamic value) => value.abs());

  MultiDimensionalArray<E> exp() =>
      _elementWiseUnaryOperation<E>((dynamic value) => value.exp());

  MultiDimensionalArray<double> inv() =>
      _elementWiseUnaryOperation<double>((dynamic value) => 1 / value);

  MultiDimensionalArray<E> log() =>
      _elementWiseUnaryOperation<E>((dynamic value) => value.log());

  MultiDimensionalArray<E> neg() =>
      _elementWiseUnaryOperation<E>((dynamic value) => -value);

  MultiDimensionalArray<E> sign() =>
      _elementWiseUnaryOperation<E>((dynamic value) => value.sign());

  MultiDimensionalArray<E> add(value) => _elementWiseBinaryOperation<E, E>(
      value, (dynamic value1, dynamic value2) => value1 + value2);

  MultiDimensionalArray<E> sub(value) => _elementWiseBinaryOperation<E, E>(
      value, (dynamic value1, dynamic value2) => value1 - value2);

  MultiDimensionalArray<E> mul(value) => _elementWiseBinaryOperation<E, E>(
      value, (dynamic value1, dynamic value2) => value1 * value2);

  MultiDimensionalArray<E> div(value) => _elementWiseBinaryOperation<E, E>(
      value, (dynamic value1, dynamic value2) => value1 / value2);

  MultiDimensionalArray<E> matrixMul(value) {
    // TODO to implement MultiDimensionalArray.matrixMul
    throw new UnimplementedError(
        "to implement MultiDimensionalArray.matrixMul: $this");
  }

  MultiDimensionalArray<bool> equal(value) =>
      _elementWiseBinaryOperation<E, bool>(
          value, (dynamic value1, dynamic value2) => value1 == value2);

  MultiDimensionalArray<bool> notEqual(value) =>
      _elementWiseBinaryOperation<E, bool>(
          value, (dynamic value1, dynamic value2) => value1 != value2);

  MultiDimensionalArray<bool> greater(value) =>
      _elementWiseBinaryOperation<E, bool>(
          value, (dynamic value1, dynamic value2) => value1 > value2);

  MultiDimensionalArray<bool> greaterEqual(value) =>
      _elementWiseBinaryOperation<E, bool>(
          value, (dynamic value1, dynamic value2) => value1 >= value2);

  MultiDimensionalArray<bool> less(value) =>
      _elementWiseBinaryOperation<E, bool>(
          value, (dynamic value1, dynamic value2) => value1 < value2);

  MultiDimensionalArray<bool> lessEqual(value) =>
      _elementWiseBinaryOperation<E, bool>(
          value, (dynamic value1, dynamic value2) => value1 <= value2);

  MultiDimensionalArray<ER> _elementWiseUnaryOperation<ER>(
          ER unaryOperation(E value)) =>
      _calculateElementWiseUnaryOperation(this, unaryOperation);

  MultiDimensionalArray<ER> _elementWiseBinaryOperation<E2, ER>(
      value, ER binaryOperation(E value1, E2 value2)) {
    var mdArray = _toMultiDimensionalArray(value);

    var calculator = new _ElementWiseBinaryOperationProcessor<E, E2, ER>(
        this, mdArray, binaryOperation);

    return calculator.process();
  }

  @override
  String toString() => "<$value, shape: $shape, rank: $rank, length: $length>";

  dynamic _toValue(int offset, int shapeIndex) {
    var shapeLength = rank - shapeIndex;
    switch (shapeLength) {
      case 0:
        return _data[offset];
      case 1:
        var length = shape[shapeIndex];
        var delta = _stride[shapeIndex];
        var values = new List<E>(length);
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          values[i] = _data[index];
        }
        return values;
      case 2:
        var length = shape[shapeIndex];
        var delta = _stride[shapeIndex];
        var values = new List<List<E>>(length);
        var shapeIndex2 = shapeIndex + 1;
        var length2 = shape[shapeIndex2];
        var delta2 = _stride[shapeIndex2];
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          var values2 = new List<E>(length2);
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
        var values = new List<List<List<E>>>(length);
        var shapeIndex2 = shapeIndex + 1;
        var length2 = shape[shapeIndex2];
        var delta2 = _stride[shapeIndex2];
        var shapeIndex3 = shapeIndex2 + 1;
        var length3 = shape[shapeIndex3];
        var delta3 = _stride[shapeIndex3];
        for (var i = 0, index = offset; i < length; i++, index += delta) {
          var values2 = new List<List<E>>(length2);
          for (var i2 = 0, index2 = index;
              i2 < length2;
              i2++, index2 += delta2) {
            var values3 = new List<E>(length3);
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
        var values = new List<List<List<List<E>>>>(length);
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
          var values2 = new List<List<List<E>>>(length2);
          for (var i2 = 0, index2 = index;
              i2 < length2;
              i2++, index2 += delta2) {
            var values3 = new List<List<E>>(length3);
            for (var i3 = 0, index3 = index2;
                i3 < length3;
                i3++, index3 += delta3) {
              var values4 = new List<E>(length3);
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
}

MultiDimensionalArray<E> _toMultiDimensionalArray<E>(value) =>
    value is MultiDimensionalArray
        ? value
        : new MultiDimensionalArray<E>(value);

List<int> _calculateShape(value) {
  var shape = [];
  dynamic element = value;
  while (element is List) {
    shape.add(element.length);
    element = element[0];
  }
  return shape;
}

int _calculateLength(List<int> shape) {
  switch (shape.length) {
    case 0:
      return 1;
    case 1:
      return shape[0];
    case 2:
      return shape[0] * shape[1];
    case 3:
      return shape[0] * shape[1] * shape[2];
    case 4:
      return shape[0] * shape[1] * shape[2] * shape[3];
    default:
      var total = shape[0] * shape[1] * shape[2] * shape[3];
      for (var i = 4; i < shape.length; i++) {
        total *= shape[i];
      }
      return total;
  }
}

List<int> _calculateDefaultStride(List<int> shape) {
  List<int> stride = new List(shape.length);
  switch (shape.length) {
    case 0:
      break;
    case 1:
      stride[0] = 1;
      break;
    default:
      var factor = 1;
      for (var i = shape.length - 1; i >= 0; i--) {
        stride[i] = factor;
        factor *= shape[i];
      }
  }
  return stride;
}

List<E> _calculateFlatData<E>(value, List<int> shape, int length) {
  var data = new List(length);
  switch (shape.length) {
    case 0:
      data[0] = value;
      break;
    case 1:
      List<E> values = value;
      data.setAll(0, values);
      break;
    case 2:
      List<List<E>> values = value;
      var dataIndex = 0;
      for (var i = 0; i < values.length; i++) {
        var values2 = values[i];
        data.setAll(dataIndex, values2);
        dataIndex += values2.length;
      }

      break;
    case 3:
      List<List<List<E>>> values = value;
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
      List<List<List<List<E>>>> values = value;
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
      var dimensionValues = new List(shape.length - 4);
      var dimensionIndexes = new List(shape.length - 4);
      var shapeIndex = 0;
      var dataIndex = 0;
      var dimensionValue = dimensionValues[0] = value;
      var dimensionIndex = dimensionIndexes[0] = 0;
      while (dataIndex < data.length) {
        if (dimensionIndex < shape[shapeIndex]) {
          if (shapeIndex == shape.length - 5) {
            List<List<List<List<E>>>> values = dimensionValue[dimensionIndex++];
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

List<int> _calculateBroadcastedShape2(
    MultiDimensionalArray array1, MultiDimensionalArray array2) {
  var broadcastedShape = new List(max(array1.rank, array2.rank));
  var broadcastedIndex = broadcastedShape.length - 1;

  var index1 = array1.rank - 1;
  var index2 = array2.rank - 1;
  while (index1 >= 0 || index2 >= 0) {
    var d1 = index1 >= 0 ? array1.shape[index1] : 1;
    var d2 = index2 >= 0 ? array2.shape[index2] : 1;
    if (d1 == 1 || d1 == d2) {
      broadcastedShape[broadcastedIndex--] = d2;
    } else if (d2 == 1) {
      broadcastedShape[broadcastedIndex--] = d1;
    } else {
      throw new ArgumentError(
          "Shapes not broadcastable: ${array1.shape} != ${array2.shape}");
    }

    index1--;
    index2--;
  }

  return broadcastedShape;
}

List<int> _calculateBroadcastedShape3(MultiDimensionalArray array1,
    MultiDimensionalArray array2, MultiDimensionalArray array3) {
  var broadcastedShape =
      new List(max(max(array1.rank, array2.rank), array3.rank));
  var broadcastedIndex = broadcastedShape.length - 1;

  var index1 = array1.rank - 1;
  var index2 = array2.rank - 1;
  var index3 = array3.rank - 1;
  while (index1 >= 0 || index2 >= 0 || index3 >= 0) {
    var d1 = index1 >= 0 ? array1.shape[index1] : 1;
    var d2 = index2 >= 0 ? array2.shape[index2] : 1;
    var d3 = index3 >= 0 ? array3.shape[index3] : 1;
    if ((d1 == 1 || d1 == d3) && (d2 == 1 || d2 == d3)) {
      broadcastedShape[broadcastedIndex--] = d3;
    } else if (d3 == 1 && (d1 == 1 || d1 == d2)) {
      broadcastedShape[broadcastedIndex--] = d2;
    } else if (d2 == 1) {
      broadcastedShape[broadcastedIndex--] = d1;
    } else {
      throw new ArgumentError(
          "Shapes not broadcastable: ${array1.shape} != ${array2.shape} != ${array3.shape}");
    }

    index1--;
    index2--;
    index3--;
  }

  return broadcastedShape;
}

MultiDimensionalArray<ER> _calculateElementWiseUnaryOperation<E, ER>(
    MultiDimensionalArray<E> array, ER unaryOperation(E value)) {
  var data = new List(array.length);
  var stride;
  switch (array.shape.length) {
    case 0:
      stride = array._stride;

      data[0] = unaryOperation(array._data[array._offset]);

      break;
    case 1:
      stride = _calculateDefaultStride(array.shape);

      var length = array.shape[0];
      var delta = array._stride[0];
      for (var i = 0, index = array._offset; i < length; i++, index += delta) {
        data[i] = unaryOperation(array._data[index]);
      }

      break;
    case 2:
      stride = _calculateDefaultStride(array.shape);

      var length = array.shape[0];
      var delta = array._stride[0];
      var resultDelta = 1;
      for (var i = 0, resultIndex = 0, index = array._offset;
          i < length;
          i++, resultIndex += resultDelta, index += delta) {
        data[resultIndex] = unaryOperation(array._data[index]);
      }

      break;
    case 3:
    case 4:
    default:
  }

  return new MultiDimensionalArray._(data, array.shape, stride, 0);
}

class _ElementWiseBinaryOperationProcessor<E1, E2, ER> {
  _ElementWiseBinaryOperationProcessor(MultiDimensionalArray<E1> array1,
      MultiDimensionalArray<E2> array2, ER operation(E1 value1, E2 value2));

  MultiDimensionalArray<ER> process() {
    //var resultLength = _calculateLength(shape);
    //var resultData = new List<E2>(resultLength);

    // TODO to implement _ElementWiseBinaryOperationProcessor.process
    throw new UnimplementedError(
        "to implement _ElementWiseBinaryOperationProcessor.process: $this");
  }
}
