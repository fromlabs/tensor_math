// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'dart:collection';
import "dart:typed_data";

import "package:collection/collection.dart";

import 'nd_descriptor.dart';
import 'nd_shape.dart';
import 'nd_data_type.dart';
import "nd_array.dart";

import "nd_array_base.dart";

class NDArrayBlockedImpl extends NDArrayBase {
  final List _data;

  final _DataInfo _dataInfo;

  final _MatrixInfo _matrixInfo;

  factory NDArrayBlockedImpl(value, NDDescriptor descriptor, NDArray reuse) {
    _checkDimension(descriptor.shape.dimension);

    var matrixInfo = new _MatrixInfo(descriptor);

    var dataInfo = _createNormalizedDataInfo(descriptor.shape, matrixInfo);

    var data = _createData(descriptor, dataInfo, matrixInfo, reuse);

    _loadData(value, data, descriptor, dataInfo, matrixInfo);

    return new NDArrayBlockedImpl._(data, descriptor, dataInfo, matrixInfo);
  }

  factory NDArrayBlockedImpl.filled(
      fillValue, NDDescriptor descriptor, NDArrayBlockedImpl reuse) {
    if (fillValue == 0) {
      _checkDimension(descriptor.shape.dimension);

      var matrixInfo = new _MatrixInfo(descriptor);

      var dataInfo = _createNormalizedDataInfo(descriptor.shape, matrixInfo);

      var data = _createData(descriptor, dataInfo, matrixInfo, reuse);

      if (reuse != null && reuse.descriptor == descriptor) {
        reuse._data.fillRange(0, reuse._data.length, new Float32x4.zero());
      }

      return new NDArrayBlockedImpl._(data, descriptor, dataInfo, matrixInfo);
    } else {
      return new NDArrayBlockedImpl.generate(
          (index) => fillValue, descriptor, reuse);
    }
  }

  factory NDArrayBlockedImpl.generate(
      generator(int index), NDDescriptor descriptor, NDArray reuse) {
    _checkDimension(descriptor.shape.dimension);

    var matrixInfo = new _MatrixInfo(descriptor);

    var dataInfo = _createNormalizedDataInfo(descriptor.shape, matrixInfo);

    var data = _createData(descriptor, dataInfo, matrixInfo, reuse);

    _generateData(generator, data, descriptor, dataInfo, matrixInfo);

    return new NDArrayBlockedImpl._(data, descriptor, dataInfo, matrixInfo);
  }

  factory NDArrayBlockedImpl.castFrom(
      NDArrayBase fromArray, NDDataType toDataType, NDArray reuse) {
    var resultDescriptor = fromArray.descriptor.cast(toDataType);

    _checkDimension(resultDescriptor.shape.dimension);

    var matrixInfo = new _MatrixInfo(resultDescriptor);

    var dataInfo =
        _createNormalizedDataInfo(resultDescriptor.shape, matrixInfo);

    var data = _createData(resultDescriptor, dataInfo, matrixInfo, reuse);

    _castData(fromArray, data, resultDescriptor, dataInfo, matrixInfo);

    return new NDArrayBlockedImpl._(
        data, resultDescriptor, dataInfo, matrixInfo);
  }

  NDArrayBlockedImpl._(
      this._data, NDDescriptor descriptor, this._dataInfo, this._matrixInfo)
      : super.raw(descriptor);

  @override
  Iterable get valueIterable =>
      _createValueIterable(_data, descriptor, _dataInfo, _matrixInfo);

  @override
  dynamic toValue() {
    var value = new List(shape[0]);

    var values = new List(descriptor.shape.dimension - 1);
    values[0] = value;
    var depth = 1;

    var indexes = new List.filled(descriptor.shape.dimension - 1, 0);

    var dataIndex = 0;
    var dataIndexes = new List(shape.dimension - 1);
    dataIndexes[0] = 0;

    var i = 0;

    while (i < descriptor.shape.length) {
      var value = values[depth - 1];

      if (depth == descriptor.shape.dimension - 1) {
        for (var row = 0; row < _matrixInfo.rows; row++) {
          List<num> rowValues = new List(_matrixInfo.columns);

          value[row] = rowValues;

          var column;
          for (column = 0;
              column < _matrixInfo.dataColumns - descriptor.dataType.blockSize;
              column += descriptor.dataType.blockSize) {
            var value4 = _data[dataIndex];

            rowValues[column] = value4.x;
            rowValues[column + 1] = value4.y;
            rowValues[column + 2] = value4.z;
            rowValues[column + 3] = value4.w;

            dataIndex += _matrixInfo.delta1;

            i += descriptor.dataType.blockSize;
          }

          var value4 = _data[dataIndex];

          switch (_matrixInfo.lastBlockColumnOffset) {
            case 3:
              rowValues[column] = value4.x;
              rowValues[column + 1] = value4.y;
              rowValues[column + 2] = value4.z;

              break;
            case 2:
              rowValues[column] = value4.x;
              rowValues[column + 1] = value4.y;

              break;
            case 1:
              rowValues[column] = value4.x;

              break;
            case 0:
              rowValues[column] = value4.x;
              rowValues[column + 1] = value4.y;
              rowValues[column + 2] = value4.z;
              rowValues[column + 3] = value4.w;

              break;
          }

          dataIndex += _matrixInfo.delta1;

          i += _matrixInfo.lastBlockColumnOffset == 0
              ? descriptor.dataType.blockSize
              : _matrixInfo.lastBlockColumnOffset;

          if (row % descriptor.dataType.blockSize <
              descriptor.dataType.blockSize - 1) {
            dataIndex += _matrixInfo.delta2;
          } else {
            dataIndex += _matrixInfo.delta3;
          }
        }

        depth--;

        if (depth > 0) {
          dataIndex = dataIndexes[depth] =
              dataIndexes[depth] + _dataInfo.headStride[depth - 1];
        }
      } else {
        var index = indexes[depth - 1];

        if (index < descriptor.shape[depth - 1]) {
          value[index] = new List(descriptor.shape[depth]);

          values[depth] = value[index];

          indexes[depth - 1]++;
          indexes[depth] = 0;

          dataIndexes[depth] = dataIndex;

          depth++;
        } else {
          depth--;

          if (depth > 0) {
            dataIndex = dataIndexes[depth] =
                dataIndexes[depth] + _dataInfo.headStride[depth - 1];
          }
        }
      }
    }

    return value;
  }

  @override
  bool get isNormalized =>
      _dataInfo == _createNormalizedDataInfo(shape, _matrixInfo);

  @override
  NDArray normalize({NDArray reuse}) {
    if (isNormalized) {
      return this;
    } else {
      return elementWiseUnaryOperationInternal(
          descriptor, reuse, (value) => value);
    }
  }

  @override
  NDArray reshape({List<int> newDimensions, NDArray reuse}) {
    if (_iterableEquality.equals(shape.dimensions, newDimensions)) {
      return this;
    } else {
      _checkDimension(newDimensions.length);

      var resultDescriptor = descriptor.reshape(newDimensions: newDimensions);

      if (isNormalized &&
          newDimensions[shape.dimension - 2] ==
              shape.dimensions[shape.dimension - 2] &&
          newDimensions[shape.dimension - 1] ==
              shape.dimensions[shape.dimension - 1]) {
        var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

        var resultDataInfo =
            _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

        return new NDArrayBlockedImpl._(
            _data, resultDescriptor, resultDataInfo, resultMatrixInfo);
      } else {
        return elementWiseUnaryOperationInternal(
            resultDescriptor, reuse, (value) => value);
      }
    }
  }

  @override
  NDArray transpose({List<int> permutationAxis, NDArray reuse}) {
    var resultDescriptor =
        descriptor.transpose(permutationAxis: permutationAxis);

    var newPermutationAxis = permutationAxis ??
        new List.generate(
            shape.dimension, (index) => shape.dimension - index - 1);

    if (newPermutationAxis[shape.dimension - 2] == shape.dimension - 2 &&
        newPermutationAxis[shape.dimension - 1] == shape.dimension - 1) {
      var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

      var resultHeadStride = new List(shape.dimension - 2);

      for (var i = 0; i < newPermutationAxis.length - 2; i++) {
        var permutationAxe = newPermutationAxis[i];
        resultHeadStride[i] = _dataInfo.headStride[permutationAxe];
      }

      var resultDataInfo = new _DataInfo(resultHeadStride);

      return new NDArrayBlockedImpl._(
          _data, resultDescriptor, resultDataInfo, resultMatrixInfo);
    } else {
      return elementWiseUnaryOperationInternal(
          resultDescriptor, reuse, (value) => value);
    }
  }

  @override
  NDArrayBlockedImpl elementWiseUnaryOperationInternal(
      NDDescriptor resultDescriptor, NDArray reuse, unaryOperation(value)) {
    var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

    var resultDataInfo =
        _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

    var resultData =
        _createData(resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

    var valueIterable =
        _createValueIterable(_data, descriptor, _dataInfo, _matrixInfo);

    var valueIterator = valueIterable.iterator;

    var depth = 1;

    var indexes = new List.filled(resultDescriptor.shape.dimension - 1, 0);

    var dataIndex = 0;
    var dataIndexes = new List(resultDescriptor.shape.dimension - 1);
    dataIndexes[0] = 0;

    var i = 0;

    while (i < resultDescriptor.shape.length) {
      if (depth == resultDescriptor.shape.dimension - 1) {
        for (var row = 0; row < resultMatrixInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  resultMatrixInfo.dataColumns -
                      resultDescriptor.dataType.blockSize;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                unaryOperation((valueIterator..moveNext()).current),
                unaryOperation((valueIterator..moveNext()).current),
                unaryOperation((valueIterator..moveNext()).current),
                unaryOperation((valueIterator..moveNext()).current));

            resultData[dataIndex] = value4;

            dataIndex += resultMatrixInfo.delta1;

            i += resultDescriptor.dataType.blockSize;
          }

          var value4;
          switch (resultMatrixInfo.lastBlockColumnOffset) {
            case 3:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  0.0);
              break;
            case 2:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  0.0,
                  0.0);
              break;
            case 1:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current),
                  0.0,
                  0.0,
                  0.0);
              break;
            case 0:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current));
              break;
          }

          resultData[dataIndex] = value4;

          dataIndex += resultMatrixInfo.delta1;

          i += resultMatrixInfo.lastBlockColumnOffset == 0
              ? resultDescriptor.dataType.blockSize
              : resultMatrixInfo.lastBlockColumnOffset;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultMatrixInfo.delta2;
          } else {
            dataIndex += resultMatrixInfo.delta3;
          }
        }

        depth--;

        if (depth > 0) {
          dataIndex = dataIndexes[depth] =
              dataIndexes[depth] + resultDataInfo.headStride[depth - 1];
        }
      } else {
        var index = indexes[depth - 1];

        if (index < resultDescriptor.shape[depth - 1]) {
          indexes[depth - 1]++;
          indexes[depth] = 0;

          dataIndexes[depth] = dataIndex;

          depth++;
        } else {
          depth--;

          // commented because is normalized
/*
        if (depth > 0) {
          dataIndex = dataIndexes[depth] =
              dataIndexes[depth] + dataInfo.headStride[depth - 1];
        }
*/
        }
      }
    }

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
  }

  @override
  NDArrayBase elementWiseBinaryOperationInternal(
      NDArrayBase array2,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      binaryOperation(value1, value2)) {
    // TODO to implement NDArrayBlockedImpl.elementWiseBinaryOperationInternal
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.elementWiseBinaryOperationInternal: $this");
  }

  @override
  NDArrayBase elementWiseTernaryOperationInternal(
      NDArrayBase array2,
      NDArrayBase array3,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      ternaryOperation(value1, value2, value3)) {
    // TODO to implement NDArrayBlockedImpl.elementWiseTernaryOperationInternal
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.elementWiseTernaryOperationInternal: $this");
  }

  @override
  NDArray reduceOperationInternal(List<int> reductionAxis, bool keepDimensions,
      NDDescriptor resultDescriptor, NDArray reuse,
      {void initReduction(),
      void onValueToReduce(int valueIndex, value),
      reduce()}) {
    // TODO to implement NDArrayBlockedImpl.reduceOperationInternal
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.reduceOperationInternal: $this");
  }

  @override
  NDArray matMul(value2, {NDArray reuse}) {
    // TODO to implement NDArrayBlockedImpl.matMul
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.matMul: $this");
  }

  @override
  NDArray tile(List<int> multiplies, {NDArray reuse}) {
    // TODO to implement NDArrayBlockedImpl.tile
    throw new UnimplementedError("to implement NDArrayBlockedImpl.tile: $this");
  }

  @override
  String toString() =>
      "<value: ${toValue()}, shape: $shape, dataType: $dataType, heasStride: ${_dataInfo.headStride}>";
}

final _iterableEquality = new IterableEquality<dynamic>();

void _checkDimension(int dimension) {
  if (dimension < 2) {
    throw new ArgumentError.value(
        dimension, "Shape dimensions at least 2 for blocked array");
  }
}

_DataInfo _createNormalizedDataInfo(NDShape shape, _MatrixInfo matrixInfo) {
  List<int> stride = new List(shape.dimension - 2);

  if (stride.isNotEmpty) {
    var factor = matrixInfo.dataLength;
    for (var i = shape.dimension - 3; i >= 0; i--) {
      stride[i] = factor;
      factor *= shape[i];
    }
  }

  return new _DataInfo(stride);
}

List _createData(NDDescriptor descriptor, _DataInfo dataInfo,
    _MatrixInfo matrixInfo, NDArrayBlockedImpl reuse) {
  if (descriptor.dataType == NDDataType.unknown) {
    throw new ArgumentError.value(descriptor.dataType.toString(), "data type");
  }

  var dataLength;
  if (dataInfo.headStride.isNotEmpty) {
    dataLength = descriptor.shape.dimensions.first * dataInfo.headStride.first;
  } else {
    dataLength = matrixInfo.dataLength;
  }

  if (reuse != null && reuse.descriptor == descriptor) {
    return reuse._data;
  } else {
    switch (descriptor.dataType) {
      case NDDataType.float32HBlocked:
      case NDDataType.float32VBlocked:
        return new Float32x4List(dataLength);
      default:
        throw new StateError("DEAD CODE");
    }
  }
}

void _loadData(value, Float32x4List data, NDDescriptor descriptor,
    _DataInfo dataInfo, _MatrixInfo matrixInfo) {
  var values = new List(descriptor.shape.dimension - 1);
  values[0] = value;
  var depth = 1;

  var indexes = new List.filled(descriptor.shape.dimension - 1, 0);

  var dataIndex = 0;
  var dataIndexes = new List(descriptor.shape.dimension - 1);
  dataIndexes[0] = 0;

  var i = 0;

  while (i < descriptor.shape.length) {
    var value = values[depth - 1];

    if (depth == descriptor.shape.dimension - 1) {
      for (var row = 0; row < matrixInfo.rows; row++) {
        List<num> rowValues = value[row];

        var column;
        for (column = 0;
            column < matrixInfo.dataColumns - descriptor.dataType.blockSize;
            column += descriptor.dataType.blockSize) {
          var value4 = new Float32x4(
              rowValues[column].toDouble(),
              rowValues[column + 1].toDouble(),
              rowValues[column + 2].toDouble(),
              rowValues[column + 3].toDouble());

          data[dataIndex] = value4;

          dataIndex += matrixInfo.delta1;

          i += descriptor.dataType.blockSize;
        }

        var value4;
        switch (matrixInfo.lastBlockColumnOffset) {
          case 3:
            value4 = new Float32x4(
                rowValues[column].toDouble(),
                rowValues[column + 1].toDouble(),
                rowValues[column + 2].toDouble(),
                0.0);
            break;
          case 2:
            value4 = new Float32x4(rowValues[column].toDouble(),
                rowValues[column + 1].toDouble(), 0.0, 0.0);
            break;
          case 1:
            value4 = new Float32x4(rowValues[column].toDouble(), 0.0, 0.0, 0.0);
            break;
          case 0:
            value4 = new Float32x4(
                rowValues[column].toDouble(),
                rowValues[column + 1].toDouble(),
                rowValues[column + 2].toDouble(),
                rowValues[column + 3].toDouble());
            break;
        }

        data[dataIndex] = value4;

        dataIndex += matrixInfo.delta1;

        i += matrixInfo.lastBlockColumnOffset == 0
            ? descriptor.dataType.blockSize
            : matrixInfo.lastBlockColumnOffset;

        if (row % descriptor.dataType.blockSize <
            descriptor.dataType.blockSize - 1) {
          dataIndex += matrixInfo.delta2;
        } else {
          dataIndex += matrixInfo.delta3;
        }
      }

      depth--;

      if (depth > 0) {
        dataIndex = dataIndexes[depth] =
            dataIndexes[depth] + dataInfo.headStride[depth - 1];
      }
    } else {
      var index = indexes[depth - 1];

      if (index < descriptor.shape[depth - 1]) {
        values[depth] = value[index];

        indexes[depth - 1]++;
        indexes[depth] = 0;

        dataIndexes[depth] = dataIndex;

        depth++;
      } else {
        depth--;

        // commented because is normalized
/*
        if (depth > 0) {
          dataIndex = dataIndexes[depth] =
              dataIndexes[depth] + dataInfo.headStride[depth - 1];
        }
*/
      }
    }
  }
}

void _generateData(generator(int index), Float32x4List data,
    NDDescriptor descriptor, _DataInfo dataInfo, _MatrixInfo matrixInfo) {
  var depth = 1;

  var indexes = new List.filled(descriptor.shape.dimension - 1, 0);

  var dataIndex = 0;
  var dataIndexes = new List(descriptor.shape.dimension - 1);
  dataIndexes[0] = 0;

  var i = 0;

  while (i < descriptor.shape.length) {
    if (depth == descriptor.shape.dimension - 1) {
      for (var row = 0; row < matrixInfo.rows; row++) {
        var column;
        for (column = 0;
            column < matrixInfo.dataColumns - descriptor.dataType.blockSize;
            column += descriptor.dataType.blockSize) {
          var value4 = new Float32x4(
              generator(i).toDouble(),
              generator(i + 1).toDouble(),
              generator(i + 2).toDouble(),
              generator(i + 3).toDouble());

          data[dataIndex] = value4;

          dataIndex += matrixInfo.delta1;

          i += descriptor.dataType.blockSize;
        }

        var value4;
        switch (matrixInfo.lastBlockColumnOffset) {
          case 3:
            value4 = new Float32x4(generator(i).toDouble(),
                generator(i + 1).toDouble(), generator(i + 2).toDouble(), 0.0);
            break;
          case 2:
            value4 = new Float32x4(
                generator(i).toDouble(), generator(i + 1).toDouble(), 0.0, 0.0);
            break;
          case 1:
            value4 = new Float32x4(generator(i).toDouble(), 0.0, 0.0, 0.0);
            break;
          case 0:
            value4 = new Float32x4(
                generator(i).toDouble(),
                generator(i + 1).toDouble(),
                generator(i + 2).toDouble(),
                generator(i + 3).toDouble());
            break;
        }

        data[dataIndex] = value4;

        dataIndex += matrixInfo.delta1;

        i += matrixInfo.lastBlockColumnOffset == 0
            ? descriptor.dataType.blockSize
            : matrixInfo.lastBlockColumnOffset;

        if (row % descriptor.dataType.blockSize <
            descriptor.dataType.blockSize - 1) {
          dataIndex += matrixInfo.delta2;
        } else {
          dataIndex += matrixInfo.delta3;
        }
      }

      depth--;

      if (depth > 0) {
        dataIndex = dataIndexes[depth] =
            dataIndexes[depth] + dataInfo.headStride[depth - 1];
      }
    } else {
      var index = indexes[depth - 1];

      if (index < descriptor.shape[depth - 1]) {
        indexes[depth - 1]++;
        indexes[depth] = 0;

        dataIndexes[depth] = dataIndex;

        depth++;
      } else {
        depth--;

        // commented because is normalized
/*
        if (depth > 0) {
          dataIndex = dataIndexes[depth] =
              dataIndexes[depth] + dataInfo.headStride[depth - 1];
        }
*/
      }
    }
  }
}

void _castData(NDArrayBase fromArray, List data, NDDescriptor descriptor,
    _DataInfo dataInfo, _MatrixInfo matrixInfo) {
  if (fromArray.dataType.isFloat && descriptor.dataType.isFloat) {
    return _castConvertedData(fromArray, data, descriptor, dataInfo, matrixInfo,
        (num value) => value);
  } else if (fromArray.dataType.isInteger && descriptor.dataType.isFloat) {
    return _castConvertedData(fromArray, data, descriptor, dataInfo, matrixInfo,
        (int value) => value.toDouble());
  } else {
    throw new UnsupportedError(
        "Cast from ${fromArray.dataType} to ${descriptor.dataType}");
  }
}

void _castConvertedData(
    NDArrayBase fromArray,
    List resultData,
    NDDescriptor resultDescriptor,
    _DataInfo resultDataInfo,
    _MatrixInfo resultMatrixInfo,
    dynamic converter(value)) {
  var valueIterator = fromArray.valueIterable.iterator;

  var depth = 1;

  var indexes = new List.filled(resultDescriptor.shape.dimension - 1, 0);

  var dataIndex = 0;
  var dataIndexes = new List(resultDescriptor.shape.dimension - 1);
  dataIndexes[0] = 0;

  var i = 0;

  while (i < resultDescriptor.shape.length) {
    if (depth == resultDescriptor.shape.dimension - 1) {
      for (var row = 0; row < resultMatrixInfo.rows; row++) {
        var column;
        for (column = 0;
            column <
                resultMatrixInfo.dataColumns -
                    resultDescriptor.dataType.blockSize;
            column += resultDescriptor.dataType.blockSize) {
          var value4 = new Float32x4(
              converter((valueIterator..moveNext()).current),
              converter((valueIterator..moveNext()).current),
              converter((valueIterator..moveNext()).current),
              converter((valueIterator..moveNext()).current));

          resultData[dataIndex] = value4;

          dataIndex += resultMatrixInfo.delta1;

          i += resultDescriptor.dataType.blockSize;
        }

        var value4;
        switch (resultMatrixInfo.lastBlockColumnOffset) {
          case 3:
            value4 = new Float32x4(
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                0.0);
            break;
          case 2:
            value4 = new Float32x4(
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                0.0,
                0.0);
            break;
          case 1:
            value4 = new Float32x4(
                converter((valueIterator..moveNext()).current), 0.0, 0.0, 0.0);
            break;
          case 0:
            value4 = new Float32x4(
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current));
            break;
        }

        resultData[dataIndex] = value4;

        dataIndex += resultMatrixInfo.delta1;

        i += resultMatrixInfo.lastBlockColumnOffset == 0
            ? resultDescriptor.dataType.blockSize
            : resultMatrixInfo.lastBlockColumnOffset;

        if (row % resultDescriptor.dataType.blockSize <
            resultDescriptor.dataType.blockSize - 1) {
          dataIndex += resultMatrixInfo.delta2;
        } else {
          dataIndex += resultMatrixInfo.delta3;
        }
      }

      depth--;

      if (depth > 0) {
        dataIndex = dataIndexes[depth] =
            dataIndexes[depth] + resultDataInfo.headStride[depth - 1];
      }
    } else {
      var index = indexes[depth - 1];

      if (index < resultDescriptor.shape[depth - 1]) {
        indexes[depth - 1]++;
        indexes[depth] = 0;

        dataIndexes[depth] = dataIndex;

        depth++;
      } else {
        depth--;

        // commented because is normalized
/*
        if (depth > 0) {
          dataIndex = dataIndexes[depth] =
              dataIndexes[depth] + dataInfo.headStride[depth - 1];
        }
*/
      }
    }
  }
}

Iterable<num> _createValueIterable(List _data, NDDescriptor descriptor,
    _DataInfo dataInfo, _MatrixInfo matrixInfo) sync* {
  var depth = 1;

  var indexes = new List.filled(descriptor.shape.dimension - 1, 0);

  var dataIndex = 0;
  var dataIndexes = new List(descriptor.shape.dimension - 1);
  dataIndexes[0] = 0;

  var i = 0;

  while (i < descriptor.shape.length) {
    if (depth == descriptor.shape.dimension - 1) {
      for (var row = 0; row < matrixInfo.rows; row++) {
        var column;
        for (column = 0;
            column < matrixInfo.dataColumns - descriptor.dataType.blockSize;
            column += descriptor.dataType.blockSize) {
          var value4 = _data[dataIndex];

          yield value4.x;
          yield value4.y;
          yield value4.z;
          yield value4.w;

          dataIndex += matrixInfo.delta1;

          i += descriptor.dataType.blockSize;
        }

        var value4 = _data[dataIndex];

        switch (matrixInfo.lastBlockColumnOffset) {
          case 3:
            yield value4.x;
            yield value4.y;
            yield value4.z;

            break;
          case 2:
            yield value4.x;
            yield value4.y;

            break;
          case 1:
            yield value4.x;

            break;
          case 0:
            yield value4.x;
            yield value4.y;
            yield value4.z;
            yield value4.w;

            break;
        }

        dataIndex += matrixInfo.delta1;

        i += matrixInfo.lastBlockColumnOffset == 0
            ? descriptor.dataType.blockSize
            : matrixInfo.lastBlockColumnOffset;

        if (row % descriptor.dataType.blockSize <
            descriptor.dataType.blockSize - 1) {
          dataIndex += matrixInfo.delta2;
        } else {
          dataIndex += matrixInfo.delta3;
        }
      }

      depth--;

      if (depth > 0) {
        dataIndex = dataIndexes[depth] =
            dataIndexes[depth] + dataInfo.headStride[depth - 1];
      }
    } else {
      var index = indexes[depth - 1];

      if (index < descriptor.shape[depth - 1]) {
        indexes[depth - 1]++;
        indexes[depth] = 0;

        dataIndexes[depth] = dataIndex;

        depth++;
      } else {
        depth--;

        if (depth > 0) {
          dataIndex = dataIndexes[depth] =
              dataIndexes[depth] + dataInfo.headStride[depth - 1];
        }
      }
    }
  }
}

class _MatrixInfo {
  final int rows;

  final int columns;

  final int blockRows;

  final int blockColumns;

  final int dataRows;

  final int dataColumns;

  final int dataLength;

  final int lastBlockRowOffset;

  final int lastBlockColumnOffset;

  final int delta1;

  final int delta2;

  final int delta3;

  factory _MatrixInfo(NDDescriptor descriptor) {
    var rows = descriptor.shape.dimensions[descriptor.shape.dimension - 2];
    var columns = descriptor.shape.dimensions[descriptor.shape.dimension - 1];

    var blockRows = (rows + descriptor.dataType.blockSize - 1) >>
        descriptor.dataType.blockDepth; // equal (/4).ceil
    var blockColumns = (columns + descriptor.dataType.blockSize - 1) >>
        descriptor.dataType.blockDepth; // equal (/4).ceil

    var dataRows = blockRows << descriptor.dataType.blockDepth;
    var dataColumns = blockColumns << descriptor.dataType.blockDepth;

    var dataLength = dataRows * blockColumns;

    var lastBlockRowOffset =
        rows & (descriptor.dataType.blockSize - 1); // equal to % 4
    var lastBlockColumnOffset =
        columns & (descriptor.dataType.blockSize - 1); // equal to % 4

    var delta1;
    if (descriptor.dataType.isHBlocked) {
      delta1 = descriptor.dataType.blockSize;
    } else {
      delta1 = dataRows;
    }

    var delta2;
    if (descriptor.dataType.isHBlocked) {
      delta2 = 1 - dataColumns;
    } else {
      delta2 = 1 - dataLength;
    }

    var delta3;
    if (descriptor.dataType.isHBlocked) {
      delta3 = 1 - descriptor.dataType.blockSize;
    } else {
      delta3 = delta2;
    }

    return new _MatrixInfo._(
        rows,
        columns,
        blockRows,
        blockColumns,
        dataRows,
        dataColumns,
        dataLength,
        lastBlockRowOffset,
        lastBlockColumnOffset,
        delta1,
        delta2,
        delta3);
  }

  _MatrixInfo._(
      this.rows,
      this.columns,
      this.blockRows,
      this.blockColumns,
      this.dataRows,
      this.dataColumns,
      this.dataLength,
      this.lastBlockRowOffset,
      this.lastBlockColumnOffset,
      this.delta1,
      this.delta2,
      this.delta3);
}

class _DataInfo {
  final List<int> headStride;

  _DataInfo(this.headStride);

  @override
  // ignore: hash_and_equals
  bool operator ==(other) {
    if (other is _DataInfo) {
      return _iterableEquality.equals(headStride, other.headStride);
    } else {
      return false;
    }
  }
}

// TODO ottimizzare le matematiche

/*
NDArray matMulFloat32x4(NDArray array1, NDArray array2) {
  var resultDescriptor = array1.descriptor.matMul(array2.descriptor);

  var list1 = _toFloat32x4ListHBlock(array1);

  var list2 = _toFloat32x4ListVBlock(array2);

  var resultList =
      _matMulFloat32x4Internal(list1, array1.shape, list2, array2.shape);

  return _fromFloat32x4ListHBlock(resultList, resultDescriptor);
}

Float32x4List _matMulFloat32x4Internal(
    Float32x4List list1, NDShape shape1, Float32x4List list2, NDShape shape2) {
  var rows1 = _toCeil4(shape1[shape1.dimension - 2]);
  var columns1 = _toCeil4(shape1[shape1.dimension - 1]);
  var rows2 = _toCeil4(shape2[shape2.dimension - 2]);
  var columns2 = _toCeil4(shape2[shape2.dimension - 1]);

  var resultList = new Float32x4List(rows1 * columns2 ~/ 4);

  var blockPerRow1 = columns1 >> 2;
  var blockPerColumn2 = rows2 >> 2;
  var resultBlockPerRow = columns2 >> 2;

  for (var row1 = 0; row1 < rows1; row1 += 4) {
    var blockRow1 = row1 ~/ 4;
    var blockOffset1 = row1 % 4;

    for (var column2 = 0; column2 < columns2; column2 += 4) {
      var blockColumn2 = column2 ~/ 4;

      var result0 = new Float32x4.zero();
      var result1 = new Float32x4.zero();
      var result2 = new Float32x4.zero();
      var result3 = new Float32x4.zero();

      for (var i = 0; i < columns1; i += 4) {
        var blockColumn1 = i ~/ 4;
        var blockIndex1 = blockRow1 * blockPerRow1 + blockColumn1;
        var i10 = blockIndex1 * 4 + blockOffset1;

        var blockRow2 = i ~/ 4;
        var blockOffset2 = i % 4;
        var blockIndex2 = blockColumn2 * blockPerColumn2 + blockRow2;
        var i20 = blockIndex2 * 4 + blockOffset2;

        var b0 = list2[i20++];
        var b1 = list2[i20++];
        var b2 = list2[i20++];
        var b3 = list2[i20++];

        var a0 = list1[i10++];
        result0 += a0.shuffle(Float32x4.XXXX) * b0 +
            a0.shuffle(Float32x4.YYYY) * b1 +
            a0.shuffle(Float32x4.ZZZZ) * b2 +
            a0.shuffle(Float32x4.WWWW) * b3;

        var a1 = list1[i10++];
        result1 += a1.shuffle(Float32x4.XXXX) * b0 +
            a1.shuffle(Float32x4.YYYY) * b1 +
            a1.shuffle(Float32x4.ZZZZ) * b2 +
            a1.shuffle(Float32x4.WWWW) * b3;

        var a2 = list1[i10++];
        result2 += a2.shuffle(Float32x4.XXXX) * b0 +
            a2.shuffle(Float32x4.YYYY) * b1 +
            a2.shuffle(Float32x4.ZZZZ) * b2 +
            a2.shuffle(Float32x4.WWWW) * b3;

        var a3 = list1[i10++];
        result3 += a3.shuffle(Float32x4.XXXX) * b0 +
            a3.shuffle(Float32x4.YYYY) * b1 +
            a3.shuffle(Float32x4.ZZZZ) * b2 +
            a3.shuffle(Float32x4.WWWW) * b3;
      }

      var resultBlockIndex = blockRow1 * resultBlockPerRow + blockColumn2;

      var ir0 = resultBlockIndex * 4 + blockOffset1;
      resultList[ir0++] = result0;
      resultList[ir0++] = result1;
      resultList[ir0++] = result2;
      resultList[ir0++] = result3;
    }
  }

  return resultList;
}

int _toCeil4(int value) => 4 * (value / 4).ceil();

Float32x4List _toFloat32x4ListHBlock(NDArrayImpl array) {
  var blockPerColumn = (array.shape[array.shape.dimension - 2] / 4).ceil();
  var blockPerRow = (array.shape[array.shape.dimension - 1] / 4).ceil();
  var lastBlockColumnOffset = array.shape[array.shape.dimension - 1] % 4;

  var data = new Float32x4List(4 * blockPerColumn * blockPerRow);

  var values = array.reshape(newDimensions: [-1]).toVector();

  var i1 = 0;
  var i2 = 0;
  while (i2 < values.length) {
    var offsetBlockColumn = i1 % blockPerRow;

    var value4;
    if (lastBlockColumnOffset == 0 || offsetBlockColumn < blockPerRow - 1) {
      value4 =
          new Float32x4(values[i2++], values[i2++], values[i2++], values[i2++]);
    } else {
      switch (lastBlockColumnOffset) {
        case 3:
          value4 = new Float32x4(values[i2++], values[i2++], values[i2++], 0.0);
          break;
        case 2:
          value4 = new Float32x4(values[i2++], values[i2++], 0.0, 0.0);
          break;
        case 1:
          value4 = new Float32x4(values[i2++], 0.0, 0.0, 0.0);
          break;
      }
    }

    var index = i1 ~/ blockPerRow;
    var blockRow = index ~/ 4;
    var blockRowOffset = index % 4;
    var blockColumn = i1 % blockPerRow;
    var blockIndex = blockRow * blockPerRow + blockColumn;
    var i3 = blockIndex * 4 + blockRowOffset;

    data[i3] = value4;

    i1++;
  }

  return data;
}

Float32x4List _toFloat32x4ListVBlock(NDArrayImpl array) {
  var blockPerColumn = (array.shape[array.shape.dimension - 2] / 4).ceil();
  var blockPerRow = (array.shape[array.shape.dimension - 1] / 4).ceil();
  var lastBlockColumnOffset = array.shape[array.shape.dimension - 1] % 4;

  var data = new Float32x4List(4 * blockPerColumn * blockPerRow);

  var values = array.reshape(newDimensions: [-1]).toVector();

  var i1 = 0;
  var i2 = 0;
  while (i2 < values.length) {
    var offsetBlockColumn = i1 % blockPerRow;

    var value4;
    if (lastBlockColumnOffset == 0 || offsetBlockColumn < blockPerRow - 1) {
      value4 =
          new Float32x4(values[i2++], values[i2++], values[i2++], values[i2++]);
    } else {
      switch (lastBlockColumnOffset) {
        case 3:
          value4 = new Float32x4(values[i2++], values[i2++], values[i2++], 0.0);
          break;
        case 2:
          value4 = new Float32x4(values[i2++], values[i2++], 0.0, 0.0);
          break;
        case 1:
          value4 = new Float32x4(values[i2++], 0.0, 0.0, 0.0);
          break;
      }
    }

    var index = i1 ~/ blockPerRow;
    var blockRow = index ~/ 4;
    var blockRowOffset = index % 4;
    var blockColumn = i1 % blockPerRow;
    var blockIndex = blockColumn * blockPerColumn + blockRow;
    var i3 = blockIndex * 4 + blockRowOffset;

    data[i3] = value4;

    i1++;
  }

  return data;
}

NDArray _fromFloat32x4ListHBlock(
    Float32x4List list, NDDescriptor resultDescriptor) {
  var resultData = new Float32List(
      resultDescriptor.shape[resultDescriptor.shape.dimension - 2] *
          resultDescriptor.shape[resultDescriptor.shape.dimension - 1]);

  var blockPerRow =
      (resultDescriptor.shape[resultDescriptor.shape.dimension - 1] / 4).ceil();
  var lastBlockColumnOffset =
      resultDescriptor.shape[resultDescriptor.shape.dimension - 1] % 4;

  var i1 = 0;
  var i2 = 0;
  while (i2 < resultData.length) {
    var offsetBlockColumn = i1 % blockPerRow;

    var index = i1 ~/ blockPerRow;
    var blockRow = index ~/ 4;
    var blockRowOffset = index % 4;
    var blockColumn = i1 % blockPerRow;
    var blockIndex = blockRow * blockPerRow + blockColumn;
    var i3 = blockIndex * 4 + blockRowOffset;

    var value4 = list[i3];
    if (lastBlockColumnOffset == 0 || offsetBlockColumn < blockPerRow - 1) {
      resultData[i2++] = value4.x;
      resultData[i2++] = value4.y;
      resultData[i2++] = value4.z;
      resultData[i2++] = value4.w;
    } else {
      switch (lastBlockColumnOffset) {
        case 3:
          resultData[i2++] = value4.x;
          resultData[i2++] = value4.y;
          resultData[i2++] = value4.z;
          break;
        case 2:
          resultData[i2++] = value4.x;
          resultData[i2++] = value4.y;
          break;
        case 1:
          resultData[i2++] = value4.x;
          break;
      }
    }

    i1++;
  }

  var resultShape = resultDescriptor.shape;
  var resultStride = _calculateDefaultStride(resultShape);

  return new NDArrayImpl._(resultData, resultDescriptor, resultStride, 0);
}
*/
