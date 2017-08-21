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
  Iterable get valueIterable => _createValueIterable(this);

  @override
  dynamic toValue() {
    var value = new List(shape[0]);

    var values = new List(shape.dimension - 1);
    values[0] = value;

    var shapeIndex = 0;

    var dimensionIndexes = new List(shape.dimension - 1);
    var dataIndexes = new List(shape.dimension - 1);

    var dataIndex = 0;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var i = 0;

    while (i < shape.length) {
      var value = values[shapeIndex];

      if (shapeIndex == shape.dimension - 2) {
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

          switch (_matrixInfo.lastBlockColumnCount) {
            case 4:
              rowValues[column] = value4.x;
              rowValues[column + 1] = value4.y;
              rowValues[column + 2] = value4.z;
              rowValues[column + 3] = value4.w;

              break;
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
          }

          dataIndex += _matrixInfo.delta1;

          i += _matrixInfo.lastBlockColumnCount;

          if (row % descriptor.dataType.blockSize <
              descriptor.dataType.blockSize - 1) {
            dataIndex += _matrixInfo.delta2;
          } else {
            dataIndex += _matrixInfo.delta3;
          }
        }

        shapeIndex--;
      } else {
        if (dimensionIndexes[shapeIndex] < shape[shapeIndex]) {
          var newList = new List(descriptor.shape[shapeIndex + 1]);

          value[dimensionIndexes[shapeIndex]] = newList;

          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          dataIndexes[shapeIndex] =
              dataIndex + _dataInfo.headStride[shapeIndex];

          shapeIndex++;

          values[shapeIndex] = newList;

          dimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = dataIndex;
        } else {
          shapeIndex--;
        }
      }
    }

    return value;
  }

  @override
  bool get isNormalized => true;

  @override
  NDArray normalize({NDArray reuse}) => this;

  NDArray identity({NDArray reuse}) {
    var targetDescriptor = descriptor;

    var targetMatrixInfo = new _MatrixInfo(targetDescriptor);

    var targetDataInfo =
        _createNormalizedDataInfo(targetDescriptor.shape, targetMatrixInfo);

    var targetData =
        _createData(targetDescriptor, targetDataInfo, targetMatrixInfo, reuse);

    _identityData(
        this, targetData, targetDescriptor, targetDataInfo, targetMatrixInfo);

    return new NDArrayBlockedImpl._(
        targetData, targetDescriptor, targetDataInfo, targetMatrixInfo);
  }

  @override
  NDArray reshape({List<int> newDimensions, NDArray reuse}) {
    if (_iterableEquality.equals(shape.dimensions, newDimensions)) {
      return this;
    } else {
      _checkDimension(newDimensions.length);

      var resultDescriptor = descriptor.reshape(newDimensions: newDimensions);

      if (newDimensions[shape.dimension - 2] ==
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
    if (permutationAxis != null &&
        !permutationAxis.every((index) => permutationAxis[index] == index)) {
      var resultDescriptor =
          descriptor.transpose(permutationAxis: permutationAxis);

      var newPermutationAxis = permutationAxis ??
          new List.generate(
              shape.dimension, (index) => shape.dimension - index - 1);

      var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

      var sourceHeadStride = new List(shape.dimension - 2);
      for (var i = 0; i < newPermutationAxis.length - 2; i++) {
        var permutationAxe = newPermutationAxis[i];
        sourceHeadStride[i] = _dataInfo.headStride[permutationAxe];
      }

      var resultDataInfo =
          _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

      var resultData = _createData(
          resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

      var matrixUnchanged =
          newPermutationAxis[shape.dimension - 2] == shape.dimension - 2 &&
              newPermutationAxis[shape.dimension - 1] == shape.dimension - 1;

      if (matrixUnchanged ||
          newPermutationAxis[shape.dimension - 2] == shape.dimension - 1 &&
              newPermutationAxis[shape.dimension - 1] == shape.dimension - 2) {
        if (matrixUnchanged) {
          _transposeData(this, sourceHeadStride, resultData, resultDescriptor,
              resultDataInfo, resultMatrixInfo);
        } else {
          _transposeSwitchedData(this, sourceHeadStride, resultData,
              resultDescriptor, resultDataInfo, resultMatrixInfo);
        }

        return new NDArrayBlockedImpl._(
            resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
      } else {
        // TODO ottimizzare con createTransposedValueIterable
        return cast(NDDataType.float32)
            .transpose(permutationAxis: newPermutationAxis)
            .cast(dataType, reuse: reuse);
      }
    } else {
      return this;
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

    var valueIterable = _createValueIterable(this);

    var valueIterator = valueIterable.iterator;

    var shapeIndex = 0;

    var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
    var dataIndexes = new List(resultDescriptor.shape.dimension - 1);

    var dataIndex = 0;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var i = 0;

    while (i < resultDescriptor.shape.length) {
      if (shapeIndex == resultDescriptor.shape.dimension - 2) {
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
          switch (resultMatrixInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current));
              break;
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
          }

          resultData[dataIndex] = value4;

          dataIndex += resultMatrixInfo.delta1;

          i += resultMatrixInfo.lastBlockColumnCount;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultMatrixInfo.delta2;
          } else {
            dataIndex += resultMatrixInfo.delta3;
          }
        }

        shapeIndex--;
      } else {
        if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          dataIndexes[shapeIndex] =
              dataIndex + resultDataInfo.headStride[shapeIndex];

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = dataIndex;
        } else {
          shapeIndex--;
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
    var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

    var resultDataInfo =
        _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

    var resultData =
        _createData(resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

    var valueIterable1 =
        _createBroadcastedValueIterable(this, resultDescriptor);
    var valueIterator1 = valueIterable1.iterator;
    var valueIterable2 =
        _createBroadcastedValueIterable(array2, resultDescriptor);
    var valueIterator2 = valueIterable2.iterator;

    var shapeIndex = 0;

    var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
    var dataIndexes = new List(resultDescriptor.shape.dimension - 1);

    var dataIndex = 0;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var i = 0;

    while (i < resultDescriptor.shape.length) {
      if (shapeIndex == resultDescriptor.shape.dimension - 2) {
        for (var row = 0; row < resultMatrixInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  resultMatrixInfo.dataColumns -
                      resultDescriptor.dataType.blockSize;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                binaryOperation((valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current),
                binaryOperation((valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current),
                binaryOperation((valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current),
                binaryOperation((valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current));

            resultData[dataIndex] = value4;

            dataIndex += resultMatrixInfo.delta1;

            i += resultDescriptor.dataType.blockSize;
          }

          var value4;
          switch (resultMatrixInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Float32x4(
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current));
              break;
            case 3:
              value4 = new Float32x4(
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  0.0);
              break;
            case 2:
              value4 = new Float32x4(
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  0.0,
                  0.0);
              break;
            case 1:
              value4 = new Float32x4(
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  0.0,
                  0.0,
                  0.0);
              break;
          }

          resultData[dataIndex] = value4;

          dataIndex += resultMatrixInfo.delta1;

          i += resultMatrixInfo.lastBlockColumnCount;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultMatrixInfo.delta2;
          } else {
            dataIndex += resultMatrixInfo.delta3;
          }
        }

        shapeIndex--;
      } else {
        if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          dataIndexes[shapeIndex] =
              dataIndex + resultDataInfo.headStride[shapeIndex];

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = dataIndex;
        } else {
          shapeIndex--;
        }
      }
    }

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
  }

  @override
  NDArrayBase elementWiseTernaryOperationInternal(
      NDArrayBase array2,
      NDArrayBase array3,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      ternaryOperation(value1, value2, value3)) {
    var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

    var resultDataInfo =
        _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

    var resultData =
        _createData(resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

    var valueIterable1 =
        _createBroadcastedValueIterable(this, resultDescriptor);
    var valueIterator1 = valueIterable1.iterator;
    var valueIterable2 =
        _createBroadcastedValueIterable(array2, resultDescriptor);
    var valueIterator2 = valueIterable2.iterator;
    var valueIterable3 =
        _createBroadcastedValueIterable(array3, resultDescriptor);
    var valueIterator3 = valueIterable3.iterator;

    var shapeIndex = 0;

    var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
    var dataIndexes = new List(resultDescriptor.shape.dimension - 1);

    var dataIndex = 0;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var i = 0;

    while (i < resultDescriptor.shape.length) {
      if (shapeIndex == resultDescriptor.shape.dimension - 2) {
        for (var row = 0; row < resultMatrixInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  resultMatrixInfo.dataColumns -
                      resultDescriptor.dataType.blockSize;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                ternaryOperation(
                    (valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current,
                    (valueIterator3..moveNext()).current),
                ternaryOperation(
                    (valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current,
                    (valueIterator3..moveNext()).current),
                ternaryOperation(
                    (valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current,
                    (valueIterator3..moveNext()).current),
                ternaryOperation(
                    (valueIterator1..moveNext()).current,
                    (valueIterator2..moveNext()).current,
                    (valueIterator3..moveNext()).current));

            resultData[dataIndex] = value4;

            dataIndex += resultMatrixInfo.delta1;

            i += resultDescriptor.dataType.blockSize;
          }

          var value4;
          switch (resultMatrixInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Float32x4(
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current));
              break;
            case 3:
              value4 = new Float32x4(
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  0.0);
              break;
            case 2:
              value4 = new Float32x4(
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  0.0,
                  0.0);
              break;
            case 1:
              value4 = new Float32x4(
                  ternaryOperation(
                      (valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current,
                      (valueIterator3..moveNext()).current),
                  0.0,
                  0.0,
                  0.0);
              break;
          }

          resultData[dataIndex] = value4;

          dataIndex += resultMatrixInfo.delta1;

          i += resultMatrixInfo.lastBlockColumnCount;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultMatrixInfo.delta2;
          } else {
            dataIndex += resultMatrixInfo.delta3;
          }
        }

        shapeIndex--;
      } else {
        if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          dataIndexes[shapeIndex] =
              dataIndex + resultDataInfo.headStride[shapeIndex];

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = dataIndex;
        } else {
          shapeIndex--;
        }
      }
    }

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
  }

  @override
  NDArray reduceOperationInternal(List<int> reductionAxis, bool keepDimensions,
      NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(), void onValue(value, int valueCount), end()}) {
    if (keepDimensions) {
      throw new UnimplementedError();
    }

    var newReductionAxis =
        convertToValidReductionAxis(reductionAxis, shape.dimension);

    if (newReductionAxis.isNotEmpty) {
      var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

      var resultDataInfo =
          _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

      var resultData = _createData(
          resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

      _reduceData(this, reductionAxis, keepDimensions, resultData,
          resultDescriptor, resultDataInfo, resultMatrixInfo,
          begin: begin, onValue: onValue, end: end);

      return new NDArrayBlockedImpl._(
          resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
    } else {
      return this;
    }
  }

  @override
  NDArray argOperationInternal(
      int axis, NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(),
      void onValue(dimensionIndex, value, int valueCount),
      end()}) {
    var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

    var resultDataInfo =
        _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

    var resultData =
        _createData(resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

    _argData(this, axis, resultData, resultDescriptor, resultDataInfo,
        resultMatrixInfo,
        begin: begin, onValue: onValue, end: end);

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
  }

  @override
  NDArray matMul(value2, {NDArray reuse}) {
    NDArrayBlockedImpl array2 =
        toNDArray(value2, dataType: NDDataType.float32VBlocked);

    var resultDescriptor = descriptor.matMul(array2.descriptor);

    var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

    var resultDataInfo =
        _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

    var resultData =
        _createData(resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

    _matMulData(this, array2, resultData, resultDescriptor, resultDataInfo,
        resultMatrixInfo);

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
  }

  @override
  NDArray tile(List<int> multiplies, {NDArray reuse}) {
    var resultDescriptor = descriptor.tile(multiplies);

    var resultMatrixInfo = new _MatrixInfo(resultDescriptor);

    var resultDataInfo =
        _createNormalizedDataInfo(resultDescriptor.shape, resultMatrixInfo);

    var resultData =
        _createData(resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

    var valueIterable = _createTiledValueIterable(this, resultDescriptor);

    var valueIterator = valueIterable.iterator;

    var shapeIndex = 0;

    var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
    var dataIndexes = new List(resultDescriptor.shape.dimension - 1);

    var dataIndex = 0;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var i = 0;

    while (i < resultDescriptor.shape.length) {
      if (shapeIndex == resultDescriptor.shape.dimension - 2) {
        for (var row = 0; row < resultMatrixInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  resultMatrixInfo.dataColumns -
                      resultDescriptor.dataType.blockSize;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                (valueIterator..moveNext()).current,
                (valueIterator..moveNext()).current,
                (valueIterator..moveNext()).current,
                (valueIterator..moveNext()).current);

            resultData[dataIndex] = value4;

            dataIndex += resultMatrixInfo.delta1;

            i += resultDescriptor.dataType.blockSize;
          }

          var value4;
          switch (resultMatrixInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Float32x4(
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current);
              break;
            case 3:
              value4 = new Float32x4(
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  0.0);
              break;
            case 2:
              value4 = new Float32x4((valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current, 0.0, 0.0);
              break;
            case 1:
              value4 = new Float32x4(
                  (valueIterator..moveNext()).current, 0.0, 0.0, 0.0);
              break;
          }

          resultData[dataIndex] = value4;

          dataIndex += resultMatrixInfo.delta1;

          i += resultMatrixInfo.lastBlockColumnCount;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultMatrixInfo.delta2;
          } else {
            dataIndex += resultMatrixInfo.delta3;
          }
        }

        shapeIndex--;
      } else {
        if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          dataIndexes[shapeIndex] =
              dataIndex + resultDataInfo.headStride[shapeIndex];

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = dataIndex;
        } else {
          shapeIndex--;
        }
      }
    }

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
  }

  @override
  NDArray reduceSum(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceSum(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var total;

    final initTotal = new Float32x4.zero();

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = initTotal;
        },
        onValue: (value, int valueCount) {
          if (value != null) {
            total += value;
          } else {
            Float32x4 currentValue = total;
            total = currentValue.x +
                currentValue.y +
                currentValue.z +
                currentValue.w;
          }
        },
        end: () => total);
  }

  @override
  NDArray reduceMean(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceMean(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var total;
    var count;

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = 0;
          count = 0;
        },
        onValue: (value, int valueCount) {
          total += value;
          count++;
        },
        end: () => total / count);
  }

  @override
  NDArray reduceMax(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceMax(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var maxValue;

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          maxValue = null;
        },
        onValue: (value, int valueCount) {
          if (maxValue == null || value > maxValue) {
            maxValue = value;
          }
        },
        end: () => maxValue);
  }

  @override
  NDArray reduceAny(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceAny(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    bool total;

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = false;
        },
        onValue: (value, int valueCount) {
          total = total || value;
        },
        end: () => total);
  }

  @override
  NDArray argMax({int axis = 0, NDArray reuse}) {
    // TODO to implement NDArrayBlockedImpl.argMax
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.argMax: $this");
  }

  @override
  String toString() =>
      "<value: ${toValue()}, shape: $shape, dataType: $dataType, heasStride: ${_dataInfo.headStride}>";

  void logData() {
    print(_data);
  }
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

  var dataLength;
  if (stride.isNotEmpty) {
    var factor = matrixInfo.dataLength;
    for (var i = shape.dimension - 3; i >= 0; i--) {
      stride[i] = factor;
      factor *= shape[i];
    }
    dataLength = shape.dimensions.first * stride.first;
  } else {
    dataLength = matrixInfo.dataLength;
  }

  return new _DataInfo(stride, dataLength);
}

List _createData(NDDescriptor descriptor, _DataInfo dataInfo,
    _MatrixInfo matrixInfo, NDArrayBlockedImpl reuse) {
  if (descriptor.dataType == NDDataType.unknown) {
    throw new ArgumentError.value(descriptor.dataType.toString(), "data type");
  }

  if (reuse != null && reuse.descriptor == descriptor) {
    return reuse._data;
  } else {
    switch (descriptor.dataType) {
      case NDDataType.float32HBlocked:
      case NDDataType.float32VBlocked:
        return new Float32x4List(dataInfo.dataLength);
      default:
        throw new StateError("DEAD CODE");
    }
  }
}

void _loadData(value, Float32x4List data, NDDescriptor descriptor,
    _DataInfo dataInfo, _MatrixInfo matrixInfo) {
  var values = new List(descriptor.shape.dimension - 1);
  values[0] = value;

  var shapeIndex = 0;

  var dimensionIndexes = new List(descriptor.shape.dimension - 1);
  var dataIndexes = new List(descriptor.shape.dimension - 1);

  var dataIndex = 0;
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;

  while (i < descriptor.shape.length) {
    var value = values[shapeIndex];

    if (shapeIndex == descriptor.shape.dimension - 2) {
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
        switch (matrixInfo.lastBlockColumnCount) {
          case 4:
            value4 = new Float32x4(
                rowValues[column].toDouble(),
                rowValues[column + 1].toDouble(),
                rowValues[column + 2].toDouble(),
                rowValues[column + 3].toDouble());
            break;
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
        }

        data[dataIndex] = value4;

        dataIndex += matrixInfo.delta1;

        i += matrixInfo.lastBlockColumnCount;

        if (row % descriptor.dataType.blockSize <
            descriptor.dataType.blockSize - 1) {
          dataIndex += matrixInfo.delta2;
        } else {
          dataIndex += matrixInfo.delta3;
        }
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < descriptor.shape[shapeIndex]) {
        var newList = value[dimensionIndexes[shapeIndex]];

        dataIndex = dataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        dataIndexes[shapeIndex] = dataIndex + dataInfo.headStride[shapeIndex];

        shapeIndex++;

        values[shapeIndex] = newList;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

void _reduceData(
    NDArrayBlockedImpl sourceArray,
    List<int> reductionAxis,
    bool keepDimensions,
    Float32x4List targetData,
    NDDescriptor targetDescriptor,
    _DataInfo targetDataInfo,
    _MatrixInfo targetMatrixInfo,
    {void begin(),
    void onValue(value, int valueCount),
    dynamic end()}) {
  var sourceDimensions = new List.from(sourceArray.descriptor.shape.dimensions
      .sublist(0, sourceArray.descriptor.shape.dimension - 2));
  var targetDimensions = new List.from(targetDescriptor.shape.dimensions
      .sublist(0, targetDescriptor.shape.dimension - 2));

  var sourceStrides = new List.from(sourceArray._dataInfo.headStride);
  var targetStrides = new List.from(targetDataInfo.headStride);

  var sourceRowIndex;
  var sourceColumnIndex;

  var axis = new Set<int>.from(reductionAxis);
  if (axis.contains(sourceArray.descriptor.shape.dimension - 1)) {
    // throw new UnimplementedError();
  }

  List<int> targetPermutedIndexes;

  if (sourceArray.dataType.isHBlocked) {
    sourceRowIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockRows);
    sourceColumnIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockColumns);
    sourceDimensions.add(sourceArray.descriptor.dataType.blockSize);
    sourceStrides.add(sourceArray._matrixInfo.dataColumns);
    sourceStrides.add(sourceArray.descriptor.dataType.blockSize);
    sourceStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(targetMatrixInfo.dataColumns);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);

    targetPermutedIndexes =
        new List.generate(targetDimensions.length, (index) => index);

    if (axis.contains(sourceArray.descriptor.shape.dimension - 2)) {
      axis.add(sourceArray.descriptor.shape.dimension);

      var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
      targetPermutedIndexes[targetPermutedIndexes.length - 1] =
          targetPermutedIndexes[targetPermutedIndexes.length - 2];
      targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
      targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 3);
    }
  } else {
    sourceColumnIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockColumns);
    sourceRowIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockRows);
    sourceDimensions.add(targetDescriptor.dataType.blockSize);
    sourceStrides.add(sourceArray._matrixInfo.dataRows);
    sourceStrides.add(sourceArray.descriptor.dataType.blockSize);
    sourceStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(targetMatrixInfo.dataRows);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);

    targetPermutedIndexes =
        new List.generate(targetDimensions.length, (index) => index);

    if (axis.contains(sourceArray.descriptor.shape.dimension - 2)) {
      axis.remove(sourceArray.descriptor.shape.dimension - 2);
      axis.add(sourceArray.descriptor.shape.dimension - 1);
      axis.add(sourceArray.descriptor.shape.dimension);

      var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
      targetPermutedIndexes[targetPermutedIndexes.length - 1] =
          targetPermutedIndexes[targetPermutedIndexes.length - 2];
      targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
      targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 3);
    }
  }

  var newReductionAxis = axis.toList(growable: false);

  var sourcePermutedIndexes = new List.generate(
      sourceArray.descriptor.shape.dimension + 1, (index) => index,
      growable: true);

  sourcePermutedIndexes = sourcePermutedIndexes
      .where((index) => !axis.contains(index))
      .toList(growable: true);

  sourcePermutedIndexes.addAll(newReductionAxis);
  sourcePermutedIndexes.add(sourcePermutedIndexes.length);

  targetStrides.add(0);
  targetPermutedIndexes = [2, 1, 3];

  print("newReductionAxis: $newReductionAxis");
  print("sourceShape: ${sourceArray.descriptor.shape}");
  print("sourceDimensions: $sourceDimensions");
  print("sourceStrides: $sourceStrides");
  print("sourcePermutedIndexes: $sourcePermutedIndexes");
  print("sourceRowIndex: $sourceRowIndex");
  print("sourceColumnIndex: $sourceColumnIndex");
  print("targetShape: ${targetDescriptor.shape}");
  print("targetDimensions: $targetDimensions");
  print("targetStrides: $targetStrides");
  print("targetPermutedIndexes: $targetPermutedIndexes");

  var sourceDimensionIndexes = new List(sourceDimensions.length + 1);
  var targetDimensionIndexes = new List(targetDimensions.length + 1);
  var sourceDataIndexes = new List(sourceDimensions.length + 1);
  var targetDataIndexes = new List(targetDimensions.length + 1);

  var shapeIndex = 0;
  sourceDimensionIndexes[sourcePermutedIndexes[0]] = 0;
  targetDimensionIndexes[targetPermutedIndexes[0]] = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;
  sourceDataIndexes[sourcePermutedIndexes[0]] = 0;
  targetDataIndexes[targetPermutedIndexes[0]] = 0;

  // TODO reduce delle colonne

  var lastRow = false;
  var lastColumn = false;
  var blockColumnCount;
  while (shapeIndex >= 0) {
    if (shapeIndex == sourceDimensions.length) {
      print(
          "ADD: ${sourceDimensionIndexes.sublist(0, shapeIndex).map((index) => index - 1)}: $sourceDataIndex:$lastRow:$lastColumn");

      print(
          "(${sourcePermutedIndexes[shapeIndex - 1]}, ${sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex - 1]] - 1}, ${sourceArray._data[sourceDataIndex]}, $blockColumnCount)");

      shapeIndex--;

      onValue(sourceArray._data[sourceDataIndex], blockColumnCount);
    } else {
      var dimension;
      if (sourcePermutedIndexes[shapeIndex] == sourceDimensions.length - 1) {
        if (lastRow) {
          dimension = sourceArray._matrixInfo.lastBlockRowCount;
        } else {
          dimension = sourceDimensions[sourcePermutedIndexes[shapeIndex]];
        }

        if (lastColumn) {
          blockColumnCount = sourceArray._matrixInfo.lastBlockColumnCount;
        } else {
          blockColumnCount =
              sourceDimensions[sourcePermutedIndexes[shapeIndex]];
        }
      } else {
        dimension = sourceDimensions[sourcePermutedIndexes[shapeIndex]];
      }

      if (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] <
          dimension) {
        if (shapeIndex < sourceDimensions.length - newReductionAxis.length) {
          targetDataIndex =
              targetDataIndexes[targetPermutedIndexes[shapeIndex]];

          targetDataIndexes[targetPermutedIndexes[shapeIndex]] =
              targetDataIndex +
                  targetStrides[targetPermutedIndexes[shapeIndex]];
        }

        if (sourcePermutedIndexes[shapeIndex] == sourceRowIndex) {
          lastRow =
              (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] ==
                  sourceDimensions[sourcePermutedIndexes[shapeIndex]] - 1);
        } else if (sourcePermutedIndexes[shapeIndex] == sourceColumnIndex) {
          lastColumn =
              (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] ==
                  sourceDimensions[sourcePermutedIndexes[shapeIndex]] - 1);
        }

        sourceDataIndex = sourceDataIndexes[sourcePermutedIndexes[shapeIndex]];

        sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]]++;
        sourceDataIndexes[sourcePermutedIndexes[shapeIndex]] =
            sourceDataIndex + sourceStrides[sourcePermutedIndexes[shapeIndex]];

        shapeIndex++;

        if (shapeIndex < sourceDimensions.length - newReductionAxis.length) {
          targetDataIndexes[targetPermutedIndexes[shapeIndex]] =
              targetDataIndex;
        }

        sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] = 0;
        sourceDataIndexes[sourcePermutedIndexes[shapeIndex]] = sourceDataIndex;

        if (shapeIndex == sourceDimensions.length - newReductionAxis.length) {
          print("INIT REDUCTION");

          begin();
        }
      } else {
        shapeIndex--;

        if (shapeIndex ==
            sourceDimensions.length - newReductionAxis.length - 1) {
          var reducedValue = end();

          print("$shapeIndex: REDUCE[$targetDataIndex]: $reducedValue");

          // TODO se la riduzione è sull'ultima colonna: devo fare una ulteriore riduzione e devo impostare il target

          // targetData[targetDataIndex] = reducedValue;
        }
      }
    }
  }
}

void _argData(
    NDArrayBlockedImpl sourceArray,
    int axis,
    Float32x4List targetData,
    NDDescriptor targetDescriptor,
    _DataInfo targetDataInfo,
    _MatrixInfo targetMatrixInfo,
    {void begin(),
    void onValue(dimensionIndex, value, int valueCount),
    dynamic end()}) {
  var sourceDimensions = new List.from(sourceArray.descriptor.shape.dimensions
      .sublist(0, sourceArray.descriptor.shape.dimension - 2));
  var targetDimensions = new List.from(targetDescriptor.shape.dimensions
      .sublist(0, targetDescriptor.shape.dimension - 2));

  var sourceStrides = new List.from(sourceArray._dataInfo.headStride);
  var targetStrides = new List.from(targetDataInfo.headStride);

  var sourceRowIndex;
  var sourceColumnIndex;

  var argAxis = new Set<int>();
  argAxis.add(axis);

  List<int> targetPermutedIndexes;

  if (sourceArray.dataType.isHBlocked) {
    sourceRowIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockRows);
    sourceColumnIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockColumns);
    sourceDimensions.add(sourceArray.descriptor.dataType.blockSize);
    sourceStrides.add(sourceArray._matrixInfo.dataColumns);
    sourceStrides.add(sourceArray.descriptor.dataType.blockSize);
    sourceStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(targetMatrixInfo.dataColumns);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);

    targetPermutedIndexes =
        new List.generate(targetDimensions.length, (index) => index);

    if (axis == sourceArray.descriptor.shape.dimension - 2) {
      argAxis.add(sourceArray.descriptor.shape.dimension);

      var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
      targetPermutedIndexes[targetPermutedIndexes.length - 1] =
          targetPermutedIndexes[targetPermutedIndexes.length - 2];
      targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
      targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 3);
    }
  } else {
    sourceColumnIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockColumns);
    sourceRowIndex = sourceDimensions.length;
    sourceDimensions.add(sourceArray._matrixInfo.blockRows);
    sourceDimensions.add(targetDescriptor.dataType.blockSize);
    sourceStrides.add(sourceArray._matrixInfo.dataRows);
    sourceStrides.add(sourceArray.descriptor.dataType.blockSize);
    sourceStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(targetMatrixInfo.dataRows);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);

    targetPermutedIndexes =
        new List.generate(targetDimensions.length, (index) => index);

    if (axis == sourceArray.descriptor.shape.dimension - 2) {
      argAxis.remove(sourceArray.descriptor.shape.dimension - 2);
      argAxis.add(sourceArray.descriptor.shape.dimension - 1);
      argAxis.add(sourceArray.descriptor.shape.dimension);

      var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
      targetPermutedIndexes[targetPermutedIndexes.length - 1] =
          targetPermutedIndexes[targetPermutedIndexes.length - 2];
      targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
      targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 3);
    }
  }

  var newArgAxis = argAxis.toList(growable: false);

  var sourcePermutedIndexes = new List.generate(
      sourceArray.descriptor.shape.dimension + 1, (index) => index,
      growable: true);

  sourcePermutedIndexes = sourcePermutedIndexes
      .where((index) => axis != index)
      .toList(growable: true);

  sourcePermutedIndexes.addAll(newArgAxis);
  sourcePermutedIndexes.add(sourcePermutedIndexes.length);

  targetStrides.add(0);
  targetPermutedIndexes = [2, 1, 3];

  print("newArgAxis: $newArgAxis");
  print("sourceShape: ${sourceArray.descriptor.shape}");
  print("sourceDimensions: $sourceDimensions");
  print("sourceStrides: $sourceStrides");
  print("sourcePermutedIndexes: $sourcePermutedIndexes");
  print("sourceRowIndex: $sourceRowIndex");
  print("sourceColumnIndex: $sourceColumnIndex");
  print("targetShape: ${targetDescriptor.shape}");
  print("targetDimensions: $targetDimensions");
  print("targetStrides: $targetStrides");
  print("targetPermutedIndexes: $targetPermutedIndexes");

  var sourceDimensionIndexes = new List(sourceDimensions.length + 1);
  var targetDimensionIndexes = new List(targetDimensions.length + 1);
  var sourceDataIndexes = new List(sourceDimensions.length + 1);
  var targetDataIndexes = new List(targetDimensions.length + 1);

  var shapeIndex = 0;
  sourceDimensionIndexes[sourcePermutedIndexes[0]] = 0;
  targetDimensionIndexes[targetPermutedIndexes[0]] = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;
  sourceDataIndexes[sourcePermutedIndexes[0]] = 0;
  targetDataIndexes[targetPermutedIndexes[0]] = 0;

  // TODO reduce delle colonne

  var lastRow = false;
  var lastColumn = false;
  var blockColumnCount;
  while (shapeIndex >= 0) {
    if (shapeIndex == sourceDimensions.length) {
      print(
          "ADD: ${sourceDimensionIndexes.sublist(0, shapeIndex).map((index) => index - 1)}: $sourceDataIndex:$lastRow:$lastColumn");

      print(
          "(${sourcePermutedIndexes[shapeIndex - 1]}, ${sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex - 1]] - 1}, ${sourceArray._data[sourceDataIndex]}, $blockColumnCount)");

      shapeIndex--;

      onValue(sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex - 1]] - 1,
          sourceArray._data[sourceDataIndex], blockColumnCount);
    } else {
      var dimension;
      if (sourcePermutedIndexes[shapeIndex] == sourceDimensions.length - 1) {
        if (lastRow) {
          dimension = sourceArray._matrixInfo.lastBlockRowCount;
        } else {
          dimension = sourceDimensions[sourcePermutedIndexes[shapeIndex]];
        }

        if (lastColumn) {
          blockColumnCount = sourceArray._matrixInfo.lastBlockColumnCount;
        } else {
          blockColumnCount =
              sourceDimensions[sourcePermutedIndexes[shapeIndex]];
        }
      } else {
        dimension = sourceDimensions[sourcePermutedIndexes[shapeIndex]];
      }

      if (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] <
          dimension) {
        if (shapeIndex < sourceDimensions.length - newArgAxis.length) {
          targetDataIndex =
              targetDataIndexes[targetPermutedIndexes[shapeIndex]];

          targetDataIndexes[targetPermutedIndexes[shapeIndex]] =
              targetDataIndex +
                  targetStrides[targetPermutedIndexes[shapeIndex]];
        }

        if (sourcePermutedIndexes[shapeIndex] == sourceRowIndex) {
          lastRow =
              (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] ==
                  sourceDimensions[sourcePermutedIndexes[shapeIndex]] - 1);
        } else if (sourcePermutedIndexes[shapeIndex] == sourceColumnIndex) {
          lastColumn =
              (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] ==
                  sourceDimensions[sourcePermutedIndexes[shapeIndex]] - 1);
        }

        sourceDataIndex = sourceDataIndexes[sourcePermutedIndexes[shapeIndex]];

        sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]]++;
        sourceDataIndexes[sourcePermutedIndexes[shapeIndex]] =
            sourceDataIndex + sourceStrides[sourcePermutedIndexes[shapeIndex]];

        shapeIndex++;

        if (shapeIndex < sourceDimensions.length - newArgAxis.length) {
          targetDataIndexes[targetPermutedIndexes[shapeIndex]] =
              targetDataIndex;
        }

        sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] = 0;
        sourceDataIndexes[sourcePermutedIndexes[shapeIndex]] = sourceDataIndex;

        if (shapeIndex == sourceDimensions.length - newArgAxis.length) {
          print("INIT REDUCTION");

          begin();
        }
      } else {
        shapeIndex--;

        if (shapeIndex == sourceDimensions.length - newArgAxis.length - 1) {
          var reducedValue = end();

          print("$shapeIndex: REDUCE[$targetDataIndex]: $reducedValue");

          // TODO se la riduzione è sull'ultima colonna: devo fare una ulteriore riduzione e devo impostare il target

          // targetData[targetDataIndex] = reducedValue;
        }
      }
    }
  }
}

void _identityData(
    NDArrayBlockedImpl sourceArray,
    Float32x4List targetData,
    NDDescriptor targetDescriptor,
    _DataInfo targetDataInfo,
    _MatrixInfo targetMatrixInfo) {
  var targetStrides = new List.from(targetDataInfo.headStride);

  var targetDimensions = new List.from(targetDescriptor.shape.dimensions
      .sublist(0, targetDescriptor.shape.dimension - 2));

  if (sourceArray.dataType.isHBlocked) {
    targetStrides.add(targetMatrixInfo.dataColumns);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
  } else {
    targetStrides.add(targetMatrixInfo.dataRows);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
  }

  var targetDimensionIndexes = new List(targetDimensions.length + 1);
  var targetDataIndexes = new List(targetDimensions.length + 1);

  var targetShapeIndex = 0;
  targetDimensionIndexes[0] = 0;
  var targetDataIndex = 0;
  targetDataIndexes[0] = 0;

  var sourceDataIndex = 0;

  while (targetShapeIndex >= 0) {
    if (targetDimensionIndexes[targetShapeIndex] <
        targetDimensions[targetShapeIndex]) {
      targetDataIndex = targetDataIndexes[targetShapeIndex];

      targetDimensionIndexes[targetShapeIndex]++;
      targetDataIndexes[targetShapeIndex] =
          targetDataIndex + targetStrides[targetShapeIndex];

      targetShapeIndex++;

      targetDimensionIndexes[targetShapeIndex] = 0;
      targetDataIndexes[targetShapeIndex] = targetDataIndex;
    } else {
      targetShapeIndex--;
    }

    if (targetShapeIndex == targetDimensions.length) {
      targetData[targetDataIndex] = sourceArray._data[sourceDataIndex++];

      targetShapeIndex--;
    } else {
      if (targetDimensionIndexes[targetShapeIndex] <
          targetDimensions[targetShapeIndex]) {
        targetDataIndex = targetDataIndexes[targetShapeIndex];

        targetDimensionIndexes[targetShapeIndex]++;
        targetDataIndexes[targetShapeIndex] =
            targetDataIndex + targetStrides[targetShapeIndex];

        targetShapeIndex++;

        targetDimensionIndexes[targetShapeIndex] = 0;
        targetDataIndexes[targetShapeIndex] = targetDataIndex;
      } else {
        targetShapeIndex--;
      }
    }
  }
}

void _identityDataOld(
    NDArrayBlockedImpl sourceArray,
    Float32x4List targetData,
    NDDescriptor targetDescriptor,
    _DataInfo targetDataInfo,
    _MatrixInfo targetMatrixInfo) {
  var targetStrides = new List.from(targetDataInfo.headStride);

  var targetDimensions = new List.from(targetDescriptor.shape.dimensions
      .sublist(0, targetDescriptor.shape.dimension - 2));

  if (sourceArray.dataType.isHBlocked) {
    targetStrides.add(targetMatrixInfo.dataColumns);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
  } else {
    targetStrides.add(targetMatrixInfo.dataRows);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
  }

  print("targetShape: ${targetDescriptor.shape}");
  print("targetDimensions: $targetDimensions");
  print("targetStrides: $targetStrides");

  var targetDimensionIndexes = new List(targetDimensions.length + 1);
  var targetDataIndexes = new List(targetDimensions.length + 1);

  var targetShapeIndex = 0;
  targetDimensionIndexes[0] = 0;
  var targetDataIndex = 0;
  targetDataIndexes[0] = 0;

  var sourceDataIndex = 0;

  while (targetShapeIndex >= 0) {
    if (targetShapeIndex == targetDimensions.length) {
      print(
          "${targetDimensionIndexes.sublist(0, targetShapeIndex).map((index) => index - 1)}: $targetDataIndex");

      targetData[targetDataIndex] = sourceArray._data[sourceDataIndex++];

      targetShapeIndex--;
    } else {
      if (targetDimensionIndexes[targetShapeIndex] <
          targetDimensions[targetShapeIndex]) {
        targetDataIndex = targetDataIndexes[targetShapeIndex];

        targetDimensionIndexes[targetShapeIndex]++;
        targetDataIndexes[targetShapeIndex] =
            targetDataIndex + targetStrides[targetShapeIndex];

        targetShapeIndex++;

        targetDimensionIndexes[targetShapeIndex] = 0;
        targetDataIndexes[targetShapeIndex] = targetDataIndex;
      } else {
        targetShapeIndex--;
      }
    }
  }
}

void _loopExample(
    NDArrayBlockedImpl sourceArray,
    List targetData,
    NDDescriptor targetDescriptor,
    _DataInfo targetDataInfo,
    _MatrixInfo targetMatrixInfo) {
  var targetStrides = new List.from(targetDataInfo.headStride);

  var targetDimensions = new List.from(targetDescriptor.shape.dimensions
      .sublist(0, targetDescriptor.shape.dimension - 2));

  if (sourceArray.dataType.isHBlocked) {
    targetStrides.add(targetMatrixInfo.dataColumns);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
  } else {
    targetStrides.add(targetMatrixInfo.dataRows);
    targetStrides.add(targetDescriptor.dataType.blockSize);
    targetStrides.add(1);
    targetDimensions.add(targetMatrixInfo.blockColumns);
    targetDimensions.add(targetMatrixInfo.blockRows);
    targetDimensions.add(targetDescriptor.dataType.blockSize);
  }

  var targetDimensionIndexes = new List(targetDimensions.length + 1);
  var targetDataIndexes = new List(targetDimensions.length + 1);

  var targetShapeIndex = 0;
  targetDimensionIndexes[0] = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;
  targetDataIndexes[0] = 0;

  while (targetShapeIndex >= 0) {
    if (targetShapeIndex == targetDimensions.length) {
      targetData[targetDataIndex] = sourceArray._data[sourceDataIndex++];

      targetShapeIndex--;
    } else {
      if (targetDimensionIndexes[targetShapeIndex] <
          targetDimensions[targetShapeIndex]) {
        targetDataIndex = targetDataIndexes[targetShapeIndex];

        targetDimensionIndexes[targetShapeIndex]++;
        targetDataIndexes[targetShapeIndex] =
            targetDataIndex + targetStrides[targetShapeIndex];

        targetShapeIndex++;

        targetDimensionIndexes[targetShapeIndex] = 0;
        targetDataIndexes[targetShapeIndex] = targetDataIndex;
      } else {
        targetShapeIndex--;
      }
    }
  }
}

void _transposeData(
    NDArrayBlockedImpl sourceArray,
    List sourceHeadStride,
    Float32x4List resultData,
    NDDescriptor resultDescriptor,
    _DataInfo resultDataInfo,
    _MatrixInfo resultMatrixInfo) {
  var dimension1s = sourceArray.dataType.isHBlocked
      ? sourceArray._matrixInfo.blockRows
      : sourceArray._matrixInfo.blockColumns;
  var dimension2s = sourceArray.dataType.isHBlocked
      ? sourceArray._matrixInfo.blockColumns
      : sourceArray._matrixInfo.blockRows;

  var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
  var sourceDataIndexes = new List(resultDescriptor.shape.dimension - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimension - 1);

  var shapeIndex = 0;
  dimensionIndexes[0] = 0;
  var sourceDataIndex = 0;
  sourceDataIndexes[0] = 0;
  var targetDataIndex = 0;
  targetDataIndexes[0] = 0;

  while (shapeIndex >= 0) {
    if (shapeIndex == resultDescriptor.shape.dimension - 2) {
      for (var dimension1 = 0; dimension1 < dimension1s; dimension1++) {
        for (var dimension2 = 0; dimension2 < dimension2s; dimension2++) {
          resultData[targetDataIndex++] = sourceArray._data[sourceDataIndex++];
          resultData[targetDataIndex++] = sourceArray._data[sourceDataIndex++];
          resultData[targetDataIndex++] = sourceArray._data[sourceDataIndex++];
          resultData[targetDataIndex++] = sourceArray._data[sourceDataIndex++];
        }
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        sourceDataIndexes[shapeIndex] =
            sourceDataIndex + sourceHeadStride[shapeIndex];
        targetDataIndexes[shapeIndex] =
            targetDataIndex + resultDataInfo.headStride[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

void _transposeSwitchedData(
    NDArrayBlockedImpl sourceArray,
    List sourceHeadStride,
    Float32x4List resultData,
    NDDescriptor resultDescriptor,
    _DataInfo resultDataInfo,
    _MatrixInfo resultMatrixInfo) {
  var dimension1s = sourceArray.dataType.isHBlocked
      ? sourceArray._matrixInfo.blockRows
      : sourceArray._matrixInfo.blockColumns;
  var dimension2s = sourceArray.dataType.isHBlocked
      ? sourceArray._matrixInfo.blockColumns
      : sourceArray._matrixInfo.blockRows;

  var delta1 = (dimension1s - 1) << resultDescriptor.dataType.blockDepth;

  var delta2 =
      resultDescriptor.dataType.blockSize - resultMatrixInfo.dataLength;

  var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
  var sourceDataIndexes = new List(resultDescriptor.shape.dimension - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimension - 1);

  var shapeIndex = 0;
  dimensionIndexes[0] = 0;
  var sourceDataIndex = 0;
  sourceDataIndexes[0] = 0;
  var targetDataIndex = 0;
  targetDataIndexes[0] = 0;

  while (shapeIndex >= 0) {
    if (shapeIndex == resultDescriptor.shape.dimension - 2) {
      for (var dimension1 = 0; dimension1 < dimension1s; dimension1++) {
        for (var dimension2 = 0; dimension2 < dimension2s; dimension2++) {
          var a0 = sourceArray._data[sourceDataIndex++];
          var a1 = sourceArray._data[sourceDataIndex++];
          var a2 = sourceArray._data[sourceDataIndex++];
          var a3 = sourceArray._data[sourceDataIndex++];

          resultData[targetDataIndex++] = new Float32x4(a0.x, a1.x, a2.x, a3.x);
          resultData[targetDataIndex++] = new Float32x4(a0.y, a1.y, a2.y, a3.y);
          resultData[targetDataIndex++] = new Float32x4(a0.z, a1.z, a2.z, a3.z);
          resultData[targetDataIndex++] = new Float32x4(a0.w, a1.w, a2.w, a3.w);

          targetDataIndex += delta1;
        }

        targetDataIndex += delta2;
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        sourceDataIndexes[shapeIndex] =
            sourceDataIndex + sourceHeadStride[shapeIndex];
        targetDataIndexes[shapeIndex] =
            targetDataIndex + resultDataInfo.headStride[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

void _matMulData(
    NDArrayBlockedImpl array1,
    NDArrayBlockedImpl array2,
    List resultData,
    NDDescriptor resultDescriptor,
    _DataInfo resultDataInfo,
    _MatrixInfo resultMatrixInfo) {
  var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
  var sourceDataIndexes1 = new List(resultDescriptor.shape.dimension - 1);
  var sourceDataIndexes2 = new List(resultDescriptor.shape.dimension - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimension - 1);

  var shapeIndex = 0;
  dimensionIndexes[0] = 0;
  var sourceDataIndex1 = 0;
  sourceDataIndexes1[0] = 0;
  var sourceDataIndex2 = 0;
  sourceDataIndexes2[0] = 0;
  var targetDataIndex = 0;
  targetDataIndexes[0] = 0;

  while (shapeIndex >= 0) {
    if (shapeIndex == resultDescriptor.shape.dimension - 2) {
      var initialSourceDataIndex2 = sourceDataIndex2;

      for (var row1 = 0;
          row1 < array1._matrixInfo.dataRows;
          row1 += array1.dataType.blockSize) {
        var initialSourceDataIndex1 = sourceDataIndex1;

        sourceDataIndex2 = initialSourceDataIndex2;

        for (var column2 = 0;
            column2 < array2._matrixInfo.dataColumns;
            column2 += array2.dataType.blockSize) {
          sourceDataIndex1 = initialSourceDataIndex1;

          var result0 = new Float32x4.zero();
          var result1 = new Float32x4.zero();
          var result2 = new Float32x4.zero();
          var result3 = new Float32x4.zero();

          for (var i = 0;
              i < array1._matrixInfo.dataColumns;
              i += array1.dataType.blockSize) {
            var b0 = array2._data[sourceDataIndex2++];
            var b1 = array2._data[sourceDataIndex2++];
            var b2 = array2._data[sourceDataIndex2++];
            var b3 = array2._data[sourceDataIndex2++];

            var a0 = array1._data[sourceDataIndex1++];
            result0 += a0.shuffle(Float32x4.XXXX) * b0 +
                a0.shuffle(Float32x4.YYYY) * b1 +
                a0.shuffle(Float32x4.ZZZZ) * b2 +
                a0.shuffle(Float32x4.WWWW) * b3;

            var a1 = array1._data[sourceDataIndex1++];
            result1 += a1.shuffle(Float32x4.XXXX) * b0 +
                a1.shuffle(Float32x4.YYYY) * b1 +
                a1.shuffle(Float32x4.ZZZZ) * b2 +
                a1.shuffle(Float32x4.WWWW) * b3;

            var a2 = array1._data[sourceDataIndex1++];
            result2 += a2.shuffle(Float32x4.XXXX) * b0 +
                a2.shuffle(Float32x4.YYYY) * b1 +
                a2.shuffle(Float32x4.ZZZZ) * b2 +
                a2.shuffle(Float32x4.WWWW) * b3;

            var a3 = array1._data[sourceDataIndex1++];
            result3 += a3.shuffle(Float32x4.XXXX) * b0 +
                a3.shuffle(Float32x4.YYYY) * b1 +
                a3.shuffle(Float32x4.ZZZZ) * b2 +
                a3.shuffle(Float32x4.WWWW) * b3;
          }

          resultData[targetDataIndex++] = result0;
          resultData[targetDataIndex++] = result1;
          resultData[targetDataIndex++] = result2;
          resultData[targetDataIndex++] = result3;
        }
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
        sourceDataIndex1 = sourceDataIndexes1[shapeIndex];
        sourceDataIndex2 = sourceDataIndexes2[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        sourceDataIndexes1[shapeIndex] =
            sourceDataIndex1 + array1._dataInfo.headStride[shapeIndex];
        sourceDataIndexes2[shapeIndex] =
            sourceDataIndex2 + array2._dataInfo.headStride[shapeIndex];
        targetDataIndexes[shapeIndex] =
            targetDataIndex + resultDataInfo.headStride[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes1[shapeIndex] = sourceDataIndex1;
        sourceDataIndexes2[shapeIndex] = sourceDataIndex2;
        targetDataIndexes[shapeIndex] = targetDataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

void _generateData(generator(int index), Float32x4List data,
    NDDescriptor descriptor, _DataInfo dataInfo, _MatrixInfo matrixInfo) {
  var shapeIndex = 0;

  var dimensionIndexes = new List(descriptor.shape.dimension - 1);
  var dataIndexes = new List(descriptor.shape.dimension - 1);

  var dataIndex = 0;
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;

  while (i < descriptor.shape.length) {
    if (shapeIndex == descriptor.shape.dimension - 2) {
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
        switch (matrixInfo.lastBlockColumnCount) {
          case 4:
            value4 = new Float32x4(
                generator(i).toDouble(),
                generator(i + 1).toDouble(),
                generator(i + 2).toDouble(),
                generator(i + 3).toDouble());
            break;
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
        }

        data[dataIndex] = value4;

        dataIndex += matrixInfo.delta1;

        i += matrixInfo.lastBlockColumnCount;

        if (row % descriptor.dataType.blockSize <
            descriptor.dataType.blockSize - 1) {
          dataIndex += matrixInfo.delta2;
        } else {
          dataIndex += matrixInfo.delta3;
        }
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < descriptor.shape[shapeIndex]) {
        dataIndex = dataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        dataIndexes[shapeIndex] = dataIndex + dataInfo.headStride[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

void _castData(NDArrayBase fromArray, List data, NDDescriptor descriptor,
    _DataInfo dataInfo, _MatrixInfo matrixInfo) {
  if (fromArray.dataType.isHBlocked && descriptor.dataType.isVBlocked) {
    _castBlockedData(fromArray, data, descriptor, dataInfo, matrixInfo);
  } else if (fromArray.dataType.isVBlocked && descriptor.dataType.isHBlocked) {
    _castBlockedData(fromArray, data, descriptor, dataInfo, matrixInfo);
  } else if (fromArray.dataType.isFloat && descriptor.dataType.isFloat) {
    _castConvertedData(fromArray, data, descriptor, dataInfo, matrixInfo,
        (num value) => value);
  } else if (fromArray.dataType.isInteger && descriptor.dataType.isFloat) {
    _castConvertedData(fromArray, data, descriptor, dataInfo, matrixInfo,
        (int value) => value.toDouble());
  } else {
    throw new UnsupportedError(
        "Cast from ${fromArray.dataType} to ${descriptor.dataType}");
  }
}

void _castBlockedData(
    NDArrayBlockedImpl fromArray,
    List resultData,
    NDDescriptor resultDescriptor,
    _DataInfo resultDataInfo,
    _MatrixInfo resultMatrixInfo) {
  var multiplier;
  if (resultDescriptor.shape.dimension > 2) {
    multiplier = resultDescriptor.shape.length ~/
        (resultMatrixInfo.rows * resultMatrixInfo.columns);
  } else {
    multiplier = 1;
  }

  var dimension1s = fromArray.dataType.isHBlocked
      ? resultMatrixInfo.blockRows
      : resultMatrixInfo.blockColumns;
  var dimension2s = fromArray.dataType.isHBlocked
      ? resultMatrixInfo.blockColumns
      : resultMatrixInfo.blockRows;

  var delta1 = (dimension1s - 1) << resultDescriptor.dataType.blockDepth;

  var delta2 =
      resultDescriptor.dataType.blockSize - resultMatrixInfo.dataLength;

  var delta3 = -(delta1 + delta2);

  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  for (var iteration = 0; iteration < multiplier; iteration++) {
    for (var dimension1 = 0; dimension1 < dimension1s; dimension1++) {
      for (var dimension2 = 0; dimension2 < dimension2s; dimension2++) {
        resultData[targetDataIndex++] = fromArray._data[sourceDataIndex++];
        resultData[targetDataIndex++] = fromArray._data[sourceDataIndex++];
        resultData[targetDataIndex++] = fromArray._data[sourceDataIndex++];
        resultData[targetDataIndex++] = fromArray._data[sourceDataIndex++];

        targetDataIndex += delta1;
      }

      targetDataIndex += delta2;
    }

    targetDataIndex += delta3;
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

  var shapeIndex = 0;

  var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
  var dataIndexes = new List(resultDescriptor.shape.dimension - 1);

  var dataIndex = 0;
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;

  while (i < resultDescriptor.shape.length) {
    if (shapeIndex == resultDescriptor.shape.dimension - 2) {
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
        switch (resultMatrixInfo.lastBlockColumnCount) {
          case 4:
            value4 = new Float32x4(
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current));
            break;
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
        }

        resultData[dataIndex] = value4;

        dataIndex += resultMatrixInfo.delta1;

        i += resultMatrixInfo.lastBlockColumnCount;

        if (row % resultDescriptor.dataType.blockSize <
            resultDescriptor.dataType.blockSize - 1) {
          dataIndex += resultMatrixInfo.delta2;
        } else {
          dataIndex += resultMatrixInfo.delta3;
        }
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
        dataIndex = dataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        dataIndexes[shapeIndex] =
            dataIndex + resultDataInfo.headStride[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

Iterable<num> _createTiledValueIterable(
    NDArrayBlockedImpl array, NDDescriptor tiledDescriptor) {
  if (array.descriptor == tiledDescriptor) {
    return _createValueIterable(array);
  } else if (array.descriptor.shape
              .dimensions[array.descriptor.shape.dimension - 2] ==
          tiledDescriptor
              .shape.dimensions[tiledDescriptor.shape.dimension - 2] &&
      array.descriptor.shape.dimensions[array.descriptor.shape.dimension - 1] ==
          tiledDescriptor
              .shape.dimensions[tiledDescriptor.shape.dimension - 1]) {
    return _createHeadTiledValueIterable(array, tiledDescriptor);
  } else {
    return _createFullTiledValueIterable(array, tiledDescriptor);
  }
}

Iterable<num> _createHeadTiledValueIterable(
    NDArrayBlockedImpl array, NDDescriptor tiledDescriptor) sync* {
  var shapeIndex = 0;

  var dimensionIndexes = new List(array.shape.dimension - 1);
  var sourceDimensionIndexes = new List(array.shape.dimension - 1);
  var dataIndexes = new List(array.shape.dimension - 1);
  var initialDataIndexes = new List(array.shape.dimension - 1);

  var dataIndex = 0;
  dimensionIndexes[0] = 0;
  sourceDimensionIndexes[0] = 0;
  dataIndexes[0] = 0;
  initialDataIndexes[0] = 0;

  var i = 0;

  while (i < tiledDescriptor.shape.length) {
    if (shapeIndex == array.shape.dimension - 2) {
      for (var row = 0; row < array._matrixInfo.rows; row++) {
        var column;
        for (column = 0;
            column < array._matrixInfo.dataColumns - array.dataType.blockSize;
            column += array.dataType.blockSize) {
          var value4 = array._data[dataIndex];

          yield value4.x;
          yield value4.y;
          yield value4.z;
          yield value4.w;

          dataIndex += array._matrixInfo.delta1;

          i += array.descriptor.dataType.blockSize;
        }

        var value4 = array._data[dataIndex];

        switch (array._matrixInfo.lastBlockColumnCount) {
          case 4:
            yield value4.x;
            yield value4.y;
            yield value4.z;
            yield value4.w;

            break;
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
        }

        dataIndex += array._matrixInfo.delta1;

        i += array._matrixInfo.lastBlockColumnCount;

        if (row % array.dataType.blockSize < array.dataType.blockSize - 1) {
          dataIndex += array._matrixInfo.delta2;
        } else {
          dataIndex += array._matrixInfo.delta3;
        }
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < tiledDescriptor.shape[shapeIndex]) {
        dataIndex = dataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        if (sourceDimensionIndexes[shapeIndex] < array.shape[shapeIndex] - 1) {
          sourceDimensionIndexes[shapeIndex]++;
          dataIndexes[shapeIndex] =
              dataIndex + array._dataInfo.headStride[shapeIndex];
        } else {
          sourceDimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = initialDataIndexes[shapeIndex];
        }

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;
        initialDataIndexes[shapeIndex] = dataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

Iterable<num> _createFullTiledValueIterable(
    NDArrayBlockedImpl array, NDDescriptor tiledDescriptor) sync* {
  var shapeIndex = 0;

  var dimensionIndexes = new List(array.shape.dimension - 1);
  var sourceDimensionIndexes = new List(array.shape.dimension - 1);
  var dataIndexes = new List(array.shape.dimension - 1);
  var initialDataIndexes = new List(array.shape.dimension - 1);

  var dataIndex = 0;
  dimensionIndexes[0] = 0;
  sourceDimensionIndexes[0] = 0;
  dataIndexes[0] = 0;
  initialDataIndexes[0] = 0;

  var rowsMultiplier =
      tiledDescriptor.shape[tiledDescriptor.shape.dimension - 2] ~/
          array._matrixInfo.rows;
  var columnsMultiplier =
      tiledDescriptor.shape[tiledDescriptor.shape.dimension - 1] ~/
          array._matrixInfo.columns;

  var i = 0;

  while (i < tiledDescriptor.shape.length) {
    if (shapeIndex == array.shape.dimension - 2) {
      var initialRowDataIndex = dataIndex;

      for (var rowIteration = 0;
          rowIteration < rowsMultiplier;
          rowIteration++) {
        for (var row = 0; row < array._matrixInfo.rows; row++) {
          var initialColumnDataIndex = dataIndex;

          var lastColumnDataIndex = dataIndex;
          for (var columnIteration = 0;
              columnIteration < columnsMultiplier;
              columnIteration++) {
            var column;
            for (column = 0;
                column <
                    array._matrixInfo.dataColumns - array.dataType.blockSize;
                column += array.dataType.blockSize) {
              var value4 = array._data[dataIndex];

              yield value4.x;
              yield value4.y;
              yield value4.z;
              yield value4.w;

              dataIndex += array._matrixInfo.delta1;

              i += array.descriptor.dataType.blockSize;
            }

            var value4 = array._data[dataIndex];

            switch (array._matrixInfo.lastBlockColumnCount) {
              case 4:
                yield value4.x;
                yield value4.y;
                yield value4.z;
                yield value4.w;

                break;
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
            }

            dataIndex += array._matrixInfo.delta1;

            i += array._matrixInfo.lastBlockColumnCount;

            if (row % array.dataType.blockSize < array.dataType.blockSize - 1) {
              dataIndex += array._matrixInfo.delta2;
            } else {
              dataIndex += array._matrixInfo.delta3;
            }

            lastColumnDataIndex = dataIndex;

            dataIndex = initialColumnDataIndex;
          }

          dataIndex = lastColumnDataIndex;
        }

        dataIndex = initialRowDataIndex;
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < tiledDescriptor.shape[shapeIndex]) {
        dataIndex = dataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        if (sourceDimensionIndexes[shapeIndex] < array.shape[shapeIndex] - 1) {
          sourceDimensionIndexes[shapeIndex]++;
          dataIndexes[shapeIndex] =
              dataIndex + array._dataInfo.headStride[shapeIndex];
        } else {
          sourceDimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = initialDataIndexes[shapeIndex];
        }

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;
        initialDataIndexes[shapeIndex] = dataIndex;
      } else {
        shapeIndex--;
      }
    }
  }
}

Iterable<num> _createBroadcastedValueIterable(
    NDArrayBlockedImpl array, NDDescriptor broadcastedDescriptor) {
  if (array.descriptor == broadcastedDescriptor) {
    return _createValueIterable(array);
  } else if (array.descriptor.shape
              .dimensions[array.descriptor.shape.dimension - 2] ==
          broadcastedDescriptor
              .shape.dimensions[broadcastedDescriptor.shape.dimension - 2] &&
      array.descriptor.shape.dimensions[array.descriptor.shape.dimension - 1] ==
          broadcastedDescriptor
              .shape.dimensions[broadcastedDescriptor.shape.dimension - 1]) {
    return _createHeadBroadcastedValueIterable(array, broadcastedDescriptor);
  } else {
    return _createFullBroadcastedValueIterable(array, broadcastedDescriptor);
  }
}

Iterable<num> _createHeadBroadcastedValueIterable(
    NDArrayBlockedImpl array, NDDescriptor broadcastedDescriptor) sync* {
  var broadcastedShape;
  var multiplier;

  if (broadcastedDescriptor.shape.dimension > array.shape.dimension) {
    broadcastedShape = new NDShape(broadcastedDescriptor.shape.dimensions
        .sublist(
            broadcastedDescriptor.shape.dimension - array.shape.dimension));
    multiplier = broadcastedDescriptor.shape.length ~/ broadcastedShape.length;
  } else {
    broadcastedShape = broadcastedDescriptor.shape;
    multiplier = 1;
  }

  for (var iteration = 0; iteration < multiplier; iteration++) {
    var shapeIndex = 0;

    var dimensionIndexes = new List(array.shape.dimension - 1);
    var sourceDimensionIndexes = new List(array.shape.dimension - 1);
    var dataIndexes = new List(array.shape.dimension - 1);
    var initialDataIndexes = new List(array.shape.dimension - 1);

    var dataIndex = 0;
    dimensionIndexes[0] = 0;
    sourceDimensionIndexes[0] = 0;
    dataIndexes[0] = 0;
    initialDataIndexes[0] = 0;

    var i = 0;

    while (i < broadcastedShape.length) {
      if (shapeIndex == array.shape.dimension - 2) {
        for (var row = 0; row < array._matrixInfo.rows; row++) {
          var column;
          for (column = 0;
              column < array._matrixInfo.dataColumns - array.dataType.blockSize;
              column += array.dataType.blockSize) {
            var value4 = array._data[dataIndex];

            yield value4.x;
            yield value4.y;
            yield value4.z;
            yield value4.w;

            dataIndex += array._matrixInfo.delta1;

            i += array.descriptor.dataType.blockSize;
          }

          var value4 = array._data[dataIndex];

          switch (array._matrixInfo.lastBlockColumnCount) {
            case 4:
              yield value4.x;
              yield value4.y;
              yield value4.z;
              yield value4.w;

              break;
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
          }

          dataIndex += array._matrixInfo.delta1;

          i += array._matrixInfo.lastBlockColumnCount;

          if (row % array.dataType.blockSize < array.dataType.blockSize - 1) {
            dataIndex += array._matrixInfo.delta2;
          } else {
            dataIndex += array._matrixInfo.delta3;
          }
        }

        shapeIndex--;
      } else {
        if (dimensionIndexes[shapeIndex] < broadcastedShape[shapeIndex]) {
          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          if (sourceDimensionIndexes[shapeIndex] <
              array.shape[shapeIndex] - 1) {
            sourceDimensionIndexes[shapeIndex]++;
            dataIndexes[shapeIndex] =
                dataIndex + array._dataInfo.headStride[shapeIndex];
          } else {
            sourceDimensionIndexes[shapeIndex] = 0;
            dataIndexes[shapeIndex] = initialDataIndexes[shapeIndex];
          }

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          sourceDimensionIndexes[shapeIndex] = 0;

          dataIndexes[shapeIndex] = dataIndex;
          initialDataIndexes[shapeIndex] = dataIndex;
        } else {
          shapeIndex--;
        }
      }
    }
  }
}

Iterable<num> _createFullBroadcastedValueIterable(
    NDArrayBlockedImpl array, NDDescriptor broadcastedDescriptor) sync* {
  var broadcastedShape;
  var multiplier;

  if (broadcastedDescriptor.shape.dimension > array.shape.dimension) {
    broadcastedShape = new NDShape(broadcastedDescriptor.shape.dimensions
        .sublist(
            broadcastedDescriptor.shape.dimension - array.shape.dimension));
    multiplier = broadcastedDescriptor.shape.length ~/ broadcastedShape.length;
  } else {
    broadcastedShape = broadcastedDescriptor.shape;
    multiplier = 1;
  }

  for (var iteration = 0; iteration < multiplier; iteration++) {
    var shapeIndex = 0;

    var dimensionIndexes = new List(array.shape.dimension - 1);
    var sourceDimensionIndexes = new List(array.shape.dimension - 1);
    var dataIndexes = new List(array.shape.dimension - 1);
    var initialDataIndexes = new List(array.shape.dimension - 1);

    var dataIndex = 0;
    dimensionIndexes[0] = 0;
    sourceDimensionIndexes[0] = 0;
    dataIndexes[0] = 0;
    initialDataIndexes[0] = 0;

    var rowsMultiplier = broadcastedShape[broadcastedShape.dimension - 2] ~/
        array._matrixInfo.rows;
    var columnsMultiplier = broadcastedShape[broadcastedShape.dimension - 1] ~/
        array._matrixInfo.columns;

    var i = 0;

    while (i < broadcastedShape.length) {
      if (shapeIndex == array.shape.dimension - 2) {
        var initialRowDataIndex = dataIndex;

        for (var rowIteration = 0;
            rowIteration < rowsMultiplier;
            rowIteration++) {
          for (var row = 0; row < array._matrixInfo.rows; row++) {
            var initialColumnDataIndex = dataIndex;

            var lastColumnDataIndex = dataIndex;
            for (var columnIteration = 0;
                columnIteration < columnsMultiplier;
                columnIteration++) {
              var column;
              for (column = 0;
                  column <
                      array._matrixInfo.dataColumns - array.dataType.blockSize;
                  column += array.dataType.blockSize) {
                var value4 = array._data[dataIndex];

                yield value4.x;
                yield value4.y;
                yield value4.z;
                yield value4.w;

                dataIndex += array._matrixInfo.delta1;

                i += array.descriptor.dataType.blockSize;
              }

              var value4 = array._data[dataIndex];

              switch (array._matrixInfo.lastBlockColumnCount) {
                case 4:
                  yield value4.x;
                  yield value4.y;
                  yield value4.z;
                  yield value4.w;

                  break;
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
              }

              dataIndex += array._matrixInfo.delta1;

              i += array._matrixInfo.lastBlockColumnCount;

              if (row % array.dataType.blockSize <
                  array.dataType.blockSize - 1) {
                dataIndex += array._matrixInfo.delta2;
              } else {
                dataIndex += array._matrixInfo.delta3;
              }

              lastColumnDataIndex = dataIndex;

              dataIndex = initialColumnDataIndex;
            }

            dataIndex = lastColumnDataIndex;
          }

          dataIndex = initialRowDataIndex;
        }

        shapeIndex--;
      } else {
        if (dimensionIndexes[shapeIndex] < broadcastedShape[shapeIndex]) {
          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          if (sourceDimensionIndexes[shapeIndex] <
              array.shape[shapeIndex] - 1) {
            sourceDimensionIndexes[shapeIndex]++;
            dataIndexes[shapeIndex] =
                dataIndex + array._dataInfo.headStride[shapeIndex];
          } else {
            sourceDimensionIndexes[shapeIndex] = 0;
            dataIndexes[shapeIndex] = initialDataIndexes[shapeIndex];
          }

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          sourceDimensionIndexes[shapeIndex] = 0;

          dataIndexes[shapeIndex] = dataIndex;
          initialDataIndexes[shapeIndex] = dataIndex;
        } else {
          shapeIndex--;
        }
      }
    }
  }
}

Iterable<num> _createValueIterable(NDArrayBlockedImpl array) sync* {
  var shapeIndex = 0;

  var dimensionIndexes = new List(array.shape.dimension - 1);
  var dataIndexes = new List(array.shape.dimension - 1);

  var dataIndex = 0;
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;

  while (i < array.shape.length) {
    if (shapeIndex == array.shape.dimension - 2) {
      for (var row = 0; row < array._matrixInfo.rows; row++) {
        var column;
        for (column = 0;
            column < array._matrixInfo.dataColumns - array.dataType.blockSize;
            column += array.dataType.blockSize) {
          var value4 = array._data[dataIndex];

          yield value4.x;
          yield value4.y;
          yield value4.z;
          yield value4.w;

          dataIndex += array._matrixInfo.delta1;

          i += array.descriptor.dataType.blockSize;
        }

        var value4 = array._data[dataIndex];

        switch (array._matrixInfo.lastBlockColumnCount) {
          case 4:
            yield value4.x;
            yield value4.y;
            yield value4.z;
            yield value4.w;

            break;
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
        }

        dataIndex += array._matrixInfo.delta1;

        i += array._matrixInfo.lastBlockColumnCount;

        if (row % array.dataType.blockSize < array.dataType.blockSize - 1) {
          dataIndex += array._matrixInfo.delta2;
        } else {
          dataIndex += array._matrixInfo.delta3;
        }
      }

      shapeIndex--;
    } else {
      if (dimensionIndexes[shapeIndex] < array.shape[shapeIndex]) {
        dataIndex = dataIndexes[shapeIndex];

        dimensionIndexes[shapeIndex]++;
        dataIndexes[shapeIndex] =
            dataIndex + array._dataInfo.headStride[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;
      } else {
        shapeIndex--;
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

  final int lastBlockRowCount;

  final int lastBlockColumnCount;

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

    var lastBlockRowCount = lastBlockRowOffset == 0
        ? descriptor.dataType.blockSize
        : lastBlockRowOffset;
    var lastBlockColumnCount = lastBlockColumnOffset == 0
        ? descriptor.dataType.blockSize
        : lastBlockColumnOffset;

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
        lastBlockRowCount,
        lastBlockColumnCount,
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
      this.lastBlockRowCount,
      this.lastBlockColumnCount,
      this.delta1,
      this.delta2,
      this.delta3);
}

class _DataInfo {
  final List<int> headStride;

  final int dataLength;

  _DataInfo(this.headStride, this.dataLength);

  @override
  // ignore: hash_and_equals
  bool operator ==(other) {
    if (other is _DataInfo) {
      return dataLength == other.dataLength &&
          _iterableEquality.equals(headStride, other.headStride);
    } else {
      return false;
    }
  }
}
