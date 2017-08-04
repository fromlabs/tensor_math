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
  final List data;

  final DataInfo dataInfo;

  final MatrixInfo matrixInfo;

  factory NDArrayBlockedImpl(value, NDDescriptor descriptor, NDArray reuse) {
    var matrixInfo = new MatrixInfo(descriptor);

    var dataInfo = new DataInfo(descriptor, matrixInfo);

    var data = _createData(descriptor, dataInfo, matrixInfo, reuse);

    _loadData(value, data, descriptor, dataInfo, matrixInfo);

    return new NDArrayBlockedImpl._(data, descriptor, dataInfo, matrixInfo);
  }

  factory NDArrayBlockedImpl.filled(
      fillValue, NDDescriptor descriptor, NDArrayBlockedImpl reuse) {
    if (fillValue == 0) {
      var matrixInfo = new MatrixInfo(descriptor);

      var dataInfo = new DataInfo(descriptor, matrixInfo);

      var data = _createData(descriptor, dataInfo, matrixInfo, reuse);

      if (reuse != null && reuse.descriptor == descriptor) {
        reuse.data.fillRange(0, reuse.data.length, new Float32x4.zero());
      }

      return new NDArrayBlockedImpl._(data, descriptor, dataInfo, matrixInfo);
    } else {
      // TODO ottimizzabile

      return new NDArrayBlockedImpl.generate(
          (index) => fillValue, descriptor, reuse);
    }
  }

  factory NDArrayBlockedImpl.generate(
      generator(int index), NDDescriptor descriptor, NDArray reuse) {
    var matrixInfo = new MatrixInfo(descriptor);

    var dataInfo = new DataInfo(descriptor, matrixInfo);

    var data = _createData(descriptor, dataInfo, matrixInfo, reuse);

    _generateData(generator, data, descriptor, dataInfo, matrixInfo);

    return new NDArrayBlockedImpl._(data, descriptor, dataInfo, matrixInfo);
  }

  factory NDArrayBlockedImpl.castFrom(
      NDArrayBase fromArray, NDDataType toDataType, NDArray reuse) {
    var resultDescriptor = fromArray.descriptor.cast(toDataType);

    var matrixInfo = new MatrixInfo(resultDescriptor);

    var dataInfo = new DataInfo(resultDescriptor, matrixInfo);

    var data = _createData(resultDescriptor, dataInfo, matrixInfo, reuse);

    _castData(fromArray, data, resultDescriptor, dataInfo, matrixInfo);

    return new NDArrayBlockedImpl._(
        data, resultDescriptor, dataInfo, matrixInfo);
  }

  NDArrayBlockedImpl._(
      this.data, NDDescriptor descriptor, this.dataInfo, this.matrixInfo)
      : super.raw(descriptor);

  @override
  bool get isNormalized => true;

  @override
  NDArray normalize({NDArray reuse}) => this;

  NDArray identity({NDArray reuse}) {
    var targetDescriptor = descriptor;

    var targetMatrixInfo = new MatrixInfo(targetDescriptor);

    var targetDataInfo = new DataInfo(targetDescriptor, targetMatrixInfo);

    var targetData =
        _createData(targetDescriptor, targetDataInfo, targetMatrixInfo, reuse);

    _identityData(
        this, targetData, targetDescriptor, targetDataInfo, targetMatrixInfo);

    return new NDArrayBlockedImpl._(
        targetData, targetDescriptor, targetDataInfo, targetMatrixInfo);
  }

  @override
  dynamic toValue() {
    var value = new List(matrixInfo.internalShape[0]);

    var values = new List(matrixInfo.internalShape.dimension - 1);
    var dimensionIndexes = new List(matrixInfo.internalShape.dimension - 1);
    var dataIndexes = new List(matrixInfo.internalShape.dimension - 1);

    values[0] = value;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var shapeIndex = 0;
    var dataIndex = 0;

    for (;;) {
      if (dimensionIndexes[shapeIndex] < matrixInfo.internalShape[shapeIndex]) {
        if (shapeIndex < matrixInfo.internalShape.dimension - 2) {
          var newList = new List(matrixInfo.internalShape[shapeIndex + 1]);

          value[dimensionIndexes[shapeIndex]] = newList;

          dataIndex = dataIndexes[shapeIndex];

          shapeIndex++;

          value = newList;
          values[shapeIndex] = value;

          dimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = dataIndex;

          continue;
        } else {
          for (var row = 0; row < matrixInfo.rows; row++) {
            List<num> rowValues = new List(matrixInfo.columns);

            value[row] = rowValues;

            var column;
            for (column = 0;
                column < matrixInfo.dataColumns - descriptor.dataType.blockSize;
                column += descriptor.dataType.blockSize) {
              var value4 = data[dataIndex];

              rowValues[column] = value4.x;
              rowValues[column + 1] = value4.y;
              rowValues[column + 2] = value4.z;
              rowValues[column + 3] = value4.w;

              dataIndex += matrixInfo.delta1;
            }

            var value4 = data[dataIndex];

            switch (matrixInfo.lastBlockColumnCount) {
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

            dataIndex += matrixInfo.delta1;

            if (row % descriptor.dataType.blockSize <
                descriptor.dataType.blockSize - 1) {
              dataIndex += matrixInfo.delta2;
            } else {
              dataIndex += matrixInfo.delta3;
            }
          }

          shapeIndex--;
        }
      } else {
        shapeIndex--;
      }

      if (shapeIndex >= 0) {
        dimensionIndexes[shapeIndex]++;
        dataIndexes[shapeIndex] += dataInfo.stride[shapeIndex];
        dataIndex = dataIndexes[shapeIndex];
        value = values[shapeIndex];
      } else {
        break;
      }
    }

    return descriptor.shape.dimension > 1
        ? value
        : (descriptor.shape.dimension == 1 ? value[0] : value[0][0]);
  }

  @override
  String toString() =>
      "<value: ${toValue()}, shape: $shape, dataType: $dataType, stride: ${dataInfo.stride}>";

  void logData() {
    print(data);
  }

  @override
  NDArray matMul(value2, {NDArray reuse}) {
    NDArrayBlockedImpl array2 =
        toNDArray(value2, dataType: NDDataType.float32VBlocked);

    var resultDescriptor = descriptor.matMul(array2.descriptor);

    var resultMatrixInfo = new MatrixInfo(resultDescriptor);

    var resultDataInfo = new DataInfo(resultDescriptor, resultMatrixInfo);

    var resultData =
        _createData(resultDescriptor, resultDataInfo, resultMatrixInfo, reuse);

    _matMulData(this, array2, resultData, resultDescriptor, resultDataInfo,
        resultMatrixInfo);

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo, resultMatrixInfo);
  }

  @override
  NDArray reduceOperationInternal(List<int> reductionAxis, bool keepDimensions,
      NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(), void onValue(value, int valueCount), dynamic end()}) {
    if (keepDimensions) {
      throw new UnimplementedError();
    }

    var newReductionAxis =
        convertToValidReductionAxis(reductionAxis, shape.dimension);

    if (newReductionAxis.isNotEmpty) {
      var resultMatrixInfo = new MatrixInfo(resultDescriptor);

      var resultDataInfo = new DataInfo(resultDescriptor, resultMatrixInfo);

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
      void onValue(int axeIndex, int dimensionIndex, value, int valueCount),
      dynamic end()}) {
    // TODO to implement NDArrayBlockedImpl.argOperationInternal
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.argOperationInternal: $this");
  }

  @override
  NDArrayBase elementWiseUnaryOperationInternal(
      NDDescriptor resultDescriptor, NDArray reuse, unaryOperation(value)) {
    // TODO to implement NDArrayBlockedImpl.elementWiseUnaryOperationInternal
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.elementWiseUnaryOperationInternal: $this");
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
  NDArray reshape({List<int> newDimensions, NDArray reuse}) {
    // TODO to implement NDArrayBlockedImpl.reshape
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.reshape: $this");
  }

  @override
  NDArray tile(List<int> multiplies, {NDArray reuse}) {
    // TODO to implement NDArrayBlockedImpl.tile
    throw new UnimplementedError("to implement NDArrayBlockedImpl.tile: $this");
  }

  @override
  NDArray transpose({List<int> permutationAxis, NDArray reuse}) {
    // TODO to implement NDArrayBlockedImpl.transpose
    throw new UnimplementedError(
        "to implement NDArrayBlockedImpl.transpose: $this");
  }

  @override
  Iterable get valueIterable => _createValueIterable(this);
}

final _iterableEquality = new IterableEquality<dynamic>();

Iterable<num> _createValueIterable(NDArrayBlockedImpl array) sync* {
  // TODO rivedere

  var dimensionIndexes = new List(array.matrixInfo.internalShape.dimension - 1);
  var dataIndexes = new List(array.matrixInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        array.matrixInfo.internalShape[shapeIndex]) {
      if (shapeIndex < array.matrixInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < array.matrixInfo.rows; row++) {
          var column;
          for (column = 0;
              column < array.matrixInfo.dataColumns - array.dataType.blockSize;
              column += array.dataType.blockSize) {
            var value4 = array.data[dataIndex];

            yield value4.x;
            yield value4.y;
            yield value4.z;
            yield value4.w;

            dataIndex += array.matrixInfo.delta1;
          }

          var value4 = array.data[dataIndex];

          switch (array.matrixInfo.lastBlockColumnCount) {
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

          dataIndex += array.matrixInfo.delta1;

          if (row % array.dataType.blockSize < array.dataType.blockSize - 1) {
            dataIndex += array.matrixInfo.delta2;
          } else {
            dataIndex += array.matrixInfo.delta3;
          }
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      dataIndexes[shapeIndex] += array.dataInfo.stride[shapeIndex];

      dataIndex = dataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

List _createData(NDDescriptor descriptor, DataInfo dataInfo,
    MatrixInfo matrixInfo, NDArrayBlockedImpl reuse) {
  if (descriptor.dataType == NDDataType.unknown) {
    throw new ArgumentError.value(descriptor.dataType.toString(), "data type");
  }

  if (reuse != null && reuse.descriptor == descriptor) {
    return reuse.data;
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
    DataInfo dataInfo, MatrixInfo matrixInfo) {
  var newValue = descriptor.shape.dimension > 1
      ? value
      : (descriptor.shape.dimension == 1
          ? [value]
          : [
              [value]
            ]);

  var values = new List(matrixInfo.internalShape.dimension - 1);
  var dimensionIndexes = new List(matrixInfo.internalShape.dimension - 1);
  var dataIndexes = new List(matrixInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  values[0] = newValue;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < matrixInfo.internalShape[shapeIndex]) {
      if (shapeIndex < matrixInfo.internalShape.dimension - 2) {
        var newList = newValue[dimensionIndexes[shapeIndex]];

        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        newValue = newList;
        values[shapeIndex] = newValue;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < matrixInfo.rows; row++) {
          List<num> rowValues = newValue[row];

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
              value4 =
                  new Float32x4(rowValues[column].toDouble(), 0.0, 0.0, 0.0);
              break;
          }

          data[dataIndex] = value4;

          dataIndex += matrixInfo.delta1;

          if (row % descriptor.dataType.blockSize <
              descriptor.dataType.blockSize - 1) {
            dataIndex += matrixInfo.delta2;
          } else {
            dataIndex += matrixInfo.delta3;
          }
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;
      dataIndexes[shapeIndex] += dataInfo.stride[shapeIndex];
      dataIndex = dataIndexes[shapeIndex];
      newValue = values[shapeIndex];
    } else {
      break;
    }
  }
}

void _generateData(generator(int index), Float32x4List data,
    NDDescriptor descriptor, DataInfo dataInfo, MatrixInfo matrixInfo) {
  var dimensionIndexes = new List(matrixInfo.internalShape.dimension - 1);
  var dataIndexes = new List(matrixInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;
  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < matrixInfo.internalShape[shapeIndex]) {
      if (shapeIndex < matrixInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
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
              value4 = new Float32x4(
                  generator(i).toDouble(),
                  generator(i + 1).toDouble(),
                  generator(i + 2).toDouble(),
                  0.0);
              break;
            case 2:
              value4 = new Float32x4(generator(i).toDouble(),
                  generator(i + 1).toDouble(), 0.0, 0.0);
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
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;
      dataIndexes[shapeIndex] += dataInfo.stride[shapeIndex];
      dataIndex = dataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

void _castData(NDArrayBase fromArray, List data, NDDescriptor descriptor,
    DataInfo dataInfo, MatrixInfo matrixInfo) {
  if (fromArray.dataType.isHBlocked && descriptor.dataType.isVBlocked) {
    _castBlockedData(fromArray, data, descriptor, dataInfo, matrixInfo);
  } else if (fromArray.dataType.isVBlocked && descriptor.dataType.isHBlocked) {
    _castBlockedData(fromArray, data, descriptor, dataInfo, matrixInfo);
  } else if (fromArray.dataType.isFloat && descriptor.dataType.isFloat) {
    _castFromFloatData(fromArray, data, descriptor, dataInfo, matrixInfo);
  } else if (fromArray.dataType.isInteger && descriptor.dataType.isFloat) {
    _castFromIntData(fromArray, data, descriptor, dataInfo, matrixInfo);
  } else {
    throw new UnsupportedError(
        "Cast from ${fromArray.dataType} to ${descriptor.dataType}");
  }
}

void _castBlockedData(
    NDArrayBlockedImpl fromArray,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    MatrixInfo resultMatrixInfo) {
  var multiplier;
  if (resultMatrixInfo.internalShape.dimension > 2) {
    multiplier = resultMatrixInfo.internalShape.length ~/
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
        resultData[targetDataIndex++] = fromArray.data[sourceDataIndex++];
        resultData[targetDataIndex++] = fromArray.data[sourceDataIndex++];
        resultData[targetDataIndex++] = fromArray.data[sourceDataIndex++];
        resultData[targetDataIndex++] = fromArray.data[sourceDataIndex++];

        targetDataIndex += delta1;
      }

      targetDataIndex += delta2;
    }

    targetDataIndex += delta3;
  }
}

void _castFromFloatData(
    NDArrayBase fromArray,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    MatrixInfo resultMatrixInfo) {
  var valueIterator = fromArray.valueIterable.iterator;

  var dimensionIndexes = new List(resultMatrixInfo.internalShape.dimension - 1);
  var dataIndexes = new List(resultMatrixInfo.internalShape.dimension - 1);
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultMatrixInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultMatrixInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
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

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultMatrixInfo.delta2;
          } else {
            dataIndex += resultMatrixInfo.delta3;
          }
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      dataIndexes[shapeIndex] += resultDataInfo.stride[shapeIndex];

      dataIndex = dataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

void _castFromIntData(
    NDArrayBase fromArray,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    MatrixInfo resultMatrixInfo) {
  Iterator<int> valueIterator = fromArray.valueIterable.iterator;

  var dimensionIndexes = new List(resultMatrixInfo.internalShape.dimension - 1);
  var dataIndexes = new List(resultMatrixInfo.internalShape.dimension - 1);
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultMatrixInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultMatrixInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultMatrixInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  resultMatrixInfo.dataColumns -
                      resultDescriptor.dataType.blockSize;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                (valueIterator..moveNext()).current.toDouble(),
                (valueIterator..moveNext()).current.toDouble(),
                (valueIterator..moveNext()).current.toDouble(),
                (valueIterator..moveNext()).current.toDouble());

            resultData[dataIndex] = value4;

            dataIndex += resultMatrixInfo.delta1;
          }

          var value4;
          switch (resultMatrixInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Float32x4(
                  (valueIterator..moveNext()).current.toDouble(),
                  (valueIterator..moveNext()).current.toDouble(),
                  (valueIterator..moveNext()).current.toDouble(),
                  (valueIterator..moveNext()).current.toDouble());
              break;
            case 3:
              value4 = new Float32x4(
                  (valueIterator..moveNext()).current.toDouble(),
                  (valueIterator..moveNext()).current.toDouble(),
                  (valueIterator..moveNext()).current.toDouble(),
                  0.0);
              break;
            case 2:
              value4 = new Float32x4(
                  (valueIterator..moveNext()).current.toDouble(),
                  (valueIterator..moveNext()).current.toDouble(),
                  0.0,
                  0.0);
              break;
            case 1:
              value4 = new Float32x4(
                  (valueIterator..moveNext()).current.toDouble(),
                  0.0,
                  0.0,
                  0.0);
              break;
          }

          resultData[dataIndex] = value4;

          dataIndex += resultMatrixInfo.delta1;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultMatrixInfo.delta2;
          } else {
            dataIndex += resultMatrixInfo.delta3;
          }
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      dataIndexes[shapeIndex] += resultDataInfo.stride[shapeIndex];

      dataIndex = dataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

void _matMulData(
    NDArrayBlockedImpl array1,
    NDArrayBlockedImpl array2,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    MatrixInfo resultMatrixInfo) {
  var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
  var sourceDataIndexes1 = new List(resultDescriptor.shape.dimension - 1);
  var sourceDataIndexes2 = new List(resultDescriptor.shape.dimension - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimension - 1);

  dimensionIndexes[0] = 0;
  sourceDataIndexes1[0] = 0;
  sourceDataIndexes2[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex1 = 0;
  var sourceDataIndex2 = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < resultDescriptor.shape[shapeIndex]) {
      if (shapeIndex < resultDescriptor.shape.dimension - 2) {
        sourceDataIndex1 = sourceDataIndexes1[shapeIndex];
        sourceDataIndex2 = sourceDataIndexes2[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        sourceDataIndexes1[shapeIndex] = sourceDataIndex1;
        sourceDataIndexes2[shapeIndex] = sourceDataIndex2;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        var initialSourceDataIndex2 = sourceDataIndex2;

        for (var row1 = 0;
            row1 < array1.matrixInfo.dataRows;
            row1 += array1.dataType.blockSize) {
          var initialSourceDataIndex1 = sourceDataIndex1;

          sourceDataIndex2 = initialSourceDataIndex2;

          for (var column2 = 0;
              column2 < array2.matrixInfo.dataColumns;
              column2 += array2.dataType.blockSize) {
            sourceDataIndex1 = initialSourceDataIndex1;

            var result0 = new Float32x4.zero();
            var result1 = new Float32x4.zero();
            var result2 = new Float32x4.zero();
            var result3 = new Float32x4.zero();

            for (var i = 0;
                i < array1.matrixInfo.dataColumns;
                i += array1.dataType.blockSize) {
              var b0 = array2.data[sourceDataIndex2++];
              var b1 = array2.data[sourceDataIndex2++];
              var b2 = array2.data[sourceDataIndex2++];
              var b3 = array2.data[sourceDataIndex2++];

              var a0 = array1.data[sourceDataIndex1++];
              result0 += a0.shuffle(Float32x4.XXXX) * b0 +
                  a0.shuffle(Float32x4.YYYY) * b1 +
                  a0.shuffle(Float32x4.ZZZZ) * b2 +
                  a0.shuffle(Float32x4.WWWW) * b3;

              var a1 = array1.data[sourceDataIndex1++];
              result1 += a1.shuffle(Float32x4.XXXX) * b0 +
                  a1.shuffle(Float32x4.YYYY) * b1 +
                  a1.shuffle(Float32x4.ZZZZ) * b2 +
                  a1.shuffle(Float32x4.WWWW) * b3;

              var a2 = array1.data[sourceDataIndex1++];
              result2 += a2.shuffle(Float32x4.XXXX) * b0 +
                  a2.shuffle(Float32x4.YYYY) * b1 +
                  a2.shuffle(Float32x4.ZZZZ) * b2 +
                  a2.shuffle(Float32x4.WWWW) * b3;

              var a3 = array1.data[sourceDataIndex1++];
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
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      sourceDataIndexes1[shapeIndex] += array1.dataInfo.stride[shapeIndex];
      sourceDataIndexes2[shapeIndex] += array2.dataInfo.stride[shapeIndex];
      targetDataIndexes[shapeIndex] += resultDataInfo.stride[shapeIndex];

      sourceDataIndex1 = sourceDataIndexes1[shapeIndex];
      sourceDataIndex2 = sourceDataIndexes2[shapeIndex];
      targetDataIndex = targetDataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

void _identityData(
    NDArrayBlockedImpl sourceArray,
    Float32x4List targetData,
    NDDescriptor targetDescriptor,
    DataInfo targetDataInfo,
    MatrixInfo targetMatrixInfo) {
  var targetDimensionIndexes = new List(targetDataInfo.dimensions.length);
  var targetDataIndexes = new List(targetDataInfo.dimensions.length);

  targetDimensionIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var sourceDataIndex = 0;
  var targetDataIndex = 0;
  var targetShapeIndex = 0;

  for (;;) {
    if (targetDimensionIndexes[targetShapeIndex] <
        targetDataInfo.dimensions[targetShapeIndex]) {
      if (targetShapeIndex < targetDataInfo.dimensions.length - 1) {
        targetDataIndex = targetDataIndexes[targetShapeIndex];

        targetShapeIndex++;

        targetDimensionIndexes[targetShapeIndex] = 0;
        targetDataIndexes[targetShapeIndex] = targetDataIndex;

        continue;
      } else {
        targetData[targetDataIndex] = sourceArray.data[sourceDataIndex++];
      }
    } else {
      targetShapeIndex--;
    }

    if (targetShapeIndex >= 0) {
      targetDimensionIndexes[targetShapeIndex]++;
      targetDataIndexes[targetShapeIndex] +=
          targetDataInfo.stride[targetShapeIndex];
      targetDataIndex = targetDataIndexes[targetShapeIndex];
    } else {
      break;
    }
  }
}

void _reduceData(
    NDArrayBlockedImpl sourceArray,
    List<int> reductionAxis,
    bool keepDimensions,
    Float32x4List targetData,
    NDDescriptor targetDescriptor,
    DataInfo targetDataInfo,
    MatrixInfo targetMatrixInfo,
    {void begin(),
    void onValue(value, int valueCount),
    dynamic end()}) {
/*
  var axis = new Set<int>.from(reductionAxis);
  if (axis.contains(sourceArray.descriptor.shape.dimension - 1)) {
    throw new UnimplementedError();
  }

  List<int> targetPermutedIndexes;

  if (sourceArray.dataType.isHBlocked) {
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
          dimension = sourceArray.matrixInfo.lastBlockRowCount;
        } else {
          dimension = sourceDimensions[sourcePermutedIndexes[shapeIndex]];
        }

        if (lastColumn) {
          blockColumnCount = sourceArray.matrixInfo.lastBlockColumnCount;
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
*/
}

class MatrixInfo {
  final NDShape internalShape;

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

  factory MatrixInfo(NDDescriptor descriptor) {
    var virtualShape = descriptor.shape.dimension > 1
        ? descriptor.shape
        : (descriptor.shape.dimension == 1
            ? new NDShape([1, descriptor.shape[0]])
            : new NDShape([1, 1]));

    var rows = virtualShape.dimensions[virtualShape.dimension - 2];
    var columns = virtualShape.dimensions[virtualShape.dimension - 1];

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

    return new MatrixInfo._(
        virtualShape,
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

  MatrixInfo._(
      this.internalShape,
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

  @override
  String toString() {
    var buffer = new StringBuffer();
    buffer.writeln("virtualShape: $internalShape");
    buffer.writeln("rows: $rows");
    buffer.writeln("columns: $columns");
    buffer.writeln("blockRows: $blockRows");
    buffer.writeln("blockColumns: $blockColumns");
    buffer.writeln("dataRows: $dataRows");
    buffer.writeln("dataColumns: $dataColumns");
    buffer.writeln("dataLength: $dataLength");
    buffer.writeln("lastBlockRowCount: $lastBlockRowCount");
    buffer.writeln("lastBlockColumnCount: $lastBlockColumnCount");
    buffer.writeln("delta1: $delta1");
    buffer.writeln("delta2: $delta2");
    buffer.writeln("delta3: $delta3");
    return buffer.toString();
  }
}

class DataInfo {
  final List<int> dimensions;

  final List<int> stride;

  final int rowIndex;

  final int columnIndex;

  final int dataLength;

  factory DataInfo(NDDescriptor descriptor, MatrixInfo matrixInfo) {
    List<int> dimensions = new List.from(matrixInfo.internalShape.dimensions
        .sublist(0, matrixInfo.internalShape.dimension - 2));

    var rowIndex;
    var columnIndex;

    if (descriptor.dataType.isHBlocked) {
      rowIndex = dimensions.length;
      dimensions.add(matrixInfo.blockRows);
      columnIndex = dimensions.length;
      dimensions.add(matrixInfo.blockColumns);
      dimensions.add(descriptor.dataType.blockSize);
    } else {
      columnIndex = dimensions.length;
      dimensions.add(matrixInfo.blockColumns);
      rowIndex = dimensions.length;
      dimensions.add(matrixInfo.blockRows);
      dimensions.add(descriptor.dataType.blockSize);
    }

    List<int> stride = new List(dimensions.length);
    var factor = 1;
    for (var i = dimensions.length - 1; i >= 0; i--) {
      stride[i] = factor;
      factor *= dimensions[i];
    }

    var dataLength = dimensions.first * stride.first;

    return new DataInfo._(
        dimensions, stride, rowIndex, columnIndex, dataLength);
  }

  DataInfo._(this.dimensions, this.stride, this.rowIndex, this.columnIndex,
      this.dataLength);

  @override
  // ignore: hash_and_equals
  bool operator ==(other) {
    if (other is DataInfo) {
      return dataLength == other.dataLength &&
          _iterableEquality.equals(dimensions, other.dimensions);
    } else {
      return false;
    }
  }

  @override
  String toString() {
    var buffer = new StringBuffer();
    buffer.writeln("dimensions: $dimensions");
    buffer.writeln("stride: $stride");
    buffer.writeln("rowIndex: $rowIndex");
    buffer.writeln("columnIndex: $columnIndex");
    buffer.writeln("dataLength: $dataLength");
    return buffer.toString();
  }
}
