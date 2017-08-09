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

  factory NDArrayBlockedImpl(value, NDDescriptor descriptor, NDArray reuse) {
    var dataInfo = new DataInfo(descriptor);

    var data = _createData(descriptor, dataInfo, reuse);

    _loadData(value, data, descriptor, dataInfo);

    return new NDArrayBlockedImpl._(data, descriptor, dataInfo);
  }

  factory NDArrayBlockedImpl.filled(
      fillValue, NDDescriptor descriptor, NDArrayBlockedImpl reuse) {
    if (fillValue == 0) {
      var dataInfo = new DataInfo(descriptor);

      var data = _createData(descriptor, dataInfo, reuse);

      if (reuse != null && reuse.descriptor == descriptor) {
        reuse.data.fillRange(0, reuse.data.length, new Float32x4.zero());
      }

      return new NDArrayBlockedImpl._(data, descriptor, dataInfo);
    } else {
      // TODO ottimizzabile

      return new NDArrayBlockedImpl.generate(
          (index) => fillValue, descriptor, reuse);
    }
  }

  factory NDArrayBlockedImpl.generate(
      generator(int index), NDDescriptor descriptor, NDArray reuse) {
    var dataInfo = new DataInfo(descriptor);

    var data = _createData(descriptor, dataInfo, reuse);

    _generateData(generator, data, descriptor, dataInfo);

    return new NDArrayBlockedImpl._(data, descriptor, dataInfo);
  }

  factory NDArrayBlockedImpl.castFrom(
      NDArrayBase fromArray, NDDataType toDataType, NDArray reuse) {
    var resultDescriptor = fromArray.descriptor.cast(toDataType);

    var dataInfo = new DataInfo(resultDescriptor);

    var data = _createData(resultDescriptor, dataInfo, reuse);

    _castData(fromArray, data, resultDescriptor, dataInfo);

    return new NDArrayBlockedImpl._(data, resultDescriptor, dataInfo);
  }

  NDArrayBlockedImpl._(this.data, NDDescriptor descriptor, this.dataInfo)
      : super.raw(descriptor);

  @override
  bool get isNormalized => true;

  @override
  NDArray normalize({NDArray reuse}) => this;

  NDArray identity({NDArray reuse}) {
    var targetDescriptor = descriptor;

    var targetDataInfo = new DataInfo(targetDescriptor);

    var targetData = _createData(targetDescriptor, targetDataInfo, reuse);

    _identityData(this, targetData, targetDescriptor, targetDataInfo);

    return new NDArrayBlockedImpl._(
        targetData, targetDescriptor, targetDataInfo);
  }

  @override
  dynamic toValue() {
    var value = new List(dataInfo.internalShape[0]);

    var values = new List(dataInfo.internalShape.dimension - 1);
    var dimensionIndexes = new List(dataInfo.internalShape.dimension - 1);
    var dataIndexes = new List(dataInfo.internalShape.dimension - 1);

    values[0] = value;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var shapeIndex = 0;
    var dataIndex = 0;

    for (;;) {
      if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
        if (shapeIndex < dataInfo.internalShape.dimension - 2) {
          var newList = new List(dataInfo.internalShape[shapeIndex + 1]);

          value[dimensionIndexes[shapeIndex]] = newList;

          dataIndex = dataIndexes[shapeIndex];

          shapeIndex++;

          value = newList;
          values[shapeIndex] = value;

          dimensionIndexes[shapeIndex] = 0;
          dataIndexes[shapeIndex] = dataIndex;

          continue;
        } else {
          for (var row = 0; row < dataInfo.rows; row++) {
            List<num> rowValues = new List(dataInfo.columns);

            value[row] = rowValues;

            var column;
            for (column = 0;
                column < dataInfo.dataColumns - descriptor.dataType.blockSize;
                column += descriptor.dataType.blockSize) {
              var value4 = data[dataIndex];

              rowValues[column] = value4.x;
              rowValues[column + 1] = value4.y;
              rowValues[column + 2] = value4.z;
              rowValues[column + 3] = value4.w;

              dataIndex += dataInfo.delta1;
            }

            var value4 = data[dataIndex];

            switch (dataInfo.lastBlockColumnCount) {
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

            dataIndex += dataInfo.delta1;

            if (row % descriptor.dataType.blockSize <
                descriptor.dataType.blockSize - 1) {
              dataIndex += dataInfo.delta2;
            } else {
              dataIndex += dataInfo.delta3;
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

    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    _matMulData(this, array2, resultData, resultDescriptor, resultDataInfo);

    return new NDArrayBlockedImpl._(
        resultData, resultDescriptor, resultDataInfo);
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
      var resultDataInfo = new DataInfo(resultDescriptor);

      var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

      _reduceData(this, reductionAxis, keepDimensions, resultData,
          resultDescriptor, resultDataInfo,
          begin: begin, onValue: onValue, end: end);

      return new NDArrayBlockedImpl._(
          resultData, resultDescriptor, resultDataInfo);
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

  var dimensionIndexes = new List(array.dataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(array.dataInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        array.dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < array.dataInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < array.dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < array.dataInfo.dataColumns - array.dataType.blockSize;
              column += array.dataType.blockSize) {
            var value4 = array.data[dataIndex];

            yield value4.x;
            yield value4.y;
            yield value4.z;
            yield value4.w;

            dataIndex += array.dataInfo.delta1;
          }

          var value4 = array.data[dataIndex];

          switch (array.dataInfo.lastBlockColumnCount) {
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

          dataIndex += array.dataInfo.delta1;

          if (row % array.dataType.blockSize < array.dataType.blockSize - 1) {
            dataIndex += array.dataInfo.delta2;
          } else {
            dataIndex += array.dataInfo.delta3;
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

List _createData(
    NDDescriptor descriptor, DataInfo dataInfo, NDArrayBlockedImpl reuse) {
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

void _loadData(
    value, Float32x4List data, NDDescriptor descriptor, DataInfo dataInfo) {
  var newValue = descriptor.shape.dimension > 1
      ? value
      : (descriptor.shape.dimension == 1
          ? [value]
          : [
              [value]
            ]);

  var values = new List(dataInfo.internalShape.dimension - 1);
  var dimensionIndexes = new List(dataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  values[0] = newValue;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimension - 2) {
        var newList = newValue[dimensionIndexes[shapeIndex]];

        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        newValue = newList;
        values[shapeIndex] = newValue;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < dataInfo.rows; row++) {
          List<num> rowValues = newValue[row];

          var column;
          for (column = 0;
              column < dataInfo.dataColumns - descriptor.dataType.blockSize;
              column += descriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                rowValues[column].toDouble(),
                rowValues[column + 1].toDouble(),
                rowValues[column + 2].toDouble(),
                rowValues[column + 3].toDouble());

            data[dataIndex] = value4;

            dataIndex += dataInfo.delta1;
          }

          var value4;
          switch (dataInfo.lastBlockColumnCount) {
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

          dataIndex += dataInfo.delta1;

          if (row % descriptor.dataType.blockSize <
              descriptor.dataType.blockSize - 1) {
            dataIndex += dataInfo.delta2;
          } else {
            dataIndex += dataInfo.delta3;
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
    NDDescriptor descriptor, DataInfo dataInfo) {
  var dimensionIndexes = new List(dataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;
  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < dataInfo.dataColumns - descriptor.dataType.blockSize;
              column += descriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                generator(i).toDouble(),
                generator(i + 1).toDouble(),
                generator(i + 2).toDouble(),
                generator(i + 3).toDouble());

            data[dataIndex] = value4;

            dataIndex += dataInfo.delta1;

            i += descriptor.dataType.blockSize;
          }

          var value4;
          switch (dataInfo.lastBlockColumnCount) {
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

          dataIndex += dataInfo.delta1;

          i += dataInfo.lastBlockColumnCount;

          if (row % descriptor.dataType.blockSize <
              descriptor.dataType.blockSize - 1) {
            dataIndex += dataInfo.delta2;
          } else {
            dataIndex += dataInfo.delta3;
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
    DataInfo dataInfo) {
  if (fromArray.dataType.isHBlocked && descriptor.dataType.isVBlocked) {
    _castBlockedData(fromArray, data, descriptor, dataInfo);
  } else if (fromArray.dataType.isVBlocked && descriptor.dataType.isHBlocked) {
    _castBlockedData(fromArray, data, descriptor, dataInfo);
  } else if (fromArray.dataType.isFloat && descriptor.dataType.isFloat) {
    _castFromFloatData(fromArray, data, descriptor, dataInfo);
  } else if (fromArray.dataType.isInteger && descriptor.dataType.isFloat) {
    _castFromIntData(fromArray, data, descriptor, dataInfo);
  } else {
    throw new UnsupportedError(
        "Cast from ${fromArray.dataType} to ${descriptor.dataType}");
  }
}

void _castBlockedData(NDArrayBlockedImpl fromArray, List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  var multiplier;
  if (resultDataInfo.internalShape.dimension > 2) {
    multiplier = resultDataInfo.internalShape.length ~/
        (resultDataInfo.rows * resultDataInfo.columns);
  } else {
    multiplier = 1;
  }

  var dimension1s = fromArray.dataType.isHBlocked
      ? resultDataInfo.blockRows
      : resultDataInfo.blockColumns;
  var dimension2s = fromArray.dataType.isHBlocked
      ? resultDataInfo.blockColumns
      : resultDataInfo.blockRows;

  var delta1 = (dimension1s - 1) << resultDescriptor.dataType.blockDepth;

  var delta2 =
      resultDescriptor.dataType.blockSize - resultDataInfo.matrixDataLength;

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

void _castFromFloatData(NDArrayBase fromArray, List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  var valueIterator = fromArray.valueIterable.iterator;

  var dimensionIndexes = new List(resultDataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimension - 1);
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  resultDataInfo.dataColumns -
                      resultDescriptor.dataType.blockSize;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                (valueIterator..moveNext()).current,
                (valueIterator..moveNext()).current,
                (valueIterator..moveNext()).current,
                (valueIterator..moveNext()).current);

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
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

          dataIndex += resultDataInfo.delta1;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultDataInfo.delta2;
          } else {
            dataIndex += resultDataInfo.delta3;
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

void _castFromIntData(NDArrayBase fromArray, List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  Iterator<int> valueIterator = fromArray.valueIterable.iterator;

  var dimensionIndexes = new List(resultDataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimension - 1);
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  resultDataInfo.dataColumns -
                      resultDescriptor.dataType.blockSize;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                (valueIterator..moveNext()).current.toDouble(),
                (valueIterator..moveNext()).current.toDouble(),
                (valueIterator..moveNext()).current.toDouble(),
                (valueIterator..moveNext()).current.toDouble());

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
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

          dataIndex += resultDataInfo.delta1;

          if (row % resultDescriptor.dataType.blockSize <
              resultDescriptor.dataType.blockSize - 1) {
            dataIndex += resultDataInfo.delta2;
          } else {
            dataIndex += resultDataInfo.delta3;
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

void _matMulData(NDArrayBlockedImpl array1, NDArrayBlockedImpl array2,
    List resultData, NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
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
            row1 < array1.dataInfo.dataRows;
            row1 += array1.dataType.blockSize) {
          var initialSourceDataIndex1 = sourceDataIndex1;

          sourceDataIndex2 = initialSourceDataIndex2;

          for (var column2 = 0;
              column2 < array2.dataInfo.dataColumns;
              column2 += array2.dataType.blockSize) {
            sourceDataIndex1 = initialSourceDataIndex1;

            var result0 = new Float32x4.zero();
            var result1 = new Float32x4.zero();
            var result2 = new Float32x4.zero();
            var result3 = new Float32x4.zero();

            for (var i = 0;
                i < array1.dataInfo.dataColumns;
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

void _identityData(NDArrayBlockedImpl sourceArray, Float32x4List targetData,
    NDDescriptor targetDescriptor, DataInfo targetDataInfo) {
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
    {void begin(),
    void onValue(value, int valueCount),
    dynamic end()}) {
  print("*** sourceDataInfo ***");
  print("shape: ${sourceArray.descriptor.shape}");
  print(sourceArray.dataInfo);
  print("*** targetDataInfo ***");
  print("shape: ${targetDescriptor.shape}");
  print(targetDataInfo);

  // TODO riduzione su un array di dimensioni = 1

  // TODO riduzione sulle colonne

  // TODO riduzione con target di dimensioni = 0

  var axis = new Set<int>.from(reductionAxis);

  bool isRowsReduction =
      axis.contains(sourceArray.descriptor.shape.dimension - 2);
  bool isColumnsReduction =
      axis.contains(sourceArray.descriptor.shape.dimension - 1);

  List<int> targetPermutedIndexes;
  if (targetDescriptor.shape.dimension > 1) {
    if (sourceArray.dataType.isHBlocked) {
      targetPermutedIndexes =
          new List.generate(targetDataInfo.dimensions.length, (index) => index);

      if (isRowsReduction) {
        var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
        targetPermutedIndexes[targetPermutedIndexes.length - 1] =
            targetPermutedIndexes[targetPermutedIndexes.length - 2];
        targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
        targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 3);
      }
    } else {
      targetPermutedIndexes =
          new List.generate(targetDataInfo.dimensions.length, (index) => index);

      if (isRowsReduction) {
        var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
        targetPermutedIndexes[targetPermutedIndexes.length - 1] =
            targetPermutedIndexes[targetPermutedIndexes.length - 3];
        targetPermutedIndexes[targetPermutedIndexes.length - 3] = tempIndex;
        targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 2);
      }
    }
  } else if (targetDescriptor.shape.dimension == 1) {
    targetPermutedIndexes = [1];
  } else {
    // TODO to implement _reduceData
    throw new UnimplementedError("to implement _reduceData");
  }

  if (sourceArray.dataType.isHBlocked) {
    if (isRowsReduction) {
      axis.add(sourceArray.descriptor.shape.dimension);
    }
  } else {
    if (isRowsReduction) {
      axis.remove(sourceArray.descriptor.shape.dimension - 2);
      axis.add(sourceArray.descriptor.shape.dimension - 1);
      axis.add(sourceArray.descriptor.shape.dimension);
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

  var targetRowIndex;
  var delta1;
  if (sourceArray.dataType.isHBlocked && isRowsReduction) {
    // TODO attenzione se è anche isColumnsReduction
    targetRowIndex = sourceArray.shape.dimension - 3;
    delta1 = sourceArray.dataInfo.dataColumns - sourceArray.dataType.blockSize;
  }

  var sourceDimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensions.length + 1);
  var sourceDataIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensions.length + 1);
  var targetDataIndexes =
      new List(targetDataInfo.internalShape.dimensions.length + 1);

  sourceDimensionIndexes[sourcePermutedIndexes[0]] = 0;
  sourceDataIndexes[sourcePermutedIndexes[0]] = 0;
  targetDataIndexes[targetPermutedIndexes[0]] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  var lastRow = false;
  var lastColumn = false;
  var blockColumnCount;

  for (;;) {
    var dimension;
    if (sourcePermutedIndexes[shapeIndex] ==
        sourceArray.dataInfo.dimensions.length - 1) {
      if (lastRow) {
        dimension = sourceArray.dataInfo.lastBlockRowCount;
      } else {
        dimension =
            sourceArray.dataInfo.dimensions[sourcePermutedIndexes[shapeIndex]];
      }

      if (lastColumn) {
        blockColumnCount = sourceArray.dataInfo.lastBlockColumnCount;
      } else {
        blockColumnCount =
            sourceArray.dataInfo.dimensions[sourcePermutedIndexes[shapeIndex]];
      }
    } else {
      dimension =
          sourceArray.dataInfo.dimensions[sourcePermutedIndexes[shapeIndex]];
    }

    if (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] < dimension) {
      if (shapeIndex < sourceArray.dataInfo.dimensions.length - 1) {
        sourceDataIndex = sourceDataIndexes[sourcePermutedIndexes[shapeIndex]];

        if (shapeIndex <
            sourceArray.dataInfo.dimensions.length - newReductionAxis.length) {
          targetDataIndex =
              targetDataIndexes[targetPermutedIndexes[shapeIndex]];
        }

        shapeIndex++;

        sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] = 0;
        sourceDataIndexes[sourcePermutedIndexes[shapeIndex]] = sourceDataIndex;

        if (shapeIndex <
            sourceArray.dataInfo.dimensions.length - newReductionAxis.length) {
          targetDataIndexes[targetPermutedIndexes[shapeIndex]] =
              targetDataIndex;
        }

        if (shapeIndex ==
            sourceArray.dataInfo.dimensions.length - newReductionAxis.length) {
          print("BEGIN");

          begin();
        }

        continue;
      } else {
        print(
            "($sourceDimensionIndexes)[$sourceDataIndex]: ONVALUE(${sourceArray.data[sourceDataIndex]}, $blockColumnCount)");

        onValue(sourceArray.data[sourceDataIndex], blockColumnCount);
      }
    } else {
      shapeIndex--;

      if (shapeIndex ==
          sourceArray.dataInfo.dimensions.length -
              newReductionAxis.length -
              1) {
        if (isColumnsReduction) {
          print("ONVALUE(null, null)");

          onValue(null, null);

          var reducedValue = end();

          print("END[$targetDataIndex]: $reducedValue");

          // TODO write in targetData
        } else {
          var reducedValue = end();

          print("END[$targetDataIndex]: $reducedValue");

          targetData[targetDataIndex] = reducedValue;
        }
      }
    }

    if (shapeIndex >= 0) {
      sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]]++;
      sourceDataIndexes[sourcePermutedIndexes[shapeIndex]] +=
          sourceArray.dataInfo.stride[sourcePermutedIndexes[shapeIndex]];
      sourceDataIndex = sourceDataIndexes[sourcePermutedIndexes[shapeIndex]];

      if (shapeIndex <
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length) {
        targetDataIndexes[targetPermutedIndexes[shapeIndex]] +=
            targetDataInfo.stride[targetPermutedIndexes[shapeIndex]];

        if (sourceArray.dataType.isHBlocked && shapeIndex == targetRowIndex) {
          var sourceDimensionIndex =
              sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]];
          if (sourceDimensionIndex % sourceArray.dataType.blockSize == 0) {
            targetDataIndexes[targetPermutedIndexes[shapeIndex]] += delta1;
          }
        }

        targetDataIndex = targetDataIndexes[targetPermutedIndexes[shapeIndex]];
      }

      if (sourcePermutedIndexes[shapeIndex] == sourceArray.dataInfo.rowIndex) {
        lastRow = (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] ==
            sourceArray.dataInfo.dimensions[sourcePermutedIndexes[shapeIndex]] -
                1);
      } else if (sourcePermutedIndexes[shapeIndex] ==
          sourceArray.dataInfo.columnIndex) {
        lastColumn =
            (sourceDimensionIndexes[sourcePermutedIndexes[shapeIndex]] ==
                sourceArray.dataInfo
                        .dimensions[sourcePermutedIndexes[shapeIndex]] -
                    1);
      }
    } else {
      break;
    }
  }
}

class DataInfo {
  final NDShape internalShape;

  final int rows;

  final int columns;

  final int blockRows;

  final int blockColumns;

  final int dataRows;

  final int dataColumns;

  final int matrixDataLength;

  final int lastBlockRowCount;

  final int lastBlockColumnCount;

  final int delta1;

  final int delta2;

  final int delta3;

  final List<int> dimensions;

  final List<int> stride;

  final int rowIndex;

  final int columnIndex;

  final int dataLength;

  factory DataInfo(NDDescriptor descriptor) {
    var internalShape = descriptor.shape.dimension > 1
        ? descriptor.shape
        : (descriptor.shape.dimension == 1
            ? new NDShape([1, descriptor.shape[0]])
            : new NDShape([1, 1]));

    var rows = internalShape.dimensions[internalShape.dimension - 2];
    var columns = internalShape.dimensions[internalShape.dimension - 1];

    var blockRows = (rows + descriptor.dataType.blockSize - 1) >>
        descriptor.dataType.blockDepth; // equal (/4).ceil
    var blockColumns = (columns + descriptor.dataType.blockSize - 1) >>
        descriptor.dataType.blockDepth; // equal (/4).ceil

    var dataRows = blockRows << descriptor.dataType.blockDepth;
    var dataColumns = blockColumns << descriptor.dataType.blockDepth;

    var matrixDataLength = dataRows * blockColumns;

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
      delta2 = 1 - matrixDataLength;
    }

    var delta3;
    if (descriptor.dataType.isHBlocked) {
      delta3 = 1 - descriptor.dataType.blockSize;
    } else {
      delta3 = delta2;
    }

    List<int> dimensions = new List.from(
        internalShape.dimensions.sublist(0, internalShape.dimension - 2));

    var rowIndex;
    var columnIndex;

    if (descriptor.dataType.isHBlocked) {
      rowIndex = dimensions.length;
      dimensions.add(blockRows);
      columnIndex = dimensions.length;
      dimensions.add(blockColumns);
      dimensions.add(descriptor.dataType.blockSize);
    } else {
      columnIndex = dimensions.length;
      dimensions.add(blockColumns);
      rowIndex = dimensions.length;
      dimensions.add(blockRows);
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
        internalShape,
        rows,
        columns,
        blockRows,
        blockColumns,
        dataRows,
        dataColumns,
        matrixDataLength,
        lastBlockRowCount,
        lastBlockColumnCount,
        delta1,
        delta2,
        delta3,
        dimensions,
        stride,
        rowIndex,
        columnIndex,
        dataLength);
  }

  DataInfo._(
      this.internalShape,
      this.rows,
      this.columns,
      this.blockRows,
      this.blockColumns,
      this.dataRows,
      this.dataColumns,
      this.matrixDataLength,
      this.lastBlockRowCount,
      this.lastBlockColumnCount,
      this.delta1,
      this.delta2,
      this.delta3,
      this.dimensions,
      this.stride,
      this.rowIndex,
      this.columnIndex,
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
    buffer.writeln("virtualShape: $internalShape");
    buffer.writeln("rows: $rows");
    buffer.writeln("columns: $columns");
    buffer.writeln("blockRows: $blockRows");
    buffer.writeln("blockColumns: $blockColumns");
    buffer.writeln("dataRows: $dataRows");
    buffer.writeln("dataColumns: $dataColumns");
    buffer.writeln("matrixDataLength: $matrixDataLength");
    buffer.writeln("lastBlockRowCount: $lastBlockRowCount");
    buffer.writeln("lastBlockColumnCount: $lastBlockColumnCount");
    buffer.writeln("delta1: $delta1");
    buffer.writeln("delta2: $delta2");
    buffer.writeln("delta3: $delta3");
    return buffer.toString();
  }
}
