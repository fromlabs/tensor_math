// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'dart:math';
import "dart:typed_data";

import "package:collection/collection.dart";

import 'nd_descriptor.dart';
import 'nd_shape.dart';
import 'nd_data_type.dart';
import "nd_array.dart";

import "nd_array_base.dart";
import "nd_array_impl.dart" as simple_impl;

final _ZERO = new Float32x4.zero();

final _iterableEquality = new IterableEquality<dynamic>();

class NDArrayBlockedImpl extends NDArrayBase {
  final List data;

  final DataInfo dataInfo;

  factory NDArrayBlockedImpl(value, NDDescriptor descriptor, NDArray reuse) {
    var dataInfo = new DataInfo(descriptor);

    var data = _createData(descriptor, dataInfo, reuse);

    _loadData(value, data, descriptor, dataInfo);

    return new NDArrayBlockedImpl.raw(data, descriptor, dataInfo);
  }

  factory NDArrayBlockedImpl.filled(
      fillValue, NDDescriptor descriptor, NDArrayBlockedImpl reuse) {
    if (fillValue == 0) {
      var dataInfo = new DataInfo(descriptor);

      var data = _createData(descriptor, dataInfo, reuse);

      if (reuse != null && reuse.descriptor == descriptor) {
        reuse.data.fillRange(0, reuse.data.length, _ZERO);
      }

      return new NDArrayBlockedImpl.raw(data, descriptor, dataInfo);
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

    return new NDArrayBlockedImpl.raw(data, descriptor, dataInfo);
  }

  factory NDArrayBlockedImpl.castFrom(
      NDArrayBase fromArray, NDDataType toDataType, NDArray reuse) {
    var resultDescriptor = fromArray.descriptor.cast(toDataType);

    var dataInfo = new DataInfo(resultDescriptor);

    var data = _createData(resultDescriptor, dataInfo, reuse);

    _castData(fromArray, data, resultDescriptor, dataInfo);

    return new NDArrayBlockedImpl.raw(data, resultDescriptor, dataInfo);
  }

  NDArrayBlockedImpl.raw(this.data, NDDescriptor descriptor, this.dataInfo)
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

    return new NDArrayBlockedImpl.raw(
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

            if (row & (descriptor.dataType.blockSize - 1) <
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
  NDArray matMul(value2, {NDArray reuse}) {
    NDArrayBlockedImpl array2 =
        toNDArray(value2, dataType: NDDataType.float32VBlocked);

    var resultDescriptor = descriptor.matMul(array2.descriptor);

    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    _matMulData(this, array2, resultData, resultDescriptor, resultDataInfo);

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArray transpose({List<int> permutationAxis, NDArray reuse}) {
    var resultDescriptor =
        descriptor.transpose(permutationAxis: permutationAxis);

    if (permutationAxis != null &&
        !permutationAxis.every((index) => permutationAxis[index] == index)) {
      var newPermutationAxis = permutationAxis ??
          new List.generate(
              shape.dimension, (index) => shape.dimension - index - 1);

      var resultDataInfo = new DataInfo(resultDescriptor);

      var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

      var matrixUnchanged =
          newPermutationAxis[shape.dimension - 2] == shape.dimension - 2 &&
              newPermutationAxis[shape.dimension - 1] == shape.dimension - 1;

      var matrixTransposed = !matrixUnchanged &&
          newPermutationAxis[shape.dimension - 2] == shape.dimension - 1 &&
          newPermutationAxis[shape.dimension - 1] == shape.dimension - 2;

      if (matrixUnchanged || matrixTransposed) {
        if (matrixUnchanged) {
          _transposeData(this, newPermutationAxis, resultData, resultDescriptor,
              resultDataInfo);
        } else {
          _transposeSwitchedData(this, newPermutationAxis, resultData,
              resultDescriptor, resultDataInfo);
        }

        return new NDArrayBlockedImpl.raw(
            resultData, resultDescriptor, resultDataInfo);
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
  NDArray tile(List<int> multiplies, {NDArray reuse}) {
    var resultDescriptor = descriptor.tile(multiplies);

    if (descriptor != resultDescriptor) {
      var resultDataInfo = new DataInfo(resultDescriptor);

      var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

      _tileData(this, resultData, resultDescriptor, resultDataInfo);

      return new NDArrayBlockedImpl.raw(
          resultData, resultDescriptor, resultDataInfo);
    } else {
      return this;
    }
  }

  @override
  NDArray reduceSum(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceSum(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var total;

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = _ZERO;
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

    var newReductionAxis =
        convertToValidReductionAxis(reductionAxis, shape.dimension);

    var isColumnsReduction = newReductionAxis.contains(shape.dimension - 1);

    var total;
    var singleCount = newReductionAxis.fold<double>(1.0,
        (count, reductionIndex) => count * shape.dimensions[reductionIndex]);

    var count = isColumnsReduction
        ? singleCount
        : new Float32x4(singleCount, singleCount, singleCount, singleCount);

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = _ZERO;
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
        end: () => total / count);
  }

  @override
  NDArray reduceMax(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceMax(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var maxValue;
    int maxValueCount;

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          maxValue = null;
          maxValueCount = null;
        },
        onValue: (value, int valueCount) {
          if (value != null) {
            if (maxValue != null) {
              Float32x4 currentValue = maxValue;
              Float32x4 newValue = currentValue.max(value);
              maxValueCount = max(maxValueCount, valueCount);

              switch (valueCount) {
                case 4:
                  maxValue = newValue;
                  break;
                case 3:
                  maxValue = new Float32x4(
                      newValue.x, newValue.y, newValue.z, currentValue.w);
                  break;
                case 2:
                  maxValue = new Float32x4(
                      newValue.x, newValue.y, currentValue.z, currentValue.w);
                  break;
                case 1:
                  maxValue = new Float32x4(newValue.x, currentValue.y,
                      currentValue.z, currentValue.w);
                  break;
              }
            } else {
              maxValue = value;
              maxValueCount = valueCount;
            }
          } else {
            Float32x4 currentValue = maxValue;

            switch (maxValueCount) {
              case 4:
                maxValue = max(max(currentValue.x, currentValue.y),
                    max(currentValue.w, currentValue.z));
                break;
              case 3:
                maxValue =
                    max(max(currentValue.x, currentValue.y), currentValue.z);
                break;
              case 2:
                maxValue = max(currentValue.x, currentValue.y);
                break;
              case 1:
                maxValue = currentValue.x;
                break;
            }
          }
        },
        end: () => maxValue);
  }

  @override
  NDArray reduceAny(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    descriptor.reduceAny(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    throw new StateError("DEAD CODE");
  }

  @override
  NDArray argMax({int axis = 0, NDArray reuse}) {
    var resultDescriptor = descriptor.argMax(axis: axis);

    var maxValueIndex;
    var maxValue;
    int maxValueCount;

    return argOperationInternal(axis, resultDescriptor, reuse,
        begin: () {
          maxValueIndex = null;
          maxValue = null;
          maxValueCount = null;
        },
        onValue: (dimensionIndex, value, int valueCount) {
          if (value != null) {
            if (maxValue != null) {
              Float32x4 currentValue = maxValue;
              Float32x4 newValue = currentValue.max(value);
              maxValueCount = max(maxValueCount, valueCount);

              switch (valueCount) {
                case 4:
                  maxValue = newValue;
                  var valueNotEqual = newValue.notEqual(currentValue);
                  maxValueIndex = new Int32x4(
                      valueNotEqual.x != 0 ? dimensionIndex.x : maxValueIndex.x,
                      valueNotEqual.y != 0 ? dimensionIndex.y : maxValueIndex.y,
                      valueNotEqual.z != 0 ? dimensionIndex.z : maxValueIndex.z,
                      valueNotEqual.w != 0
                          ? dimensionIndex.w
                          : maxValueIndex.w);
                  break;
                case 3:
                  maxValue = new Float32x4(
                      newValue.x, newValue.y, newValue.z, currentValue.w);
                  maxValueIndex = new Int32x4(
                      newValue.x != currentValue.x
                          ? dimensionIndex.x
                          : maxValueIndex.x,
                      newValue.y != currentValue.y
                          ? dimensionIndex.y
                          : maxValueIndex.y,
                      newValue.z != currentValue.z
                          ? dimensionIndex.z
                          : maxValueIndex.z,
                      maxValueIndex.w);
                  break;
                case 2:
                  maxValue = new Float32x4(
                      newValue.x, newValue.y, newValue.z, currentValue.w);
                  maxValueIndex = new Int32x4(
                      newValue.x != currentValue.x
                          ? dimensionIndex.x
                          : maxValueIndex.x,
                      newValue.y != currentValue.y
                          ? dimensionIndex.y
                          : maxValueIndex.y,
                      maxValueIndex.z,
                      maxValueIndex.w);
                  break;
                case 1:
                  maxValue = new Float32x4(
                      newValue.x, newValue.y, newValue.z, currentValue.w);
                  maxValueIndex = new Int32x4(
                      newValue.x != currentValue.x
                          ? dimensionIndex.x
                          : maxValueIndex.x,
                      maxValueIndex.y,
                      maxValueIndex.z,
                      maxValueIndex.w);
                  break;
              }
            } else {
              maxValueIndex = dimensionIndex;
              maxValue = value;
              maxValueCount = valueCount;
            }
          } else {
            Float32x4 currentValue = maxValue;

            switch (maxValueCount) {
              case 4:
                maxValue = max(max(currentValue.x, currentValue.y),
                    max(currentValue.w, currentValue.z));
                if (maxValue == currentValue.x) {
                  maxValueIndex = maxValueIndex.x;
                } else if (maxValue == currentValue.y) {
                  maxValueIndex = maxValueIndex.y;
                } else if (maxValue == currentValue.z) {
                  maxValueIndex = maxValueIndex.z;
                } else if (maxValue == currentValue.w) {
                  maxValueIndex = maxValueIndex.w;
                }
                break;
              case 3:
                maxValue =
                    max(max(currentValue.x, currentValue.y), currentValue.z);
                if (maxValue == currentValue.x) {
                  maxValueIndex = maxValueIndex.x;
                } else if (maxValue == currentValue.y) {
                  maxValueIndex = maxValueIndex.y;
                } else if (maxValue == currentValue.z) {
                  maxValueIndex = maxValueIndex.z;
                }
                break;
              case 2:
                maxValue = max(currentValue.x, currentValue.y);
                if (maxValue == currentValue.x) {
                  maxValueIndex = maxValueIndex.x;
                } else if (maxValue == currentValue.y) {
                  maxValueIndex = maxValueIndex.y;
                }
                break;
              case 1:
                maxValue = currentValue.x;
                maxValueIndex = maxValueIndex.x;
                break;
            }
          }
        },
        end: () => maxValueIndex);
  }

  @override
  NDArray reshape({List<int> newDimensions, NDArray reuse}) {
    if (_iterableEquality.equals(shape.dimensions, newDimensions)) {
      return this;
    } else {
      var resultDescriptor = descriptor.reshape(newDimensions: newDimensions);

      if (newDimensions[shape.dimension - 1] ==
              shape.dimensions[shape.dimension - 1] &&
          (newDimensions.length == 1 ||
              newDimensions[shape.dimension - 2] ==
                  shape.dimensions[shape.dimension - 2])) {
        var resultDataInfo = new DataInfo(resultDescriptor);

        return new NDArrayBlockedImpl.raw(
            data, resultDescriptor, resultDataInfo);
      } else {
        return elementWiseUnaryOperationInternal(
            resultDescriptor, reuse, (value) => value);
      }
    }
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

      return new NDArrayBlockedImpl.raw(
          resultData, resultDescriptor, resultDataInfo);
    } else {
      return this;
    }
  }

  @override
  NDArray argOperationInternal(
      int axis, NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(),
      void onValue(dimensionIndex, value, int valueCount),
      dynamic end()}) {
    var resultDataInfo = new simple_impl.DataInfo.normalized(resultDescriptor);

    var resultData = simple_impl.createData(resultDescriptor, reuse);

    _argData(this, axis, resultData, resultDescriptor, resultDataInfo,
        begin: begin, onValue: onValue, end: end);

    return new simple_impl.NDArrayImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayBase elementWiseUnaryOperationInternal(
      NDDescriptor resultDescriptor, NDArray reuse, unaryOperation(value)) {
    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    var dimensionIndexes = new List(resultDataInfo.internalShape.dimension - 1);
    var dataIndexes = new List(resultDataInfo.internalShape.dimension - 1);

    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var shapeIndex = 0;
    var dataIndex = 0;

    var valueIterable = _createValueIterable(this);

    var valueIterator = valueIterable.iterator;

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
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current),
                  unaryOperation((valueIterator..moveNext()).current));

              resultData[dataIndex] = value4;

              dataIndex += resultDataInfo.delta1;
            }

            var value4;
            switch (resultDataInfo.lastBlockColumnCount) {
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

            dataIndex += resultDataInfo.delta1;

            if (row & (resultDescriptor.dataType.blockSize - 1) <
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

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayBase elementWiseBinaryOperationInternal(
      NDArrayBase array2,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      binaryOperation(value1, value2)) {
    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    var dimensionIndexes = new List(resultDataInfo.internalShape.dimension - 1);
    var dataIndexes = new List(resultDataInfo.internalShape.dimension - 1);

    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var shapeIndex = 0;
    var dataIndex = 0;

    var valueIterable1 =
        _createBroadCastedValueIterable(this, resultDescriptor, resultDataInfo);
    var valueIterator1 = valueIterable1.iterator;

    var valueIterable2 = _createBroadCastedValueIterable(
        array2, resultDescriptor, resultDataInfo);
    var valueIterator2 = valueIterable2.iterator;

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
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current),
                  binaryOperation((valueIterator1..moveNext()).current,
                      (valueIterator2..moveNext()).current));

              resultData[dataIndex] = value4;

              dataIndex += resultDataInfo.delta1;
            }

            var value4;
            switch (resultDataInfo.lastBlockColumnCount) {
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

            dataIndex += resultDataInfo.delta1;

            if (row & (resultDescriptor.dataType.blockSize - 1) <
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

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayBase elementWiseTernaryOperationInternal(
      NDArrayBase array2,
      NDArrayBase array3,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      ternaryOperation(value1, value2, value3)) {
    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    var dimensionIndexes = new List(resultDataInfo.internalShape.dimension - 1);
    var dataIndexes = new List(resultDataInfo.internalShape.dimension - 1);

    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var shapeIndex = 0;
    var dataIndex = 0;

    var valueIterable1 =
        _createBroadCastedValueIterable(this, resultDescriptor, resultDataInfo);
    var valueIterator1 = valueIterable1.iterator;

    var valueIterable2 = _createBroadCastedValueIterable(
        array2, resultDescriptor, resultDataInfo);
    var valueIterator2 = valueIterable2.iterator;

    var valueIterable3 = _createBroadCastedValueIterable(
        array3, resultDescriptor, resultDataInfo);
    var valueIterator3 = valueIterable2.iterator;

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

              dataIndex += resultDataInfo.delta1;
            }

            var value4;
            switch (resultDataInfo.lastBlockColumnCount) {
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

            dataIndex += resultDataInfo.delta1;

            if (row & (resultDescriptor.dataType.blockSize - 1) <
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

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  Iterable get valueIterable => _createValueIterable(this);

  @override
  String toString() =>
      "<value: ${toValue()}, shape: $shape, dataType: $dataType, stride: ${dataInfo.stride}>";

  void logData() {
    print(data);
  }

  void _transposeData(
      NDArrayBlockedImpl sourceArray,
      List<int> permutationAxis,
      Float32x4List resultData,
      NDDescriptor resultDescriptor,
      DataInfo resultDataInfo) {
    var dimension1s = sourceArray.dataType.isHBlocked
        ? sourceArray.dataInfo.blockRows
        : sourceArray.dataInfo.blockColumns;
    var dimension2s = sourceArray.dataType.isHBlocked
        ? sourceArray.dataInfo.blockColumns
        : sourceArray.dataInfo.blockRows;

    var newPermutationAxis =
        permutationAxis.sublist(0, permutationAxis.length - 2);

    var sourceStride =
        _permute(sourceArray.dataInfo.stride, newPermutationAxis);
    var sourceDimensions =
        _permute(sourceArray.dataInfo.dimensions, newPermutationAxis);

    var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
    var sourceDataIndexes = new List(resultDescriptor.shape.dimension - 1);
    var targetDataIndexes = new List(resultDescriptor.shape.dimension - 1);

    dimensionIndexes[0] = 0;
    sourceDataIndexes[0] = 0;
    targetDataIndexes[0] = 0;

    var shapeIndex = 0;
    var sourceDataIndex = 0;
    var targetDataIndex = 0;

    for (;;) {
      if (dimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
        if (shapeIndex < resultDescriptor.shape.dimension - 2) {
          sourceDataIndex = sourceDataIndexes[shapeIndex];
          targetDataIndex = targetDataIndexes[shapeIndex];

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          sourceDataIndexes[shapeIndex] = sourceDataIndex;
          targetDataIndexes[shapeIndex] = targetDataIndex;

          continue;
        } else {
          for (var dimension1 = 0; dimension1 < dimension1s; dimension1++) {
            for (var dimension2 = 0; dimension2 < dimension2s; dimension2++) {
              resultData[targetDataIndex++] =
                  sourceArray.data[sourceDataIndex++];
              resultData[targetDataIndex++] =
                  sourceArray.data[sourceDataIndex++];
              resultData[targetDataIndex++] =
                  sourceArray.data[sourceDataIndex++];
              resultData[targetDataIndex++] =
                  sourceArray.data[sourceDataIndex++];
            }
          }

          shapeIndex--;
        }
      } else {
        shapeIndex--;
      }

      if (shapeIndex >= 0) {
        dimensionIndexes[shapeIndex]++;
        sourceDataIndexes[shapeIndex] += sourceStride[shapeIndex];
        targetDataIndexes[shapeIndex] += resultDataInfo.stride[shapeIndex];
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];
      } else {
        break;
      }
    }
  }

  void _transposeSwitchedData(
      NDArrayBlockedImpl sourceArray,
      List<int> permutationAxis,
      Float32x4List resultData,
      NDDescriptor resultDescriptor,
      DataInfo resultDataInfo) {
    var dimension1s = sourceArray.dataType.isHBlocked
        ? sourceArray.dataInfo.blockRows
        : sourceArray.dataInfo.blockColumns;
    var dimension2s = sourceArray.dataType.isHBlocked
        ? sourceArray.dataInfo.blockColumns
        : sourceArray.dataInfo.blockRows;

    var delta1 = (dimension1s - 1) << resultDescriptor.dataType.blockDepth;

    var delta2 =
        resultDescriptor.dataType.blockSize - dataInfo.matrixDataLength;

    var newPermutationAxis =
        permutationAxis.sublist(0, permutationAxis.length - 2);

    var sourceStride =
        _permute(sourceArray.dataInfo.stride, newPermutationAxis);
    var sourceDimensions =
        _permute(sourceArray.dataInfo.dimensions, newPermutationAxis);

    var dimensionIndexes = new List(resultDescriptor.shape.dimension - 1);
    var sourceDataIndexes = new List(resultDescriptor.shape.dimension - 1);
    var targetDataIndexes = new List(resultDescriptor.shape.dimension - 1);

    dimensionIndexes[0] = 0;
    sourceDataIndexes[0] = 0;
    targetDataIndexes[0] = 0;

    var shapeIndex = 0;
    var sourceDataIndex = 0;
    var targetDataIndex = 0;

    for (;;) {
      if (dimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
        if (shapeIndex < resultDescriptor.shape.dimension - 2) {
          sourceDataIndex = sourceDataIndexes[shapeIndex];
          targetDataIndex = targetDataIndexes[shapeIndex];

          shapeIndex++;

          dimensionIndexes[shapeIndex] = 0;
          sourceDataIndexes[shapeIndex] = sourceDataIndex;
          targetDataIndexes[shapeIndex] = targetDataIndex;

          continue;
        } else {
          for (var dimension1 = 0; dimension1 < dimension1s; dimension1++) {
            for (var dimension2 = 0; dimension2 < dimension2s; dimension2++) {
              var a0 = sourceArray.data[sourceDataIndex++];
              var a1 = sourceArray.data[sourceDataIndex++];
              var a2 = sourceArray.data[sourceDataIndex++];
              var a3 = sourceArray.data[sourceDataIndex++];

              resultData[targetDataIndex++] =
                  new Float32x4(a0.x, a1.x, a2.x, a3.x);
              resultData[targetDataIndex++] =
                  new Float32x4(a0.y, a1.y, a2.y, a3.y);
              resultData[targetDataIndex++] =
                  new Float32x4(a0.z, a1.z, a2.z, a3.z);
              resultData[targetDataIndex++] =
                  new Float32x4(a0.w, a1.w, a2.w, a3.w);

              targetDataIndex += delta1;
            }

            targetDataIndex += delta2;
          }

          shapeIndex--;
        }
      } else {
        shapeIndex--;
      }

      if (shapeIndex >= 0) {
        dimensionIndexes[shapeIndex]++;
        sourceDataIndexes[shapeIndex] += sourceStride[shapeIndex];
        targetDataIndexes[shapeIndex] += resultDataInfo.stride[shapeIndex];
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];
      } else {
        break;
      }
    }
  }
}

void _tileData(NDArrayBlockedImpl fromArray, List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  var dimensionIndexes = new List(resultDataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var valueIterable =
      _createTiledValueIterable(fromArray, resultDescriptor, resultDataInfo);

  var valueIterator = valueIterable.iterator;

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

          if (row & (resultDescriptor.dataType.blockSize - 1) <
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
    buffer.writeln("internalShape: $internalShape");
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

          if (row & (array.dataType.blockSize - 1) <
              array.dataType.blockSize - 1) {
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

Iterable<num> _createBroadCastedValueIterable(NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  if (sourceArray.dataInfo.internalShape
              .dimensions[sourceArray.dataInfo.internalShape.dimension - 1] ==
          resultDataInfo.internalShape
              .dimensions[resultDataInfo.internalShape.dimension - 1] &&
      sourceArray.dataInfo.internalShape
              .dimensions[sourceArray.dataInfo.internalShape.dimension - 2] ==
          resultDataInfo.internalShape
              .dimensions[resultDataInfo.internalShape.dimension - 2]) {
    return _createHeadBroadCastedValueIterable(
        sourceArray, resultDescriptor, resultDataInfo);
  } else {
    return _createFullBroadCastedValueIterable(
        sourceArray, resultDescriptor, resultDataInfo);
  }
}

Iterable<num> _createHeadBroadCastedValueIterable(
    NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo) sync* {
  // TODO to implement _createHeadBroadCastedValueIterable
  throw new UnimplementedError(
      "to implement _createHeadBroadCastedValueIterable");
}

Iterable<num> _createFullBroadCastedValueIterable(
    NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo) sync* {
  // TODO to implement _createFullBroadCastedValueIterable
  throw new UnimplementedError(
      "to implement _createFullBroadCastedValueIterable");
}

Iterable<num> _createTiledValueIterable(NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  if (sourceArray.dataInfo.internalShape
              .dimensions[sourceArray.dataInfo.internalShape.dimension - 1] ==
          resultDataInfo.internalShape
              .dimensions[resultDataInfo.internalShape.dimension - 1] &&
      sourceArray.dataInfo.internalShape
              .dimensions[sourceArray.dataInfo.internalShape.dimension - 2] ==
          resultDataInfo.internalShape
              .dimensions[resultDataInfo.internalShape.dimension - 2]) {
    return _createHeadTiledValueIterable(
        sourceArray, resultDescriptor, resultDataInfo);
  } else {
    return _createFullTiledValueIterable(
        sourceArray, resultDescriptor, resultDataInfo);
  }
}

Iterable<num> _createHeadTiledValueIterable(NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) sync* {
  var dimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimension - 1);
  var sourceDimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(sourceArray.dataInfo.internalShape.dimension - 1);
  var initialDataIndexes =
      new List(sourceArray.dataInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  sourceDimensionIndexes[0] = 0;
  dataIndexes[0] = 0;
  initialDataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < sourceArray.dataInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;
        initialDataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < sourceArray.dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column <
                  sourceArray.dataInfo.dataColumns -
                      sourceArray.dataType.blockSize;
              column += sourceArray.dataType.blockSize) {
            var value4 = sourceArray.data[dataIndex];

            yield value4.x;
            yield value4.y;
            yield value4.z;
            yield value4.w;

            dataIndex += sourceArray.dataInfo.delta1;
          }

          var value4 = sourceArray.data[dataIndex];

          switch (sourceArray.dataInfo.lastBlockColumnCount) {
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

          dataIndex += sourceArray.dataInfo.delta1;

          if (row & (sourceArray.dataType.blockSize - 1) <
              sourceArray.dataType.blockSize - 1) {
            dataIndex += sourceArray.dataInfo.delta2;
          } else {
            dataIndex += sourceArray.dataInfo.delta3;
          }
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      if (sourceDimensionIndexes[shapeIndex] <
          sourceArray.shape[shapeIndex] - 1) {
        sourceDimensionIndexes[shapeIndex]++;
        dataIndexes[shapeIndex] += sourceArray.dataInfo.stride[shapeIndex];
      } else {
        sourceDimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = initialDataIndexes[shapeIndex];
      }

      dataIndex = dataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

Iterable<num> _createFullTiledValueIterable(NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) sync* {
  var dimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimension - 1);
  var sourceDimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimension - 1);
  var dataIndexes = new List(sourceArray.dataInfo.internalShape.dimension - 1);
  var initialDataIndexes =
      new List(sourceArray.dataInfo.internalShape.dimension - 1);

  dimensionIndexes[0] = 0;
  sourceDimensionIndexes[0] = 0;
  dataIndexes[0] = 0;
  initialDataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var rowsMultiplier = resultDataInfo
          .internalShape[resultDataInfo.internalShape.dimension - 2] ~/
      sourceArray.dataInfo.rows;
  var columnsMultiplier = resultDataInfo
          .internalShape[resultDataInfo.internalShape.dimension - 1] ~/
      sourceArray.dataInfo.columns;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < sourceArray.dataInfo.internalShape.dimension - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;
        initialDataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        var initialRowDataIndex = dataIndex;

        for (var rowIteration = 0;
            rowIteration < rowsMultiplier;
            rowIteration++) {
          for (var row = 0; row < sourceArray.dataInfo.rows; row++) {
            var initialColumnDataIndex = dataIndex;

            var lastColumnDataIndex = dataIndex;
            for (var columnIteration = 0;
                columnIteration < columnsMultiplier;
                columnIteration++) {
              var column;
              for (column = 0;
                  column <
                      sourceArray.dataInfo.dataColumns -
                          sourceArray.dataType.blockSize;
                  column += sourceArray.dataType.blockSize) {
                var value4 = sourceArray.data[dataIndex];

                yield value4.x;
                yield value4.y;
                yield value4.z;
                yield value4.w;

                dataIndex += sourceArray.dataInfo.delta1;
              }

              var value4 = sourceArray.data[dataIndex];

              switch (sourceArray.dataInfo.lastBlockColumnCount) {
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

              dataIndex += sourceArray.dataInfo.delta1;

              if (row & (sourceArray.dataType.blockSize - 1) <
                  sourceArray.dataType.blockSize - 1) {
                dataIndex += sourceArray.dataInfo.delta2;
              } else {
                dataIndex += sourceArray.dataInfo.delta3;
              }

              lastColumnDataIndex = dataIndex;

              dataIndex = initialColumnDataIndex;
            }

            dataIndex = lastColumnDataIndex;
          }

          dataIndex = initialRowDataIndex;
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      if (sourceDimensionIndexes[shapeIndex] <
          sourceArray.shape[shapeIndex] - 1) {
        sourceDimensionIndexes[shapeIndex]++;
        dataIndexes[shapeIndex] += sourceArray.dataInfo.stride[shapeIndex];
      } else {
        sourceDimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = initialDataIndexes[shapeIndex];
      }

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

          if (row & (descriptor.dataType.blockSize - 1) <
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

          if (row & (descriptor.dataType.blockSize - 1) <
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

          if (row & (resultDescriptor.dataType.blockSize - 1) <
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

          if (row & (resultDescriptor.dataType.blockSize - 1) <
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

            var result0 = _ZERO;
            var result1 = _ZERO;
            var result2 = _ZERO;
            var result3 = _ZERO;

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
  var axis = new Set<int>.from(reductionAxis);

  bool isRowsReduction =
      axis.contains(sourceArray.descriptor.shape.dimension - 2);

  bool isColumnsReduction =
      axis.contains(sourceArray.descriptor.shape.dimension - 1);

  if (sourceArray.dataType.isHBlocked) {
    if (sourceArray.shape.dimension == 1) {
      axis = new Set.from([1]);
    } else if (isRowsReduction) {
      axis.add(sourceArray.descriptor.shape.dimension);
    }
  } else {
    if (sourceArray.shape.dimension == 1) {
      axis = new Set.from([0]);
    } else if (isRowsReduction && isColumnsReduction) {
      axis.remove(sourceArray.descriptor.shape.dimension - 2);
      axis.remove(sourceArray.descriptor.shape.dimension - 1);
      axis.add(sourceArray.descriptor.shape.dimension - 1);
      axis.add(sourceArray.descriptor.shape.dimension);
      axis.add(sourceArray.descriptor.shape.dimension - 2);
    } else if (isRowsReduction) {
      axis.remove(sourceArray.descriptor.shape.dimension - 2);
      axis.add(sourceArray.descriptor.shape.dimension - 1);
      axis.add(sourceArray.descriptor.shape.dimension);
    } else if (isColumnsReduction) {
      axis.remove(sourceArray.descriptor.shape.dimension - 1);
      axis.add(sourceArray.descriptor.shape.dimension - 2);
    }
  }

  var newReductionAxis = axis.toList(growable: false);

  List<int> targetPermutedIndexes =
      new List.generate(targetDataInfo.dimensions.length, (index) => index);
  if (targetDescriptor.shape.dimension > 1) {
    if (sourceArray.dataType.isHBlocked) {
      if (isRowsReduction || isColumnsReduction) {
        var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
        targetPermutedIndexes[targetPermutedIndexes.length - 1] =
            targetPermutedIndexes[targetPermutedIndexes.length - 2];
        targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
        tempIndex =
            targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 3);
        targetPermutedIndexes.add(tempIndex);
      }
    } else {
      if (isRowsReduction || isColumnsReduction) {
        var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
        targetPermutedIndexes[targetPermutedIndexes.length - 1] =
            targetPermutedIndexes[targetPermutedIndexes.length - 3];
        targetPermutedIndexes[targetPermutedIndexes.length - 3] = tempIndex;
        tempIndex =
            targetPermutedIndexes.removeAt(targetPermutedIndexes.length - 2);
        targetPermutedIndexes.add(tempIndex);
      }
    }
  } else if (targetDescriptor.shape.dimension == 1) {
    if (sourceArray.dataType.isHBlocked) {
      if (isRowsReduction || isColumnsReduction) {
        targetPermutedIndexes = [1, 0, 2];
      }
    } else {
      if (isRowsReduction || isColumnsReduction) {
        targetPermutedIndexes = [0, 1, 2];
      }
    }
  }

  var sourcePermutedIndexes = new List.generate(
      sourceArray.dataInfo.dimensions.length, (index) => index);

  sourcePermutedIndexes =
      sourcePermutedIndexes.where((index) => !axis.contains(index)).toList();

  sourcePermutedIndexes.addAll(newReductionAxis);

  var sourceTargetRowIndex;
  var delta1;
  var sourceTargetColumnIndex;
  var delta2;

  if (targetDescriptor.shape.dimension > 1) {
    if (sourceArray.dataType.isHBlocked) {
      if (isRowsReduction && isColumnsReduction) {
        sourceTargetRowIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            2;
        delta1 =
            targetDataInfo.dataColumns - targetDescriptor.dataType.blockSize;

        sourceTargetColumnIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            1;
        delta2 = targetDescriptor.dataType.blockSize;
      } else if (isRowsReduction) {
        sourceTargetRowIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            2;
        delta1 =
            targetDataInfo.dataColumns - targetDescriptor.dataType.blockSize;
      } else if (isColumnsReduction) {
        sourceTargetRowIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            3;
        delta1 =
            targetDataInfo.dataColumns - targetDescriptor.dataType.blockSize;
      }
    } else {
      if (isRowsReduction && isColumnsReduction) {
        sourceTargetRowIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            2;
        delta1 = 0;

        sourceTargetColumnIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            1;
        delta2 = targetDataInfo.dataRows;
      } else if (isRowsReduction) {
        sourceTargetRowIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            2;
        delta1 = 0;
      } else if (isColumnsReduction) {
        sourceTargetRowIndex = sourceArray.dataInfo.dimensions.length -
            newReductionAxis.length -
            3;
        delta1 = 0;
      }
    }
  } else if (targetDescriptor.shape.dimension == 1) {
    if (isRowsReduction && isColumnsReduction) {
      sourceTargetColumnIndex =
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1;
      delta2 = targetDataInfo.dataRows;
    }
  }

  var targetBeginIndex;
  if (targetDescriptor.shape.dimension >= 1) {
    targetBeginIndex =
        sourceArray.dataInfo.dimensions.length - newReductionAxis.length;
  } else {
    targetBeginIndex =
        sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1;
  }

  var targetLimitIndex =
      isColumnsReduction ? targetBeginIndex - 1 : targetBeginIndex;

  var lastTargetColumnIndex;
  if (targetDescriptor.shape.dimension > 1) {
    if (isRowsReduction && isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1];
    } else if (isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 2];
    }
  } else if (targetDescriptor.shape.dimension == 1) {
    if (isRowsReduction && isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1];
    } else if (isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 2];
    }
  } else {
    if (isRowsReduction && isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 0];
    } else if (isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 2];
    }
  }

  var permutedLastTargetColumnIndex = lastTargetColumnIndex != null
      ? sourcePermutedIndexes.indexOf(lastTargetColumnIndex)
      : null;

  var lastSourceIndex = sourceArray.dataInfo.dimensions.length - 1;
  var sourceRowIndex = sourceArray.dataInfo.rowIndex;
  var sourceColumnIndex = sourceArray.dataInfo.columnIndex;
  var targetColumnIndex = targetDataInfo.columnIndex;

  var permutedLastSourceIndex = sourcePermutedIndexes.indexOf(lastSourceIndex);
  var permutedSourceRowIndex = sourcePermutedIndexes.indexOf(sourceRowIndex);
  var permutedSourceColumnIndex =
      sourcePermutedIndexes.indexOf(sourceColumnIndex);
  var permutedTargetColumnIndex =
      targetPermutedIndexes.indexOf(targetColumnIndex);

  var sourceStride =
      _permute(sourceArray.dataInfo.stride, sourcePermutedIndexes);
  var targetStride = _permute(targetDataInfo.stride, targetPermutedIndexes);
  var sourceDimensions =
      _permute(sourceArray.dataInfo.dimensions, sourcePermutedIndexes);
  var targetDimensions =
      _permute(targetDataInfo.dimensions, targetPermutedIndexes);

  var sourceDimensionIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var sourceDataIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var targetDataIndexes = new List(targetDataInfo.dimensions.length);

  sourceDimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  var lastRow =
      sourceArray.dataInfo.dimensions[sourceArray.dataInfo.rowIndex] == 1;
  var lastColumn =
      sourceArray.dataInfo.dimensions[sourceArray.dataInfo.columnIndex] == 1;
  var lastTargetColumn = lastTargetColumnIndex != null &&
      targetDataInfo.dimensions[targetDataInfo.columnIndex] == 1;

  Float32x4 columnValue;
  int columnIndex;

  if (isColumnsReduction) {
    columnValue = _ZERO;
    columnIndex = 0;
  }

  if (targetDescriptor.shape.dimension == 0 &&
      sourceArray.shape.dimension > 1) {
    begin();
  }

  for (;;) {
    var maxDimension;
    if (shapeIndex == permutedLastSourceIndex) {
      if (lastRow) {
        maxDimension = sourceArray.dataInfo.lastBlockRowCount;
      } else {
        maxDimension = sourceArray.dataType.blockSize;
      }
    } else {
      maxDimension = sourceDimensions[shapeIndex];
    }

    if (sourceDimensionIndexes[shapeIndex] < maxDimension) {
      if (shapeIndex < sourceArray.dataInfo.dimensions.length - 1) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];

        if (shapeIndex < targetBeginIndex) {
          targetDataIndex = targetDataIndexes[shapeIndex];
        }

        shapeIndex++;

        sourceDimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;

        if (shapeIndex < targetBeginIndex) {
          targetDataIndexes[shapeIndex] = targetDataIndex;

          if (isColumnsReduction &&
              targetDataInfo.dimensions[targetDataInfo.columnIndex] > 1) {
            if (targetPermutedIndexes[shapeIndex] ==
                targetDataInfo.columnIndex) {
              lastTargetColumn = false;
            }
          }
        }

        if (shapeIndex == targetBeginIndex) {
          begin();
        }

        continue;
      } else {
        onValue(
            sourceArray.data[sourceDataIndex],
            lastColumn
                ? sourceArray.dataInfo.lastBlockColumnCount
                : sourceArray.dataType.blockSize);
      }
    } else {
      shapeIndex--;

      if (shapeIndex ==
          sourceArray.dataInfo.dimensions.length -
              newReductionAxis.length -
              1) {
        if (isColumnsReduction) {
          onValue(null, null);

          var reducedValue = end();

          switch (columnIndex) {
            case 0:
              columnValue = columnValue.withX(reducedValue);
              columnIndex++;
              break;
            case 1:
              columnValue = columnValue.withY(reducedValue);
              columnIndex++;
              break;
            case 2:
              columnValue = columnValue.withZ(reducedValue);
              columnIndex++;
              break;
            case 3:
              columnValue = columnValue.withW(reducedValue);
              columnIndex++;
              break;
          }

          if (columnIndex ==
              (lastTargetColumn
                  ? targetDataInfo.lastBlockColumnCount
                  : targetDescriptor.dataType.blockSize)) {
            targetData[targetDataIndex] = columnValue;

            columnValue = _ZERO;
            columnIndex = 0;
          }
        } else {
          var reducedValue = end();

          targetData[targetDataIndex] = reducedValue;
        }
      }
    }

    if (shapeIndex >= 0) {
      sourceDimensionIndexes[shapeIndex]++;
      sourceDataIndexes[shapeIndex] += sourceStride[shapeIndex];
      sourceDataIndex = sourceDataIndexes[shapeIndex];

      if (shapeIndex < targetBeginIndex) {
        if (shapeIndex < targetLimitIndex) {
          targetDataIndexes[shapeIndex] += targetStride[shapeIndex];
        }

        if (shapeIndex == sourceTargetRowIndex) {
          var sourceDimensionIndex = sourceDimensionIndexes[shapeIndex];
          if (sourceDimensionIndex & (sourceArray.dataType.blockSize - 1) ==
              0) {
            targetDataIndexes[shapeIndex] += delta1;
          }
        } else if (shapeIndex == sourceTargetColumnIndex) {
          var sourceDimensionIndex = sourceDimensionIndexes[shapeIndex];
          if (sourceDimensionIndex & (sourceArray.dataType.blockSize - 1) ==
              0) {
            targetDataIndexes[shapeIndex] += delta2;
          }
        }

        targetDataIndex = targetDataIndexes[shapeIndex];

        if (isColumnsReduction &&
            targetDataInfo.dimensions[targetDataInfo.columnIndex] > 1) {
          if (shapeIndex == permutedTargetColumnIndex) {
            if (lastTargetColumnIndex == sourceArray.dataInfo.columnIndex) {
              lastTargetColumn =
                  (sourceDimensionIndexes[permutedLastTargetColumnIndex] ==
                      targetDimensions[shapeIndex] - 1);
            } else if (lastTargetColumnIndex == sourceArray.dataInfo.rowIndex) {
              lastTargetColumn =
                  (sourceDimensionIndexes[permutedLastTargetColumnIndex] ==
                      targetDimensions[shapeIndex] - 1);
            } else {
              lastTargetColumn =
                  (sourceDimensionIndexes[permutedLastTargetColumnIndex] >>
                          sourceArray.dataType.blockDepth ==
                      targetDimensions[shapeIndex] - 1);
            }
          }
        }
      }

      if (sourceArray.dataInfo.dimensions[sourceArray.dataInfo.rowIndex] > 1) {
        if (shapeIndex == permutedSourceRowIndex) {
          lastRow = (sourceDimensionIndexes[shapeIndex] ==
              sourceDimensions[shapeIndex] - 1);
        }
      }
      if (sourceArray.dataInfo.dimensions[sourceArray.dataInfo.columnIndex] >
          1) {
        if (shapeIndex == permutedSourceColumnIndex) {
          lastColumn = (sourceDimensionIndexes[shapeIndex] ==
              sourceDimensions[shapeIndex] - 1);
        }
      }
    } else {
      break;
    }
  }
}

void _argData(NDArrayBlockedImpl sourceArray, int axis, Uint32List targetData,
    NDDescriptor targetDescriptor, simple_impl.DataInfo targetDataInfo,
    {void begin(),
    void onValue(dimensionIndex, value, int valueCount),
    dynamic end()}) {
  var axisSet = new Set<int>.from([axis]);

  bool isRowsReduction =
      axisSet.contains(sourceArray.descriptor.shape.dimension - 2);

  bool isColumnsReduction =
      axisSet.contains(sourceArray.descriptor.shape.dimension - 1);

  if (sourceArray.dataType.isHBlocked) {
    if (sourceArray.shape.dimension == 1) {
      axisSet = new Set.from([1]);
    } else if (isRowsReduction) {
      axisSet.add(sourceArray.descriptor.shape.dimension);
    }
  } else {
    if (sourceArray.shape.dimension == 1) {
      axisSet = new Set.from([0]);
    } else if (isRowsReduction && isColumnsReduction) {
      axisSet.remove(sourceArray.descriptor.shape.dimension - 2);
      axisSet.remove(sourceArray.descriptor.shape.dimension - 1);
      axisSet.add(sourceArray.descriptor.shape.dimension - 1);
      axisSet.add(sourceArray.descriptor.shape.dimension);
      axisSet.add(sourceArray.descriptor.shape.dimension - 2);
    } else if (isRowsReduction) {
      axisSet.remove(sourceArray.descriptor.shape.dimension - 2);
      axisSet.add(sourceArray.descriptor.shape.dimension - 1);
      axisSet.add(sourceArray.descriptor.shape.dimension);
    } else if (isColumnsReduction) {
      axisSet.remove(sourceArray.descriptor.shape.dimension - 1);
      axisSet.add(sourceArray.descriptor.shape.dimension - 2);
    }
  }

  var newReductionAxis = axisSet.toList(growable: false);

  var sourcePermutedIndexes = new List.generate(
      sourceArray.dataInfo.dimensions.length, (index) => index);

  sourcePermutedIndexes =
      sourcePermutedIndexes.where((index) => !axisSet.contains(index)).toList();

  if (!isRowsReduction && !isColumnsReduction) {
    if (sourceArray.dataType.isHBlocked) {
      var tempIndex = sourcePermutedIndexes[sourcePermutedIndexes.length - 1];
      sourcePermutedIndexes[sourcePermutedIndexes.length - 1] =
          sourcePermutedIndexes[sourcePermutedIndexes.length - 2];
      sourcePermutedIndexes[sourcePermutedIndexes.length - 2] = tempIndex;
    } else {
      var tempIndex =
          sourcePermutedIndexes.removeAt(sourcePermutedIndexes.length - 3);
      sourcePermutedIndexes.add(tempIndex);
    }
  }

  sourcePermutedIndexes.addAll(newReductionAxis);

  var targetBeginIndex;
  if (targetDescriptor.shape.dimension >= 1) {
    targetBeginIndex =
        sourceArray.dataInfo.dimensions.length - newReductionAxis.length;
  } else {
    targetBeginIndex =
        sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1;
  }

  var lastSourceIndex = sourceArray.dataInfo.dimensions.length - 1;
  var sourceRowIndex = sourceArray.dataInfo.rowIndex;
  var sourceColumnIndex = sourceArray.dataInfo.columnIndex;

  var permutedLastSourceIndex = sourcePermutedIndexes.indexOf(lastSourceIndex);
  var permutedSourceRowIndex = sourcePermutedIndexes.indexOf(sourceRowIndex);
  var permutedSourceColumnIndex =
      sourcePermutedIndexes.indexOf(sourceColumnIndex);

  var sourceStride =
      _permute(sourceArray.dataInfo.stride, sourcePermutedIndexes);
  var sourceDimensions =
      _permute(sourceArray.dataInfo.dimensions, sourcePermutedIndexes);

  var sourceDimensionIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var sourceDataIndexes = new List(sourceArray.dataInfo.dimensions.length);

  sourceDimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  var lastRow =
      sourceArray.dataInfo.dimensions[sourceArray.dataInfo.rowIndex] == 1;
  var lastColumn =
      sourceArray.dataInfo.dimensions[sourceArray.dataInfo.columnIndex] == 1;

  var sourceDimensionIndex;

  if (targetDescriptor.shape.dimension == 0 &&
      sourceArray.shape.dimension > 1) {
    begin();

    sourceDimensionIndex = 0;
  }

  for (;;) {
    var maxDimension;
    if (shapeIndex == permutedLastSourceIndex) {
      if (lastRow) {
        maxDimension = sourceArray.dataInfo.lastBlockRowCount;
      } else {
        maxDimension = sourceArray.dataType.blockSize;
      }
    } else {
      maxDimension = sourceDimensions[shapeIndex];
    }

    if (sourceDimensionIndexes[shapeIndex] < maxDimension) {
      if (shapeIndex < sourceArray.dataInfo.dimensions.length - 1) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];

        shapeIndex++;

        sourceDimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;

        if (shapeIndex == targetBeginIndex) {
          begin();

          sourceDimensionIndex = 0;
        }

        continue;
      } else {
        var valueCount = lastColumn
            ? sourceArray.dataInfo.lastBlockColumnCount
            : sourceArray.dataType.blockSize;

        var dimensionValue;
        if (isColumnsReduction) {
          switch (valueCount) {
            case 4:
              dimensionValue = new Int32x4(
                  sourceDimensionIndex++,
                  sourceDimensionIndex++,
                  sourceDimensionIndex++,
                  sourceDimensionIndex++);
              break;
            case 3:
              dimensionValue = new Int32x4(sourceDimensionIndex++,
                  sourceDimensionIndex++, sourceDimensionIndex++, 0);
              sourceDimensionIndex++;
              break;
            case 2:
              dimensionValue = new Int32x4(
                  sourceDimensionIndex++, sourceDimensionIndex++, 0, 0);
              sourceDimensionIndex++;
              sourceDimensionIndex++;
              break;
            case 1:
              dimensionValue = new Int32x4(sourceDimensionIndex++, 0, 0, 0);
              sourceDimensionIndex++;
              sourceDimensionIndex++;
              sourceDimensionIndex++;
              break;
          }
        } else {
          switch (valueCount) {
            case 4:
              dimensionValue = new Int32x4(
                  sourceDimensionIndex,
                  sourceDimensionIndex,
                  sourceDimensionIndex,
                  sourceDimensionIndex);
              break;
            case 3:
              dimensionValue = new Int32x4(sourceDimensionIndex,
                  sourceDimensionIndex, sourceDimensionIndex, 0);
              break;
            case 2:
              dimensionValue =
                  new Int32x4(sourceDimensionIndex, sourceDimensionIndex, 0, 0);
              break;
            case 1:
              dimensionValue = new Int32x4(sourceDimensionIndex, 0, 0, 0);
              break;
          }

          sourceDimensionIndex++;
        }

        onValue(dimensionValue, sourceArray.data[sourceDataIndex], valueCount);
      }
    } else {
      shapeIndex--;

      if (shapeIndex ==
          sourceArray.dataInfo.dimensions.length -
              newReductionAxis.length -
              1) {
        if (isColumnsReduction) {
          onValue(null, null, null);

          var reducedValue = end();

          targetData[targetDataIndex++] = reducedValue;
        } else {
          var reducedValue = end();

          var valueCount = lastColumn
              ? sourceArray.dataInfo.lastBlockColumnCount
              : sourceArray.dataType.blockSize;

          switch (valueCount) {
            case 4:
              targetData[targetDataIndex++] = reducedValue.x;
              targetData[targetDataIndex++] = reducedValue.y;
              targetData[targetDataIndex++] = reducedValue.z;
              targetData[targetDataIndex++] = reducedValue.w;
              break;
            case 3:
              targetData[targetDataIndex++] = reducedValue.x;
              targetData[targetDataIndex++] = reducedValue.y;
              targetData[targetDataIndex++] = reducedValue.z;
              break;
            case 2:
              targetData[targetDataIndex++] = reducedValue.x;
              targetData[targetDataIndex++] = reducedValue.y;
              break;
            case 1:
              targetData[targetDataIndex++] = reducedValue.x;
              break;
          }
        }
      }
    }

    if (shapeIndex >= 0) {
      sourceDimensionIndexes[shapeIndex]++;
      sourceDataIndexes[shapeIndex] += sourceStride[shapeIndex];
      sourceDataIndex = sourceDataIndexes[shapeIndex];

      if (sourceArray.dataInfo.dimensions[sourceArray.dataInfo.rowIndex] > 1) {
        if (shapeIndex == permutedSourceRowIndex) {
          lastRow = (sourceDimensionIndexes[shapeIndex] ==
              sourceDimensions[shapeIndex] - 1);
        }
      }
      if (sourceArray.dataInfo.dimensions[sourceArray.dataInfo.columnIndex] >
          1) {
        if (shapeIndex == permutedSourceColumnIndex) {
          lastColumn = (sourceDimensionIndexes[shapeIndex] ==
              sourceDimensions[shapeIndex] - 1);
        }
      }
    } else {
      break;
    }
  }
}

List<int> _permute(List<int> list, List<int> permutedIndexes) {
  var newList = permutedIndexes.map((index) => list[index]).toList();
  for (var index = newList.length; index < list.length; index++) {
    newList.add(list[index]);
  }
  return newList;
}
