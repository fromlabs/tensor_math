// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'dart:math' as math;
import "dart:typed_data";

import "package:collection/collection.dart";

import 'nd_descriptor.dart';
import 'nd_shape.dart';
import 'nd_data_type.dart';
import "nd_array.dart";

import "nd_array_base.dart";

import "nd_util.dart";

final _zeroFloat = new Float32x4.zero();
final _zeroInt = new Int32x4(0, 0, 0, 0);

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
    if (fillValue == 0.0 || fillValue == 0 || fillValue == false) {
      var dataInfo = new DataInfo(descriptor);

      var data = _createData(descriptor, dataInfo, reuse);

      if (reuse != null && reuse.descriptor == descriptor) {
        reuse.data.fillRange(0, reuse.data.length,
            descriptor.dataType.isFloat ? _zeroFloat : _zeroInt);
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

  @override
  dynamic toValue() {
    if (dataType.isBoolean) {
      return _toValueBoolean();
    } else {
      return _toValueNum();
    }
  }

  dynamic _toValueNum() {
    var value = new List(dataInfo.internalShape[0]);

    var values = new List(dataInfo.internalShape.dimensionCount - 1);
    var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
    var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

    values[0] = value;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var shapeIndex = 0;
    var dataIndex = 0;

    var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

    for (;;) {
      if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
        if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
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
                column < lastColumnIndex;
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

    return descriptor.shape.dimensionCount > 1
        ? value
        : (descriptor.shape.dimensionCount == 1 ? value[0] : value[0][0]);
  }

  dynamic _toValueBoolean() {
    var value = new List(dataInfo.internalShape[0]);

    var values = new List(dataInfo.internalShape.dimensionCount - 1);
    var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
    var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

    values[0] = value;
    dimensionIndexes[0] = 0;
    dataIndexes[0] = 0;

    var shapeIndex = 0;
    var dataIndex = 0;

    var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

    for (;;) {
      if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
        if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
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
                column < lastColumnIndex;
                column += descriptor.dataType.blockSize) {
              var value4 = data[dataIndex];

              rowValues[column] = value4.flagX;
              rowValues[column + 1] = value4.flagY;
              rowValues[column + 2] = value4.flagZ;
              rowValues[column + 3] = value4.flagW;

              dataIndex += dataInfo.delta1;
            }

            var value4 = data[dataIndex];

            switch (dataInfo.lastBlockColumnCount) {
              case 4:
                rowValues[column] = value4.flagX;
                rowValues[column + 1] = value4.flagY;
                rowValues[column + 2] = value4.flagZ;
                rowValues[column + 3] = value4.flagW;

                break;
              case 3:
                rowValues[column] = value4.flagX;
                rowValues[column + 1] = value4.flagY;
                rowValues[column + 2] = value4.flagZ;

                break;
              case 2:
                rowValues[column] = value4.flagX;
                rowValues[column + 1] = value4.flagY;

                break;
              case 1:
                rowValues[column] = value4.flagX;

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

    return descriptor.shape.dimensionCount > 1
        ? value
        : (descriptor.shape.dimensionCount == 1 ? value[0] : value[0][0]);
  }

  @override
  NDArray matMul(value2, {NDArray reuse}) {
    NDArrayBlockedImpl array2 =
        toNDArray(value2, dataType: NDDataType.float32Blocked);
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
          new List.generate(shape.dimensionCount,
              (index) => shape.dimensionCount - index - 1);

      var resultDataInfo = new DataInfo(resultDescriptor);

      var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

      var matrixUnchanged = newPermutationAxis[shape.dimensionCount - 2] ==
              shape.dimensionCount - 2 &&
          newPermutationAxis[shape.dimensionCount - 1] ==
              shape.dimensionCount - 1;

      var matrixTransposed = !matrixUnchanged &&
          newPermutationAxis[shape.dimensionCount - 2] ==
              shape.dimensionCount - 1 &&
          newPermutationAxis[shape.dimensionCount - 1] ==
              shape.dimensionCount - 2;

      // TODO generalizzare matrixUnchanged nel caso in cui le colonne non sono cambiate

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
        var simpleDataType;
        if (dataType.isFloat) {
          simpleDataType = NDDataType.float32;
        } else if (dataType.isInteger) {
          simpleDataType = NDDataType.int32;
        } else if (dataType.isBoolean) {
          simpleDataType = NDDataType.boolean;
        }

        // TODO ottimizzare
        return cast(simpleDataType)
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

      if (multiplies[multiplies.length - 1] == 1) {
        _tileDataBlocked(this, resultData, resultDescriptor, resultDataInfo);
      } else {
        _tileData(this, resultData, resultDescriptor, resultDataInfo);
      }

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
      reductionAxis: reductionAxis,
      keepDimensions: false, // TODO implementare keepDimensions
    );

    final zero = resultDescriptor.dataType.isFloat ? _zeroFloat : _zeroInt;

    var total;

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = zero;
        },
        onValue: (value, int valueCount) {
          if (value != null) {
            total += value;
          } else {
            var currentValue = total;
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
      reductionAxis: reductionAxis,
      keepDimensions: false, // TODO implementare keepDimensions
    );

    var newReductionAxis =
        convertToValidReductionAxis(reductionAxis, shape.dimensionCount);

    var isColumnsReduction =
        newReductionAxis.contains(shape.dimensionCount - 1);

    var total;
    var singleCount = newReductionAxis.fold<double>(1.0,
        (count, reductionIndex) => count * shape.dimensions[reductionIndex]);

    var count = isColumnsReduction
        ? singleCount
        : new Float32x4(singleCount, singleCount, singleCount, singleCount);

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = _zeroFloat;
        },
        onValue: (value, int valueCount) {
          if (value != null) {
            total += value;
          } else {
            var currentValue = total;
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
      reductionAxis: reductionAxis,
      keepDimensions: false, // TODO implementare keepDimensions
    );

    var maxValue;
    int maxValueCount;

    if (resultDescriptor.dataType.isFloat) {
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
                maxValueCount = math.max(maxValueCount, valueCount);

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
                  maxValue = math.max(math.max(currentValue.x, currentValue.y),
                      math.max(currentValue.w, currentValue.z));
                  break;
                case 3:
                  maxValue = math.max(
                      math.max(currentValue.x, currentValue.y), currentValue.z);
                  break;
                case 2:
                  maxValue = math.max(currentValue.x, currentValue.y);
                  break;
                case 1:
                  maxValue = currentValue.x;
                  break;
              }
            }
          },
          end: () => maxValue);
    } else {
      return reduceOperationInternal(
          reductionAxis, keepDimensions, resultDescriptor, reuse,
          begin: () {
            maxValue = null;
            maxValueCount = null;
          },
          onValue: (value, int valueCount) {
            if (value != null) {
              if (maxValue != null) {
                Int32x4 currentValue = maxValue;
                maxValueCount = math.max(maxValueCount, valueCount);

                switch (valueCount) {
                  case 4:
                    maxValue = new Int32x4(
                        math.max(value.x, currentValue.x),
                        math.max(value.y, currentValue.y),
                        math.max(value.z, currentValue.z),
                        math.max(value.w, currentValue.w));
                    break;
                  case 3:
                    maxValue = new Int32x4(
                        math.max(value.x, currentValue.x),
                        math.max(value.y, currentValue.y),
                        math.max(value.z, currentValue.z),
                        currentValue.w);
                    break;
                  case 2:
                    maxValue = new Int32x4(
                        math.max(value.x, currentValue.x),
                        math.max(value.y, currentValue.y),
                        currentValue.z,
                        currentValue.w);
                    break;
                  case 1:
                    maxValue = new Int32x4(math.max(value.x, currentValue.x),
                        currentValue.y, currentValue.z, currentValue.w);
                    break;
                }
              } else {
                maxValue = value;
                maxValueCount = valueCount;
              }
            } else {
              Int32x4 currentValue = maxValue;

              switch (maxValueCount) {
                case 4:
                  maxValue = math.max(math.max(currentValue.x, currentValue.y),
                      math.max(currentValue.w, currentValue.z));
                  break;
                case 3:
                  maxValue = math.max(
                      math.max(currentValue.x, currentValue.y), currentValue.z);
                  break;
                case 2:
                  maxValue = math.max(currentValue.x, currentValue.y);
                  break;
                case 1:
                  maxValue = currentValue.x;
                  break;
              }
            }
          },
          end: () => maxValue);
    }
  }

  @override
  NDArray reduceAny(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceAny(
      reductionAxis: reductionAxis,
      keepDimensions: false, // TODO implementare keepDimensions
    );

    var total;

    return reduceOperationInternal(
        reductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = _zeroInt;
        },
        onValue: (value, int valueCount) {
          if (value != null) {
            var newValue = total | value;

            switch (valueCount) {
              case 4:
                total = newValue;
                break;
              case 3:
                total = new Int32x4.bool(
                    newValue.flagX, newValue.flagY, newValue.flagZ, false);
                break;
              case 2:
                total = new Int32x4.bool(
                    newValue.flagX, newValue.flagY, false, false);
                break;
              case 1:
                total = new Int32x4.bool(newValue.flagX, false, false, false);
            }
          } else {
            var currentValue = total;
            total = currentValue.flagX ||
                currentValue.flagY ||
                currentValue.flagZ ||
                currentValue.flagW;
          }
        },
        end: () => total);
  }

  @override
  NDArray argMax({int axis = 0, NDArray reuse}) {
    var resultDescriptor = descriptor.argMax(axis: axis);

    var maxValueIndex;
    var maxValue;
    int maxValueCount;

    if (descriptor.dataType.isFloat) {
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
                maxValueCount = math.max(maxValueCount, valueCount);

                switch (valueCount) {
                  case 4:
                    maxValue = newValue;
                    var valueNotEqual = newValue.notEqual(currentValue);
                    maxValueIndex = new Int32x4(
                        valueNotEqual.x != 0
                            ? dimensionIndex.x
                            : maxValueIndex.x,
                        valueNotEqual.y != 0
                            ? dimensionIndex.y
                            : maxValueIndex.y,
                        valueNotEqual.z != 0
                            ? dimensionIndex.z
                            : maxValueIndex.z,
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
                  maxValue = math.max(math.max(currentValue.x, currentValue.y),
                      math.max(currentValue.w, currentValue.z));
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
                  maxValue = math.max(
                      math.max(currentValue.x, currentValue.y), currentValue.z);
                  if (maxValue == currentValue.x) {
                    maxValueIndex = maxValueIndex.x;
                  } else if (maxValue == currentValue.y) {
                    maxValueIndex = maxValueIndex.y;
                  } else if (maxValue == currentValue.z) {
                    maxValueIndex = maxValueIndex.z;
                  }
                  break;
                case 2:
                  maxValue = math.max(currentValue.x, currentValue.y);
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
    } else {
      return argOperationInternal(axis, resultDescriptor, reuse,
          begin: () {
            maxValueIndex = null;
            maxValue = null;
            maxValueCount = null;
          },
          onValue: (dimensionIndex, value, int valueCount) {
            if (value != null) {
              if (maxValue != null) {
                Int32x4 currentValue = maxValue;
                maxValueCount = math.max(maxValueCount, valueCount);

                switch (valueCount) {
                  case 4:
                    var newValueX;
                    var newValueIndexX;
                    if (value.x > currentValue.x) {
                      newValueX = value.x;
                      newValueIndexX = dimensionIndex.x;
                    } else {
                      newValueX = currentValue.x;
                      newValueIndexX = maxValueIndex.x;
                    }

                    var newValueY;
                    var newValueIndexY;
                    if (value.y > currentValue.y) {
                      newValueY = value.y;
                      newValueIndexY = dimensionIndex.y;
                    } else {
                      newValueY = currentValue.y;
                      newValueIndexY = maxValueIndex.y;
                    }

                    var newValueZ;
                    var newValueIndexZ;
                    if (value.z > currentValue.z) {
                      newValueZ = value.z;
                      newValueIndexZ = dimensionIndex.z;
                    } else {
                      newValueZ = currentValue.z;
                      newValueIndexZ = maxValueIndex.z;
                    }

                    var newValueW;
                    var newValueIndexW;
                    if (value.w > currentValue.w) {
                      newValueW = value.w;
                      newValueIndexW = dimensionIndex.w;
                    } else {
                      newValueW = currentValue.w;
                      newValueIndexW = maxValueIndex.w;
                    }

                    maxValue =
                        new Int32x4(newValueX, newValueY, newValueZ, newValueW);
                    maxValueIndex = new Int32x4(newValueIndexX, newValueIndexY,
                        newValueIndexZ, newValueIndexW);
                    break;
                  case 3:
                    var newValueX;
                    var newValueIndexX;
                    if (value.x > currentValue.x) {
                      newValueX = value.x;
                      newValueIndexX = dimensionIndex.x;
                    } else {
                      newValueX = currentValue.x;
                      newValueIndexX = maxValueIndex.x;
                    }

                    var newValueY;
                    var newValueIndexY;
                    if (value.y > currentValue.y) {
                      newValueY = value.y;
                      newValueIndexY = dimensionIndex.y;
                    } else {
                      newValueY = currentValue.y;
                      newValueIndexY = maxValueIndex.y;
                    }

                    var newValueZ;
                    var newValueIndexZ;
                    if (value.z > currentValue.z) {
                      newValueZ = value.z;
                      newValueIndexZ = dimensionIndex.z;
                    } else {
                      newValueZ = currentValue.z;
                      newValueIndexZ = maxValueIndex.z;
                    }

                    maxValue = new Int32x4(
                        newValueX, newValueY, newValueZ, currentValue.w);
                    maxValueIndex = new Int32x4(newValueIndexX, newValueIndexY,
                        newValueIndexZ, maxValueIndex.w);
                    break;
                  case 2:
                    var newValueX;
                    var newValueIndexX;
                    if (value.x > currentValue.x) {
                      newValueX = value.x;
                      newValueIndexX = dimensionIndex.x;
                    } else {
                      newValueX = currentValue.x;
                      newValueIndexX = maxValueIndex.x;
                    }

                    var newValueY;
                    var newValueIndexY;
                    if (value.y > currentValue.y) {
                      newValueY = value.y;
                      newValueIndexY = dimensionIndex.y;
                    } else {
                      newValueY = currentValue.y;
                      newValueIndexY = maxValueIndex.y;
                    }

                    maxValue = new Int32x4(
                        newValueX, newValueY, currentValue.z, currentValue.w);
                    maxValueIndex = new Int32x4(newValueIndexX, newValueIndexY,
                        maxValueIndex.z, maxValueIndex.w);
                    break;
                  case 1:
                    var newValueX;
                    var newValueIndexX;
                    if (value.x > currentValue.x) {
                      newValueX = value.x;
                      newValueIndexX = dimensionIndex.x;
                    } else {
                      newValueX = currentValue.x;
                      newValueIndexX = maxValueIndex.x;
                    }

                    maxValue = new Int32x4(newValueX, currentValue.y,
                        currentValue.z, currentValue.w);
                    maxValueIndex = new Int32x4(newValueIndexX, maxValueIndex.z,
                        maxValueIndex.z, maxValueIndex.w);
                    break;
                }
              } else {
                maxValueIndex = dimensionIndex;
                maxValue = value;
                maxValueCount = valueCount;
              }
            } else {
              Int32x4 currentValue = maxValue;

              switch (maxValueCount) {
                case 4:
                  maxValue = math.max(math.max(currentValue.x, currentValue.y),
                      math.max(currentValue.w, currentValue.z));
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
                  maxValue = math.max(
                      math.max(currentValue.x, currentValue.y), currentValue.z);
                  if (maxValue == currentValue.x) {
                    maxValueIndex = maxValueIndex.x;
                  } else if (maxValue == currentValue.y) {
                    maxValueIndex = maxValueIndex.y;
                  } else if (maxValue == currentValue.z) {
                    maxValueIndex = maxValueIndex.z;
                  }
                  break;
                case 2:
                  maxValue = math.max(currentValue.x, currentValue.y);
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
  }

  @override
  NDArray reshape({List<int> newDimensions, NDArray reuse}) {
    if (_iterableEquality.equals(shape.dimensions, newDimensions)) {
      return this;
    } else {
      var resultDescriptor = descriptor.reshape(newDimensions: newDimensions);

      var resultDataInfo = new DataInfo(resultDescriptor);

      var resultData;
      if (resultDataInfo.internalShape[
                  resultDataInfo.internalShape.dimensionCount - 1] ==
              dataInfo
                  .internalShape[dataInfo.internalShape.dimensionCount - 1] &&
          resultDataInfo.internalShape[
                  resultDataInfo.internalShape.dimensionCount - 2] ==
              dataInfo
                  .internalShape[dataInfo.internalShape.dimensionCount - 2]) {
        resultData = data;
      } else {
        // TODO verificare casi in cui si possa ottimizzare
        resultData = _createData(resultDescriptor, resultDataInfo, reuse);

        _elementWiseUnaryOperationData(this, resultData, resultDescriptor,
            resultDataInfo, (value, valueCount) => value);
      }

      return new NDArrayBlockedImpl.raw(
          resultData, resultDescriptor, resultDataInfo);
    }
  }

  @override
  NDArray reduceOperationInternal(List<int> reductionAxis, bool keepDimensions,
      NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(), void onValue(value, int valueCount), dynamic end()}) {
    var newReductionAxis =
        convertToValidReductionAxis(reductionAxis, shape.dimensionCount);

    if (newReductionAxis.isEmpty) {
      return this;
    } else if (keepDimensions) {
      // TODO gestione reduce con keepDimensions

      var resultDataInfo = new DataInfo(resultDescriptor);

      var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

      _reduceData(this, newReductionAxis, false, resultData, resultDescriptor,
          resultDataInfo,
          begin: begin, onValue: onValue, end: end);

      var result = new NDArrayBlockedImpl.raw(
          resultData, resultDescriptor, resultDataInfo);

      var newDimensions = new List.generate(
          shape.dimensions.length,
          (index) =>
              newReductionAxis.contains(index) ? 1 : shape.dimensions[index]);

      return result.reshape(newDimensions: newDimensions);
    } else {
      var resultDataInfo = new DataInfo(resultDescriptor);

      var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

      _reduceData(this, newReductionAxis, keepDimensions, resultData,
          resultDescriptor, resultDataInfo,
          begin: begin, onValue: onValue, end: end);

      return new NDArrayBlockedImpl.raw(
          resultData, resultDescriptor, resultDataInfo);
    }
  }

  @override
  NDArray argOperationInternal(
      int axis, NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(),
      void onValue(dimensionIndex, value, int valueCount),
      dynamic end()}) {
    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    _argData(this, axis, resultData, resultDescriptor, resultDataInfo,
        begin: begin, onValue: onValue, end: end);

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayBase elementWiseUnaryOperationInternal(NDDescriptor resultDescriptor,
      NDArray reuse, unaryOperation(value, int valueCount)) {
    if (!resultDescriptor.dataType.isBlocked || !dataType.isBlocked) {
      // TODO gestire nel descrittore dell'operazione
      throw new UnsupportedError("DEAD CODE");
    }

    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    _elementWiseUnaryOperationDataBlocked(
        this, resultData, resultDescriptor, resultDataInfo, unaryOperation);

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayBase elementWiseBinaryOperationInternal(
      NDArrayBase array2,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      binaryOperation(value1, value2, int valueCount)) {
    if (!resultDescriptor.dataType.isBlocked ||
        !dataType.isBlocked ||
        !array2.dataType.isBlocked) {
      // TODO gestire nel descrittore dell'operazione
      throw new UnsupportedError("DEAD CODE");
    }

    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    _elementWiseBinaryOperationDataBlocked(this, array2, resultData,
        resultDescriptor, resultDataInfo, binaryOperation);

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayBase elementWiseTernaryOperationInternal(
      NDArrayBase array2,
      NDArrayBase array3,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      ternaryOperation(value1, value2, value3, int valueCount)) {
    if (!resultDescriptor.dataType.isBlocked ||
        !dataType.isBlocked ||
        !array2.dataType.isBlocked ||
        !array3.dataType.isBlocked) {
      // TODO gestire nel descrittore dell'operazione
      throw new UnsupportedError("DEAD CODE");
    }

    var resultDataInfo = new DataInfo(resultDescriptor);

    var resultData = _createData(resultDescriptor, resultDataInfo, reuse);

    _elementWiseTernaryOperationDataBlocked(this, array2, array3, resultData,
        resultDescriptor, resultDataInfo, ternaryOperation);

    return new NDArrayBlockedImpl.raw(
        resultData, resultDescriptor, resultDataInfo);
  }

  @override
  Iterable get valueIterable => _createValueIterable(this);

  @override
  String toString() =>
      "<value: ${toValue()}, shape: $shape, dataType: $dataType>";

  void logData() {
    print(data);
  }

  @override
  NDArray abs({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.abs(), reuse, (value, valueCount) => value.abs());

  @override
  NDArray exp({NDArray reuse}) =>
      elementWiseUnaryOperationInternal(descriptor.exp(), reuse,
          (Float32x4 value, valueCount) {
        switch (valueCount) {
          case 4:
            return new Float32x4(math.exp(value.x), math.exp(value.y),
                math.exp(value.z), math.exp(value.w));
          case 3:
            return new Float32x4(
                math.exp(value.x), math.exp(value.y), math.exp(value.z), 0.0);
          case 2:
            return new Float32x4(
                math.exp(value.x), math.exp(value.y), 0.0, 0.0);
          case 1:
            return new Float32x4(math.exp(value.x), 0.0, 0.0, 0.0);
        }
      });

  @override
  NDArray log({NDArray reuse}) =>
      elementWiseUnaryOperationInternal(descriptor.log(), reuse,
          (Float32x4 value, valueCount) {
        switch (valueCount) {
          case 4:
            return new Float32x4(math.log(value.x), math.log(value.y),
                math.log(value.z), math.log(value.w));
          case 3:
            return new Float32x4(
                math.log(value.x), math.log(value.y), math.log(value.z), 0.0);
          case 2:
            return new Float32x4(
                math.log(value.x), math.log(value.y), 0.0, 0.0);
          case 1:
            return new Float32x4(math.log(value.x), 0.0, 0.0, 0.0);
        }
      });

  @override
  NDArray neg({NDArray reuse}) => elementWiseUnaryOperationInternal(
      descriptor.neg(), reuse, (value, valueCount) => -value);

  @override
  NDArray reciprocal({NDArray reuse}) =>
      elementWiseUnaryOperationInternal(descriptor.reciprocal(), reuse,
          (Float32x4 value, valueCount) {
        var reciprocalValue = value.reciprocal();

        switch (valueCount) {
          case 4:
            return reciprocalValue;
          case 3:
            return new Float32x4(
                reciprocalValue.x, reciprocalValue.y, reciprocalValue.z, 0.0);
          case 2:
            return new Float32x4(
                reciprocalValue.x, reciprocalValue.y, 0.0, 0.0);
          case 1:
            return new Float32x4(reciprocalValue.x, 0.0, 0.0, 0.0);
        }
      });

  @override
  NDArray sign({NDArray reuse}) {
    if (dataType.isFloat) {
      return elementWiseUnaryOperationInternal(descriptor.sign(), reuse,
          (Float32x4 value, valueCount) {
        switch (valueCount) {
          case 4:
            return new Float32x4(
                value.x.sign, value.y.sign, value.z.sign, value.w.sign);
          case 3:
            return new Float32x4(value.x.sign, value.y.sign, value.z.sign, 0.0);
          case 2:
            return new Float32x4(value.x.sign, value.y.sign, 0.0, 0.0);
          case 1:
            return new Float32x4(value.x.sign, 0.0, 0.0, 0.0);
        }
      });
    } else {
      return elementWiseUnaryOperationInternal(descriptor.sign(), reuse,
          (Int32x4 value, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4(
                value.x.sign, value.y.sign, value.z.sign, value.w.sign);
          case 3:
            return new Int32x4(value.x.sign, value.y.sign, value.z.sign, 0);
          case 2:
            return new Int32x4(value.x.sign, value.y.sign, 0, 0);
          case 1:
            return new Int32x4(value.x.sign, 0, 0, 0);
        }
      });
    }
  }

  @override
  NDArray not({NDArray reuse}) {
    return elementWiseUnaryOperationInternal(descriptor.not(), reuse,
        (Int32x4 value, valueCount) {
      switch (valueCount) {
        case 4:
          return new Int32x4.bool(
              !value.flagX, !value.flagY, !value.flagZ, !value.flagW);
        case 3:
          return new Int32x4.bool(
              !value.flagX, !value.flagY, !value.flagZ, false);
        case 2:
          return new Int32x4.bool(!value.flagX, !value.flagY, false, false);
        case 1:
          return new Int32x4.bool(!value.flagX, false, false, false);
      }
    });
  }

  @override
  NDArray add(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.add(array2.descriptor),
        reuse,
        (Float32x4 value1, Float32x4 value2, valueCount) => value1 + value2);
  }

  @override
  NDArray sub(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.sub(array2.descriptor),
        reuse,
        (Float32x4 value1, Float32x4 value2, valueCount) => value1 - value2);
  }

  @override
  NDArray mul(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2,
        descriptor.mul(array2.descriptor),
        reuse,
        (Float32x4 value1, Float32x4 value2, valueCount) => value1 * value2);
  }

  @override
  NDArray div(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    return elementWiseBinaryOperationInternal(
        array2, descriptor.div(array2.descriptor), reuse,
        (Float32x4 value1, Float32x4 value2, valueCount) {
      var divValue = value1 / value2;

      switch (valueCount) {
        case 4:
          return divValue;
        case 3:
          return new Float32x4(divValue.x, divValue.y, divValue.z, 0.0);
        case 2:
          return new Float32x4(divValue.x, divValue.y, 0.0, 0.0);
        case 1:
          return new Float32x4(divValue.x, 0.0, 0.0, 0.0);
      }
    });
  }

  @override
  NDArray isGreater(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    if (dataType.isFloat) {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (Float32x4 value1, Float32x4 value2, valueCount) {
        var result = value1.greaterThan(value2);

        switch (valueCount) {
          case 4:
            return result;
          case 3:
            return new Int32x4.bool(
                result.flagX, result.flagY, result.flagZ, false);
          case 2:
            return new Int32x4.bool(result.flagX, result.flagY, false, false);
          case 1:
            return new Int32x4.bool(result.flagX, false, false, false);
        }
      });
    } else {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (value1, value2, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4.bool(value1.x > value2.x, value1.y > value2.y,
                value1.z > value2.z, value1.w > value2.w);
          case 3:
            return new Int32x4.bool(value1.x > value2.x, value1.y > value2.y,
                value1.z > value2.z, false);
          case 2:
            return new Int32x4.bool(
                value1.x > value2.x, value1.y > value2.y, false, false);
          case 1:
            return new Int32x4.bool(value1.x > value2.x, false, false, false);
        }
      });
    }
  }

  @override
  NDArray isGreaterOrEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    if (dataType.isFloat) {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (Float32x4 value1, Float32x4 value2, valueCount) {
        var result = value1.greaterThanOrEqual(value2);

        switch (valueCount) {
          case 4:
            return result;
          case 3:
            return new Int32x4.bool(
                result.flagX, result.flagY, result.flagZ, false);
          case 2:
            return new Int32x4.bool(result.flagX, result.flagY, false, false);
          case 1:
            return new Int32x4.bool(result.flagX, false, false, false);
        }
      });
    } else {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (value1, value2, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4.bool(value1.x >= value2.x, value1.y >= value2.y,
                value1.z >= value2.z, value1.w >= value2.w);
          case 3:
            return new Int32x4.bool(value1.x >= value2.x, value1.y >= value2.y,
                value1.z >= value2.z, false);
          case 2:
            return new Int32x4.bool(
                value1.x >= value2.x, value1.y >= value2.y, false, false);
          case 1:
            return new Int32x4.bool(value1.x >= value2.x, false, false, false);
        }
      });
    }
  }

  @override
  NDArray isLess(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    if (dataType.isFloat) {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (Float32x4 value1, Float32x4 value2, valueCount) {
        var result = value1.lessThan(value2);

        switch (valueCount) {
          case 4:
            return result;
          case 3:
            return new Int32x4.bool(
                result.flagX, result.flagY, result.flagZ, false);
          case 2:
            return new Int32x4.bool(result.flagX, result.flagY, false, false);
          case 1:
            return new Int32x4.bool(result.flagX, false, false, false);
        }
      });
    } else {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (value1, value2, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4.bool(value1.x < value2.x, value1.y < value2.y,
                value1.z < value2.z, value1.w < value2.w);
          case 3:
            return new Int32x4.bool(value1.x < value2.x, value1.y < value2.y,
                value1.z < value2.z, false);
          case 2:
            return new Int32x4.bool(
                value1.x < value2.x, value1.y < value2.y, false, false);
          case 1:
            return new Int32x4.bool(value1.x < value2.x, false, false, false);
        }
      });
    }
  }

  @override
  NDArray isLessOrEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    if (dataType.isFloat) {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (Float32x4 value1, Float32x4 value2, valueCount) {
        var result = value1.lessThanOrEqual(value2);

        switch (valueCount) {
          case 4:
            return result;
          case 3:
            return new Int32x4.bool(
                result.flagX, result.flagY, result.flagZ, false);
          case 2:
            return new Int32x4.bool(result.flagX, result.flagY, false, false);
          case 1:
            return new Int32x4.bool(result.flagX, false, false, false);
        }
      });
    } else {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (value1, value2, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4.bool(value1.x <= value2.x, value1.y <= value2.y,
                value1.z <= value2.z, value1.w <= value2.w);
          case 3:
            return new Int32x4.bool(value1.x <= value2.x, value1.y <= value2.y,
                value1.z <= value2.z, false);
          case 2:
            return new Int32x4.bool(
                value1.x <= value2.x, value1.y <= value2.y, false, false);
          case 1:
            return new Int32x4.bool(value1.x <= value2.x, false, false, false);
        }
      });
    }
  }

  @override
  NDArray isEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    if (dataType.isFloat) {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (Float32x4 value1, Float32x4 value2, valueCount) {
        var result = value1.equal(value2);

        switch (valueCount) {
          case 4:
            return result;
          case 3:
            return new Int32x4.bool(
                result.flagX, result.flagY, result.flagZ, false);
          case 2:
            return new Int32x4.bool(result.flagX, result.flagY, false, false);
          case 1:
            return new Int32x4.bool(result.flagX, false, false, false);
        }
      });
    } else {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (value1, value2, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4.bool(value1.x == value2.x, value1.y == value2.y,
                value1.z == value2.z, value1.w == value2.w);
          case 3:
            return new Int32x4.bool(value1.x == value2.x, value1.y == value2.y,
                value1.z == value2.z, false);
          case 2:
            return new Int32x4.bool(
                value1.x == value2.x, value1.y == value2.y, false, false);
          case 1:
            return new Int32x4.bool(value1.x == value2.x, false, false, false);
        }
      });
    }
  }

  @override
  NDArray isNotEqual(value2, {NDArray reuse}) {
    var array2 = toNDArray(value2, dataType: dataType);

    if (dataType.isFloat) {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (Float32x4 value1, Float32x4 value2, valueCount) {
        var result = value1.notEqual(value2);

        switch (valueCount) {
          case 4:
            return result;
          case 3:
            return new Int32x4.bool(
                result.flagX, result.flagY, result.flagZ, false);
          case 2:
            return new Int32x4.bool(result.flagX, result.flagY, false, false);
          case 1:
            return new Int32x4.bool(result.flagX, false, false, false);
        }
      });
    } else {
      return elementWiseBinaryOperationInternal(
          array2, descriptor.isNotEqual(array2.descriptor), reuse,
          (value1, value2, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4.bool(value1.x != value2.x, value1.y != value2.y,
                value1.z != value2.z, value1.w != value2.w);
          case 3:
            return new Int32x4.bool(value1.x != value2.x, value1.y != value2.y,
                value1.z != value2.z, false);
          case 2:
            return new Int32x4.bool(
                value1.x != value2.x, value1.y != value2.y, false, false);
          case 1:
            return new Int32x4.bool(value1.x != value2.x, false, false, false);
        }
      });
    }
  }

  NDDataType _calculateBlockedDataType(value) {
    if (value is NDArray) {
      return value.dataType;
    } else {
      var valueDataType = calculateDataType(value);

      if (valueDataType.isFloat) {
        return NDDataType.float32Blocked;
      } else if (valueDataType.isInteger) {
        return NDDataType.int32Blocked;
      } else if (valueDataType.isBoolean) {
        return NDDataType.booleanBlocked;
      } else {
        throw new StateError("DEAD CODE");
      }
    }
  }

  @override
  NDArray select(thenValue, elseValue, {NDDataType dataType, NDArray reuse}) {
    var thenArray = toNDArray(thenValue,
        dataType: dataType ?? _calculateBlockedDataType(thenValue));
    var elseArray = toNDArray(elseValue,
        dataType: dataType ?? _calculateBlockedDataType(elseValue));

    var resultDescriptor =
        descriptor.select(thenArray.descriptor, elseArray.descriptor);

    if (thenArray.dataType.isFloat) {
      return elementWiseTernaryOperationInternal(
          thenArray, elseArray, resultDescriptor, reuse,
          (Int32x4 value1, Float32x4 value2, Float32x4 value3, valueCount) {
        switch (valueCount) {
          case 4:
            return new Float32x4(
                value1.flagX ? value2.x : value3.x,
                value1.flagY ? value2.y : value3.y,
                value1.flagZ ? value2.z : value3.z,
                value1.flagW ? value2.w : value3.w);
          case 3:
            return new Float32x4(
                value1.flagX ? value2.x : value3.x,
                value1.flagY ? value2.y : value3.y,
                value1.flagZ ? value2.z : value3.z,
                0.0);
          case 2:
            return new Float32x4(value1.flagX ? value2.x : value3.x,
                value1.flagY ? value2.y : value3.y, 0.0, 0.0);
          case 1:
            return new Float32x4(
                value1.flagX ? value2.x : value3.x, 0.0, 0.0, 0.0);
        }
      });
    } else {
      return elementWiseTernaryOperationInternal(
          thenArray, elseArray, resultDescriptor, reuse,
          (Int32x4 value1, Int32x4 value2, Int32x4 value3, valueCount) {
        switch (valueCount) {
          case 4:
            return new Int32x4(
                value1.flagX ? value2.x : value3.x,
                value1.flagY ? value2.y : value3.y,
                value1.flagZ ? value2.z : value3.z,
                value1.flagW ? value2.w : value3.w);
          case 3:
            return new Int32x4(
                value1.flagX ? value2.x : value3.x,
                value1.flagY ? value2.y : value3.y,
                value1.flagZ ? value2.z : value3.z,
                0);
          case 2:
            return new Int32x4(value1.flagX ? value2.x : value3.x,
                value1.flagY ? value2.y : value3.y, 0, 0);
          case 1:
            return new Int32x4(value1.flagX ? value2.x : value3.x, 0, 0, 0);
        }
      });
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
    var internalShape = descriptor.shape.dimensionCount > 1
        ? descriptor.shape
        : (descriptor.shape.dimensionCount == 1
            ? new NDShape([1, descriptor.shape[0]])
            : new NDShape([1, 1]));

    var rows = internalShape.dimensions[internalShape.dimensionCount - 2];
    var columns = internalShape.dimensions[internalShape.dimensionCount - 1];

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

    var delta1 = dataRows;
    var delta2 = 1 - matrixDataLength;
    var delta3 = delta2;

    List<int> dimensions = new List.from(
        internalShape.dimensions.sublist(0, internalShape.dimensionCount - 2));

    var columnIndex = dimensions.length;
    dimensions.add(blockColumns);
    var rowIndex = dimensions.length;
    dimensions.add(blockRows);
    dimensions.add(descriptor.dataType.blockSize);

    List<int> stride = new List(dimensions.length);
    var factor = 1;
    for (var i = dimensions.length - 1; i >= 0; i--) {
      stride[i] = factor;
      factor *= dimensions[i];
    }

    List<int> dimensions2 = new List.from(
        internalShape.dimensions.sublist(0, internalShape.dimensionCount - 2));

    dimensions2.add(blockColumns);
    dimensions2.add(dataRows);

    List<int> stride2 = new List(dimensions.length - 1);
    var factor2 = 1;
    for (var i = dimensions2.length - 1; i >= 0; i--) {
      stride2[i] = factor2;
      factor2 *= dimensions2[i];
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
        dimensions2,
        stride2,
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

Iterable _createValueIterable(NDArrayBlockedImpl array) {
  if (array.dataType.isBoolean) {
    return _createValueIterableBoolean(array);
  } else {
    return _createValueIterableNum(array);
  }
}

Iterable<num> _createValueIterableNum(NDArrayBlockedImpl array) sync* {
  var dimensionIndexes =
      new List(array.dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(array.dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = array.dataInfo.dataColumns - array.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        array.dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < array.dataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < array.dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
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

Iterable<bool> _createValueIterableBoolean(NDArrayBlockedImpl array) sync* {
  var dimensionIndexes =
      new List(array.dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(array.dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = array.dataInfo.dataColumns - array.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        array.dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < array.dataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < array.dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += array.dataType.blockSize) {
            Int32x4 value4 = array.data[dataIndex];

            yield value4.flagX;
            yield value4.flagY;
            yield value4.flagZ;
            yield value4.flagW;

            dataIndex += array.dataInfo.delta1;
          }

          Int32x4 value4 = array.data[dataIndex];

          switch (array.dataInfo.lastBlockColumnCount) {
            case 4:
              yield value4.flagX;
              yield value4.flagY;
              yield value4.flagZ;
              yield value4.flagW;

              break;
            case 3:
              yield value4.flagX;
              yield value4.flagY;
              yield value4.flagZ;

              break;
            case 2:
              yield value4.flagX;
              yield value4.flagY;

              break;
            case 1:
              yield value4.flagX;

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

Iterable _createTiledValueIterable(NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  if (sourceArray.dataType.isBoolean) {
    return _createTiledValueIterableBoolean(
        sourceArray, resultDescriptor, resultDataInfo);
  } else {
    return _createTiledValueIterableNum(
        sourceArray, resultDescriptor, resultDataInfo);
  }
}

Iterable _createTiledValueIterableNum(NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) sync* {
  var dimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);
  var sourceDimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);
  var initialDataIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  sourceDimensionIndexes[0] = 0;
  dataIndexes[0] = 0;
  initialDataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var rowsMultiplier = resultDataInfo
          .internalShape[resultDataInfo.internalShape.dimensionCount - 2] ~/
      sourceArray.dataInfo.rows;
  var columnsMultiplier = resultDataInfo
          .internalShape[resultDataInfo.internalShape.dimensionCount - 1] ~/
      sourceArray.dataInfo.columns;

  var lastColumnIndex =
      sourceArray.dataInfo.dataColumns - sourceArray.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < sourceArray.dataInfo.internalShape.dimensionCount - 2) {
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
                  column < lastColumnIndex;
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

Iterable _createTiledValueIterableBoolean(NDArrayBlockedImpl sourceArray,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) sync* {
  var dimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);
  var sourceDimensionIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);
  var initialDataIndexes =
      new List(sourceArray.dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  sourceDimensionIndexes[0] = 0;
  dataIndexes[0] = 0;
  initialDataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var rowsMultiplier = resultDataInfo
          .internalShape[resultDataInfo.internalShape.dimensionCount - 2] ~/
      sourceArray.dataInfo.rows;
  var columnsMultiplier = resultDataInfo
          .internalShape[resultDataInfo.internalShape.dimensionCount - 1] ~/
      sourceArray.dataInfo.columns;

  var lastColumnIndex =
      sourceArray.dataInfo.dataColumns - sourceArray.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < sourceArray.dataInfo.internalShape.dimensionCount - 2) {
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
                  column < lastColumnIndex;
                  column += sourceArray.dataType.blockSize) {
                var value4 = sourceArray.data[dataIndex];

                yield value4.flagX;
                yield value4.flagY;
                yield value4.flagZ;
                yield value4.flagW;

                dataIndex += sourceArray.dataInfo.delta1;
              }

              var value4 = sourceArray.data[dataIndex];

              switch (sourceArray.dataInfo.lastBlockColumnCount) {
                case 4:
                  yield value4.flagX;
                  yield value4.flagY;
                  yield value4.flagZ;
                  yield value4.flagW;

                  break;
                case 3:
                  yield value4.flagX;
                  yield value4.flagY;
                  yield value4.flagZ;

                  break;
                case 2:
                  yield value4.flagX;
                  yield value4.flagY;

                  break;
                case 1:
                  yield value4.flagX;

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
      case NDDataType.float32Blocked:
        return new Float32x4List(dataInfo.dataLength);
      case NDDataType.int32Blocked:
        return new Int32x4List(dataInfo.dataLength);
      case NDDataType.booleanBlocked:
        return new Int32x4List(dataInfo.dataLength);
      default:
        throw new StateError("DEAD CODE");
    }
  }
}

void _loadData(value, List data, NDDescriptor descriptor, DataInfo dataInfo) {
  switch (descriptor.dataType) {
    case NDDataType.float32Blocked:
      return _loadDataFloat(value, data, descriptor, dataInfo);
    case NDDataType.int32Blocked:
      return _loadDataInt(value, data, descriptor, dataInfo);
    case NDDataType.booleanBlocked:
      return _loadDataBoolean(value, data, descriptor, dataInfo);
    default:
      throw new StateError("DEAD CODE");
  }
}

void _loadDataFloat(
    value, Float32x4List data, NDDescriptor descriptor, DataInfo dataInfo) {
  var newValue = descriptor.shape.dimensionCount > 1
      ? value
      : (descriptor.shape.dimensionCount == 1
          ? [value]
          : [
              [value]
            ]);

  var values = new List(dataInfo.internalShape.dimensionCount - 1);
  var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  values[0] = newValue;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
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
              column < lastColumnIndex;
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

void _loadDataInt(
    value, Int32x4List data, NDDescriptor descriptor, DataInfo dataInfo) {
  var newValue = descriptor.shape.dimensionCount > 1
      ? value
      : (descriptor.shape.dimensionCount == 1
          ? [value]
          : [
              [value]
            ]);

  var values = new List(dataInfo.internalShape.dimensionCount - 1);
  var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  values[0] = newValue;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
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
          List<int> rowValues = newValue[row];

          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += descriptor.dataType.blockSize) {
            var value4 = new Int32x4(rowValues[column], rowValues[column + 1],
                rowValues[column + 2], rowValues[column + 3]);

            data[dataIndex] = value4;

            dataIndex += dataInfo.delta1;
          }

          var value4;
          switch (dataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4(rowValues[column], rowValues[column + 1],
                  rowValues[column + 2], rowValues[column + 3]);
              break;
            case 3:
              value4 = new Int32x4(rowValues[column], rowValues[column + 1],
                  rowValues[column + 2], 0);
              break;
            case 2:
              value4 =
                  new Int32x4(rowValues[column], rowValues[column + 1], 0, 0);
              break;
            case 1:
              value4 = new Int32x4(rowValues[column], 0, 0, 0);
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

void _loadDataBoolean(
    value, Int32x4List data, NDDescriptor descriptor, DataInfo dataInfo) {
  var newValue = descriptor.shape.dimensionCount > 1
      ? value
      : (descriptor.shape.dimensionCount == 1
          ? [value]
          : [
              [value]
            ]);

  var values = new List(dataInfo.internalShape.dimensionCount - 1);
  var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  values[0] = newValue;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
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
          List<bool> rowValues = newValue[row];

          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += descriptor.dataType.blockSize) {
            var value4 = new Int32x4.bool(
                rowValues[column],
                rowValues[column + 1],
                rowValues[column + 2],
                rowValues[column + 3]);

            data[dataIndex] = value4;

            dataIndex += dataInfo.delta1;
          }

          var value4;
          switch (dataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4.bool(
                  rowValues[column],
                  rowValues[column + 1],
                  rowValues[column + 2],
                  rowValues[column + 3]);
              break;
            case 3:
              value4 = new Int32x4.bool(rowValues[column],
                  rowValues[column + 1], rowValues[column + 2], false);
              break;
            case 2:
              value4 = new Int32x4.bool(
                  rowValues[column], rowValues[column + 1], false, false);
              break;
            case 1:
              value4 = new Int32x4.bool(rowValues[column], false, false, false);
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

void _generateData(generator(int index), List data, NDDescriptor descriptor,
    DataInfo dataInfo) {
  switch (descriptor.dataType) {
    case NDDataType.float32Blocked:
      return _generateDataFloat(generator, data, descriptor, dataInfo);
    case NDDataType.int32Blocked:
      return _generateDataInt(generator, data, descriptor, dataInfo);
    case NDDataType.booleanBlocked:
      return _generateDataBoolean(generator, data, descriptor, dataInfo);
    default:
      throw new StateError("DEAD CODE");
  }
}

void _generateDataFloat(generator(int index), Float32x4List data,
    NDDescriptor descriptor, DataInfo dataInfo) {
  var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;
  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
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

void _generateDataInt(int generator(int index), Int32x4List data,
    NDDescriptor descriptor, DataInfo dataInfo) {
  var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;
  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += descriptor.dataType.blockSize) {
            var value4 = new Int32x4(generator(i), generator(i + 1),
                generator(i + 2), generator(i + 3));

            data[dataIndex] = value4;

            dataIndex += dataInfo.delta1;

            i += descriptor.dataType.blockSize;
          }

          var value4;
          switch (dataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4(generator(i), generator(i + 1),
                  generator(i + 2), generator(i + 3));
              break;
            case 3:
              value4 = new Int32x4(
                  generator(i), generator(i + 1), generator(i + 2), 0);
              break;
            case 2:
              value4 = new Int32x4(generator(i), generator(i + 1), 0, 0);
              break;
            case 1:
              value4 = new Int32x4(generator(i), 0, 0, 0);
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

void _generateDataBoolean(bool generator(int index), Int32x4List data,
    NDDescriptor descriptor, DataInfo dataInfo) {
  var dimensionIndexes = new List(dataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(dataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var i = 0;
  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex = dataInfo.dataColumns - descriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < dataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < dataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < dataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += descriptor.dataType.blockSize) {
            var value4 = new Int32x4.bool(generator(i), generator(i + 1),
                generator(i + 2), generator(i + 3));

            data[dataIndex] = value4;

            dataIndex += dataInfo.delta1;

            i += descriptor.dataType.blockSize;
          }

          var value4;
          switch (dataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4.bool(generator(i), generator(i + 1),
                  generator(i + 2), generator(i + 3));
              break;
            case 3:
              value4 = new Int32x4.bool(
                  generator(i), generator(i + 1), generator(i + 2), false);
              break;
            case 2:
              value4 = new Int32x4.bool(
                  generator(i), generator(i + 1), false, false);
              break;
            case 1:
              value4 = new Int32x4.bool(generator(i), false, false, false);
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
  if (fromArray.dataType.isBlocked) {
    switch (descriptor.dataType) {
      case NDDataType.float32Blocked:
        switch (fromArray.descriptor.dataType) {
          case NDDataType.int32Blocked:
            return _elementWiseUnaryOperationDataBlocked(
                fromArray,
                data,
                descriptor,
                dataInfo,
                (Int32x4 value, int valueCount) => new Float32x4(
                    value.x.toDouble(),
                    value.y.toDouble(),
                    value.z.toDouble(),
                    value.w.toDouble()));
          case NDDataType.booleanBlocked:
            return _elementWiseUnaryOperationDataBlocked(
                fromArray,
                data,
                descriptor,
                dataInfo,
                (Int32x4 value, int valueCount) => new Float32x4(
                    value.flagX ? 1.0 : 0.0,
                    value.flagY ? 1.0 : 0.0,
                    value.flagZ ? 1.0 : 0.0,
                    value.flagW ? 1.0 : 0.0));
          default:
            throw new StateError("DEAD CODE");
        }
        break;
      case NDDataType.int32Blocked:
        switch (fromArray.descriptor.dataType) {
          case NDDataType.float32Blocked:
            return _elementWiseUnaryOperationDataBlocked(
                fromArray,
                data,
                descriptor,
                dataInfo,
                (Float32x4 value, int valueCount) => new Int32x4(
                    value.x.toInt(),
                    value.y.toInt(),
                    value.z.toInt(),
                    value.w.toInt()));
          case NDDataType.booleanBlocked:
            return _elementWiseUnaryOperationDataBlocked(
                fromArray,
                data,
                descriptor,
                dataInfo,
                (Int32x4 value, int valueCount) => new Int32x4(
                    value.flagX ? 1 : 0,
                    value.flagY ? 1 : 0,
                    value.flagZ ? 1 : 0,
                    value.flagW ? 1 : 0));
          default:
            throw new StateError("DEAD CODE");
        }
        break;
      case NDDataType.booleanBlocked:
        switch (fromArray.descriptor.dataType) {
          case NDDataType.float32Blocked:
            return _elementWiseUnaryOperationDataBlocked(
                fromArray,
                data,
                descriptor,
                dataInfo,
                (Float32x4 value, int valueCount) => new Int32x4.bool(
                    value.x != 0.0,
                    value.y != 0.0,
                    value.z != 0.0,
                    value.w != 0.0));
          case NDDataType.int32Blocked:
            return _elementWiseUnaryOperationDataBlocked(
                fromArray,
                data,
                descriptor,
                dataInfo,
                (Int32x4 value, int valueCount) => new Int32x4.bool(
                    value.x != 0, value.y != 0, value.z != 0, value.w != 0));
          default:
            throw new StateError("DEAD CODE");
        }
        break;
      default:
        throw new StateError("DEAD CODE");
    }
  } else {
    switch (descriptor.dataType) {
      case NDDataType.float32Blocked:
        if (fromArray.dataType.isFloat) {
          return _castConvertedDataFloat(
              fromArray, data, descriptor, dataInfo, (double value) => value);
        } else if (fromArray.dataType.isInteger) {
          return _castConvertedDataFloat(fromArray, data, descriptor, dataInfo,
              (int value) => value.toDouble());
        } else if (fromArray.dataType.isBoolean) {
          return _castConvertedDataFloat(fromArray, data, descriptor, dataInfo,
              (bool value) => value ? 1.0 : 0.0);
        } else {
          throw new StateError("DEAD CODE");
        }
        break;
      case NDDataType.int32Blocked:
        if (fromArray.dataType.isFloat) {
          return _castConvertedDataInt(fromArray, data, descriptor, dataInfo,
              (double value) => value.toInt());
        } else if (fromArray.dataType.isInteger) {
          return _castConvertedDataInt(
              fromArray, data, descriptor, dataInfo, (int value) => value);
        } else if (fromArray.dataType.isBoolean) {
          return _castConvertedDataInt(fromArray, data, descriptor, dataInfo,
              (bool value) => value ? 1 : 0);
        } else {
          throw new StateError("DEAD CODE");
        }
        break;
      case NDDataType.booleanBlocked:
        if (fromArray.dataType.isFloat) {
          return _castConvertedDataBoolean(fromArray, data, descriptor,
              dataInfo, (double value) => value != 0.0);
        } else if (fromArray.dataType.isInteger) {
          return _castConvertedDataBoolean(
              fromArray, data, descriptor, dataInfo, (int value) => value != 0);
        } else if (fromArray.dataType.isBoolean) {
          return _castConvertedDataBoolean(
              fromArray, data, descriptor, dataInfo, (bool value) => value);
        } else {
          throw new StateError("DEAD CODE");
        }
        break;
      default:
        throw new StateError("DEAD CODE");
    }
  }
}

void _castConvertedDataFloat(
    NDArrayBase fromArray,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    double converter(value)) {
  Iterator valueIterator = fromArray.valueIterable.iterator;

  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current));

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
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
                  converter((valueIterator..moveNext()).current),
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

void _castConvertedDataInt(
    NDArrayBase fromArray,
    Int32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    int converter(value)) {
  Iterator valueIterator = fromArray.valueIterable.iterator;

  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Int32x4(
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current));

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4(
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current));
              break;
            case 3:
              value4 = new Int32x4(
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  0);
              break;
            case 2:
              value4 = new Int32x4(
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  0,
                  0);
              break;
            case 1:
              value4 = new Int32x4(
                  converter((valueIterator..moveNext()).current), 0, 0, 0);
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

void _castConvertedDataBoolean(
    NDArrayBase fromArray,
    Int32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    bool converter(value)) {
  Iterator valueIterator = fromArray.valueIterable.iterator;

  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);
  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Int32x4.bool(
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current),
                converter((valueIterator..moveNext()).current));

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4.bool(
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current));
              break;
            case 3:
              value4 = new Int32x4.bool(
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  false);
              break;
            case 2:
              value4 = new Int32x4.bool(
                  converter((valueIterator..moveNext()).current),
                  converter((valueIterator..moveNext()).current),
                  false,
                  false);
              break;
            case 1:
              value4 = new Int32x4.bool(
                  converter((valueIterator..moveNext()).current),
                  false,
                  false,
                  false);
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
  var multiplier;
  if (resultDataInfo.internalShape.dimensionCount > 2) {
    multiplier = resultDataInfo.internalShape.length ~/
        (resultDataInfo.rows * resultDataInfo.columns);
  } else {
    multiplier = 1;
  }

  var delta1 = (array1.dataInfo.blockRows - 1) << array1.dataType.blockDepth;

  var delta2 = array1.dataType.blockSize - array1.dataInfo.matrixDataLength;

  var delta3 = -(delta1 + delta2);

  var sourceDataIndex1 = 0;
  var sourceDataIndex2 = 0;
  var targetDataIndex = 0;

  final dynamic zero =
      resultDescriptor.dataType.isFloat ? _zeroFloat : _zeroInt;

  // TODO costanti per Float32x4.XXXX e Int32x4.XXXX sono stranamente lente

  for (var iteration = 0; iteration < multiplier; iteration++) {
    var initialSourceDataIndex1 = sourceDataIndex1;

    for (var column2 = 0;
        column2 < array2.dataInfo.dataColumns;
        column2 += array2.dataType.blockSize) {
      var initialSourceDataIndex2 = sourceDataIndex2;

      sourceDataIndex1 = initialSourceDataIndex1;

      for (var row1 = 0;
          row1 < array1.dataInfo.dataRows;
          row1 += array1.dataType.blockSize) {
        sourceDataIndex2 = initialSourceDataIndex2;

        var result0 = zero;
        var result1 = zero;
        var result2 = zero;
        var result3 = zero;

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

          sourceDataIndex1 += delta1;
        }

        resultData[targetDataIndex++] = result0;
        resultData[targetDataIndex++] = result1;
        resultData[targetDataIndex++] = result2;
        resultData[targetDataIndex++] = result3;

        sourceDataIndex1 += delta2;
      }
    }

    sourceDataIndex1 += delta3;
  }
}

void _reduceData(
    NDArrayBlockedImpl sourceArray,
    List<int> reductionAxis,
    bool keepDimensions,
    List targetData,
    NDDescriptor targetDescriptor,
    DataInfo targetDataInfo,
    {void begin(),
    void onValue(value, int valueCount),
    dynamic end()}) {
  if (targetDescriptor.dataType.isBoolean) {
    _reduceDataBoolean(sourceArray, reductionAxis, keepDimensions, targetData,
        targetDescriptor, targetDataInfo,
        begin: begin, onValue: onValue, end: end);
  } else {
    _reduceDataNum(sourceArray, reductionAxis, keepDimensions, targetData,
        targetDescriptor, targetDataInfo,
        begin: begin, onValue: onValue, end: end);
  }
}

void _reduceDataNum(
    NDArrayBlockedImpl sourceArray,
    List<int> reductionAxis,
    bool keepDimensions,
    List targetData,
    NDDescriptor targetDescriptor,
    DataInfo targetDataInfo,
    {void begin(),
    void onValue(value, int valueCount),
    dynamic end()}) {
  var axis = new Set<int>.from(reductionAxis);

  bool isRowsReduction =
      axis.contains(sourceArray.descriptor.shape.dimensionCount - 2);

  bool isColumnsReduction =
      axis.contains(sourceArray.descriptor.shape.dimensionCount - 1);

  if (sourceArray.shape.dimensionCount == 1) {
    // skip
  } else if (isRowsReduction && isColumnsReduction) {
    // skip
  } else if (isRowsReduction) {
    axis.remove(sourceArray.descriptor.shape.dimensionCount - 2);
    axis.add(sourceArray.descriptor.shape.dimensionCount - 1);
  } else if (isColumnsReduction) {
    axis.remove(sourceArray.descriptor.shape.dimensionCount - 1);
    axis.add(sourceArray.descriptor.shape.dimensionCount - 2);
  }

  var newReductionAxis = axis.toList(growable: false);

  List<int> targetPermutedIndexes =
      new List.generate(targetDataInfo.dimensions.length, (index) => index);

  if (targetDescriptor.shape.dimensionCount > 1) {
    if (isRowsReduction || isColumnsReduction) {
      var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
      targetPermutedIndexes[targetPermutedIndexes.length - 1] =
          targetPermutedIndexes[targetPermutedIndexes.length - 2];
      targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
    }
  }

  var sourcePermutedIndexes = new List.generate(
      sourceArray.dataInfo.dimensions.length, (index) => index);

  sourcePermutedIndexes =
      sourcePermutedIndexes.where((index) => !axis.contains(index)).toList();

  sourcePermutedIndexes.addAll(newReductionAxis);

  var targetBeginIndex =
      sourceArray.dataInfo.dimensions.length - newReductionAxis.length;

  var targetLimitIndex =
      isColumnsReduction ? targetBeginIndex - 1 : targetBeginIndex;

  var sourceTargetColumnIndex;
  var delta;
  if (targetDescriptor.shape.dimensionCount >= 1) {
    if (isColumnsReduction) {
      sourceTargetColumnIndex =
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1;
      delta = targetDataInfo.dataRows;
    }
  }

  var lastTargetColumnIndex;
  if (targetDescriptor.shape.dimensionCount >= 1) {
    if (isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1];
    }
  } else {
    if (isRowsReduction && isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 0];
    } else if (isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1];
    }
  }

  var sourceRowIndex = sourceArray.dataInfo.rowIndex;
  var sourceColumnIndex = sourceArray.dataInfo.columnIndex;
  var targetColumnIndex = targetDataInfo.columnIndex;

  var permutedLastTargetColumnIndex = lastTargetColumnIndex != null
      ? sourcePermutedIndexes.indexOf(lastTargetColumnIndex)
      : null;
  var permutedSourceRowIndex = sourcePermutedIndexes.indexOf(sourceRowIndex);
  var permutedSourceColumnIndex =
      sourcePermutedIndexes.indexOf(sourceColumnIndex);
  var permutedTargetColumnIndex =
      targetPermutedIndexes.indexOf(targetColumnIndex);

  var sourceStride =
      permute(sourceArray.dataInfo.stride, sourcePermutedIndexes);
  var targetStride = permute(targetDataInfo.stride, targetPermutedIndexes);
  var sourceDimensions =
      permute(sourceArray.dataInfo.dimensions, sourcePermutedIndexes);
  var targetDimensions =
      permute(targetDataInfo.dimensions, targetPermutedIndexes);

  sourceDimensions[permutedSourceRowIndex] = sourceArray.dataInfo.rows;

  var sourceDimensionIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var sourceDataIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var targetDataIndexes = new List(targetDataInfo.dimensions.length);

  sourceDimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  var lastColumn =
      sourceArray.dataInfo.dimensions[sourceArray.dataInfo.columnIndex] == 1;
  var lastTargetColumn = isColumnsReduction &&
      targetDataInfo.dimensions[targetDataInfo.columnIndex] == 1;
  var isLastTargetColumnToCalculate = isColumnsReduction &&
      targetDataInfo.dimensions[targetDataInfo.columnIndex] > 1;

  final zero = targetDescriptor.dataType.isFloat ? _zeroFloat : _zeroInt;

  var columnValue;
  int columnIndex;

  if (isColumnsReduction) {
    columnValue = zero;
    columnIndex = 0;
  }

  if (targetDescriptor.shape.dimensionCount == 0 &&
      sourceArray.shape.dimensionCount > 1) {
    begin();
  }

  for (;;) {
    if (sourceDimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
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

          if (isLastTargetColumnToCalculate) {
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

            columnValue = zero;
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

        if (shapeIndex == sourceTargetColumnIndex) {
          if (sourceDimensionIndexes[shapeIndex] &
                  (sourceArray.dataType.blockSize - 1) ==
              0) {
            targetDataIndexes[shapeIndex] += delta;
          }
        }

        targetDataIndex = targetDataIndexes[shapeIndex];

        if (isLastTargetColumnToCalculate) {
          if (shapeIndex == permutedTargetColumnIndex) {
            if (lastTargetColumnIndex == sourceArray.dataInfo.rowIndex) {
              lastTargetColumn =
                  (sourceDimensionIndexes[permutedLastTargetColumnIndex] >>
                          sourceArray.dataType.blockDepth ==
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

void _reduceDataBoolean(
    NDArrayBlockedImpl sourceArray,
    List<int> reductionAxis,
    bool keepDimensions,
    List targetData,
    NDDescriptor targetDescriptor,
    DataInfo targetDataInfo,
    {void begin(),
    void onValue(value, int valueCount),
    dynamic end()}) {
  var axis = new Set<int>.from(reductionAxis);

  bool isRowsReduction =
      axis.contains(sourceArray.descriptor.shape.dimensionCount - 2);

  bool isColumnsReduction =
      axis.contains(sourceArray.descriptor.shape.dimensionCount - 1);

  if (sourceArray.shape.dimensionCount == 1) {
    // skip
  } else if (isRowsReduction && isColumnsReduction) {
    // skip
  } else if (isRowsReduction) {
    axis.remove(sourceArray.descriptor.shape.dimensionCount - 2);
    axis.add(sourceArray.descriptor.shape.dimensionCount - 1);
  } else if (isColumnsReduction) {
    axis.remove(sourceArray.descriptor.shape.dimensionCount - 1);
    axis.add(sourceArray.descriptor.shape.dimensionCount - 2);
  }

  var newReductionAxis = axis.toList(growable: false);

  List<int> targetPermutedIndexes =
      new List.generate(targetDataInfo.dimensions.length, (index) => index);

  if (targetDescriptor.shape.dimensionCount > 1) {
    if (isRowsReduction || isColumnsReduction) {
      var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
      targetPermutedIndexes[targetPermutedIndexes.length - 1] =
          targetPermutedIndexes[targetPermutedIndexes.length - 2];
      targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
    }
  }

  var sourcePermutedIndexes = new List.generate(
      sourceArray.dataInfo.dimensions.length, (index) => index);

  sourcePermutedIndexes =
      sourcePermutedIndexes.where((index) => !axis.contains(index)).toList();

  sourcePermutedIndexes.addAll(newReductionAxis);

  var targetBeginIndex =
      sourceArray.dataInfo.dimensions.length - newReductionAxis.length;

  var targetLimitIndex =
      isColumnsReduction ? targetBeginIndex - 1 : targetBeginIndex;

  var sourceTargetColumnIndex;
  var delta;
  if (targetDescriptor.shape.dimensionCount >= 1) {
    if (isColumnsReduction) {
      sourceTargetColumnIndex =
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1;
      delta = targetDataInfo.dataRows;
    }
  }

  var lastTargetColumnIndex;
  if (targetDescriptor.shape.dimensionCount >= 1) {
    if (isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1];
    }
  } else {
    if (isRowsReduction && isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 0];
    } else if (isColumnsReduction) {
      lastTargetColumnIndex = sourcePermutedIndexes[
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1];
    }
  }

  var sourceRowIndex = sourceArray.dataInfo.rowIndex;
  var sourceColumnIndex = sourceArray.dataInfo.columnIndex;
  var targetColumnIndex = targetDataInfo.columnIndex;

  var permutedLastTargetColumnIndex = lastTargetColumnIndex != null
      ? sourcePermutedIndexes.indexOf(lastTargetColumnIndex)
      : null;
  var permutedSourceRowIndex = sourcePermutedIndexes.indexOf(sourceRowIndex);
  var permutedSourceColumnIndex =
      sourcePermutedIndexes.indexOf(sourceColumnIndex);
  var permutedTargetColumnIndex =
      targetPermutedIndexes.indexOf(targetColumnIndex);

  var sourceStride =
      permute(sourceArray.dataInfo.stride, sourcePermutedIndexes);
  var targetStride = permute(targetDataInfo.stride, targetPermutedIndexes);
  var sourceDimensions =
      permute(sourceArray.dataInfo.dimensions, sourcePermutedIndexes);
  var targetDimensions =
      permute(targetDataInfo.dimensions, targetPermutedIndexes);

  sourceDimensions[permutedSourceRowIndex] = sourceArray.dataInfo.rows;

  var sourceDimensionIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var sourceDataIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var targetDataIndexes = new List(targetDataInfo.dimensions.length);

  sourceDimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  var lastColumn =
      sourceArray.dataInfo.dimensions[sourceArray.dataInfo.columnIndex] == 1;
  var lastTargetColumn = isColumnsReduction &&
      targetDataInfo.dimensions[targetDataInfo.columnIndex] == 1;
  var isLastTargetColumnToCalculate = isColumnsReduction &&
      targetDataInfo.dimensions[targetDataInfo.columnIndex] > 1;

  final zero = targetDescriptor.dataType.isFloat ? _zeroFloat : _zeroInt;

  var columnValue;
  int columnIndex;

  if (isColumnsReduction) {
    columnValue = zero;
    columnIndex = 0;
  }

  if (targetDescriptor.shape.dimensionCount == 0 &&
      sourceArray.shape.dimensionCount > 1) {
    begin();
  }

  for (;;) {
    if (sourceDimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
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

          if (isLastTargetColumnToCalculate) {
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
              columnValue = columnValue.withFlagX(reducedValue);
              columnIndex++;
              break;
            case 1:
              columnValue = columnValue.withFlagY(reducedValue);
              columnIndex++;
              break;
            case 2:
              columnValue = columnValue.withFlagZ(reducedValue);
              columnIndex++;
              break;
            case 3:
              columnValue = columnValue.withFlagW(reducedValue);
              columnIndex++;
              break;
          }

          if (columnIndex ==
              (lastTargetColumn
                  ? targetDataInfo.lastBlockColumnCount
                  : targetDescriptor.dataType.blockSize)) {
            targetData[targetDataIndex] = columnValue;

            columnValue = zero;
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

        if (shapeIndex == sourceTargetColumnIndex) {
          if (sourceDimensionIndexes[shapeIndex] &
                  (sourceArray.dataType.blockSize - 1) ==
              0) {
            targetDataIndexes[shapeIndex] += delta;
          }
        }

        targetDataIndex = targetDataIndexes[shapeIndex];

        if (isLastTargetColumnToCalculate) {
          if (shapeIndex == permutedTargetColumnIndex) {
            if (lastTargetColumnIndex == sourceArray.dataInfo.rowIndex) {
              lastTargetColumn =
                  (sourceDimensionIndexes[permutedLastTargetColumnIndex] >>
                          sourceArray.dataType.blockDepth ==
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

void _argData(NDArrayBlockedImpl sourceArray, int axis, Int32x4List targetData,
    NDDescriptor targetDescriptor, DataInfo targetDataInfo,
    {void begin(),
    void onValue(dimensionIndex, value, int valueCount),
    dynamic end()}) {
  var axisSet = new Set<int>.from([axis]);

  bool isRowsReduction =
      axisSet.contains(sourceArray.descriptor.shape.dimensionCount - 2);

  bool isColumnsReduction =
      axisSet.contains(sourceArray.descriptor.shape.dimensionCount - 1);

  if (sourceArray.shape.dimensionCount == 1) {
    // skip
  } else if (isRowsReduction) {
    axisSet.remove(sourceArray.descriptor.shape.dimensionCount - 2);
    axisSet.add(sourceArray.descriptor.shape.dimensionCount - 1);
  } else if (isColumnsReduction) {
    axisSet.remove(sourceArray.descriptor.shape.dimensionCount - 1);
    axisSet.add(sourceArray.descriptor.shape.dimensionCount - 2);
  }

  var newReductionAxis = axisSet.toList(growable: false);

  List<int> targetPermutedIndexes =
      new List.generate(targetDataInfo.dimensions.length, (index) => index);

  if (targetDescriptor.shape.dimensionCount > 1) {
    if (isRowsReduction || isColumnsReduction) {
      var tempIndex = targetPermutedIndexes[targetPermutedIndexes.length - 1];
      targetPermutedIndexes[targetPermutedIndexes.length - 1] =
          targetPermutedIndexes[targetPermutedIndexes.length - 2];
      targetPermutedIndexes[targetPermutedIndexes.length - 2] = tempIndex;
    }
  }

  var sourcePermutedIndexes = new List.generate(
      sourceArray.dataInfo.dimensions.length, (index) => index);

  sourcePermutedIndexes =
      sourcePermutedIndexes.where((index) => !axisSet.contains(index)).toList();

  sourcePermutedIndexes.addAll(newReductionAxis);

  var targetBeginIndex =
      sourceArray.dataInfo.dimensions.length - newReductionAxis.length;

  var targetLimitIndex =
      isColumnsReduction ? targetBeginIndex - 1 : targetBeginIndex;

  var sourceTargetColumnIndex;
  var delta;
  if (targetDescriptor.shape.dimensionCount >= 1) {
    if (isColumnsReduction) {
      sourceTargetColumnIndex =
          sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1;
      delta = targetDataInfo.dataRows;
    }
  }

  var lastTargetColumnIndex = sourcePermutedIndexes[
      sourceArray.dataInfo.dimensions.length - newReductionAxis.length - 1];

  var sourceRowIndex = sourceArray.dataInfo.rowIndex;
  var sourceColumnIndex = sourceArray.dataInfo.columnIndex;
  var targetColumnIndex = targetDataInfo.columnIndex;

  var permutedLastTargetColumnIndex = lastTargetColumnIndex != null
      ? sourcePermutedIndexes.indexOf(lastTargetColumnIndex)
      : null;
  var permutedSourceRowIndex = sourcePermutedIndexes.indexOf(sourceRowIndex);
  var permutedSourceColumnIndex =
      sourcePermutedIndexes.indexOf(sourceColumnIndex);
  var permutedTargetColumnIndex =
      targetPermutedIndexes.indexOf(targetColumnIndex);

  var sourceStride =
      permute(sourceArray.dataInfo.stride, sourcePermutedIndexes);
  var targetStride = permute(targetDataInfo.stride, targetPermutedIndexes);
  var sourceDimensions =
      permute(sourceArray.dataInfo.dimensions, sourcePermutedIndexes);
  var targetDimensions =
      permute(targetDataInfo.dimensions, targetPermutedIndexes);

  sourceDimensions[permutedSourceRowIndex] = sourceArray.dataInfo.rows;

  var sourceDimensionIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var sourceDataIndexes = new List(sourceArray.dataInfo.dimensions.length);
  var targetDataIndexes = new List(targetDataInfo.dimensions.length);

  sourceDimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  var lastColumn =
      sourceArray.dataInfo.dimensions[sourceArray.dataInfo.columnIndex] == 1;
  var lastTargetColumn = isColumnsReduction &&
      targetDataInfo.dimensions[targetDataInfo.columnIndex] == 1;
  var isLastTargetColumnToCalculate = isColumnsReduction &&
      targetDataInfo.dimensions[targetDataInfo.columnIndex] > 1;

  var columnValue;
  int columnIndex;

  if (isColumnsReduction) {
    columnValue = _zeroInt;
    columnIndex = 0;
  }

  var sourceDimensionIndex;
  if (targetDescriptor.shape.dimensionCount == 0 &&
      sourceArray.shape.dimensionCount > 1) {
    begin();

    sourceDimensionIndex = 0;
  }

  for (;;) {
    if (sourceDimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
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

          if (isLastTargetColumnToCalculate) {
            if (targetPermutedIndexes[shapeIndex] ==
                targetDataInfo.columnIndex) {
              lastTargetColumn = false;
            }
          }
        }

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

            columnValue = _zeroInt;
            columnIndex = 0;
          }
        } else {
          var reducedValue = end();

          var valueCount = lastColumn
              ? sourceArray.dataInfo.lastBlockColumnCount
              : sourceArray.dataType.blockSize;

          switch (valueCount) {
            case 4:
              targetData[targetDataIndex] = new Int32x4(reducedValue.x,
                  reducedValue.y, reducedValue.z, reducedValue.w);
              break;
            case 3:
              targetData[targetDataIndex] = new Int32x4(
                  reducedValue.x, reducedValue.y, reducedValue.z, 0);
              break;
            case 2:
              targetData[targetDataIndex] =
                  new Int32x4(reducedValue.x, reducedValue.y, 0, 0);
              break;
            case 1:
              targetData[targetDataIndex] =
                  new Int32x4(reducedValue.x, 0, 0, 0);
              break;
          }
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

        if (shapeIndex == sourceTargetColumnIndex) {
          if (sourceDimensionIndexes[shapeIndex] &
                  (sourceArray.dataType.blockSize - 1) ==
              0) {
            targetDataIndexes[shapeIndex] += delta;
          }
        }

        targetDataIndex = targetDataIndexes[shapeIndex];

        if (isLastTargetColumnToCalculate) {
          if (shapeIndex == permutedTargetColumnIndex) {
            if (lastTargetColumnIndex == sourceArray.dataInfo.rowIndex) {
              lastTargetColumn =
                  (sourceDimensionIndexes[permutedLastTargetColumnIndex] >>
                          sourceArray.dataType.blockDepth ==
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

void _transposeData(
    NDArrayBlockedImpl sourceArray,
    List<int> permutationAxis,
    Float32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo) {
  var newPermutationAxis =
      permutationAxis.sublist(0, permutationAxis.length - 2);

  var sourceStride = permute(sourceArray.dataInfo.stride, newPermutationAxis);
  var sourceDimensions =
      permute(sourceArray.dataInfo.dimensions, newPermutationAxis);

  var dimensionIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var sourceDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
      if (shapeIndex < resultDescriptor.shape.dimensionCount - 2) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        for (var dimension1 = 0;
            dimension1 < sourceArray.dataInfo.blockColumns;
            dimension1++) {
          for (var dimension2 = 0;
              dimension2 < sourceArray.dataInfo.blockRows;
              dimension2++) {
            resultData[targetDataIndex++] = sourceArray.data[sourceDataIndex++];
            resultData[targetDataIndex++] = sourceArray.data[sourceDataIndex++];
            resultData[targetDataIndex++] = sourceArray.data[sourceDataIndex++];
            resultData[targetDataIndex++] = sourceArray.data[sourceDataIndex++];
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
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo) {
  switch (resultDescriptor.dataType) {
    case NDDataType.float32Blocked:
      return _transposeSwitchedDataFloat(sourceArray, permutationAxis,
          resultData, resultDescriptor, resultDataInfo);
    case NDDataType.int32Blocked:
      return _transposeSwitchedDataInt(sourceArray, permutationAxis, resultData,
          resultDescriptor, resultDataInfo);
    case NDDataType.booleanBlocked:
      return _transposeSwitchedDataBoolean(sourceArray, permutationAxis,
          resultData, resultDescriptor, resultDataInfo);
    default:
      throw new StateError("DEAD CODE");
  }
}

void _transposeSwitchedDataFloat(
    NDArrayBlockedImpl sourceArray,
    List<int> permutationAxis,
    Float32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo) {
  var delta1 = (sourceArray.dataInfo.blockColumns - 1) <<
      resultDescriptor.dataType.blockDepth;

  var delta2 = resultDescriptor.dataType.blockSize -
      sourceArray.dataInfo.matrixDataLength;

  var newPermutationAxis =
      permutationAxis.sublist(0, permutationAxis.length - 2);

  var sourceStride = permute(sourceArray.dataInfo.stride, newPermutationAxis);
  var sourceDimensions =
      permute(sourceArray.dataInfo.dimensions, newPermutationAxis);

  var dimensionIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var sourceDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
      if (shapeIndex < resultDescriptor.shape.dimensionCount - 2) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        for (var dimension1 = 0;
            dimension1 < sourceArray.dataInfo.blockColumns;
            dimension1++) {
          for (var dimension2 = 0;
              dimension2 < sourceArray.dataInfo.blockRows;
              dimension2++) {
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

void _transposeSwitchedDataInt(
    NDArrayBlockedImpl sourceArray,
    List<int> permutationAxis,
    Int32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo) {
  var delta1 = (sourceArray.dataInfo.blockColumns - 1) <<
      resultDescriptor.dataType.blockDepth;

  var delta2 = resultDescriptor.dataType.blockSize -
      sourceArray.dataInfo.matrixDataLength;

  var newPermutationAxis =
      permutationAxis.sublist(0, permutationAxis.length - 2);

  var sourceStride = permute(sourceArray.dataInfo.stride, newPermutationAxis);
  var sourceDimensions =
      permute(sourceArray.dataInfo.dimensions, newPermutationAxis);

  var dimensionIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var sourceDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
      if (shapeIndex < resultDescriptor.shape.dimensionCount - 2) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        for (var dimension1 = 0;
            dimension1 < sourceArray.dataInfo.blockColumns;
            dimension1++) {
          for (var dimension2 = 0;
              dimension2 < sourceArray.dataInfo.blockRows;
              dimension2++) {
            var a0 = sourceArray.data[sourceDataIndex++];
            var a1 = sourceArray.data[sourceDataIndex++];
            var a2 = sourceArray.data[sourceDataIndex++];
            var a3 = sourceArray.data[sourceDataIndex++];

            resultData[targetDataIndex++] = new Int32x4(a0.x, a1.x, a2.x, a3.x);
            resultData[targetDataIndex++] = new Int32x4(a0.y, a1.y, a2.y, a3.y);
            resultData[targetDataIndex++] = new Int32x4(a0.z, a1.z, a2.z, a3.z);
            resultData[targetDataIndex++] = new Int32x4(a0.w, a1.w, a2.w, a3.w);

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

void _transposeSwitchedDataBoolean(
    NDArrayBlockedImpl sourceArray,
    List<int> permutationAxis,
    Int32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo) {
  var delta1 = (sourceArray.dataInfo.blockColumns - 1) <<
      resultDescriptor.dataType.blockDepth;

  var delta2 = resultDescriptor.dataType.blockSize -
      sourceArray.dataInfo.matrixDataLength;

  var newPermutationAxis =
      permutationAxis.sublist(0, permutationAxis.length - 2);

  var sourceStride = permute(sourceArray.dataInfo.stride, newPermutationAxis);
  var sourceDimensions =
      permute(sourceArray.dataInfo.dimensions, newPermutationAxis);

  var dimensionIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var sourceDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);
  var targetDataIndexes = new List(resultDescriptor.shape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] < sourceDimensions[shapeIndex]) {
      if (shapeIndex < resultDescriptor.shape.dimensionCount - 2) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = sourceDataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        for (var dimension1 = 0;
            dimension1 < sourceArray.dataInfo.blockColumns;
            dimension1++) {
          for (var dimension2 = 0;
              dimension2 < sourceArray.dataInfo.blockRows;
              dimension2++) {
            Int32x4 a0 = sourceArray.data[sourceDataIndex++];
            Int32x4 a1 = sourceArray.data[sourceDataIndex++];
            Int32x4 a2 = sourceArray.data[sourceDataIndex++];
            Int32x4 a3 = sourceArray.data[sourceDataIndex++];

            resultData[targetDataIndex++] =
                new Int32x4.bool(a0.flagX, a1.flagX, a2.flagX, a3.flagX);
            resultData[targetDataIndex++] =
                new Int32x4.bool(a0.flagY, a1.flagY, a2.flagY, a3.flagY);
            resultData[targetDataIndex++] =
                new Int32x4.bool(a0.flagZ, a1.flagZ, a2.flagZ, a3.flagZ);
            resultData[targetDataIndex++] =
                new Int32x4.bool(a0.flagW, a1.flagW, a2.flagW, a3.flagW);

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

void _tileDataBlocked(NDArrayBlockedImpl fromArray, List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  var rowsMultiplier = resultDataInfo.rows ~/ fromArray.dataInfo.rows;

  var targetDelta =
      resultDescriptor.dataType.blockSize - resultDataInfo.lastBlockRowCount;

  var sourceDelta1 =
      fromArray.dataType.blockSize - fromArray.dataInfo.lastBlockRowCount;
  var sourceDelta2 = 1;

  var sourceDimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var targetDimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);

  var startSourceDataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var sourceDataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var targetDataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);

  sourceDimensionIndexes[0] = 0;
  targetDimensionIndexes[0] = 0;
  startSourceDataIndexes[0] = 0;
  sourceDataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var startSourceDataIndex = 0;
  var sourceDataIndex = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (targetDimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        sourceDataIndex = sourceDataIndexes[shapeIndex];
        startSourceDataIndex = sourceDataIndex;
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        sourceDimensionIndexes[shapeIndex] = 0;
        targetDimensionIndexes[shapeIndex] = 0;

        sourceDataIndexes[shapeIndex] = sourceDataIndex;
        startSourceDataIndexes[shapeIndex] = startSourceDataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        for (var column = 0;
            column < resultDataInfo.dataColumns;
            column += resultDescriptor.dataType.blockSize) {
          var startSourceDataIndex = sourceDataIndex;

          for (var iteration = 0; iteration < rowsMultiplier; iteration++) {
            sourceDataIndex = startSourceDataIndex;

            for (var row = 0; row < fromArray.dataInfo.rows; row++) {
              resultData[targetDataIndex] = fromArray.data[sourceDataIndex];

              sourceDataIndex += sourceDelta2;
              targetDataIndex++;
            }
          }

          sourceDataIndex += sourceDelta1;
          targetDataIndex += targetDelta;
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      sourceDimensionIndexes[shapeIndex]++;

      if (sourceDimensionIndexes[shapeIndex] <
          fromArray.dataInfo.internalShape[shapeIndex]) {
        sourceDataIndexes[shapeIndex] += fromArray.dataInfo.stride[shapeIndex];
      } else {
        sourceDimensionIndexes[shapeIndex] = 0;
        sourceDataIndexes[shapeIndex] = startSourceDataIndexes[shapeIndex];
      }

      targetDimensionIndexes[shapeIndex]++;
      targetDataIndexes[shapeIndex] += resultDataInfo.stride[shapeIndex];

      sourceDataIndex = sourceDataIndexes[shapeIndex];
      targetDataIndex = targetDataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

void _tileData(NDArrayBlockedImpl fromArray, List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  switch (resultDescriptor.dataType) {
    case NDDataType.float32Blocked:
      return _tileDataFloat(
          fromArray, resultData, resultDescriptor, resultDataInfo);
    case NDDataType.int32Blocked:
      return _tileDataInt(
          fromArray, resultData, resultDescriptor, resultDataInfo);
    case NDDataType.booleanBlocked:
      return _tileDataBoolean(
          fromArray, resultData, resultDescriptor, resultDataInfo);
    default:
      throw new StateError("DEAD CODE");
  }
}

void _tileDataFloat(NDArrayBlockedImpl fromArray, Float32x4List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var valueIterable =
      _createTiledValueIterable(fromArray, resultDescriptor, resultDataInfo);

  var valueIterator = valueIterable.iterator;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
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

void _tileDataInt(NDArrayBlockedImpl fromArray, Int32x4List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var valueIterable =
      _createTiledValueIterable(fromArray, resultDescriptor, resultDataInfo);

  var valueIterator = valueIterable.iterator;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Int32x4(
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
              value4 = new Int32x4(
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current);
              break;
            case 3:
              value4 = new Int32x4(
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  0);
              break;
            case 2:
              value4 = new Int32x4((valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current, 0, 0);
              break;
            case 1:
              value4 =
                  new Int32x4((valueIterator..moveNext()).current, 0, 0, 0);
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

void _tileDataBoolean(NDArrayBlockedImpl fromArray, Int32x4List resultData,
    NDDescriptor resultDescriptor, DataInfo resultDataInfo) {
  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var valueIterable =
      _createTiledValueIterable(fromArray, resultDescriptor, resultDataInfo);

  var valueIterator = valueIterable.iterator;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Int32x4.bool(
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
              value4 = new Int32x4.bool(
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current);
              break;
            case 3:
              value4 = new Int32x4.bool(
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current,
                  false);
              break;
            case 2:
              value4 = new Int32x4.bool((valueIterator..moveNext()).current,
                  (valueIterator..moveNext()).current, false, false);
              break;
            case 1:
              value4 = new Int32x4.bool(
                  (valueIterator..moveNext()).current, false, false, false);
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

// TODO rinminare in una specie di identity
void _elementWiseUnaryOperationData(
    NDArrayBlockedImpl sourceArray,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    unaryOperation(value, int valueCount)) {
  switch (resultDescriptor.dataType) {
    case NDDataType.float32Blocked:
      return _elementWiseUnaryOperationDataFloat(sourceArray, resultData,
          resultDescriptor, resultDataInfo, unaryOperation);
    case NDDataType.int32Blocked:
      return _elementWiseUnaryOperationDataInt(sourceArray, resultData,
          resultDescriptor, resultDataInfo, unaryOperation);
    case NDDataType.booleanBlocked:
      return _elementWiseUnaryOperationDataBoolean(sourceArray, resultData,
          resultDescriptor, resultDataInfo, unaryOperation);
    default:
      throw new StateError("DEAD CODE");
  }
}

void _elementWiseUnaryOperationDataFloat(
    NDArrayBlockedImpl sourceArray,
    Float32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    double unaryOperation(value, int valueCount)) {
  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var valueIterable = _createValueIterable(sourceArray);

  var valueIterator = valueIterable.iterator;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Float32x4(
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1));

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1));
              break;
            case 3:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  0.0);
              break;
            case 2:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  0.0,
                  0.0);
              break;
            case 1:
              value4 = new Float32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
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

void _elementWiseUnaryOperationDataInt(
    NDArrayBlockedImpl sourceArray,
    Int32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    int unaryOperation(value, int valueCount)) {
  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var valueIterable = _createValueIterable(sourceArray);

  var valueIterator = valueIterable.iterator;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Int32x4(
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1));

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1));
              break;
            case 3:
              value4 = new Int32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  0);
              break;
            case 2:
              value4 = new Int32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  0,
                  0);
              break;
            case 1:
              value4 = new Int32x4(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  0,
                  0,
                  0);
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

void _elementWiseUnaryOperationDataBoolean(
    NDArrayBlockedImpl sourceArray,
    Int32x4List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    bool unaryOperation(value, int valueCount)) {
  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var dataIndexes = new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  dataIndexes[0] = 0;

  var shapeIndex = 0;
  var dataIndex = 0;

  var valueIterable = _createValueIterable(sourceArray);

  var valueIterator = valueIterable.iterator;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        dataIndex = dataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        dataIndexes[shapeIndex] = dataIndex;

        continue;
      } else {
        for (var row = 0; row < resultDataInfo.rows; row++) {
          var column;
          for (column = 0;
              column < lastColumnIndex;
              column += resultDescriptor.dataType.blockSize) {
            var value4 = new Int32x4.bool(
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1),
                unaryOperation((valueIterator..moveNext()).current, 1));

            resultData[dataIndex] = value4;

            dataIndex += resultDataInfo.delta1;
          }

          var value4;
          switch (resultDataInfo.lastBlockColumnCount) {
            case 4:
              value4 = new Int32x4.bool(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1));
              break;
            case 3:
              value4 = new Int32x4.bool(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  false);
              break;
            case 2:
              value4 = new Int32x4.bool(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  false,
                  false);
              break;
            case 1:
              value4 = new Int32x4.bool(
                  unaryOperation((valueIterator..moveNext()).current, 1),
                  false,
                  false,
                  false);
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

void _elementWiseUnaryOperationDataBlocked(
    NDArrayBlockedImpl sourceArray,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    unaryOperation(value, int valueCount)) {
  var multiplier;
  if (resultDataInfo.internalShape.dimensionCount > 2) {
    multiplier = resultDataInfo.internalShape.length ~/
        (resultDataInfo.rows * resultDataInfo.columns);
  } else {
    multiplier = 1;
  }

  var delta1 =
      resultDescriptor.dataType.blockSize - resultDataInfo.lastBlockRowCount;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  var dataIndex = 0;

  for (var iteration = 0; iteration < multiplier; iteration++) {
    for (var column = 0;
        column < resultDataInfo.dataColumns;
        column += resultDescriptor.dataType.blockSize) {
      var columnCount = column < lastColumnIndex
          ? resultDescriptor.dataType.blockSize
          : resultDataInfo.lastBlockColumnCount;

      for (var row = 0; row < resultDataInfo.rows; row++) {
        resultData[dataIndex] =
            unaryOperation(sourceArray.data[dataIndex], columnCount);

        dataIndex++;
      }

      dataIndex += delta1;
    }
  }
}

void _elementWiseBinaryOperationDataBlocked(
    NDArrayBlockedImpl array1,
    NDArrayBlockedImpl array2,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    binaryOperation(value1, value2, int valueCount)) {
  var source1Stride =
      _calculateBroadcastedStride(resultDataInfo.internalShape, array1);
  var source2Stride =
      _calculateBroadcastedStride(resultDataInfo.internalShape, array2);
  var targetStride = new List.from(resultDataInfo.stride
      .sublist(0, resultDataInfo.internalShape.dimensionCount - 2));

  var isSource1BroadcastedRows = source1Stride[source1Stride.length - 2] == 0;
  var isSource1BroadcastedColumns =
      source1Stride[source1Stride.length - 1] == 0;

  var isSource2BroadcastedRows = source2Stride[source2Stride.length - 2] == 0;
  var isSource2BroadcastedColumns =
      source2Stride[source2Stride.length - 1] == 0;

  var targetDelta =
      resultDescriptor.dataType.blockSize - resultDataInfo.lastBlockRowCount;

  int source1Delta1 = isSource1BroadcastedRows
      ? (isSource1BroadcastedColumns ? 0 : array1.dataType.blockSize)
      : (isSource1BroadcastedColumns ? -array1.dataInfo.rows : targetDelta);
  int source1Delta2 = isSource1BroadcastedRows ? 0 : 1;

  int source2Delta1 = isSource2BroadcastedRows
      ? (isSource2BroadcastedColumns ? 0 : array2.dataType.blockSize)
      : (isSource2BroadcastedColumns ? -array2.dataInfo.rows : targetDelta);
  int source2Delta2 = isSource2BroadcastedRows ? 0 : 1;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);

  var source1DataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var source2DataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var targetDataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  source1DataIndexes[0] = 0;
  source2DataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var source1DataIndex = 0;
  var source2DataIndex = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        source1DataIndex = source1DataIndexes[shapeIndex];
        source2DataIndex = source2DataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        source1DataIndexes[shapeIndex] = source1DataIndex;
        source2DataIndexes[shapeIndex] = source2DataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        for (var column = 0;
            column < resultDataInfo.dataColumns;
            column += resultDescriptor.dataType.blockSize) {
          var columnCount = column < lastColumnIndex
              ? resultDescriptor.dataType.blockSize
              : resultDataInfo.lastBlockColumnCount;

          for (var row = 0; row < resultDataInfo.rows; row++) {
            var value1 = array1.data[source1DataIndex];
            var value2 = array2.data[source2DataIndex];

            if (isSource1BroadcastedColumns) {
              switch (columnCount) {
                case 4:
                  value1 = new Float32x4.splat(value1.x);
                  break;
                case 3:
                  value1 = new Float32x4(value1.x, value1.x, value1.x, 0.0);
                  break;
                case 2:
                  value1 = new Float32x4(value1.x, value1.x, 0.0, 0.0);
                  break;
                case 1:
                  value1 = new Float32x4(value1.x, 0.0, 0.0, 0.0);
                  break;
              }
            }

            if (isSource2BroadcastedColumns) {
              switch (columnCount) {
                case 4:
                  value2 = new Float32x4.splat(value2.x);
                  break;
                case 3:
                  value2 = new Float32x4(value2.x, value2.x, value2.x, 0.0);
                  break;
                case 2:
                  value2 = new Float32x4(value2.x, value2.x, 0.0, 0.0);
                  break;
                case 1:
                  value2 = new Float32x4(value2.x, 0.0, 0.0, 0.0);
                  break;
              }
            }

            resultData[targetDataIndex] =
                binaryOperation(value1, value2, columnCount);

            source1DataIndex += source1Delta2;
            source2DataIndex += source2Delta2;
            targetDataIndex++;
          }

          source1DataIndex += source1Delta1;
          source2DataIndex += source2Delta1;
          targetDataIndex += targetDelta;
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      source1DataIndexes[shapeIndex] += source1Stride[shapeIndex];
      source2DataIndexes[shapeIndex] += source2Stride[shapeIndex];
      targetDataIndexes[shapeIndex] += targetStride[shapeIndex];

      source1DataIndex = source1DataIndexes[shapeIndex];
      source2DataIndex = source2DataIndexes[shapeIndex];
      targetDataIndex = targetDataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

void _elementWiseTernaryOperationDataBlocked(
    NDArrayBlockedImpl array1,
    NDArrayBlockedImpl array2,
    NDArrayBlockedImpl array3,
    List resultData,
    NDDescriptor resultDescriptor,
    DataInfo resultDataInfo,
    ternaryOperation(value1, value2, value3, int valueCount)) {
  var source1Stride =
      _calculateBroadcastedStride(resultDataInfo.internalShape, array1);
  var source2Stride =
      _calculateBroadcastedStride(resultDataInfo.internalShape, array2);
  var source3Stride =
      _calculateBroadcastedStride(resultDataInfo.internalShape, array3);
  var targetStride = new List.from(resultDataInfo.stride
      .sublist(0, resultDataInfo.internalShape.dimensionCount - 2));

  var isSource1BroadcastedRows = source1Stride[source1Stride.length - 2] == 0;
  var isSource1BroadcastedColumns =
      source1Stride[source1Stride.length - 1] == 0;

  var isSource2BroadcastedRows = source2Stride[source2Stride.length - 2] == 0;
  var isSource2BroadcastedColumns =
      source2Stride[source2Stride.length - 1] == 0;

  var isSource3BroadcastedRows = source3Stride[source3Stride.length - 2] == 0;
  var isSource3BroadcastedColumns =
      source3Stride[source3Stride.length - 1] == 0;

  var targetDelta =
      resultDescriptor.dataType.blockSize - resultDataInfo.lastBlockRowCount;

  int source1Delta1 = isSource1BroadcastedRows
      ? (isSource1BroadcastedColumns ? 0 : array1.dataType.blockSize)
      : (isSource1BroadcastedColumns ? -array1.dataInfo.rows : targetDelta);
  int source1Delta2 = isSource1BroadcastedRows ? 0 : 1;

  int source2Delta1 = isSource2BroadcastedRows
      ? (isSource2BroadcastedColumns ? 0 : array2.dataType.blockSize)
      : (isSource2BroadcastedColumns ? -array2.dataInfo.rows : targetDelta);
  int source2Delta2 = isSource2BroadcastedRows ? 0 : 1;

  int source3Delta1 = isSource3BroadcastedRows
      ? (isSource3BroadcastedColumns ? 0 : array3.dataType.blockSize)
      : (isSource3BroadcastedColumns ? -array3.dataInfo.rows : targetDelta);
  int source3Delta2 = isSource3BroadcastedRows ? 0 : 1;

  var lastColumnIndex =
      resultDataInfo.dataColumns - resultDescriptor.dataType.blockSize;

  var dimensionIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);

  var source1DataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var source2DataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var source3DataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);
  var targetDataIndexes =
      new List(resultDataInfo.internalShape.dimensionCount - 1);

  dimensionIndexes[0] = 0;
  source1DataIndexes[0] = 0;
  source2DataIndexes[0] = 0;
  source3DataIndexes[0] = 0;
  targetDataIndexes[0] = 0;

  var shapeIndex = 0;
  var source1DataIndex = 0;
  var source2DataIndex = 0;
  var source3DataIndex = 0;
  var targetDataIndex = 0;

  for (;;) {
    if (dimensionIndexes[shapeIndex] <
        resultDataInfo.internalShape[shapeIndex]) {
      if (shapeIndex < resultDataInfo.internalShape.dimensionCount - 2) {
        source1DataIndex = source1DataIndexes[shapeIndex];
        source2DataIndex = source2DataIndexes[shapeIndex];
        source3DataIndex = source3DataIndexes[shapeIndex];
        targetDataIndex = targetDataIndexes[shapeIndex];

        shapeIndex++;

        dimensionIndexes[shapeIndex] = 0;

        source1DataIndexes[shapeIndex] = source1DataIndex;
        source2DataIndexes[shapeIndex] = source2DataIndex;
        source3DataIndexes[shapeIndex] = source3DataIndex;
        targetDataIndexes[shapeIndex] = targetDataIndex;

        continue;
      } else {
        for (var column = 0;
            column < resultDataInfo.dataColumns;
            column += resultDescriptor.dataType.blockSize) {
          var columnCount = column < lastColumnIndex
              ? resultDescriptor.dataType.blockSize
              : resultDataInfo.lastBlockColumnCount;

          for (var row = 0; row < resultDataInfo.rows; row++) {
            var value1 = array1.data[source1DataIndex];
            var value2 = array2.data[source2DataIndex];
            var value3 = array3.data[source3DataIndex];

            if (isSource1BroadcastedColumns) {
              switch (columnCount) {
                case 4:
                  value1 = new Float32x4.splat(value1.x);
                  break;
                case 3:
                  value1 = new Float32x4(value1.x, value1.x, value1.x, 0.0);
                  break;
                case 2:
                  value1 = new Float32x4(value1.x, value1.x, 0.0, 0.0);
                  break;
                case 1:
                  value1 = new Float32x4(value1.x, 0.0, 0.0, 0.0);
                  break;
              }
            }

            if (isSource2BroadcastedColumns) {
              switch (columnCount) {
                case 4:
                  value2 = new Float32x4.splat(value2.x);
                  break;
                case 3:
                  value2 = new Float32x4(value2.x, value2.x, value2.x, 0.0);
                  break;
                case 2:
                  value2 = new Float32x4(value2.x, value2.x, 0.0, 0.0);
                  break;
                case 1:
                  value2 = new Float32x4(value2.x, 0.0, 0.0, 0.0);
                  break;
              }
            }

            if (isSource3BroadcastedColumns) {
              switch (columnCount) {
                case 4:
                  value3 = new Float32x4.splat(value3.x);
                  break;
                case 3:
                  value3 = new Float32x4(value3.x, value3.x, value3.x, 0.0);
                  break;
                case 2:
                  value3 = new Float32x4(value3.x, value3.x, 0.0, 0.0);
                  break;
                case 1:
                  value3 = new Float32x4(value3.x, 0.0, 0.0, 0.0);
                  break;
              }
            }

            resultData[targetDataIndex] =
                ternaryOperation(value1, value2, value3, columnCount);

            source1DataIndex += source1Delta2;
            source2DataIndex += source2Delta2;
            source3DataIndex += source3Delta2;
            targetDataIndex++;
          }

          source1DataIndex += source1Delta1;
          source2DataIndex += source2Delta1;
          source3DataIndex += source3Delta1;
          targetDataIndex += targetDelta;
        }

        shapeIndex--;
      }
    } else {
      shapeIndex--;
    }

    if (shapeIndex >= 0) {
      dimensionIndexes[shapeIndex]++;

      source1DataIndexes[shapeIndex] += source1Stride[shapeIndex];
      source2DataIndexes[shapeIndex] += source2Stride[shapeIndex];
      source3DataIndexes[shapeIndex] += source3Stride[shapeIndex];
      targetDataIndexes[shapeIndex] += targetStride[shapeIndex];

      source1DataIndex = source1DataIndexes[shapeIndex];
      source2DataIndex = source2DataIndexes[shapeIndex];
      source3DataIndex = source3DataIndexes[shapeIndex];
      targetDataIndex = targetDataIndexes[shapeIndex];
    } else {
      break;
    }
  }
}

List<int> _calculateBroadcastedStride(
    NDShape broadcastedShape, NDArrayBlockedImpl array) {
  var dimensionDelta =
      broadcastedShape.dimensionCount - array.shape.dimensionCount;

  return new List.generate(broadcastedShape.dimensionCount, (index) {
    if (index < dimensionDelta || array.shape[index - dimensionDelta] == 1) {
      return 0;
    } else {
      return array.dataInfo.stride[index - dimensionDelta];
    }
  });
}
