// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:typed_data";

import "package:collection/collection.dart";

import 'nd_descriptor.dart';
import 'nd_shape.dart';
import 'nd_data_type.dart';
import "nd_array.dart";

import "nd_array_base.dart";

import "nd_util.dart";

class NDArrayImpl extends NDArrayBase {
  final List _data;

  final DataInfo _dataInfo;

  factory NDArrayImpl(value, NDDescriptor descriptor, NDArray reuse) {
    var dataInfo = new DataInfo.normalized(descriptor);

    var data = createData(descriptor, reuse);

    _loadData(value, data, descriptor);

    return new NDArrayImpl.raw(data, descriptor, dataInfo);
  }

  factory NDArrayImpl.filled(
          fillValue, NDDescriptor descriptor, NDArray reuse) =>
      new NDArrayImpl.generate((index) => fillValue, descriptor, reuse);

  factory NDArrayImpl.generate(
      generator(int index), NDDescriptor descriptor, NDArray reuse) {
    var dataInfo = new DataInfo.normalized(descriptor);

    var data = createData(descriptor, reuse);

    _generateData(generator, data, descriptor);

    return new NDArrayImpl.raw(data, descriptor, dataInfo);
  }

  factory NDArrayImpl.castFrom(
      NDArrayBase fromArray, NDDataType toDataType, NDArray reuse) {
    var resultDescriptor = fromArray.descriptor.cast(toDataType);

    var dataInfo = new DataInfo.normalized(resultDescriptor);

    var data = createData(resultDescriptor, reuse);

    _castData(fromArray, data, resultDescriptor);

    return new NDArrayImpl.raw(data, resultDescriptor, dataInfo);
  }

  NDArrayImpl.raw(this._data, NDDescriptor descriptor, this._dataInfo)
      : super.raw(descriptor);

  @override
  Iterable get valueIterable =>
      _createValueIterable(_data, descriptor, _dataInfo);

  @override
  dynamic toValue() {
    if (shape.isScalar) {
      return _data[_dataInfo.offset];
    } else {
      var shapeIndex = 0;
      var dimensionValues = new List(shape.dimensionCount);
      var dimensionIndexes = new List(shape.dimensionCount);
      var dataIndexes = new List(shape.dimensionCount);
      var dataIndex = dataIndexes[shapeIndex] = _dataInfo.offset;
      var resultValues =
          dimensionValues[shapeIndex] = new List(shape[shapeIndex]);
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      var i = 0;
      while (i < shape.length) {
        if (dimensionIndex < shape[shapeIndex]) {
          if (shapeIndex == shape.dimensionCount - 1) {
            var axeDimension = shape[shapeIndex];
            var axeStride = _dataInfo.stride[shapeIndex];
            while (dimensionIndex < axeDimension) {
              resultValues[dimensionIndex++] = _data[dataIndex];
              dataIndex += axeStride;
              i++;
            }
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
              dataIndexes[shapeIndex] + _dataInfo.stride[shapeIndex];
          resultValues = dimensionValues[shapeIndex];
          dimensionIndex =
              dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        }
      }
      return dimensionValues[0];
    }
  }

  @override
  bool get isNormalized => _dataInfo == new DataInfo.normalized(descriptor);

  @override
  NDArray normalize({NDArray reuse}) {
    if (isNormalized) {
      return this;
    } else {
      var resultDescriptor = descriptor;

      var resultDataInfo = new DataInfo.normalized(resultDescriptor);

      var resultData = elementWiseUnaryOperationInternal(
          resultDescriptor, reuse, (value) => value)._data;

      return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
    }
  }

  @override
  NDArray reshape({List<int> newDimensions, NDArray reuse}) {
    if (_iterableEquality.equals(shape.dimensions, newDimensions)) {
      return this;
    } else {
      var resultDescriptor = descriptor.reshape(newDimensions: newDimensions);

      var resultDataInfo = new DataInfo.normalized(resultDescriptor);

      var resultData;
      if (isNormalized) {
        resultData = _data;
      } else {
        resultData = elementWiseUnaryOperationInternal(
            resultDescriptor, reuse, (value) => value)._data;
      }

      return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
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
              shape.dimensionCount, (index) => shape.dimensionCount - index - 1);

      var resultStride = new List(shape.dimensionCount);

      for (var i = 0; i < newPermutationAxis.length; i++) {
        var permutationAxe = newPermutationAxis[i];
        resultStride[i] = _dataInfo.stride[permutationAxe];
      }

      var resultDataInfo = new DataInfo(resultStride, _dataInfo.offset);

      return new NDArrayImpl.raw(_data, resultDescriptor, resultDataInfo);
    } else {
      return this;
    }
  }

  @override
  NDArray tile(List<int> multiplies, {NDArray reuse}) {
    var resultDescriptor = descriptor.tile(multiplies);

    if (descriptor != resultDescriptor) {
      var resultShape = resultDescriptor.shape;
      var resultData = createData(resultDescriptor, reuse);
      var resultDataInfo = new DataInfo.normalized(resultDescriptor);

      var shapeIndex = 0;
      var dimensionIndexes = new List(resultShape.dimensionCount);
      var dimensionSourceIndexes = new List(resultShape.dimensionCount);
      var data1Indexes = new List(resultShape.dimensionCount);
      var startData1Indexes = new List(resultShape.dimensionCount);
      var stride1 = _dataInfo.stride;
      var data1Index = data1Indexes[shapeIndex] = _dataInfo.offset;
      var startData1Index = startData1Indexes[shapeIndex] = data1Index;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      var dimensionSourceIndex = dimensionSourceIndexes[shapeIndex] = 0;
      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < resultShape[shapeIndex]) {
          if (shapeIndex == resultShape.dimensionCount - 1) {
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
            data1Index =
                data1Indexes[shapeIndex] = startData1Indexes[shapeIndex];
            dimensionSourceIndex = dimensionSourceIndexes[shapeIndex] = 0;
          }
        }
      }

      return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
    } else {
      return this;
    }
  }

  @override
  NDArray matMul(value2, {NDArray reuse}) {
    NDArrayImpl array2 = toNDArray(value2, dataType: dataType);

    var resultDescriptor = descriptor.matMul(array2.descriptor);
    var resultShape = resultDescriptor.shape;
    var resultData = createData(resultDescriptor, reuse);
    var resultDataInfo = new DataInfo.normalized(resultDescriptor);

    var shapeIndex = 0;
    var dimensionIndexes = new List(resultShape.dimensionCount);
    var data1Indexes = new List(resultShape.dimensionCount);
    var data1Index = data1Indexes[shapeIndex] = _dataInfo.offset;
    var data2Indexes = new List(resultShape.dimensionCount);
    var data2Index = data2Indexes[shapeIndex] = array2._dataInfo.offset;
    var resultDataIndex = 0;
    var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
    while (resultDataIndex < resultData.length) {
      if (dimensionIndex < resultShape[shapeIndex]) {
        if (shapeIndex == resultShape.dimensionCount - 2) {
          var shapeDimension1 = shape[shapeIndex];
          var shapeDimensionInternal = shape[shapeIndex + 1];
          var shapeDimension2 = array2.shape[shapeIndex + 1];
          var stride1 = _dataInfo.stride[shapeIndex];
          var stride1Internal = _dataInfo.stride[shapeIndex + 1];
          var stride2Internal = array2._dataInfo.stride[shapeIndex];
          var stride2 = array2._dataInfo.stride[shapeIndex + 1];

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
              while (innerIndex < shapeDimensionInternal) {
                sumValue += _data[data1Index] * array2._data[data2Index];
                innerIndex++;
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
            data1Indexes[shapeIndex] + _dataInfo.stride[shapeIndex];
        data2Index = data2Indexes[shapeIndex] =
            data2Indexes[shapeIndex] + array2._dataInfo.stride[shapeIndex];
        dimensionIndex =
            dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
      }
    }

    return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
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
          total = 0;
        },
        onValue: (value, int valueCount) {
          total += value;
        },
        end: () => total);
  }

  @override
  NDArray reduceMean(
      {List<int> reductionAxis, bool keepDimensions = false, NDArray reuse}) {
    var resultDescriptor = descriptor.reduceMean(
        reductionAxis: reductionAxis, keepDimensions: keepDimensions);

    var newReductionAxis =
        convertToValidReductionAxis(reductionAxis, shape.dimensionCount);

    var total;
    var count = newReductionAxis.fold<int>(
        1, (count, reductionIndex) => count * shape.dimensions[reductionIndex]);

    return reduceOperationInternal(
        newReductionAxis, keepDimensions, resultDescriptor, reuse,
        begin: () {
          total = 0;
        },
        onValue: (value, int valueCount) {
          total += value;
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
    var resultDescriptor = descriptor.argMax(axis: axis);

    var maxValueIndex;
    var maxValue;

    return argOperationInternal(axis, resultDescriptor, reuse,
        begin: () {
          maxValueIndex = null;
          maxValue = null;
        },
        onValue: (dimensionIndex, value, int valueCount) {
          if (maxValue == null || value > maxValue) {
            maxValueIndex = dimensionIndex;
            maxValue = value;
          }
        },
        end: () => maxValueIndex);
  }

  @override
  String toString() =>
      "<value: ${toValue()}, shape: $shape, dataType: $dataType, stride: $_dataInfo.stride, offset: $_dataInfo.offset>";

  @override
  NDArrayImpl elementWiseUnaryOperationInternal(
      NDDescriptor resultDescriptor, NDArray reuse, unaryOperation(value)) {
    var resultData = createData(resultDescriptor, reuse);
    var resultDataInfo;

    if (shape.isScalar) {
      resultDataInfo = new DataInfo(_dataInfo.stride, 0);

      resultData[0] = unaryOperation(_data[_dataInfo.offset]);
    } else {
      resultDataInfo = new DataInfo.normalized(resultDescriptor);

      var shapeIndex = 0;
      var dimensionIndexes = new List(shape.dimensionCount);
      var dataIndexes = new List(shape.dimensionCount);
      var dataIndex = dataIndexes[shapeIndex] = _dataInfo.offset;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < shape[shapeIndex]) {
          if (shapeIndex == shape.dimensionCount - 1) {
            var axeDimension = shape[shapeIndex];
            var axeStride = _dataInfo.stride[shapeIndex];
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
              dataIndexes[shapeIndex] + _dataInfo.stride[shapeIndex];
          dimensionIndex =
              dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
        }
      }
    }

    return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayImpl elementWiseBinaryOperationInternal(
      covariant NDArrayImpl array2,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      binaryOperation(value1, value2)) {
    var resultShape = resultDescriptor.shape;
    var resultData = createData(resultDescriptor, reuse);
    var resultDataInfo;

    if (resultShape.isScalar) {
      resultDataInfo = new DataInfo(_dataInfo.stride, 0);

      resultData[0] = binaryOperation(
          _data[_dataInfo.offset], array2._data[array2._dataInfo.offset]);
    } else {
      resultDataInfo = new DataInfo.normalized(resultDescriptor);

      var shapeIndex = 0;
      var dimensionIndexes = new List(resultShape.dimensionCount);
      var data1Indexes = new List(resultShape.dimensionCount);
      var stride1 = _calculateBroadcastedStride(resultShape, this);
      var data1Index = data1Indexes[shapeIndex] = _dataInfo.offset;
      var data2Indexes = new List(resultShape.dimensionCount);
      var stride2 = _calculateBroadcastedStride(resultShape, array2);
      var data2Index = data2Indexes[shapeIndex] = array2._dataInfo.offset;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < resultShape[shapeIndex]) {
          if (shapeIndex == resultShape.dimensionCount - 1) {
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

    return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArrayImpl elementWiseTernaryOperationInternal(
      covariant NDArrayImpl array2,
      covariant NDArrayImpl array3,
      NDDescriptor resultDescriptor,
      NDArray reuse,
      ternaryOperation(value1, value2, value3)) {
    var resultShape = resultDescriptor.shape;
    var resultData = createData(resultDescriptor, reuse);
    var resultDataInfo;

    if (resultShape.isScalar) {
      resultDataInfo = new DataInfo(_dataInfo.stride, 0);

      resultData[0] = ternaryOperation(
          _data[_dataInfo.offset],
          array2._data[array2._dataInfo.offset],
          array3._data[array3._dataInfo.offset]);
    } else {
      resultDataInfo = new DataInfo.normalized(resultDescriptor);

      var shapeIndex = 0;
      var dimensionIndexes = new List(resultShape.dimensionCount);
      var data1Indexes = new List(resultShape.dimensionCount);
      var stride1 = _calculateBroadcastedStride(resultShape, this);
      var data1Delta = stride1[shapeIndex];
      var data1Index = data1Indexes[shapeIndex] = _dataInfo.offset;
      var data2Indexes = new List(resultShape.dimensionCount);
      var stride2 = _calculateBroadcastedStride(resultShape, array2);
      var data2Delta = stride2[shapeIndex];
      var data2Index = data2Indexes[shapeIndex] = array2._dataInfo.offset;
      var data3Indexes = new List(resultShape.dimensionCount);
      var stride3 = _calculateBroadcastedStride(resultShape, array3);
      var data3Delta = stride3[shapeIndex];
      var data3Index = data3Indexes[shapeIndex] = array3._dataInfo.offset;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < resultShape[shapeIndex]) {
          if (shapeIndex == resultShape.dimensionCount - 1) {
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

    return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
  }

  @override
  NDArray reduceOperationInternal(List<int> reductionAxis, bool keepDimensions,
      NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(), void onValue(value, int valueCount), dynamic end()}) {
    var newReductionAxis =
        convertToValidReductionAxis(reductionAxis, shape.dimensionCount);

    if (newReductionAxis.isNotEmpty) {
      var resultData = createData(resultDescriptor, reuse);
      var resultDataInfo = new DataInfo.normalized(resultDescriptor);

      var shapeIndex = 0;
      var permutedIndexes = new List(shape.dimensionCount);
      var axis = new Set.from(newReductionAxis);
      var resultIndex = 0;
      for (var i = 0; i < shape.dimensionCount; i++) {
        if (!axis.contains(i)) {
          permutedIndexes[resultIndex++] = i;
        }
      }
      for (var i = 0; i < newReductionAxis.length; i++) {
        permutedIndexes[resultIndex++] = newReductionAxis[i];
      }

      var stride = permute(_dataInfo.stride, permutedIndexes);
      var dimensions = permute(shape.dimensions, permutedIndexes);

      var dimensionIndexes = new List(shape.dimensionCount);
      var dataIndexes = new List(shape.dimensionCount);
      var dataIndex = dataIndexes[0] = _dataInfo.offset;
      var resultDataIndex = 0;
      var dimensionIndex = dimensionIndexes[0] = 0;

      if (resultDescriptor.shape.dimensionCount == 0) {
        begin();
      }

      while (resultDataIndex < resultData.length) {
        if (dimensionIndex < dimensions[shapeIndex]) {
          if (shapeIndex == shape.dimensionCount - newReductionAxis.length - 1) {
            begin();
          }

          if (shapeIndex == shape.dimensionCount - 1) {
            onValue(_data[dataIndex], 1);

            dataIndex += stride[shapeIndex];
            dimensionIndex++;
          } else {
            shapeIndex++;
            dataIndexes[shapeIndex] = dataIndex;
            dimensionIndexes[shapeIndex] = 0;
            dimensionIndex = 0;
          }
        } else {
          shapeIndex--;

          if (shapeIndex == shape.dimensionCount - newReductionAxis.length - 1) {
            resultData[resultDataIndex++] = end();
          }

          if (shapeIndex >= 0) {
            dataIndexes[shapeIndex] += stride[shapeIndex];
            dataIndex = dataIndexes[shapeIndex];

            dimensionIndexes[shapeIndex]++;
            dimensionIndex = dimensionIndexes[shapeIndex];
          }
        }
      }

      return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
    } else {
      return this;
    }
  }

  @override
  NDArray argOperationInternal(
      axis, NDDescriptor resultDescriptor, NDArray reuse,
      {void begin(),
      void onValue(dimensionIndex, value, int valueCount),
      dynamic end()}) {
    var resultData = createData(resultDescriptor, reuse);
    var resultDataInfo = new DataInfo.normalized(resultDescriptor);

    var shapeIndex = 0;
    var permutedIndexes = new List(shape.dimensionCount);
    var resultIndex = 0;
    for (var i = 0; i < shape.dimensionCount; i++) {
      if (i != axis) {
        permutedIndexes[resultIndex++] = i;
      }
    }
    permutedIndexes[resultIndex++] = axis;

    var stride = permute(_dataInfo.stride, permutedIndexes);
    var dimensions = permute(shape.dimensions, permutedIndexes);

    var dimensionIndexes = new List(shape.dimensionCount);
    var dataIndexes = new List(shape.dimensionCount);
    var dataIndex = dataIndexes[0] = _dataInfo.offset;
    var resultDataIndex = 0;
    var dimensionIndex = dimensionIndexes[0] = 0;

    if (resultDescriptor.shape.dimensionCount == 0) {
      begin();
    }

    while (resultDataIndex < resultData.length) {
      if (dimensionIndex < dimensions[shapeIndex]) {
        if (shapeIndex == shape.dimensionCount - 2) {
          begin();
        }

        if (shapeIndex == shape.dimensionCount - 1) {
          onValue(dimensionIndex, _data[dataIndex], 1);

          dataIndex += stride[shapeIndex];
          dimensionIndex++;
        } else {
          shapeIndex++;
          dataIndexes[shapeIndex] = dataIndex;
          dimensionIndexes[shapeIndex] = 0;
          dimensionIndex = 0;
        }
      } else {
        shapeIndex--;

        if (shapeIndex == shape.dimensionCount - 2) {
          resultData[resultDataIndex++] = end();
        }

        if (shapeIndex >= 0) {
          dataIndexes[shapeIndex] += stride[shapeIndex];
          dataIndex = dataIndexes[shapeIndex];

          dimensionIndexes[shapeIndex]++;
          dimensionIndex = dimensionIndexes[shapeIndex];
        }
      }
    }

    return new NDArrayImpl.raw(resultData, resultDescriptor, resultDataInfo);
  }
}

final _iterableEquality = new IterableEquality<dynamic>();

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
    var dimensionValues = new List(shape.dimensionCount);
    var dimensionIndexes = new List(shape.dimensionCount);
    var shapeIndex = 0;
    var dataIndex = 0;
    var dimensionValue = dimensionValues[0] = value;
    var dimensionIndex = dimensionIndexes[0] = 0;
    while (dataIndex < data.length) {
      if (dimensionIndex < shape[shapeIndex]) {
        if (shapeIndex == shape.dimensionCount - 1) {
          var axeDimension = shape[shapeIndex];
          while (dimensionIndex < axeDimension) {
            data[dataIndex++] = converter(dimensionValue[dimensionIndex++]);
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
}

void _generateData(generator(int index), List data, NDDescriptor descriptor) {
  if (descriptor.dataType.isFloat) {
    _generateConvertedData(
        generator, data, descriptor, (num value) => value.toDouble());
  } else if (descriptor.dataType.isInteger) {
    _generateConvertedData(
        generator, data, descriptor, (num value) => value.toInt());
  } else {
    _generateConvertedData(generator, data, descriptor, (value) => value);
  }
}

void _generateConvertedData(generator(int index), List data,
    NDDescriptor descriptor, dynamic converter(value)) {
  for (var index = 0; index < descriptor.shape.length; index++) {
    data[index] = converter(generator(index));
  }
}

void _castData(NDArrayBase fromArray, List data, NDDescriptor descriptor) {
  if ((fromArray.dataType.isFloat && descriptor.dataType.isFloat) ||
      (fromArray.dataType.isInteger && descriptor.dataType.isInteger)) {
    return _castConvertedData(
        fromArray, data, descriptor, (num value) => value);
  } else if (fromArray.dataType.isFloat && descriptor.dataType.isInteger) {
    return _castConvertedData(
        fromArray, data, descriptor, (double value) => value.toInt());
  } else if (fromArray.dataType.isInteger && descriptor.dataType.isFloat) {
    return _castConvertedData(
        fromArray, data, descriptor, (int value) => value.toDouble());
  } else if (fromArray.dataType.isNumeric && descriptor.dataType.isBoolean) {
    return _castConvertedData(
        fromArray, data, descriptor, (num value) => value != 0);
  } else if (fromArray.dataType.isBoolean && descriptor.dataType.isFloat) {
    return _castConvertedData(
        fromArray, data, descriptor, (bool value) => value ? 1.0 : 0.0);
  } else if (fromArray.dataType.isBoolean && descriptor.dataType.isInteger) {
    return _castConvertedData(
        fromArray, data, descriptor, (bool value) => value ? 1 : 0);
  } else {
    throw new StateError("DEAD CODE");
  }
}

void _castConvertedData(NDArrayBase fromArray, List data,
    NDDescriptor descriptor, dynamic converter(value)) {
  var valueIterator = fromArray.valueIterable.iterator;

  var shape = descriptor.shape;
  if (shape.isScalar) {
    data[0] = converter((valueIterator..moveNext()).current);
  } else {
    var dataIndex = 0;
    while (valueIterator.moveNext()) {
      data[dataIndex++] = converter(valueIterator.current);
    }
  }
}

List<int> _calculateBroadcastedStride(
        NDShape broadcastedShape, NDArrayImpl array) =>
    new List.generate(broadcastedShape.dimensionCount, (index) {
      var dimensionDelta = broadcastedShape.dimensionCount - array.shape.dimensionCount;
      if (index < dimensionDelta || array.shape[index - dimensionDelta] == 1) {
        return 0;
      } else {
        return array._dataInfo.stride[index - dimensionDelta];
      }
    }, growable: false);

List createData(NDDescriptor descriptor, NDArrayImpl reuse) {
  if (descriptor.dataType == NDDataType.unknown) {
    throw new ArgumentError.value(descriptor.dataType.toString(), "data type");
  }

  if (reuse != null && reuse.descriptor == descriptor) {
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

Iterable<num> _createValueIterable(
    List _data, NDDescriptor descriptor, DataInfo dataInfo) sync* {
  if (descriptor.shape.isScalar) {
    yield _data[dataInfo.offset];
  } else {
    var shapeIndex = 0;
    var dimensionIndexes = new List(descriptor.shape.dimensionCount);
    var dataIndexes = new List(descriptor.shape.dimensionCount);
    var dataIndex = dataIndexes[shapeIndex] = dataInfo.offset;
    var dimensionIndex = dimensionIndexes[shapeIndex] = 0;
    var i = 0;
    while (i < descriptor.shape.length) {
      if (dimensionIndex < descriptor.shape[shapeIndex]) {
        if (shapeIndex == descriptor.shape.dimensionCount - 1) {
          var axeDimension = descriptor.shape[shapeIndex];
          var axeStride = dataInfo.stride[shapeIndex];
          while (dimensionIndex < axeDimension) {
            yield _data[dataIndex];
            dimensionIndex++;
            dataIndex += axeStride;
            i++;
          }
        } else {
          shapeIndex++;
          dataIndexes[shapeIndex] = dataIndex;
          dimensionIndex = dimensionIndexes[shapeIndex] = 0;
        }
      } else {
        shapeIndex--;
        dataIndex = dataIndexes[shapeIndex] =
            dataIndexes[shapeIndex] + dataInfo.stride[shapeIndex];
        dimensionIndex =
            dimensionIndexes[shapeIndex] = dimensionIndexes[shapeIndex] + 1;
      }
    }
  }
}

class DataInfo {
  final List<int> stride;

  final int offset;

  DataInfo(this.stride, this.offset);

  factory DataInfo.normalized(NDDescriptor descriptor) {
    List<int> stride = new List(descriptor.shape.dimensionCount);
    var factor = 1;
    for (var i = descriptor.shape.dimensionCount - 1; i >= 0; i--) {
      stride[i] = factor;
      factor *= descriptor.shape[i];
    }
    return new DataInfo(stride, 0);
  }

  @override
  // ignore: hash_and_equals
  bool operator ==(other) {
    if (other is DataInfo) {
      return offset == other.offset &&
          _iterableEquality.equals(stride, other.stride);
    } else {
      return false;
    }
  }
}
