import "dart:math" as math;

import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  test([30, 1], 1);
  test([1, 30], 0);
}

void test2() {
  var array = new tm.NDArray([
    [2],
    [2]
  ], dataType: tm.NDDataType.float32);

  array = array.transpose(permutationAxis: [1, 0]);

  array.normalize();

  print(array);

  print(array.oneHot(
      axis: 0, dimensionCount: 3, resultDataType: tm.NDDataType.float32));
}

void test(List<int> shape, int axis) {
  var random = new math.Random();

  var dimensionCount = 10;

  var input = new tm.NDArray.generate(
      shape, (index) => random.nextInt(dimensionCount),
      dataType: tm.NDDataType.float32);

  print("dimensionCount: $dimensionCount");
  print("axis: $axis");
  print("input: $input");

  var oneHot = input.oneHot(
      axis: axis,
      dimensionCount: dimensionCount,
      resultDataType: tm.NDDataType.float32);

  print("oneHot: $oneHot");

  var input2 = oneHot.argMax(axis: axis);

  input2 = input2.reshape(newDimensions: input.shape.dimensions);

  print("input2: $input2");
}

void test1(
    {int batchSize,
    int inputHeight,
    int inputWidth,
    int inputDepth,
    int kernelHeight,
    int kernelWidth,
    int outputDepth,
    int vStride = 1,
    int hStride = 1}) {
  var input = new tm.NDArray.generate(
      [batchSize, inputHeight, inputWidth, inputDepth], (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(input.shape);

  var inputColumns = input.im2col(
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      vStride: vStride,
      hStride: hStride,
      keepInputDepth: false);

  print(inputColumns.shape);

  var inputColumns2 = inputColumns.transpose(permutationAxis: [1, 0]);

  inputColumns2 = inputColumns2.normalize();

  inputColumns2 = inputColumns2.transpose(permutationAxis: [1, 0]);

  if (!iterableEquality.equals(
      inputColumns.toValue(), inputColumns2.toValue())) {
    print(inputColumns.toValue());
    print(inputColumns2.toValue());

    throw new StateError("not equals");
  }

  var output1 = inputColumns.col2im(
      inputDimensions: input.shape.dimensions,
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      vStride: vStride,
      hStride: hStride);

  var output2 = inputColumns2.col2im(
      inputDimensions: input.shape.dimensions,
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      vStride: vStride,
      hStride: hStride);

  if (!iterableEquality.equals(output1.toValue(), output2.toValue())) {
    print(output1.toValue());
    print(output2.toValue());

    throw new StateError("not equals");
  }
}
