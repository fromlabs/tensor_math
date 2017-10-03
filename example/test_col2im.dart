import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {

  test1(
      batchSize: 2,
      inputHeight: 3,
      inputWidth: 3,
      inputDepth: 3,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 4);

  test2(
      batchSize: 2,
      inputHeight: 3,
      inputWidth: 3,
      inputDepth: 3,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 4);
}

void test1(
    {int batchSize,
    int inputHeight,
    int inputWidth,
    int inputDepth,
    int kernelHeight,
    int kernelWidth,
    int outputDepth,
    int heightStride = 1,
    int widthStride = 1}) {
  var input = new tm.NDArray.generate(
      [batchSize, inputHeight, inputWidth, inputDepth], (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(input.shape);

  var inputColumns = input.im2col(
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      heightStride: heightStride,
      widthStride: widthStride,
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
      imageDimensions: input.shape.dimensions,
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      heightStride: heightStride,
      widthStride: widthStride);

  var output2 = inputColumns2.col2im(
      imageDimensions: input.shape.dimensions,
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      heightStride: heightStride,
      widthStride: widthStride);

  if (!iterableEquality.equals(output1.toValue(), output2.toValue())) {
    print(output1.toValue());
    print(output2.toValue());

    throw new StateError("not equals");
  }
}

void test2(
    {int batchSize,
    int inputHeight,
    int inputWidth,
    int inputDepth,
    int kernelHeight,
    int kernelWidth,
    int outputDepth,
    int heightStride = 1,
    int widthStride = 1}) {
  var input = new tm.NDArray.generate(
      [batchSize, inputHeight, inputWidth, inputDepth], (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(input.shape);

  var inputColumns = input.im2col(
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      heightStride: heightStride,
      widthStride: widthStride,
      keepInputDepth: true);

  var inputColumns2 = inputColumns.transpose(permutationAxis: [1, 2, 0]);

  inputColumns2 = inputColumns2.normalize();

  inputColumns2 = inputColumns2.transpose(permutationAxis: [2, 0, 1]);

  if (!iterableEquality.equals(
      inputColumns.toValue(), inputColumns2.toValue())) {
    print(inputColumns.toValue());
    print(inputColumns2.toValue());

    throw new StateError("not equals");
  }

  var output1 = inputColumns.col2im(
      imageDimensions: input.shape.dimensions,
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      heightStride: heightStride,
      widthStride: widthStride);

  var output2 = inputColumns2.col2im(
      imageDimensions: input.shape.dimensions,
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      heightStride: heightStride,
      widthStride: widthStride);

  if (!iterableEquality.equals(output1.toValue(), output2.toValue())) {
    print(output1.toValue());
    print(output2.toValue());

    throw new StateError("not equals");
  }
}
