import "dart:math";

import "package:tensor_math/tensor_math.dart" as tm;

import "package:tensor_math/src/nd_array_impl.dart";

void main() {

  testPool(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 2,
      kernelHeight: 2,
      kernelWidth: 2);

  testConv(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 1,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 1,
      strides: [2, 2]);

  testConv(
      batchSize: 2,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 2,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 3);

  testConv(
      batchSize: 2,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 2,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 3,
      strides: [2, 2]);

  testConv(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 1,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 1);

  testConv(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 1,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 1,
      strides: [2, 2]);

  testConv(
      batchSize: 1,
      inputHeight: 2,
      inputWidth: 2,
      inputDepth: 2,
      kernelHeight: 2,
      kernelWidth: 2,
      outputDepth: 3);

  testConv(
      batchSize: 1,
      inputHeight: 2,
      inputWidth: 2,
      inputDepth: 2,
      kernelHeight: 2,
      kernelWidth: 2,
      outputDepth: 3,
      strides: [2, 2]);

  testConv(
      batchSize: 1,
      inputHeight: 2,
      inputWidth: 2,
      inputDepth: 2,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 1);

  testPool(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 1,
      kernelHeight: 2,
      kernelWidth: 2);

  testPool(
      batchSize: 2,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 1,
      kernelHeight: 2,
      kernelWidth: 2);
}

void testConv(
    {int batchSize,
    int inputHeight,
    int inputWidth,
    int inputDepth,
    int kernelHeight,
    int kernelWidth,
    int outputDepth,
    List<int> strides = const [1, 1]}) {
  var input = new tm.NDArray.generate(
      [batchSize, inputHeight, inputWidth, inputDepth], (index) => index + 1,
      dataType: tm.NDDataType.float32);

  var kernel = new tm.NDArray.generate([
    kernelHeight,
    kernelWidth,
    inputDepth,
    outputDepth
  ], (index) => index + 1, dataType: tm.NDDataType.float32);

  var bias = new tm.NDArray.generate([outputDepth], (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print("input shape: ${input.shape}");
  print("kernel shape: ${kernel.shape}");
  print("bias shape: ${bias.shape}");
  print("strides: $strides");

  var output = input.conv2d(kernel: kernel, strides: strides);

  print("output shape: ${output.shape}");

  print("output: $output");
}

void testPool(
    {int batchSize,
    int inputHeight,
    int inputWidth,
    int inputDepth,
    int kernelHeight,
    int kernelWidth}) {
  var input = new tm.NDArray.generate(
      [batchSize, inputHeight, inputWidth, inputDepth], (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print("input shape: ${input.shape}");
  print("kernel shape: [$kernelHeight, $kernelWidth]");
  print("strides: [$kernelHeight, $kernelWidth]");

  var output = input.maxPool(kernelShape: [kernelHeight, kernelWidth]);

  print("output shape: ${output.shape}");

  print("output: $output");
}
