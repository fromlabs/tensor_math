import "dart:math" as math;

import "package:tensor_math/tensor_math.dart" as tm;

void main() {
  testConv(
      batchSize: 2,
      inputHeight: 3,
      inputWidth: 3,
      inputDepth: 3,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 4);

  testPool(
      batchSize: 2,
      inputHeight: 3,
      inputWidth: 3,
      inputDepth: 3,
      blockHeight: 3,
      blockWidth: 3);

  return;

  testPool(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 2,
      blockHeight: 2,
      blockWidth: 2);

  testConv(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 2,
      kernelHeight: 2,
      kernelWidth: 2,
      outputDepth: 2);

  return;

  testConv(
      batchSize: 1,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 1,
      kernelHeight: 3,
      kernelWidth: 3,
      outputDepth: 1,
      vStride: 2,
      hStride: 2);

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
      vStride: 2,
      hStride: 2);

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
      vStride: 2,
      hStride: 2);

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
      vStride: 2,
      hStride: 2);

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
      blockHeight: 2,
      blockWidth: 2);

  testPool(
      batchSize: 2,
      inputHeight: 5,
      inputWidth: 5,
      inputDepth: 1,
      blockHeight: 2,
      blockWidth: 2);
}

void testConv(
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
  print("vStride: $vStride");
  print("hStride: $hStride");

  var output =
      conv2d(input, kernel: kernel, vStride: vStride, hStride: hStride);

  print("output shape: ${output.shape}");
  print("output: $output");

  var backPropagatedGradient = new tm.NDArray.ones(output.shape.dimensions,
      dataType: tm.NDDataType.float32);

  var gradients = conv2dGradients(
      backPropagatedGradient: backPropagatedGradient,
      output: output,
      input: input,
      bias: bias,
      kernel: kernel,
      vStride: vStride,
      hStride: hStride);

  print("gradients: $gradients");
}

void testPool(
    {int batchSize,
    int inputHeight,
    int inputWidth,
    int inputDepth,
    int blockHeight,
    int blockWidth}) {
  var input = new tm.NDArray.generate(
      [batchSize, inputHeight, inputWidth, inputDepth], (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print("input shape: ${input.shape}");
  print("kernel shape: [$blockHeight, $blockWidth]");
  print("strides: [$blockHeight, $blockWidth]");

  var output = maxPool(input, blockHeight: blockHeight, blockWidth: blockWidth);

  print("output shape: ${output.shape}");
  print("output: $output");

  var backPropagatedGradient = new tm.NDArray.ones(output.shape.dimensions,
      dataType: tm.NDDataType.float32);

  var gradients = maxPoolGradient(
      backPropagatedGradient: backPropagatedGradient,
      output: output,
      input: input,
      blockHeight: blockHeight,
      blockWidth: blockWidth);

  print("gradients: $gradients");
}

tm.NDArray conv2d(tm.NDArray input,
    {tm.NDArray kernel, tm.NDArray bias, int vStride = 1, int hStride = 1}) {
  var batchSize = input.shape[0];
  var inputHeight = input.shape[1];
  var inputWidth = input.shape[2];
  var inputDepth = input.shape[3];

  var kernelHeight = kernel.shape[0];
  var kernelWidth = kernel.shape[1];

  var outputHeight =
      inputHeight != null ? (inputHeight / vStride).ceil() : null;
  var outputWidth = inputWidth != null ? (inputWidth / hStride).ceil() : null;
  var outputDepth = kernel.shape[3];

  tm.NDArray inputColumns = input.im2col(
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      vStride: vStride,
      hStride: hStride,
      keepInputDepth: false);

  // OK reshape sui primi livelli (no ultimo), comunque matrice piccola
  var kernelReshaped = kernel.reshape(
      newDimensions: [kernelHeight * kernelWidth * inputDepth, outputDepth]);

  var convolution = inputColumns.matMul(kernelReshaped);

  if (bias != null) {
    convolution = convolution.add(bias);
  }

  // OK reshape sui primi livelli (no ultimo)
  return convolution.reshape(
      newDimensions: [batchSize, outputHeight, outputWidth, outputDepth]);
}

Map<String, tm.NDArray> conv2dGradients(
    {tm.NDArray input,
    tm.NDArray output,
    tm.NDArray kernel,
    tm.NDArray bias,
    int vStride = 1,
    int hStride = 1,
    tm.NDArray backPropagatedGradient}) {
  var kernelHeight = kernel.shape[0];
  var kernelWidth = kernel.shape[1];
  var kernelInputDepth = kernel.shape[2];
  var kernelOutputDepth = kernel.shape[3];

  // OK reshape sui primi livelli (no ultimo)
  var backPropagatedGradientReshaped =
      backPropagatedGradient.reshape(newDimensions: [-1, kernelOutputDepth]);

  // TODO cache del kernelReshaped
  // OK reshape sui primi livelli (no ultimo), comunque matrice piccola
  var kernelReshaped = kernel.reshape(newDimensions: [
    kernelHeight * kernelWidth * kernelInputDepth,
    kernelOutputDepth
  ]);

  // TODO cache del inputColumns
  tm.NDArray inputColumns = input.im2col(
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      vStride: vStride,
      hStride: hStride,
      keepInputDepth: false);

  // OK transpose ultimi due livelli
  var kernelGradient = inputColumns.transpose(
      permutationAxis: [1, 0]).matMul(backPropagatedGradientReshaped);

  // OK reshape sui primi livelli (no ultimo)
  kernelGradient =
      kernelGradient.reshape(newDimensions: kernel.shape.dimensions);

  // OK transpose ultimi due livelli
  var inputColumnsGradient = backPropagatedGradientReshaped
      .matMul(kernelReshaped.transpose(permutationAxis: [1, 0]));

  var inputGradient = inputColumnsGradient.col2im(
      inputDimensions: input.shape.dimensions,
      blockHeight: kernelHeight,
      blockWidth: kernelWidth,
      vStride: vStride,
      hStride: hStride);

  // TODO calcolo gradiente del bias
  var biasGradient;

  return {
    "input": inputGradient,
    "kernel": kernelGradient,
    "bias": biasGradient
  };
}

tm.NDArray maxPool(tm.NDArray input, {int blockHeight, int blockWidth}) {
  var batchSize = input.shape[0];
  var inputHeight = input.shape[1];
  var inputWidth = input.shape[2];
  var inputDepth = input.shape[3];

  int vStride = blockHeight;
  int hStride = blockWidth;

  var outputHeight =
      inputHeight != null ? (inputHeight / vStride).ceil() : null;
  var outputWidth = inputWidth != null ? (inputWidth / hStride).ceil() : null;

  tm.NDArray inputColumns = input.im2col(
      blockHeight: blockHeight,
      blockWidth: blockWidth,
      vStride: vStride,
      hStride: hStride,
      keepInputDepth: true);

  var reduction = inputColumns.reduceMax(reductionAxis: [1]);

  // OK reshape sui primi livelli (no ultimo)
  var maxPool = reduction.reshape(
      newDimensions: [batchSize, outputHeight, outputWidth, inputDepth]);

  return maxPool;
}

tm.NDArray maxPoolGradient(
    {tm.NDArray input,
    tm.NDArray output,
    int blockHeight,
    int blockWidth,
    tm.NDArray backPropagatedGradient}) {
  int vStride = blockHeight;
  int hStride = blockWidth;

  // TODO cache del inputColumns
  tm.NDArray inputColumns = input.im2col(
      blockHeight: blockHeight,
      blockWidth: blockWidth,
      vStride: vStride,
      hStride: hStride,
      keepInputDepth: true);

  var reductionForGradient =
      inputColumns.reduceMax(reductionAxis: [1], keepDimensions: true);

  var maxForGradient = inputColumns.elementWiseBinaryOperation(
      reductionForGradient,
      resultDataType: tm.NDDataType.float32,
      binaryOperation: (value1, value2, valueCount) {
    return value1 == value2 ? 1.0 : 0.0;
  });

  var sumForGradient =
      maxForGradient.reduceSum(reductionAxis: [1], keepDimensions: true);

  return maxForGradient.div(sumForGradient);
}
