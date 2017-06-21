import "dart:math" as math;

import "package:collection/collection.dart";

import "package:tensor_math_simd/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  test([3, 5, 11, 13]);

  test([3, 5, 4, 8]);

  test([5, 11, 13]);

  test([11, 13]);
}

void test(List<int> shape) {
  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32HBlocked)
          .cast(tm.NDDataType.float32)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32VBlocked)
          .cast(tm.NDDataType.float32)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32HBlocked)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32)
          .cast(tm.NDDataType.float32HBlocked)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32VBlocked)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32)
          .cast(tm.NDDataType.float32VBlocked)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.int32)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32HBlocked)
          .cast(tm.NDDataType.int32)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.int32)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32VBlocked)
          .cast(tm.NDDataType.int32)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32HBlocked)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.int32)
          .cast(tm.NDDataType.float32HBlocked)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32VBlocked)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.int32)
          .cast(tm.NDDataType.float32VBlocked)
          .toValue()));
}
