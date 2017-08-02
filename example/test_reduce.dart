import 'package:collection/collection.dart';

import "package:tensor_math_simd/tensor_math.dart" as tm;

import "package:tensor_math_simd/src/nd_array_blocked_impl.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest3();

  // performanceTest();
}

void functionalTest3() {
  var shape = [2, 2, 6, 6];
  var reductionAxis = [2, 3];

  var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  var value = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    throw new StateError("not equals: $value $expectedValue");
  }

  value = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    throw new StateError("not equals: $value $expectedValue");
  }
}

void functionalTest2() {
  var shape = [2, 2, 6, 6];
  var reductionAxis = [2];

  var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  var value = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    throw new StateError("not equals: $value $expectedValue");
  }

  value = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    throw new StateError("not equals: $value $expectedValue");
  }
}

void functionalTest1() {
  var shape = [2, 2, 6, 6];
  var reductionAxis = [0];

  var expectedValue = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  var value = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    throw new StateError("not equals: $value $expectedValue");
  }

  value = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    throw new StateError("not equals: $value $expectedValue");
  }
}