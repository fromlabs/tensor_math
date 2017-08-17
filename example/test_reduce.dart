import 'package:collection/collection.dart';

import "package:tensor_math_simd/tensor_math.dart" as tm;

import "package:tensor_math_simd/src/nd_array_blocked_impl2.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  // functionalTest([2, 4, 1, 1], [2, 3]); // disabilitato reset
  functionalTest([2, 4, 1, 1], [2, 3]); // disabilitato reset

  functionalTest([5, 2, 1], [2]); // abilitato reset

  functionalTest([5, 2, 1, 1], [2, 3]);


  functionalTest([1, 1, 1, 1], [2, 3]);

  functionalTest([1, 2, 1], [2]);

  functionalTest([1, 2, 1], [0]);
  functionalTest([1, 1, 2], [1, 2]);
  functionalTest([2, 1, 1], [0]);
  functionalTest([1, 1, 1], [0]);
  functionalTest([2, 1], [1]);
  functionalTest([1, 1], [0, 1]);
  functionalTest([1, 1], [1]);
  functionalTest([1], [0]);
  functionalTest([1, 1, 5], [1]);

  functionalTest([1, 1], [0]);

  // performanceTest();
}

void functionalTest(List<int> shape, List<int> reductionAxis) {
  var array = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(array.toValue());

  var expectedValue = array.reduceSum(reductionAxis: reductionAxis).toValue();

  print(expectedValue);

  var value;

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32HBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32VBlocked)
      .reduceSum(reductionAxis: reductionAxis)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }
}
