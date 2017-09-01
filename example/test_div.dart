import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest([10, 10, 10], [10, 1, 1]);
  functionalTest([8, 8, 8], [8, 1, 8]);
  functionalTest([10, 10, 10], [10, 10, 10]);
  functionalTest([10, 10, 10], [1, 10, 10]);
  functionalTest([10, 10, 10], [10, 10]);
  functionalTest([10, 10, 10], [10, 1, 10]);
  functionalTest([10, 10, 10], [1, 1, 10]);
  functionalTest([10, 10, 10], [10, 1]);
  functionalTest([10, 10], [10, 1]);
  functionalTest([], []);
  functionalTest([], [10]);
  functionalTest([10], [10, 10]);

  functionalTest([128, 10], [10]);
  functionalTest([128, 10], [128, 1]);
  functionalTest([128, 10], [128, 10]);
  functionalTest([784, 10], []);
  functionalTest([784, 10], [784, 10]);
  functionalTest([10], []);
  functionalTest([10], [10]);
}

void functionalTest(List<int> shape1, List<int> shape2) {
  var shapeLength2 = shape2.isNotEmpty ? shape2.reduce((v1, v2) => v1 * v2) : 1;

  var expectedValue = (new tm.NDArray.generate(shape1, (index) => index + 1,
              dataType: tm.NDDataType.float32) /
          new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
              dataType: tm.NDDataType.float32))
      .toValue();

  print(expectedValue);

  var value = (new tm.NDArray.generate(shape1, (index) => index + 1,
              dataType: tm.NDDataType.float32Blocked) /
          new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
              dataType: tm.NDDataType.float32Blocked))
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(expectedValue);
    print(value);
    throw new StateError("not equals");
  }
}
