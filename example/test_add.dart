import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest([10, 10, 10], [10, 10, 10]);
  functionalTest([10, 10, 10], [10, 10]);
  functionalTest([10, 10, 10], [10, 1]);

  functionalTest([], []);
  functionalTest([], [10]);
  functionalTest([10], [10, 10]);
}

void functionalTest(List<int> shape1, List<int> shape2) {
  var shapeLength2 = shape2.isNotEmpty ? shape2.reduce((v1, v2) => v1 * v2) : 1;

  var expectedValue = (new tm.NDArray.generate(shape1, (index) => index + 1,
              dataType: tm.NDDataType.float32) +
          new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
              dataType: tm.NDDataType.float32))
      .toValue();

  print(expectedValue);

  var value = (new tm.NDArray.generate(shape1, (index) => index + 1,
              dataType: tm.NDDataType.float32HBlocked) +
          new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
              dataType: tm.NDDataType.float32HBlocked))
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(expectedValue);
    print(value);
    throw new StateError("not equals");
  }

  value = (new tm.NDArray.generate(shape1, (index) => index + 1,
              dataType: tm.NDDataType.float32VBlocked) +
          new tm.NDArray.generate(shape2, (index) => shapeLength2 - index,
              dataType: tm.NDDataType.float32VBlocked))
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(expectedValue);
    print(value);
    throw new StateError("not equals");
  }
}
