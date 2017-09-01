import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as tm;

final iterableEquality = new DeepCollectionEquality();

void main() {
  functionalTest([], [1]);
  functionalTest([], [1, 1]);
  functionalTest([1], []);
  functionalTest([1, 1], []);
  functionalTest([1, 1], [1]);
  functionalTest([1], [1, 1]);

  functionalTest([2, 2, 2, 2], [2, 2, 2, 2]);

  functionalTest([2, 2, 2, 2], [1, 4, 2, 2]); // modificata solo testa

  functionalTest([2, 2, 2, 2], [2, 2, 1, 4]);

  functionalTest([2, 2, 2, 2], [2, 2, 4, 1]);
}

void functionalTest(List<int> shape, List<int> newShape) {
  var array = new tm.NDArray.generate(shape, (index) => index + 1,
      dataType: tm.NDDataType.float32);

  print(array.toValue());

  var expectedValue = array.reshape(newDimensions: newShape).toValue();

  print(expectedValue);

  var value;

  value = new tm.NDArray.generate(shape, (index) => index + 1,
          dataType: tm.NDDataType.float32Blocked)
      .reshape(newDimensions: newShape)
      .toValue();

  if (!iterableEquality.equals(value, expectedValue)) {
    print(value);
    print(expectedValue);

    throw new StateError("not equals");
  }
}
