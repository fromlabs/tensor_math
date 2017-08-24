import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  test([2, 2, 2, 2], [2, 2, 2, 2]);

  test([2, 2, 2, 2], [1, 4, 2, 2]); // modificata solo testa

  test([2, 2, 2, 2], [2, 2, 1, 4]);

  test([2, 2, 2, 2], [2, 2, 4, 1]);
}

void test(List<int> shape, List<int> newShape) {
  print("Reshape from $shape to $newShape");

  var list = generateValue(shape);
  var list2 =
      new tm.NDArray(generateValue(newShape), dataType: tm.NDDataType.float32)
          .toValue();

  print(list);
  print(list2);

  testReshape(
      new tm.NDArray(list, dataType: tm.NDDataType.float32), newShape, list2);
  testReshape(new tm.NDArray(list, dataType: tm.NDDataType.float32HBlocked),
      newShape, list2);
  testReshape(new tm.NDArray(list, dataType: tm.NDDataType.float32VBlocked),
      newShape, list2);
}

void testReshape(tm.NDArray array, List<int> newShape, List newValue) {
  var array2 = array.reshape(newDimensions: newShape);
  var value2 = array2.toValue();
  if (!equality.equals(value2, newValue)) {
    print(value2);
    print(newValue);

    throw new StateError("Not equals");
  }
}

List generateValue(List<int> shape) =>
    new tm.NDArray.generate(shape, (index) => index + 1).toValue();
