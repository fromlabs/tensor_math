import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  List<int> shape = [3, 5, 11, 13];

  print(equality.equals(
      new tm.NDArray.zeros(shape, dataType: tm.NDDataType.float32).toValue(),
      new tm.NDArray.zeros(shape, dataType: tm.NDDataType.float32Blocked)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.ones(shape, dataType: tm.NDDataType.float32).toValue(),
      new tm.NDArray.ones(shape, dataType: tm.NDDataType.float32Blocked)
          .toValue()));

  print(equality.equals(
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32)
          .toValue(),
      new tm.NDArray.generate(shape, (index) => index + 1,
              dataType: tm.NDDataType.float32Blocked)
          .toValue()));
}
