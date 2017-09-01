import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {

  print(new tm.NDArray.generate([1, 5], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([2, 1]).toValue());

  return;

  print(new tm.NDArray.generate([2, 10, 14], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([2, 2, 1]).toValue());

  print(new tm.NDArray.generate([2, 8, 8], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([2, 1, 1]).toValue());

  print(new tm.NDArray.generate([2, 10, 14], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([2, 1, 1]).toValue());

  print(new tm.NDArray.generate([8, 8], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([2, 1]).toValue());

  // caso head tiled
  print(new tm.NDArray.generate([10, 10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([3, 2, 1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([1, 1, 3, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([1, 1, 3, 2]).toValue());
  // caso head tiled
  print(new tm.NDArray.generate([10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([3, 1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([1, 3, 2]).toValue());
  // nope
  print(new tm.NDArray.generate([10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([3, 2]).toValue());
  // nope
  print(new tm.NDArray.generate([10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10], (index) => index + 1,
      dataType: tm.NDDataType.float32Blocked).tile([3]).toValue());
}
