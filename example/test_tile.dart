import "package:collection/collection.dart";

import "package:tensor_math_simd/tensor_math.dart" as tm;

final equality = new DeepCollectionEquality();

void main() {
  // caso head tiled
  print(new tm.NDArray.generate([10, 10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([3, 2, 1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([1, 1, 3, 2]).toValue());
  // caso head tiled
  print(new tm.NDArray.generate([10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([3, 1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([1, 3, 2]).toValue());
  // nope
  print(new tm.NDArray.generate([10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([3, 2]).toValue());
  // nope
  print(new tm.NDArray.generate([10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10], (index) => index + 1,
      dataType: tm.NDDataType.float32HBlocked).tile([3]).toValue());

  // caso head tiled
  print(new tm.NDArray.generate([10, 10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([3, 2, 1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([1, 1, 3, 2]).toValue());
  // caso head tiled
  print(new tm.NDArray.generate([10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([3, 1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([1, 3, 2]).toValue());
  // nope
  print(new tm.NDArray.generate([10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([1, 1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10, 10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([3, 2]).toValue());
  // nope
  print(new tm.NDArray.generate([10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([1]).toValue());
  // caso full tiled
  print(new tm.NDArray.generate([10], (index) => index + 1,
      dataType: tm.NDDataType.float32VBlocked).tile([3]).toValue());
}
