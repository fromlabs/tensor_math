import 'package:collection/collection.dart';

import "package:tensor_math/tensor_math.dart" as tm;

import "package:tensor_math/src/nd_array_blocked_impl.dart";

final iterableEquality = new DeepCollectionEquality();

void main() {
  // TODO implementare operazioni con booleani

  var array = new tm.NDArray([[true, false], [false, true]]);

  print(array);

  print(array.cast(tm.NDDataType.float32));
  print(array.cast(tm.NDDataType.int32));

  print(array.cast(tm.NDDataType.float32Blocked));
  print(array.cast(tm.NDDataType.int32Blocked));
  print(array.cast(tm.NDDataType.booleanBlocked));

  array = new tm.NDArray([[true, false], [false, true]], dataType: tm.NDDataType.booleanBlocked);

  print(array);

  print(array.cast(tm.NDDataType.float32));
  print(array.cast(tm.NDDataType.int32));
  print(array.cast(tm.NDDataType.boolean));

  print(array.cast(tm.NDDataType.float32Blocked));
  print(array.cast(tm.NDDataType.int32Blocked));

}