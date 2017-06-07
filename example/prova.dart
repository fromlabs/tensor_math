import "package:tensor_math/tensor_math.dart";

void main() {
  var a = new NDArray.generate([2, 2, 2], (index) => index,
      dataType: NDDataType.float32);

  print(a);

  print(-a.inv());

  print(a.elementWiseUnaryOperation(
      resultDataType: NDDataType.float32,
      unaryOperation: (value) => -1 / value));
}