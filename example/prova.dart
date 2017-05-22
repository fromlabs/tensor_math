import "package:tensor_math/tensor_math.dart";

void main() {
  var a = new NDArray.generate([2, 2, 2], (index) => index.toDouble());

  print(a);

  print(new NDArray([1.0, 2.0, 3.5], dataType: NDDataType.int64));

  print(new NDArray([1.0, 2, 3]));

  print(new NDArray.zeros([2, 2], dataType: NDDataType.float64) +
      new NDArray.zeros([2, 2], dataType: NDDataType.float32));


}
