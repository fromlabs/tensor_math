import "package:tensor_math/tensor_math.dart";

void main() {
  var watch = new Stopwatch();
  watch.start();

  var array1 = new NDArray.generate([4, 4], (index) => index, dataType: NDDataType.float32);
  var array2 = new NDArray.generate([4, 4], (index) => index, dataType: NDDataType.float32);

  for (var i = 0; i < 1000000; i++) {
    array1.matMul(array2);
  }

  // TODO provare ricorsione

  print("Finish in ${watch.elapsedMilliseconds} ms");
}
