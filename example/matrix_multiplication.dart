import "package:tensor_math/tensor_math.dart";

import "dart:typed_data";

void main() {
  var watch = new Stopwatch();
  watch.start();

  var array1 = new NDArray.generate([4, 4], (index) => index, dataType: NDDataType.float32);
  var array2 = new NDArray.generate([4, 4], (index) => index, dataType: NDDataType.float32);

  for (var i = 0; i < 1000000; i++) {
    matMul1(array1, array2);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  for (var i = 0; i < 1000000; i++) {
    matMul2(array1, array2);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  var A = toList(array1);
  var B = toList(array2);
  var R = new Float32x4List(4);
  for (var i = 0; i < 1000000; i++) {
    matMul3(A, B, R);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
}

NDArray matMul1(NDArray a, NDArray b) {
  return a.matMul(b);
}

NDArray matMul2(NDArray array1, NDArray array2) {

  var A = toList(array1);
  var B = toList(array2);
  var R = new Float32x4List(4);

  var a0 = A[0];
  var a1 = A[1];
  var a2 = A[2];
  var a3 = A[3];

  var b0 = B[0];
  R[0] = b0.shuffle(Float32x4.XXXX) * a0 +
      b0.shuffle(Float32x4.YYYY) * a1 +
      b0.shuffle(Float32x4.ZZZZ) * a2 +
      b0.shuffle(Float32x4.WWWW) * a3;
  var b1 = B[1];
  R[1] = b1.shuffle(Float32x4.XXXX) * a0 +
      b1.shuffle(Float32x4.YYYY) * a1 +
      b1.shuffle(Float32x4.ZZZZ) * a2 +
      b1.shuffle(Float32x4.WWWW) * a3;
  var b2 = B[2];
  R[2] = b2.shuffle(Float32x4.XXXX) * a0 +
      b2.shuffle(Float32x4.YYYY) * a1 +
      b2.shuffle(Float32x4.ZZZZ) * a2 +
      b2.shuffle(Float32x4.WWWW) * a3;
  var b3 = B[3];
  R[3] = b3.shuffle(Float32x4.XXXX) * a0 +
      b3.shuffle(Float32x4.YYYY) * a1 +
      b3.shuffle(Float32x4.ZZZZ) * a2 +
      b3.shuffle(Float32x4.WWWW) * a3;

  return toArray(R);
}

void matMul3(Float32x4List A, Float32x4List B, Float32x4List R) {
  var a0 = A[0];
  var a1 = A[1];
  var a2 = A[2];
  var a3 = A[3];

  var b0 = B[0];
  R[0] = b0.shuffle(Float32x4.XXXX) * a0 +
      b0.shuffle(Float32x4.YYYY) * a1 +
      b0.shuffle(Float32x4.ZZZZ) * a2 +
      b0.shuffle(Float32x4.WWWW) * a3;
  var b1 = B[1];
  R[1] = b1.shuffle(Float32x4.XXXX) * a0 +
      b1.shuffle(Float32x4.YYYY) * a1 +
      b1.shuffle(Float32x4.ZZZZ) * a2 +
      b1.shuffle(Float32x4.WWWW) * a3;
  var b2 = B[2];
  R[2] = b2.shuffle(Float32x4.XXXX) * a0 +
      b2.shuffle(Float32x4.YYYY) * a1 +
      b2.shuffle(Float32x4.ZZZZ) * a2 +
      b2.shuffle(Float32x4.WWWW) * a3;
  var b3 = B[3];
  R[3] = b3.shuffle(Float32x4.XXXX) * a0 +
      b3.shuffle(Float32x4.YYYY) * a1 +
      b3.shuffle(Float32x4.ZZZZ) * a2 +
      b3.shuffle(Float32x4.WWWW) * a3;
}

List<Float32x4> toList(NDArray array) {
  List<Float32x4> l = new Float32x4List(4);

  var m = array.toMatrix();
  var row = m[0];
  l[0] = new Float32x4(row[0], row[1], row[2], row[3]);
  row = m[1];
  l[1] = new Float32x4(row[0], row[1], row[2], row[3]);
  row = m[2];
  l[2] = new Float32x4(row[0], row[1], row[2], row[3]);
  row = m[3];
  l[3] = new Float32x4(row[0], row[1], row[2], row[3]);

  return l;
}

NDArray toArray(List<Float32x4> list) {
  var m = [];

  var row = list[0];
  m.add([row.x, row.y, row.z, row.w]);
  row = list[1];
  m.add([row.x, row.y, row.z, row.w]);
  row = list[2];
  m.add([row.x, row.y, row.z, row.w]);
  row = list[3];
  m.add([row.x, row.y, row.z, row.w]);

  return new NDArray(m);
}
