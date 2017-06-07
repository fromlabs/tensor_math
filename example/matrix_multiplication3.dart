import "package:tensor_math/tensor_math.dart";

import "dart:typed_data";

void main() {
  var array1 = new NDArray.generate([4, 4], (index) => index + 1,
      dataType: NDDataType.float32);
  var array2 = new NDArray.generate([4, 4], (index) => 16 - index,
      dataType: NDDataType.float32);
  array1 = array1.transpose();

  print(array1);
  print(array2);

  print(matMul1(array1, array2));

  array1 = new NDArray.generate([4, 4], (index) => index + 1,
      dataType: NDDataType.float32);
  array2 = new NDArray.generate([4, 4], (index) => 16 - index,
      dataType: NDDataType.float32);

  print(matMul2b(array1, array2));
}

NDArray matMul1(NDArray a, NDArray b) {
  return a.matMul(b);
}

NDArray matMul2(NDArray array1, NDArray array2) {
  var A = toList(array1);
  var B = toList(array2);
  var R = new Float32x4List(4);

  var b0 = B[0];
  var b1 = B[1];
  var b2 = B[2];
  var b3 = B[3];

  var a0 = A[0];
  R[0] = a0.shuffle(Float32x4.XXXX) * b0 +
      a0.shuffle(Float32x4.YYYY) * b1 +
      a0.shuffle(Float32x4.ZZZZ) * b2 +
      a0.shuffle(Float32x4.WWWW) * b3;
  var a1 = A[1];
  R[1] = a1.shuffle(Float32x4.XXXX) * b0 +
      a1.shuffle(Float32x4.YYYY) * b1 +
      a1.shuffle(Float32x4.ZZZZ) * b2 +
      a1.shuffle(Float32x4.WWWW) * b3;
  var a2 = A[2];
  R[2] = a2.shuffle(Float32x4.XXXX) * b0 +
      a2.shuffle(Float32x4.YYYY) * b1 +
      a2.shuffle(Float32x4.ZZZZ) * b2 +
      a2.shuffle(Float32x4.WWWW) * b3;
  var a3 = A[3];
  R[3] = a3.shuffle(Float32x4.XXXX) * b0 +
      a3.shuffle(Float32x4.YYYY) * b1 +
      a3.shuffle(Float32x4.ZZZZ) * b2 +
      a3.shuffle(Float32x4.WWWW) * b3;

  return toArray(R);
}

NDArray matMul2b(NDArray array1, NDArray array2) {
  var A = toList(array1);
  var B = toList(array2);
  var R = new Float32x4List(4);

  var a0 = A[0];
  var a1 = A[1];
  var a2 = A[2];
  var a3 = A[3];

  var b0 = B[0];
  var b1 = B[1];
  var b2 = B[2];
  var b3 = B[3];

  print(a0);
  print(b0);

  R[0] = a0 * b0 +
      a1 * b1 +
      a2 * b2 +
      a0 * b3;

  R[1] = a1.shuffle(Float32x4.XXXX) * b0 +
      a1.shuffle(Float32x4.YYYY) * b1 +
      a1.shuffle(Float32x4.ZZZZ) * b2 +
      a1.shuffle(Float32x4.WWWW) * b3;

  R[2] = a2.shuffle(Float32x4.XXXX) * b0 +
      a2.shuffle(Float32x4.YYYY) * b1 +
      a2.shuffle(Float32x4.ZZZZ) * b2 +
      a2.shuffle(Float32x4.WWWW) * b3;

  R[3] = a3.shuffle(Float32x4.XXXX) * b0 +
      a3.shuffle(Float32x4.YYYY) * b1 +
      a3.shuffle(Float32x4.ZZZZ) * b2 +
      a3.shuffle(Float32x4.WWWW) * b3;

  return toArray(R);
}

void matMul3(Float32x4List A, Float32x4List B, Float32x4List R) {
  var b0 = B[0];
  var b1 = B[1];
  var b2 = B[2];
  var b3 = B[3];

  var a0 = A[0];
  R[0] = a0.shuffle(Float32x4.XXXX) * b0 +
      a0.shuffle(Float32x4.YYYY) * b1 +
      a0.shuffle(Float32x4.ZZZZ) * b2 +
      a0.shuffle(Float32x4.WWWW) * b3;
  var a1 = A[1];
  R[1] = a1.shuffle(Float32x4.XXXX) * b0 +
      a1.shuffle(Float32x4.YYYY) * b1 +
      a1.shuffle(Float32x4.ZZZZ) * b2 +
      a1.shuffle(Float32x4.WWWW) * b3;
  var a2 = A[2];
  R[2] = a2.shuffle(Float32x4.XXXX) * b0 +
      a2.shuffle(Float32x4.YYYY) * b1 +
      a2.shuffle(Float32x4.ZZZZ) * b2 +
      a2.shuffle(Float32x4.WWWW) * b3;
  var a3 = A[3];
  R[3] = a3.shuffle(Float32x4.XXXX) * b0 +
      a3.shuffle(Float32x4.YYYY) * b1 +
      a3.shuffle(Float32x4.ZZZZ) * b2 +
      a3.shuffle(Float32x4.WWWW) * b3;
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
