import "dart:typed_data";

import "package:tensor_math/tensor_math.dart" as tm;

import "data_converter2.dart";

void main() {
  //test0(8, 8, 8, 8, true);

  // test0(8, 4, 4, 16, true);

  // test0(7, 7, 7, 7, true);

  // test0(32, 32, 32, 32, false);

  test(10, 512, 512, 512, 512);

  //test(1024);
}

void test0(int rows1, int columns1, int rows2, int columns2, bool check) {
  var list1 =
      new List.generate(rows1 * columns1, (index) => (index + 1).toDouble());
  var list2 = new List.generate(
      rows2 * columns2, (index) => (rows2 * columns2 - index).toDouble());

  var array1 = new tm.NDArray.generate(
      [rows1, columns1], (index) => list1[index],
      dataType: tm.NDDataType.float32);

  var array2 = new tm.NDArray.generate(
      [rows2, columns2], (index) => list2[index],
      dataType: tm.NDDataType.float32);

  var resultArray = array1.matMul(array2);

  var resultList = resultArray.reshape(newDimensions: [-1]).toVector();

  if (check) {
    print(resultList);
  }

  var a = toFloat32x4List1(rows1, columns1, list1);
  var b = toFloat32x4List1(rows2, columns2, list2);

  var c = new Float32x4List(4 * (rows1 / 4).ceil() * (columns2 / 4).ceil());

  matMul1(a, rows1, columns1, b, rows2, columns2, c);

  var resultList1 = fromFloat32x4List1(rows1, columns2, c);

  if (check) {
    print(resultList1);
    checkEquals(resultList1, resultList);
  }

  a = toFloat32x4List2(rows1, columns1, list1);
  b = toFloat32x4List2(rows2, columns2, list2);

  c = new Float32x4List(4 * (rows1 / 4).ceil() * (columns2 / 4).ceil());

  matMul2(a, rows1, columns1, b, rows2, columns2, c);

  var resultList2 = fromFloat32x4List2(rows1, columns2, c);

  if (check) {
    print(resultList2);
    checkEquals(resultList2, resultList);
  }

  a = toFloat32x4List2(rows1, columns1, list1);
  b = toFloat32x4List3(rows2, columns2, list2);

  c = new Float32x4List(4 * (rows1 / 4).ceil() * (columns2 / 4).ceil());
  matMul23(a, rows1, columns1, b, rows2, columns2, c);

  var resultList3 = fromFloat32x4List2(rows1, columns2, c);

  if (check) {
    print(resultList3);
    checkEquals(resultList3, resultList);
  }
}

void test(int steps, int rows1, int columns1, int rows2, int columns2) {
  var list1 =
      new List.generate(rows1 * columns1, (index) => (index + 1).toDouble());
  var list2 = new List.generate(
      rows2 * columns2, (index) => (rows2 * columns2 - index).toDouble());

  var watch = new Stopwatch();
  watch.start();

  var array1 = new tm.NDArray.generate(
      [rows1, columns1], (index) => list1[index],
      dataType: tm.NDDataType.float32);

  var array2 = new tm.NDArray.generate(
      [rows2, columns2], (index) => list2[index],
      dataType: tm.NDDataType.float32);

  var resultArray;
  for (var i = 0; i < steps; i++) {
    //resultArray = array1.matMul(array2, reuse: resultArray);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  var a = toFloat32x4List1(rows1, columns1, list1);
  var b = toFloat32x4List1(rows2, columns2, list2);

  var c = new Float32x4List(4 * (rows1 / 4).ceil() * (columns2 / 4).ceil());

  for (var i = 0; i < steps; i++) {
    matMul1(a, rows1, columns1, b, rows2, columns2, c);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  a = toFloat32x4List2(rows1, columns1, list1);
  b = toFloat32x4List2(rows2, columns2, list2);

  c = new Float32x4List(4 * (rows1 / 4).ceil() * (columns2 / 4).ceil());

  for (var i = 0; i < steps; i++) {
    matMul2(a, rows1, columns1, b, rows2, columns2, c);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  a = toFloat32x4List2(rows1, columns1, list1);
  b = toFloat32x4List3(rows2, columns2, list2);

  c = new Float32x4List(4 * (rows1 / 4).ceil() * (columns2 / 4).ceil());

  for (var i = 0; i < steps; i++) {
    matMul23(a, rows1, columns1, b, rows2, columns2, c);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
}

void checkEquals(List<double> list1, List<double> list2) {
  if (list1.length != list2.length) {
    throw new ArgumentError(
        "Different lengths: ${list1.length} != ${list2.length}");
  }

  for (var i = 0; i < list1.length; i++) {
    if (list1[i] != list2[i]) {
      throw new ArgumentError(
          "Different values: ${list1[i]} != ${list2[i]} at $i");
    }
  }
}
