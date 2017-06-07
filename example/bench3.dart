import "dart:typed_data";

import "package:tensor_math/tensor_math.dart";

void main() {
  test2(1024);
}

void test2(int n) {
  var watch = new Stopwatch();
  watch.start();

  var aData = new List<Float32x4>.generate(n * n >> 2, (index) {
    var i = (index * 4).toDouble() + 1;
    return new Float32x4(i, i + 1, i + 2, i + 3);
  });
  var a = new Float32x4List.fromList(aData);

  var bData = new List<Float32x4>.generate(n * n >> 2, (index) {
    var i = (n * n) - (index * 4).toDouble();
    return new Float32x4(i, i - 1, i - 2, i - 3);
  });
  var b = new Float32x4List.fromList(bData);

  var c = new Float32x4List(n * n >> 2);
  for (var i = 0; i < 10; i++) {
    matMul2(a, n, n, b, n, n, c, n, n);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  for (var i = 0; i < 10; i++) {
    matMul3(a, n, n, b, n, n, c, n, n);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  for (var i = 0; i < 10; i++) {
    matMul4(a, n, n, b, n, n, c, n, n);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
  watch.reset();

  for (var i = 0; i < 10; i++) {
    matMul5(a, n, n, b, n, n, c, n, n);
  }

  print("Finish in ${watch.elapsedMilliseconds} ms");
}

void matMul2(
    Float32x4List list1,
    int rows1,
    int columns1,
    Float32x4List list2,
    int rows2,
    int columns2,
    Float32x4List result,
    int resultRows,
    int resultColumns) {
  var stride1 = columns1 >> 2;
  var stride2 = columns2 >> 2;
  var resultStride = stride2;

  for (var row1 = 0; row1 < rows1; row1 += 4) {
    for (var column2 = 0; column2 < columns2; column2 += 4) {
      for (var i = 0; i < columns1; i += 4) {
        var i10 = row1 * stride1 + (i >> 2);
        var i11 = i10 + stride1;
        var i12 = i11 + stride1;
        var i13 = i12 + stride1;

        var i20 = i * stride2 + (column2 >> 2);
        var i21 = i20 + stride2;
        var i22 = i21 + stride2;
        var i23 = i22 + stride2;

        var ir0 = row1 * resultStride + (column2 >> 2);
        var ir1 = ir0 + resultStride;
        var ir2 = ir1 + resultStride;
        var ir3 = ir2 + resultStride;

        matMulInternal(list1, i10, i11, i12, i13, list2, i20, i21, i22, i23,
            result, ir0, ir1, ir2, ir3);
      }
    }
  }
}

void matMul3(
    Float32x4List list1,
    int rows1,
    int columns1,
    Float32x4List list2,
    int rows2,
    int columns2,
    Float32x4List result,
    int resultRows,
    int resultColumns) {
  var stride1 = columns1 >> 2;
  var stride2 = columns2 >> 2;
  var resultStride = stride2;

  for (var row1 = 0; row1 < rows1; row1 += 4) {
    for (var column2 = 0; column2 < columns2; column2 += 4) {
      for (var i = 0; i < columns1; i += 4) {
        var i10 = row1 * stride1 + (i >> 2);
        var i11 = i10 + stride1;
        var i12 = i11 + stride1;
        var i13 = i12 + stride1;

        var i20 = i * stride2 + (column2 >> 2);
        var i21 = i20 + stride2;
        var i22 = i21 + stride2;
        var i23 = i22 + stride2;

        var ir0 = row1 * resultStride + (column2 >> 2);
        var ir1 = ir0 + resultStride;
        var ir2 = ir1 + resultStride;
        var ir3 = ir2 + resultStride;
      }
    }
  }
}

void matMul4(
    Float32x4List list1,
    int rows1,
    int columns1,
    Float32x4List list2,
    int rows2,
    int columns2,
    Float32x4List result,
    int resultRows,
    int resultColumns) {
  var stride1 = columns1 >> 2;
  var stride2 = columns2 >> 2;
  var resultStride = stride2;

  for (var row1 = 0; row1 < rows1; row1 += 4) {
    for (var column2 = 0; column2 < columns2; column2 += 4) {
      for (var i = 0; i < columns1; i += 4) {
        var i10 = 0;
        var i11 = i10 + stride1;
        var i12 = i11 + stride1;
        var i13 = i12 + stride1;

        var i20 = 0;
        var i21 = i20 + stride2;
        var i22 = i21 + stride2;
        var i23 = i22 + stride2;

        var ir0 = 0;
        var ir1 = ir0 + resultStride;
        var ir2 = ir1 + resultStride;
        var ir3 = ir2 + resultStride;

        matMulInternal(list1, i10, i11, i12, i13, list2, i20, i21, i22, i23,
            result, ir0, ir1, ir2, ir3);
      }
    }
  }
}

void matMul5(
    Float32x4List list1,
    int rows1,
    int columns1,
    Float32x4List list2,
    int rows2,
    int columns2,
    Float32x4List result,
    int resultRows,
    int resultColumns) {
  var stride1 = columns1 >> 2;
  var stride2 = columns2 >> 2;
  var resultStride = stride2;

  for (var row1 = 0; row1 < rows1; row1 += 4) {
    for (var column2 = 0; column2 < columns2; column2 += 4) {
      for (var i = 0; i < columns1; i += 4) {
        var i10 = 0;
        var i11 = 1;
        var i12 = 2;
        var i13 = 3;

        var i20 = 0;
        var i21 = 1;
        var i22 = 2;
        var i23 = 3;

        var ir0 = 0;
        var ir1 = 1;
        var ir2 = 2;
        var ir3 = 3;

        matMulInternal(list1, i10, i11, i12, i13, list2, i20, i21, i22, i23,
            result, ir0, ir1, ir2, ir3);
      }
    }
  }
}

void matMulInternal(
    Float32x4List list1,
    int i10,
    int i11,
    int i12,
    int i13,
    Float32x4List list2,
    int i20,
    int i21,
    int i22,
    int i23,
    Float32x4List result,
    int ir0,
    int ir1,
    int ir2,
    int ir3) {
  var b0 = list2[i20];
  var b1 = list2[i21];
  var b2 = list2[i22];
  var b3 = list2[i23];

  var a0 = list1[i10];

  result[ir0] += a0.shuffle(Float32x4.XXXX) * b0 +
      a0.shuffle(Float32x4.YYYY) * b1 +
      a0.shuffle(Float32x4.ZZZZ) * b2 +
      a0.shuffle(Float32x4.WWWW) * b3;

  var a1 = list1[i11];
  result[ir1] += a1.shuffle(Float32x4.XXXX) * b0 +
      a1.shuffle(Float32x4.YYYY) * b1 +
      a1.shuffle(Float32x4.ZZZZ) * b2 +
      a1.shuffle(Float32x4.WWWW) * b3;

  var a2 = list1[i12];
  result[ir2] += a2.shuffle(Float32x4.XXXX) * b0 +
      a2.shuffle(Float32x4.YYYY) * b1 +
      a2.shuffle(Float32x4.ZZZZ) * b2 +
      a2.shuffle(Float32x4.WWWW) * b3;

  var a3 = list1[i13];
  result[ir3] += a3.shuffle(Float32x4.XXXX) * b0 +
      a3.shuffle(Float32x4.YYYY) * b1 +
      a3.shuffle(Float32x4.ZZZZ) * b2 +
      a3.shuffle(Float32x4.WWWW) * b3;
}
