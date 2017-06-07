import "dart:typed_data";

Float32x4List toFloat32x4List1(int rows, int columns, List<double> values) {
  var data = new List<Float32x4>.generate((rows * columns) >> 2, (index) {
    var i = index * 4;

    return new Float32x4(
        values[i], values[i + 1], values[i + 2], values[i + 3]);
  });

  return new Float32x4List.fromList(data);
}

List<double> fromFloat32x4List1(int rows, int columns, Float32x4List values) {
  return new List<double>.generate(rows * columns, (index) {
    var i = index ~/ 4;
    var offset = index % 4;

    var value = values[i];

    switch (offset) {
      case 0:
        return value.x;
      case 1:
        return value.y;
      case 2:
        return value.z;
      case 3:
        return value.w;
    }
  });
}

Float32x4List toFloat32x4List2(int rows, int columns, List<double> values) {
  var blockPerRow = columns >> 2;

  var data = new List<Float32x4>.generate(rows * blockPerRow, (index) {
    var blockIndex = index ~/ 4;
    var blockOffset = index % 4;

    var blockRow = blockIndex ~/ blockPerRow;
    var blockColumn = blockIndex % blockPerRow;

    var row = blockRow * 4 + blockOffset;
    var column = blockColumn * 4;

    var i = row * columns + column;

    return new Float32x4(
        values[i], values[i + 1], values[i + 2], values[i + 3]);
  });

  return new Float32x4List.fromList(data);
}

Float32x4List toFloat32x4List3(int rows, int columns, List<double> values) {
  var blockPerColumn = rows >> 2;

  var data = new List<Float32x4>.generate(columns * blockPerColumn, (index) {
    var blockIndex = index ~/ 4;
    var blockOffset = index % 4;

    var blockColumn = blockIndex ~/ blockPerColumn;
    var blockRow = blockIndex % blockPerColumn;

    var row = blockRow * 4 + blockOffset;
    var column = blockColumn * 4;

    var i = row * columns + column;

    return new Float32x4(
        values[i], values[i + 1], values[i + 2], values[i + 3]);
  });

  return new Float32x4List.fromList(data);
}

List<double> fromFloat32x4List2(int rows, int columns, Float32x4List values) {
  var blockPerRow = columns >> 2;

  return new List<double>.generate(rows * columns, (index) {
    var row = index ~/ columns;
    var column = index % columns;

    var blockRow = row ~/ 4;
    var blockOffset = row % 4;
    var blockColumn = column ~/ 4;

    var blockIndex = blockRow * blockPerRow + blockColumn;

    var i = blockIndex * 4 + blockOffset;
    var offset = column % 4;

    var value = values[i];

    switch (offset) {
      case 0:
        return value.x;
      case 1:
        return value.y;
      case 2:
        return value.z;
      case 3:
        return value.w;
    }
  });
}

List<double> fromFloat32x4List3(int rows, int columns, Float32x4List values) {
  var blockPerColumn = rows >> 2;

  return new List<double>.generate(rows * columns, (index) {
    var row = index ~/ columns;
    var column = index % columns;

    var blockRow = row ~/ 4;
    var blockOffset = row % 4;
    var blockColumn = column ~/ 4;

    var blockIndex = blockColumn * blockPerColumn + blockRow;

    var i = blockIndex * 4 + blockOffset;
    var offset = column % 4;

    var value = values[i];

    switch (offset) {
      case 0:
        return value.x;
      case 1:
        return value.y;
      case 2:
        return value.z;
      case 3:
        return value.w;
    }
  });
}

void matMul1(
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
  var blockPerRow1 = columns1 >> 2;
  var blockPerRow2 = columns2 >> 2;
  var resultBlockPerRow = columns2 >> 2;

  for (var row1 = 0; row1 < rows1; row1 += 4) {
    for (var column2 = 0; column2 < columns2; column2 += 4) {
      for (var i = 0; i < columns1; i += 4) {
        var blockRow1 = row1 ~/ 4;
        var blockOffset1 = row1 % 4;
        var blockColumn1 = i ~/ 4;
        var blockIndex1 = blockRow1 * blockPerRow1 + blockColumn1;
        var i10 = blockIndex1 * 4 + blockOffset1;
        var i11 = i10 + 1;
        var i12 = i11 + 1;
        var i13 = i12 + 1;

        var blockRow2 = i ~/ 4;
        var blockOffset2 = i % 4;
        var blockColumn2 = column2 ~/ 4;
        var blockIndex2 = blockRow2 * blockPerRow2 + blockColumn2;
        var i20 = blockIndex2 * 4 + blockOffset2;
        var i21 = i20 + 1;
        var i22 = i21 + 1;
        var i23 = i22 + 1;

        var resultBlockRow = row1 ~/ 4;
        var resultBlockOffset = row1 % 4;
        var resultBlockColumn = column2 ~/ 4;
        var resultBlockIndex =
            resultBlockRow * resultBlockPerRow + resultBlockColumn;
        var ir0 = resultBlockIndex * 4 + resultBlockOffset;
        var ir1 = ir0 + 1;
        var ir2 = ir1 + 1;
        var ir3 = ir2 + 1;

        matMulInternal(list1, i10, i11, i12, i13, list2, i20, i21, i22, i23,
            result, ir0, ir1, ir2, ir3);
      }
    }
  }
}

void matMul23(
    Float32x4List list1,
    int rows1,
    int columns1,
    Float32x4List list2,
    int rows2,
    int columns2,
    Float32x4List result,
    int resultRows,
    int resultColumns) {
  var blockPerRow1 = columns1 >> 2;
  var blockPerColumn2 = rows2 >> 2;
  var resultBlockPerRow = columns2 >> 2;

  for (var row1 = 0; row1 < rows1; row1 += 4) {
    for (var column2 = 0; column2 < columns2; column2 += 4) {
      for (var i = 0; i < columns1; i += 4) {
        var blockRow1 = row1 ~/ 4;
        var blockOffset1 = row1 % 4;
        var blockColumn1 = i ~/ 4;
        var blockIndex1 = blockRow1 * blockPerRow1 + blockColumn1;
        var i10 = blockIndex1 * 4 + blockOffset1;
        var i11 = i10 + 1;
        var i12 = i11 + 1;
        var i13 = i12 + 1;

        var blockRow2 = i ~/ 4;
        var blockOffset2 = i % 4;
        var blockColumn2 = column2 ~/ 4;
        var blockIndex2 = blockColumn2 * blockPerColumn2 + blockRow2;
        var i20 = blockIndex2 * 4 + blockOffset2;
        var i21 = i20 + 1;
        var i22 = i21 + 1;
        var i23 = i22 + 1;

        var resultBlockRow = row1 ~/ 4;
        var resultBlockOffset = row1 % 4;
        var resultBlockColumn = column2 ~/ 4;
        var resultBlockIndex =
            resultBlockRow * resultBlockPerRow + resultBlockColumn;
        var ir0 = resultBlockIndex * 4 + resultBlockOffset;
        var ir1 = ir0 + 1;
        var ir2 = ir1 + 1;
        var ir3 = ir2 + 1;

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
