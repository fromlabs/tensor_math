// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as math;

import 'package:meta/meta.dart';

NDPadding2d createSamePadding2d(
    {@required int inputHeight,
    @required int inputWidth,
    @required int blockHeight,
    @required int blockWidth,
    int heightStride = 1,
    int widthStride = 1}) {
  var padAlongHeight;
  if (inputHeight % heightStride == 0) {
    padAlongHeight = math.max(blockHeight - heightStride, 0);
  } else {
    padAlongHeight = math.max(blockHeight - (inputHeight % heightStride), 0);
  }

  var padAlongWidth;
  if (inputWidth % widthStride == 0) {
    padAlongWidth = math.max(blockWidth - widthStride, 0);
  } else {
    padAlongWidth = math.max(blockWidth - (inputWidth % widthStride), 0);
  }

  var padTop = padAlongHeight ~/ 2;
  var padBottom = padAlongHeight - padTop;
  var padLeft = padAlongWidth ~/ 2;
  var padRight = padAlongWidth - padLeft;

  return new NDPadding2d(
      top: padTop, bottom: padBottom, left: padLeft, right: padRight);
}

class NDPadding2d {
  final int top;

  final int bottom;

  final int left;

  final int right;

  NDPadding2d({this.top = 0, this.bottom = 0, this.left = 0, this.right = 0});
}
