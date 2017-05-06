// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

export "nd_shape_impl.dart" show broadcastIterable;

abstract class NDShape {
  int get dimension;

  int get length;

  bool get isUnknownDimension;

  bool get isUnknownLength;

  bool get isScalar;

  bool get isVector;

  bool get isMatrix;

  bool get isTensor3D;

  bool get isTensor4D;

  List<int> get dimensions;

  int get(int axe);

  int operator [](int axe);

  NDShape transpose({List<int> permutationAxis});

  NDShape reduce({List<int> reductionAxis});

  NDShape merge(NDShape shape2);

  NDShape broadcast(NDShape shape2);

  NDShape matMul(NDShape shape2);

  NDShape reshape({List<int> newDimensions});

  NDShape tile(List<int> multiplies);
}
