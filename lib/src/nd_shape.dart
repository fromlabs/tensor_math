// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "nd_shapeable.dart";

import "nd_shape_impl.dart";

// export "nd_shape_impl.dart" show broadcastIterable;

abstract class NDShape implements NDShapeable {
  factory NDShape(List<int> dimensions) =>
      new NDShapeImpl(new List.from(dimensions));

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

  NDShape merge(NDShape shape2);

  NDShape broadcast(NDShape shape2);

  NDShape reduce({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDShape reshape({List<int> newDimensions});

  @override
  NDShape tile(List<int> multiplies);

  @override
  NDShape transpose({List<int> permutationAxis});

  @override
  NDShape matMul(covariant NDShape shape2);

  @override
  NDShape reduceSum({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDShape reduceMean({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDShape reduceMax({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDShape reduceAny({List<int> reductionAxis, bool keepDimensions = false});

  @override
  NDShape argMax({int axis});

  @override
  NDShape abs();

  @override
  NDShape exp();

  @override
  NDShape inv();

  @override
  NDShape log();

  @override
  NDShape neg();

  @override
  NDShape sign();

  @override
  NDShape add(covariant NDShape shape2);

  @override
  NDShape sub(covariant NDShape shape2);

  @override
  NDShape mul(covariant NDShape shape2);

  @override
  NDShape div(covariant NDShape shape2);

  @override
  NDShape not();

  @override
  NDShape isEqual(covariant NDShape shape2);

  @override
  NDShape isNotEqual(covariant NDShape shape2);

  @override
  NDShape isGreater(covariant NDShape shape2);

  @override
  NDShape isGreaterOrEqual(covariant NDShape shape2);

  @override
  NDShape isLess(covariant NDShape shape2);

  @override
  NDShape isLessOrEqual(covariant NDShape shape2);

  @override
  NDShape select(covariant NDShape thenShape, covariant NDShape elseShape);

  @override
  NDShape operator -();

  @override
  NDShape operator +(covariant NDShape shape2);

  @override
  NDShape operator -(covariant NDShape shape2);

  @override
  NDShape operator *(covariant NDShape shape2);

  @override
  NDShape operator /(covariant NDShape shape2);

  @override
  NDShape operator >(covariant NDShape shape2);

  @override
  NDShape operator >=(covariant NDShape shape2);

  @override
  NDShape operator <(covariant NDShape shape2);

  @override
  NDShape operator <=(covariant NDShape shape2);
}
