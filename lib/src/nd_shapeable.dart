// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "nd_shape.dart";

abstract class NDShapeable {
  NDShape get shape;

  NDShapeable reshape({List<int> newDimensions});

  NDShapeable tile(List<int> multiplies);

  NDShapeable transpose({List<int> permutationAxis});

  NDShapeable matMul(value2);

  NDShapeable reduceSum({List<int> reductionAxis, bool keepDimensions = false});

  NDShapeable reduceMean({List<int> reductionAxis, bool keepDimensions = false});

  NDShapeable reduceMax({List<int> reductionAxis, bool keepDimensions = false});

  NDShapeable reduceAny({List<int> reductionAxis, bool keepDimensions = false});

  NDShapeable argMax({int axis});

  NDShapeable abs();

  NDShapeable exp();

  NDShapeable inv();

  NDShapeable log();

  NDShapeable neg();

  NDShapeable sign();

  NDShapeable add(value2);

  NDShapeable sub(value2);

  NDShapeable mul(value2);

  NDShapeable div(value2);

  NDShapeable not();

  NDShapeable isEqual(value2);

  NDShapeable isNotEqual(value2);

  NDShapeable isGreater(value2);

  NDShapeable isGreaterOrEqual(value2);

  NDShapeable isLess(value2);

  NDShapeable isLessOrEqual(value2);

  NDShapeable select(thenValue, elseValue);

  NDShapeable operator -();

  NDShapeable operator +(value2);

  NDShapeable operator -(value2);

  NDShapeable operator *(value2);

  NDShapeable operator /(value2);

  NDShapeable operator >(value2);

  NDShapeable operator >=(value2);

  NDShapeable operator <(value2);

  NDShapeable operator <=(value2);
}
