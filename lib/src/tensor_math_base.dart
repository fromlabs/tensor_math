// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as dart_math;

dynamic add(value1, value2) => value1 + value2;

dynamic adds(List values) => values.reduce((total, value) => add(total, value));

dynamic sub(value1, value2) => value1 - value2;

dynamic neg(value) => -value;

dynamic mul(value1, value2) => value1 * value2;

dynamic div(numerator, denominator) => numerator / denominator;

dynamic inv(value) => 1 / value;

dynamic exp(value) => dart_math.exp(value);

dynamic log(value) => dart_math.log(value);

dynamic select(bool condition, thenValue, elseValue) =>
    condition ? thenValue : elseValue;

dynamic abs(value) {
  num numValue = value;
  return numValue.abs();
}

dynamic sign(value) {
  num numValue = value;
  return numValue.sign;
}

dynamic sum(value) => value;

bool equal(value1, value2) => value1 == value2;

bool notEqual(value1, value2) => value1 != value2;

bool less(value1, value2) => value1 < value2;

bool lessEqual(value1, value2) => value1 <= value2;

bool greater(value1, value2) => value1 > value2;

bool greaterEqual(value1, value2) => value1 >= value2;
