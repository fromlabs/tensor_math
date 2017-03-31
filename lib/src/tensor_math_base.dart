// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as dart_math;

import "dart:async";

typedef R UnaryFunction<R, V1>(V1 value1);

typedef R BinaryFunction<R, V1, V2>(V1 value1, V2 value2);

typedef R TernaryFunction<R, V1, V2, V3>(V1 value1, V2 value2, V3 value3);

FutureOr add(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, _addInternal);

FutureOr adds(List<FutureOr> values) =>
    values.reduce((total, value) => add(total, value));

FutureOr neg(FutureOr value1) => _unaryFunction(value1, _negInternal);

FutureOr sub(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, _subInternal);

FutureOr mul(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, _mulInternal);

FutureOr div(FutureOr numerator, FutureOr denominator) =>
    _binaryFunction(numerator, denominator, _divInternal);

dynamic inv(value) => 1 / value;

dynamic exp(value) => dart_math.exp(value);

dynamic log(value) => dart_math.log(value);

FutureOr select(
        FutureOr<bool> condition, FutureOr thenValue, FutureOr elseValue) =>
    _ternaryFunction(condition, thenValue, elseValue, _selectInternal);

FutureOr abs(FutureOr value1) => _unaryFunction(value1, _absInternal);

dynamic sign(value) {
  num numValue = value;
  return numValue.sign;
}

dynamic sum(value) => value;

bool equal(value1, value2) => value1 == value2;

bool notEqual(value1, value2) => value1 != value2;

bool less(value1, value2) => value1 < value2;

bool lessEqual(value1, value2) => value1 <= value2;

FutureOr<bool> greater(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, _greaterInternal);

bool greaterEqual(value1, value2) => value1 >= value2;

FutureOr<R> _unaryFunction<R, V1>(FutureOr<V1> value1, UnaryFunction<R, V1> function) {
  if (value1 is Future<V1>) {
    return value1.then((value) => function(value));
  } else {
    return function(value1);
  }
}

FutureOr<R> _binaryFunction<R, V1, V2>(
    FutureOr<V1> value1, FutureOr<V2> value2, BinaryFunction<R, V1, V2> function) {
  if (value1 is Future<V1> || value2 is Future<V2>) {
    var value1Future = value1 is Future<V1> ? value1 : new Future.value(value1);
    var value2Future = value2 is Future<V2> ? value2 : new Future.value(value2);
    return Future.wait([value1Future, value2Future]).then(
        (values) {
          print("values[0]: ${values[0]}");
          print("values[1]: ${values[1]}");
          return function(values[0], values[1]);
        });
  } else {
    return function(value1, value2);
  }
}

FutureOr<R> _ternaryFunction<R, V1, V2, V3>(FutureOr<V1> value1, FutureOr<V2> value2,
    FutureOr<V3> value3, TernaryFunction<R, V1, V2, V3> function) {
  if (value1 is Future<V1> || value2 is Future<V2> || value3 is Future<V3>) {
    var value1Future = value1 is Future<V1> ? value1 : new Future.value(value1);
    var value2Future = value2 is Future<V2> ? value2 : new Future.value(value2);
    var value3Future = value3 is Future<V3> ? value3 : new Future.value(value3);
    return Future.wait([value1Future, value2Future, value3Future]).then(
        (values) => function(values[0], values[1], values[2]));
  } else {
    return function(value1, value2, value3);
  }
}

dynamic _negInternal(value1) => -value1;

dynamic _addInternal(value1, value2) => value1 + value2;

dynamic _subInternal(value1, value2) => value1 - value2;

dynamic _mulInternal(value1, value2) => value1 * value2;

dynamic _divInternal(numerator, denominator) => numerator / denominator;

dynamic _absInternal(value) {
  num numValue = value;
  return numValue.abs();
}

bool _greaterInternal(value1, value2) => value1 > value2;

FutureOr _selectInternal(bool condition, thenValue, elseValue) =>
    condition ? thenValue : elseValue;
