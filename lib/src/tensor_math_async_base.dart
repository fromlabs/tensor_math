// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as dart_math;

import "dart:async";

import "tensor_math_base.dart" as sync_math;

typedef R UnaryFunction<R, V1>(V1 value1);

typedef R BinaryFunction<R, V1, V2>(V1 value1, V2 value2);

typedef R TernaryFunction<R, V1, V2, V3>(V1 value1, V2 value2, V3 value3);

FutureOr add(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync_math.add);

FutureOr adds(List<FutureOr> values) =>
    values.reduce((total, value) => add(total, value));

FutureOr neg(FutureOr value1) => _unaryFunction(value1, sync_math.neg);

FutureOr sub(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync_math.sub);

FutureOr mul(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync_math.mul);

FutureOr div(FutureOr numerator, FutureOr denominator) =>
    _binaryFunction(numerator, denominator, sync_math.div);

FutureOr inv(FutureOr value) => _unaryFunction(value1, sync_math.inv);

FutureOr exp(FutureOr value) => _unaryFunction(value1, sync_math.exp);

FutureOr log(FutureOr value) => _unaryFunction(value1, sync_math.log);

FutureOr select(
        FutureOr<bool> condition, FutureOr thenValue, FutureOr elseValue) =>
    _ternaryFunction(condition, thenValue, elseValue, sync_math.select);

FutureOr abs(FutureOr value1) => _unaryFunction(value1, sync_math.abs);

FutureOr sign(FutureOr value) => _unaryFunction(value1, sync_math.sign);

// FutureOr sum(FutureOr value) => value;

FutureOr<bool> equal(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync.equal);

FutureOr<bool> notEqual(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync.notEqual);

FutureOr<bool> less(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync.less);

FutureOr<bool> lessEqual(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync.lessEqual);

FutureOr<bool> greater(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync.greater);

FutureOr<bool> greaterEqual(FutureOr value1, FutureOr value2) =>
    _binaryFunction(value1, value2, sync.greaterEqual);

FutureOr<R> _unaryFunction<R, V1>(
    FutureOr<V1> value1, UnaryFunction<R, V1> function) {
  if (value1 is Future<V1>) {
    return value1.then((value) => function(value));
  } else {
    return function(value1);
  }
}

FutureOr<R> _binaryFunction<R, V1, V2>(FutureOr<V1> value1, FutureOr<V2> value2,
    BinaryFunction<R, V1, V2> function) {
  if (value1 is Future<V1> || value2 is Future<V2>) {
    var value1Future = value1 is Future<V1> ? value1 : new Future.value(value1);
    var value2Future = value2 is Future<V2> ? value2 : new Future.value(value2);
    return Future.wait([value1Future, value2Future]).then((values) {
      print("values[0]: ${values[0]}");
      print("values[1]: ${values[1]}");
      return function(values[0], values[1]);
    });
  } else {
    return function(value1, value2);
  }
}

FutureOr<R> _ternaryFunction<R, V1, V2, V3>(
    FutureOr<V1> value1,
    FutureOr<V2> value2,
    FutureOr<V3> value3,
    TernaryFunction<R, V1, V2, V3> function) {
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
