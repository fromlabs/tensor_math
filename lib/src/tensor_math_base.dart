// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as dart_math;

import 'package:collection/collection.dart';

const DeepCollectionEquality _deepCollectionEquality =
    const DeepCollectionEquality();

int rank(value) {
  var rank = 0;
  var element = value;
  while (element is List) {
    rank++;
    element = element[0];
  }
  return rank;
}

int length(value) =>
    shape(value).reduce((value, dimensions) => value * dimensions);

List<int> shape(value) {
  var shape = <int>[];
  dynamic element = value;
  while (element is List) {
    shape.add(element.length);
    element = element[0];
  }
  return shape;
}

dynamic add(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 + value2);
}

dynamic adds(List values) => values.reduce((total, value) => add(total, value));

dynamic sub(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 - value2);
}

dynamic neg(value) =>
    _elementWiseUnaryOperation(value, shape(value), (value) => -value);

dynamic mul(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 * value2);
}

dynamic div(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 / value2);
}

dynamic inv(value) =>
    _elementWiseUnaryOperation(value, shape(value), (value) => 1 / value);

dynamic exp(value) => _elementWiseUnaryOperation(
    value, shape(value), (value) => dart_math.exp(value));

dynamic log(value) => _elementWiseUnaryOperation(
    value, shape(value), (value) => dart_math.log(value));

dynamic abs(value) => _elementWiseUnaryOperation(value, shape(value), (value) {
      num numValue = value;
      return numValue.abs();
    });

dynamic sign(value) => _elementWiseUnaryOperation(value, shape(value), (value) {
      num numValue = value;
      return numValue.sign;
    });

dynamic sum(value, {List<num> axes}) => _sum(value, shape(value));

dynamic mean(value, {List<num> axes}) => sum(value) / length(value);

dynamic transpose(value, {List<num> axes}) {
  var s = shape(value);

  var axes2 = axes;
  if (axes != null) {
    if (axes.length != s.length || new Set.from(axes).length != axes.length) {
      throw new ArgumentError("Invalid axes $axes for matrix of shape $shape");
    }
  } else {
    axes2 = new List.generate(s.length, (index) => s.length - index - 1);
  }

  // TODO to implement transpose
  throw new UnimplementedError("to implement transpose");
}

dynamic matMul(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  if (shape1.length < 2) {
    throw new ArgumentError("Shape rank < 2: $shape1");
  } else if (shape2.length < 2) {
    throw new ArgumentError("Shape rank < 2: $shape2");
  } else if (shape1.length != shape2.length) {
    throw new ArgumentError("Shapes different: $shape1 != $shape2");
  } else {
    var headShape1 =
        shape1.length > 2 ? shape1.sublist(0, shape1.length - 3) : [];
    var headShape2 =
        shape2.length > 2 ? shape2.sublist(0, shape2.length - 3) : [];

    var matShape1 =
        headShape1.isEmpty ? shape1 : shape1.sublist(headShape1.length);
    var matShape2 =
        headShape2.isEmpty ? shape2 : shape2.sublist(headShape2.length);

    if (headShape1.isEmpty ||
        _deepCollectionEquality.equals(headShape1, headShape2)) {
      if (matShape1.last == matShape2.first) {
        if (headShape1.isEmpty) {
          return _matMul2d(value1, value2, matShape1, matShape2);
        } else {
          // TODO to implement matMul
          throw new UnimplementedError("to implement matMul");
        }
      } else {
        throw new ArgumentError(
            "Shape tail dimensions incompatible: $matShape1, $matShape2 [${matShape1.last} != ${matShape2.first}]");
      }
    } else {
      throw new ArgumentError(
          "Shape head dimensions different: $headShape1 != $headShape2");
    }
  }
}

dynamic select(condition, thenValue, elseValue) {
  var conditionShape = shape(condition);
  var thenShape = shape(thenValue);
  var elseShape = shape(elseValue);

  var resultShape = _broadcastedShape(
      conditionShape, _broadcastedShape(thenShape, elseShape));

  return _elementWiseTernaryOperation(
      condition,
      thenValue,
      elseValue,
      conditionShape,
      thenShape,
      elseShape,
      resultShape,
      (condition, thenValue, elseValue) => condition ? thenValue : elseValue);
}

dynamic equal(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 == value2);
}

dynamic notEqual(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 != value2);
}

dynamic less(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 < value2);
}

dynamic lessEqual(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 <= value2);
}

dynamic greater(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 > value2);
}

dynamic greaterEqual(value1, value2) {
  var shape1 = shape(value1);
  var shape2 = shape(value2);

  var resultShape = _broadcastedShape(shape1, shape2);

  return _elementWiseBinaryOperation(value1, value2, shape1, shape2,
      resultShape, (value1, value2) => value1 >= value2);
}

dynamic _broadcastedShape(shape1, shape2) {
  var broadcastedShape = [];

  var index1 = shape1.length - 1;
  var index2 = shape2.length - 1;
  while (index1 >= 0 || index2 >= 0) {
    var d1 = index1 >= 0 ? shape1[index1] : 1;
    var d2 = index2 >= 0 ? shape2[index2] : 1;
    if (d1 == 1 || d1 == d2) {
      broadcastedShape.add(d2);
    } else if (d2 == 1) {
      broadcastedShape.add(d1);
    } else {
      throw new ArgumentError("Shapes not broadcastable: $shape1 != $shape2");
    }

    index1--;
    index2--;
  }
  return broadcastedShape.reversed.toList();
}

dynamic _matMul2d(value1, value2, List<int> shape1, List<int> shape2) {
  var result = [];
  for (int row1 = 0; row1 < shape1.first; row1++) {
    var row = [];
    for (int column2 = 0; column2 < shape2.last; column2++) {
      var sumValue = 0;
      for (int index = 0; index < shape1.last; index++) {
        sumValue += value1[row1][index] * value2[index][column2];
      }
      row.add(sumValue);
    }
    result.add(row);
  }
  return result;
}

dynamic _elementWiseUnaryOperation(
    value, List<int> resultShape, unaryOperation(value)) {
  if (resultShape.isEmpty) {
    return unaryOperation(value);
  } else {
    var result = [];

    for (int index = 0; index < resultShape.first; index++) {
      result.add(_elementWiseUnaryOperation(
          value[index], resultShape.sublist(1), unaryOperation));
    }

    return result;
  }
}

dynamic _elementWiseBinaryOperation(value1, value2, List<int> shape1,
    List<int> shape2, List<int> resultShape, binaryOperation(value1, value2)) {
  if (resultShape.isEmpty) {
    return binaryOperation(value1, value2);
  } else {
    var result = [];

    for (int index = 0; index < resultShape.first; index++) {
      var element1;
      var elementShape1;
      if (shape1.length == resultShape.length) {
        element1 = value1[shape1.first > 1 ? index : 0];
        elementShape1 = shape1.sublist(1);
      } else {
        element1 = value1;
        elementShape1 = shape1;
      }

      var element2;
      var elementShape2;
      if (shape2.length == resultShape.length) {
        element2 = value2[shape2.first > 1 ? index : 0];
        elementShape2 = shape2.sublist(1);
      } else {
        element2 = value2;
        elementShape2 = shape2;
      }

      var value = _elementWiseBinaryOperation(element1, element2, elementShape1,
          elementShape2, resultShape.sublist(1), binaryOperation);

      result.add(value);
    }

    return result;
  }
}

dynamic _elementWiseTernaryOperation(
    value1,
    value2,
    value3,
    List<int> shape1,
    List<int> shape2,
    List<int> shape3,
    List<int> resultShape,
    ternaryOperation(value1, value2, value3)) {
  if (resultShape.isEmpty) {
    return ternaryOperation(value1, value2, value3);
  } else {
    var result = [];

    for (int index = 0; index < resultShape.first; index++) {
      var element1;
      var elementShape1;
      if (shape1.length == resultShape.length) {
        element1 = value1[shape1.first > 1 ? index : 0];
        elementShape1 = shape1.sublist(1);
      } else {
        element1 = value1;
        elementShape1 = shape1;
      }

      var element2;
      var elementShape2;
      if (shape2.length == resultShape.length) {
        element2 = value2[shape2.first > 1 ? index : 0];
        elementShape2 = shape2.sublist(1);
      } else {
        element2 = value2;
        elementShape2 = shape2;
      }

      var element3;
      var elementShape3;
      if (shape3.length == resultShape.length) {
        element3 = value3[shape3.first > 1 ? index : 0];
        elementShape3 = shape3.sublist(1);
      } else {
        element3 = value3;
        elementShape3 = shape3;
      }

      var value = _elementWiseTernaryOperation(
          element1,
          element2,
          element3,
          elementShape1,
          elementShape2,
          elementShape3,
          resultShape.sublist(1),
          ternaryOperation);

      result.add(value);
    }

    return result;
  }
}

dynamic _sum(value, List<int> resultShape) {
  if (resultShape.isEmpty) {
    return value;
  } else {
    var result = 0;

    for (int index = 0; index < resultShape.first; index++) {
      result += _sum(value[index], resultShape.sublist(1));
    }

    return result;
  }
}
