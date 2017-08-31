// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

class NDDataType {
  final int depth;
  final int blockDepth;
  final int blockSize;

  final bool isFloat;
  final bool isInteger;
  final bool isBoolean;
  final bool isString;
  final bool isGeneric;
  final bool isUnsigned;
  final bool isClamped;

  static const NDDataType float32 =
      const NDDataType._(isFloat: true, depth: 32);
  static const NDDataType float32VBlocked =
      const NDDataType._(isFloat: true, depth: 32, blockDepth: 2);
  static const NDDataType float64 =
      const NDDataType._(isFloat: true, depth: 64);
  static const NDDataType int8 = const NDDataType._(isInteger: true, depth: 8);
  static const NDDataType uint8 =
      const NDDataType._(isInteger: true, isUnsigned: true, depth: 8);
  static const NDDataType uint8Clamped = const NDDataType._(
      isInteger: true, isUnsigned: true, isClamped: true, depth: 8);
  static const NDDataType int16 =
      const NDDataType._(isInteger: true, depth: 16);
  static const NDDataType uint16 =
      const NDDataType._(isInteger: true, isUnsigned: true, depth: 16);
  static const NDDataType int32 =
      const NDDataType._(isInteger: true, depth: 32);
  static const NDDataType uint32 =
      const NDDataType._(isInteger: true, isUnsigned: true, depth: 32);
  static const NDDataType int64 =
      const NDDataType._(isInteger: true, depth: 64);
  static const NDDataType uint64 =
      const NDDataType._(isInteger: true, isUnsigned: true, depth: 64);
  static const NDDataType boolean = const NDDataType._(isBoolean: true);
  static const NDDataType string = const NDDataType._(isString: true);
  static const NDDataType generic = const NDDataType._(isGeneric: true);
  static const NDDataType unknown = const NDDataType._();

  const NDDataType._(
      {this.isFloat = false,
      this.isInteger = false,
      this.isBoolean = false,
      this.isClamped = false,
      this.isGeneric = false,
      this.isString = false,
      this.isUnsigned = false,
      this.depth,
      this.blockDepth = 0})
      : this.blockSize = 1 << blockDepth;

  bool get isUnknown => this == unknown;

  bool get isNumeric => isFloat || isInteger;

  bool get isBlocked => blockSize > 1;

  bool isCastableTo(NDDataType toDataType) =>
      this == toDataType ||
      (isNumeric && toDataType.isNumeric) ||
      (isNumeric && toDataType.isBoolean) ||
      (isBoolean && toDataType.isNumeric);

  bool isCompatibleWith(NDDataType dataType2) =>
      isUnknown || dataType2.isUnknown || this == dataType2;

  NDDataType mergeWith(NDDataType dataType2) {
    if (isUnknown) {
      return dataType2;
    } else if (dataType2.isUnknown) {
      return this;
    } else if (this == dataType2) {
      return this;
    } else {
      throw new UnsupportedError("Merge $this and $dataType2 data type");
    }
  }

  @override
  String toString() {
    switch (this) {
      case float32:
        return "float32";
      case float32VBlocked:
        return "float32VBlocked";
      case float64:
        return "float64";
      case uint8:
        return "uint8";
      case uint8Clamped:
        return "uint8Clamped";
      case int16:
        return "int16";
      case uint16:
        return "uint16";
      case int32:
        return "int32";
      case uint32:
        return "uint32";
      case int64:
        return "int64";
      case uint64:
        return "uint64";
      case boolean:
        return "boolean";
      case string:
        return "string";
      case generic:
        return "generic";
      case unknown:
        return "unknown";
      default:
        throw new StateError("DEAD CODE");
    }
  }
}
