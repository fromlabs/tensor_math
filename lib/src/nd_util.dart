// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

List<T> permute<T>(List<T> list, List<int> permutedIndexes) {
  var newList = permutedIndexes.map((index) => list[index]).toList();
  for (var index = newList.length; index < list.length; index++) {
    newList.add(list[index]);
  }
  return newList;
}

bool debug(Object object) {
  print("DEBUG: $object");
  return true;
}
