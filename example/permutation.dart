void main() {
  print(generate(5).toList());
}

void main2() {
  var n = 4;

  for (var d = 1; d <= n; d++) {
    test(n, d);
  }
}

void test(int n, int d) {
  var i = 0;
  var indexes = new List(d);
  indexes[0] = 0;

  for (;;) {
    if (indexes[i] > n - 1) {
      i--;

      if (i >= 0) {
        indexes[i]++;
      } else {
        break;
      }
    } else if (i < d - 1) {
      // inner
      i++;

      indexes[i] = indexes[i - 1] + 1;
    } else {
      // last
      print(indexes);

      indexes[i]++;
    }
  }
}

Iterable<List<int>> generate(int n) sync* {
  for (var d = 1; d <= n; d++) {
    var i = 0;
    var indexes = new List(d);
    indexes[0] = 0;

    for (;;) {
      if (indexes[i] > n - 1) {
        i--;

        if (i >= 0) {
          indexes[i]++;
        } else {
          break;
        }
      } else if (i < d - 1) {
        // inner
        i++;

        indexes[i] = indexes[i - 1] + 1;
      } else {
        // last
        yield new List.from(indexes);

        indexes[i]++;
      }
    }
  }
}
