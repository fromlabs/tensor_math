import "dart:typed_data";

void main() {
  var a0 = 32988912.0;
  var b0 = 342287.0;
  var expectedC0 = a0 + b0;

  var a = new Float32x4.splat(a0);
  var b = new Float32x4.splat(b0);
  var c = a + b;

  var expectedC = new Float32x4.splat(expectedC0);

  print(a);
  print(b);
  print(c);
  print(expectedC);
  print(expectedC0);
  
  var cs = new Float32List(1);
  cs[0] = expectedC0;
  print(cs);
}
