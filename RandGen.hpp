#pragma once
// 乱数は、xorを使ってますが、メルセンヌツイスターの方がよいかも知れません。
class RandXor
{
private:
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
public:
  RandXor();
  unsigned int random();
  static int rand_range_1_10();
  static int rand_range_1_32();
};
