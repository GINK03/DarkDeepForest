#pragma once
class RandXor
{
  private:
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
  public:
  RandXor()
  {
    init();
  }
  void init()
  {
    x=123456789;
    y=362436069;
    z=521288629;
    w= 88675123;
  }
 
  unsigned int random()
  {
    unsigned int t;
    t= (x^(x<<11)) ;
    x=y;
    y=z;
    z=w; 
    return( w=(w^(w>>19))^(t^(t>>8)) );
  }
};
