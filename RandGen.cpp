#include "RandGen.hpp"
RandXor::RandXor(){
  x=123456789;
  y=362436069;
  z=521288629;
  w= 88675123; 
}
unsigned int RandXor::random(){
  unsigned int t=(x^(x<<11));
  x=y;
  y=z;
  z=w; 
  return( w=(w^(w>>19))^(t^(t>>8)) );
}
// this is test
int main() {
  int i = 0;

}
