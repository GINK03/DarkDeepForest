#pragma once
#include "RandGen.hpp"
#include <random>
#define SZ(a) ((int)(a).size())
static const int NUM_CLASSES = 3; // 分類のときのクラス数
static RandXor  randxor;    // マルチスレッド対応にするなら、木ごとに乱数用オブジェクトを用意して、シードを変えましょう。
enum 
{
  LEFT,
  RIGHT,
  NUM_LR,
};
namespace Alice {
  static std::random_device rd;
  static std::mt19937 mt(rd());
}
