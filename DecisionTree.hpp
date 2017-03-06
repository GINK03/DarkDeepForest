#pragma once
#include <vector>
#include <assert.h>
#include "TreeNode.hpp"
#include "Defines.hpp"
#include "RandGen.hpp"
using namespace std;
#define SZ(a) ((int)(a).size())

typedef double FeatureType;
typedef int AnswerType;
static RandXor  randxor;    // マルチスレッド対応にするなら、木ごとに乱数用オブジェクトを用意して、シードを変えましょう。
static const int NUM_CLASSES = 3; // 分類のときのクラス数
class DecisionTree
{
public:
  DecisionTree();
 
  // 学習。訓練データをいれて、決定木を作成する。
  // features           説明変数x0,x1,x2...
  // answers            目的変数y
  // minNodeSize        ノード内
  // maxLevel           ノードの深さの最大値
  // numRandomFeatures  領域を分けるときに試す説明変数（グラフでは軸）の数
  // numRandomPositions 領域を分けるときに試すデータ（グラフでは点）の数
  DecisionTree(const vector <vector <FeatureType> >& features,
    const vector <AnswerType>& answers,
    int minNodeSize,
    int maxLevel,
    int numRandomFeatures,
    int numRandomPositions);
 
  // 予測
  // features テスト用の説明変数x0,x1,x2のセット
  // 返り値   目的変数yの予測値
  AnswerType estimate(const vector<FeatureType>& features);
private:
 
  // nodeを葉にして、curNodeを次のノードへ進める
  void setLeaf( TreeNode& node, int& curNode, const vector<AnswerType>& answers );
 
  vector < TreeNode > m_nodes;    // 決定木のノードたち。m_nodes[0]がルート
};
