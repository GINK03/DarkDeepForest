#pragma once
#include "Defines.hpp"
#include <vector>
#include <assert.h>
#include "Tree.hpp"
#include "TreeNode.hpp"
#include "RandGen.hpp"
using namespace std;

typedef double FeatureType;
typedef int AnswerType;
class RandomTree : public Tree
{
public:
  RandomTree();
 
  // 学習。訓練データをいれて、決定木を作成する。
  // features           説明変数x0,x1,x2...
  // answers            目的変数y
  // minNodeSize        ノード内
  // maxLevel           ノードの深さの最大値
  // numRandomFeatures  領域を分けるときに試す説明変数（グラフでは軸）の数
  // numRandomPositions 領域を分けるときに試すデータ（グラフでは点）の数
  RandomTree(const vector<vector<FeatureType>>& features,
    const vector<AnswerType>& answers,
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
 
  vector<TreeNode> m_nodes;    // 決定木のノードたち。m_nodes[0]がルート
};
