#pragma once
#include <vector>
#include "DecisionTree.hpp"
#define SZ(a) ((int)(a).size())
typedef double FeatureType;
typedef int AnswerType;
using namespace std;

class RandomForest {
private:
  vector<DecisionTree> m_trees;    // たくさんの決定木
public:
  RandomForest();
 
  void clear()
  {
    m_trees.clear();
  }
 
  // 訓練
  // 繰り返し呼ぶことで木を追加することもできる。
  // features           説明変数x0,x1,x2...のセット
  // answers            目的変数yのセット
  // treesNo　　　　　　追加する木の数
  // minNodeSize        ノード内
 
  void train(const vector <vector <FeatureType> >& features,
    const vector <AnswerType>& answers,
    int treesNo,
    int minNodeSize)
  {
    for(int i=0;i<treesNo;i++)
    {
      m_trees.emplace_back(DecisionTree(features, answers, minNodeSize, 16, 2, 5));
    }
  }
 
 
  // 分類の予測
  // features テスト用の説明変数x0,x1,x2のセット
  // 返り値   目的変数yの予測値
  AnswerType estimateClassification(vector <FeatureType> &features)
  {
    if (SZ(m_trees) == 0)
    {
      return 0;
    }
 
    // 多数決
    int freq[NUM_CLASSES]={};
    for(int i=0;i<SZ(m_trees);i++)
    {
      freq[m_trees[i].estimate(features)]++;
    }
 
    int bestFreq = -1;
    int bestC = -1;
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
      if(freq[c]>bestFreq)
      {
        bestFreq = freq[c];
        bestC = c;
      }
    }
 
    return bestC;
  }
 
};
