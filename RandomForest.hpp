#pragma once
#include <vector>
#include "Tree.hpp"
#include "DecisionTree.hpp"
#include <boost/variant.hpp>
typedef double FeatureType;
typedef int AnswerType;
namespace RandomForest {

class RandomForest {
private:
  std::vector<Tree*> m_trees;    // たくさんの決定木
public:
  RandomForest();
  ~RandomForest(); 
  void clear();
 
  // 訓練
  // 繰り返し呼ぶことで木を追加することもできる。
  // features           説明変数x0,x1,x2...のセット
  // answers            目的変数yのセット
  // treesNo　　　　　　追加する木の数
  // minNodeSize        ノード内
  void train(const std::vector<std::vector<FeatureType>>& features,
    const std::vector<AnswerType>& answers,
    int treesNo,
    int minNodeSize);
 
  // ランダムネスを高めた、RandomForest
  // 多分こんな感じでいいだと思う
  // 訓練
  // 繰り返し呼ぶことで木を追加することもできる。
  // features           説明変数x0,x1,x2...のセット
  // answers            目的変数yのセット
  // treesNo　　　　　　追加する木の数
  // minNodeSize        ノード内
  void train_noisy(const std::vector<std::vector<FeatureType>>& features,
    const std::vector<AnswerType>& answers,
    int treesNo,
    int minNodeSize, 
    int maxLevel,
    int numRandomFeatures,
    int numRandomPositions);

 
  // 分類の予測
  // features テスト用の説明変数x0,x1,x2のセット
  // 返り値   目的変数yの予測値
  AnswerType estimateClassification(std::vector<FeatureType> &features);
  std::vector<double> predict(std::vector<FeatureType> &features);
 
};

};
