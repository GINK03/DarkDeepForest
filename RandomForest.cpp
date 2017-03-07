#include "RandomForest.hpp"
#include <vector>
namespace RandomForest {

template <typename T>
int sanitize(T& t) {
  return static_cast<int>(t.size());
}

RandomForest::RandomForest() {
  clear();
}

void RandomForest::clear() {
  m_trees.clear();
}

void RandomForest::train(const std::vector<std::vector<FeatureType>>& features,
    const std::vector<AnswerType>& answers,
    int treesNo,
    int minNodeSize)
{
  for(int i=0;i<treesNo;i++)
  {
    m_trees.emplace_back(DecisionTree(features, answers, minNodeSize, 16, 2, 5));
  }
}

void RandomForest::train_noisy(const std::vector<std::vector<FeatureType>>& features,
    const std::vector<AnswerType>& answers,
    int treesNo,
    int minNodeSize, 
    int maxLevel,
    int numRandomFeatures,
    int numRandomPositions)
{
  for(int i=0;i<treesNo;i++)
  {
    //五番目の引数がnumRandomFeatures(乱雑さを表現している。これを素性の数の限界値まで引き上げる)
    m_trees.emplace_back(DecisionTree(features, answers, minNodeSize, maxLevel, numRandomFeatures, numRandomPositions));
  }
}
 
AnswerType RandomForest::estimateClassification(std::vector <FeatureType> &features)
{
  if (sanitize(m_trees) == 0)
  {
    return 0;
  }
  // 多数決
  int freq[NUM_CLASSES]={};
  for(int i=0;i<sanitize(m_trees);i++)
  {
    freq[m_trees[i].estimate(features)]++;
  }
  int bestFreq = -1;
  int bestC    = -1;
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

std::vector<double> RandomForest::predict(std::vector<FeatureType> &features) 
{
  if (sanitize(m_trees) == 0)
  {
    return std::vector<double>();
  }
  // 多数決
  int freq[NUM_CLASSES]={};
  for(int i=0;i < sanitize(m_trees);i++)
  {
    freq[m_trees[i].estimate(features)]++;
  }
  std::vector<double> result(NUM_CLASSES);
  for(int i=0;i < NUM_CLASSES; i++) {
    result[i] = freq[i];

  }
  /*int bestFreq = -1;
  int bestC    = -1;
  for (int c = 0; c < NUM_CLASSES; ++c)
  {
    if(freq[c]>bestFreq)
    {
      bestFreq = freq[c];
      bestC = c;
    }
  }*/
  /*for(int i = 0;i < NUM_CLASSES; i++) {
    if( bestC == i ) {
      result[i] = 1.;
    } else {
      result[i] = 0.;
    }
  }*/
  return result;
}

};
