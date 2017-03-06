#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "Defines.hpp"
#include "DecisionTree.cpp"
using namespace std;
template<typename T>
int SZ(T& a){
  return (int)(a.size());
}
 
class RandomForest {
  public:
  RandomForest()
  {
    clear();
  }
  void clear()
  {
    m_trees.clear();
  }
  void train(const vector<vector<FeatureType>>& features,
    const vector<AnswerType>& answers,
    int treesNo,
    int minNodeSize)
  {
    for(int i=0;i<treesNo;i++)
    {
      m_trees.emplace_back(DecisionTree(features, answers, minNodeSize, 16, 2, 5));
    }
  }
  AnswerType estimateClassification(vector <FeatureType> &features)
  {
    if (SZ(m_trees) == 0)
    {
      return 0;
    }
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
  private:
  vector<DecisionTree> m_trees; 
};
 
int main()
{
  int numAll;
  int numTrainings;
  int numTests;
  int numFeatures; 
  vector<vector<FeatureType>> allFeatures(numAll, vector<FeatureType>(numFeatures));
  vector<AnswerType>          allAnswers(numAll);
 
  for(int i = 0 ; i < numAll; ++i)
  {
    for (int k = 0; k < numFeatures; ++k)
    {
      cin >> allFeatures[i][k];
    }
    cin >> allAnswers[i];
    assert(allAnswers[i]>=0);
    assert(allAnswers[i]<NUM_CLASSES);
  }
  vector<int> shuffleTable;
  for (int i = 0; i < numTrainings+numTests; ++i)
  {
    shuffleTable.emplace_back(i);
  }
  random_shuffle(shuffleTable.begin(), shuffleTable.end());
  vector<vector<FeatureType>> trainingFeatures(numTrainings, vector <FeatureType>(numFeatures));
  vector<AnswerType>          trainingAnswers(numTrainings);
  for (int i = 0; i < numTrainings; ++i)
  {
    trainingFeatures[i] = allFeatures[shuffleTable[i]];
    trainingAnswers[i]  = allAnswers[shuffleTable[i]];
  }
  vector<vector<FeatureType>> testFeatures(numTests, vector<FeatureType>(numFeatures));
  vector<AnswerType>          testAnswers(numTests);
  for (int i = 0; i < numTests; ++i)
  {
    testFeatures[i] = allFeatures[shuffleTable[numTrainings+i]];
    testAnswers[i]  = allAnswers[shuffleTable[numTrainings+i]];
  }
  RandomForest* rf = new RandomForest();
  int numTrees = 0;
  for (int k = 0; k < 20; ++k)
  {
    const int numAdditionalTrees = 1;
    rf->train(trainingFeatures, trainingAnswers, numAdditionalTrees, 1);
    numTrees += numAdditionalTrees;
    cout << "-----" << endl;
    cout << "numTrees=" << numTrees << endl;
    double totalError = 0.0;
    for (int i = 0; i < numTests; ++i)
    {
      const AnswerType myAnswer = rf->estimateClassification(testFeatures[i]);
      int diff = 0;
      if(myAnswer!=testAnswers[i])
      {
        diff = 1;
      }
      totalError += diff;
    }
    cout << "totalError=" << totalError << endl;
  }
  delete rf;
  return 0; 
}
