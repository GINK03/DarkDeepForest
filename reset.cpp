#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "RandGen.hpp"
#include "TreeNode.hpp"
#include "DecisionTree.hpp"
#include "RandomForest.hpp"
using namespace std;
#define SZ(a) ((int)(a).size())
 
typedef double FeatureType;
typedef int AnswerType;
 
 
 
int main()
{
  int numAll;         // 全データ数
  int numTrainings;   // 訓練データ数
  int numTests;       // テストデータ数
  int numFeatures;    // 説明変数の数
 
  // y = f(x0,x1,x2,...)
  // x0,x1,x2は説明変数です。コード上ではfeatureと命名してます。
  // yは目的変数です。コード上ではanswerという命名をしてます。
 
 
  cin >> numAll >> numTrainings >> numTests >> numFeatures;
  assert(numTrainings+numTests<=numAll);
 
  // 全データ
  vector < vector <FeatureType> > allFeatures(numAll, vector <FeatureType> (numFeatures));
  vector < AnswerType >       allAnswers(numAll);
 
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
 
  // シャッフル用
  vector < int > shuffleTable;
  for (int i = 0; i < numTrainings+numTests; ++i)
  {
    shuffleTable.emplace_back(i);
  }
  random_shuffle(shuffleTable.begin(), shuffleTable.end());
 
  // 訓練データ
  vector < vector <FeatureType> > trainingFeatures(numTrainings, vector <FeatureType>(numFeatures));
  vector < AnswerType >       trainingAnswers(numTrainings);
  for (int i = 0; i < numTrainings; ++i)
  {
    trainingFeatures[i] = allFeatures[shuffleTable[i]];
    trainingAnswers[i]  = allAnswers[shuffleTable[i]];
  }
 
  // テストデータ
  vector < vector <FeatureType> > testFeatures(numTests, vector <FeatureType>(numFeatures));
  vector < AnswerType >       testAnswers(numTests);
  for (int i = 0; i < numTests; ++i)
  {
    testFeatures[i] = allFeatures[shuffleTable[numTrainings+i]];
    testAnswers[i]  = allAnswers[shuffleTable[numTrainings+i]];
  }
 
  // ランダムフォレストを使って予測
  RandomForest* rf = new RandomForest();
 
  // 木を徐々に増やしていく
  int numTrees = 0;
  for (int k = 0; k < 20; ++k)
  {
    // 学習
    const int numAdditionalTrees = 1;
    rf->train(trainingFeatures, trainingAnswers, numAdditionalTrees, 1);
    numTrees += numAdditionalTrees;
 
    // 予測と結果表示
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
