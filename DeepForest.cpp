#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "RandGen.hpp"
#include <memory>
#include <list>
#include <boost/algorithm/string.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/functions.hpp>
#include <boost/range/as_literal.hpp>
#include "TreeNode.hpp"
#include "DecisionTree.hpp"
#include "RandomForest.hpp"
#include <utility> 
typedef double FeatureType;
typedef int AnswerType;

using namespace boost;
struct LayerContainer {
public:
  std::vector<std::pair<AnswerType, std::vector<FeatureType>>> answer_features;
  std::vector<std::shared_ptr<RandomForest::RandomForest>> rf_container;
  LayerContainer() {
    rf_container = std::vector<std::shared_ptr<RandomForest::RandomForest>>(5);
    answer_features = std::vector<std::pair<AnswerType, std::vector<FeatureType>>>();
  }
};

int ensemble(const vector<double>& in1, 
             const vector<double>& in2,
             const vector<double>& in3,
	     const vector<double>& in4) {
  vector<double> tmp(in1.size());
  for(int i=0; i< in1.size(); i++ ) {
    tmp[i] = (in1[i] + in2[i] + in3[i] + in4[i])/4.;
  }
  double max = *std::max_element(tmp.begin(), tmp.end());
  int res = -1;
  for(int i=0; i< in1.size(); i++) {
    if(tmp[i] == max) {
       res = i;
    }
  }
  return res;
}

int main()
{
  int numAll;         // 全データ数
  int numTrainings;   // 訓練データ数
  int numTests;       // テストデータ数
  int numFeatures;    // 説明変数の数
 
  // y = f(x0,x1,x2,...)
  // x0,x1,x2は説明変数です。コード上ではfeatureと命名してます。
  // yは目的変数です。コード上ではanswerという命名をしてます。
  std::cin >> numAll >> numTrainings >> numTests >> numFeatures;
  assert(numTrainings+numTests<=numAll);
 
  //NOTE; DeepForestに対応するため、featの数は先頭に,NUM_CLASSES分追加する
  // 全データ
  std::vector<std::vector<FeatureType>> allFeatures(numAll, std::vector<FeatureType>(numFeatures + NUM_CLASSES*4));
  std::vector<AnswerType> allAnswers(numAll);
 
  for(int i = 0 ; i < numAll; ++i)
  {
    //for (int k = 0; k < numFeatures; ++k)
    //NOTE: NUM_CLASSを初期値に
    for_each( irange(0, numFeatures),  [&](int k) {
        cin >> allFeatures[i][k+NUM_CLASSES*4];
    } );
    cin >> allAnswers[i];
    assert(allAnswers[i]>=0);
    assert(allAnswers[i]<NUM_CLASSES);
  }
 
  // シャッフル用
  std::vector<int> shuffleTable;
  for (int i = 0; i < numTrainings+numTests; ++i)
  {
    shuffleTable.emplace_back(i);
  }
  std::random_shuffle(shuffleTable.begin(), shuffleTable.end());

  std::vector<std::shared_ptr<LayerContainer>> layer_container = {};
  //シード用のRF作成
  auto seed = std::shared_ptr<LayerContainer>(new LayerContainer());
  // 訓練データ
  std::vector<std::vector<FeatureType>> trainingFeatures(numTrainings, std::vector<FeatureType>(numFeatures));
  std::vector<AnswerType> trainingAnswers(numTrainings);
  for (int i = 0; i < numTrainings; ++i)
  {
    trainingFeatures[i] = allFeatures[shuffleTable[i]];
    trainingAnswers[i]  = allAnswers[shuffleTable[i]];
    std::pair<AnswerType, std::vector<FeatureType>> pair;
    pair.first = allAnswers[shuffleTable[i]];
    pair.second = allFeatures[shuffleTable[i]];
  }

  // テストデータ
  std::vector<std::vector<FeatureType>> testFeatures(numTests, std::vector<FeatureType>(numFeatures));
  std::vector<AnswerType> testAnswers(numTests);
  for (int i = 0; i < numTests; ++i)
  {
    testFeatures[i] = allFeatures[shuffleTable[numTrainings+i]];
    testAnswers[i]  = allAnswers[shuffleTable[numTrainings+i]];
  }
  // ランダムフォレストを使って予測
  auto rf1 = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
  auto rf2 = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
  auto rf1_noisy = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
  auto rf2_noisy = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
  for(auto tuple: { std::make_tuple(250, rf1, true), std::make_tuple(250, rf2, true), std::make_tuple(250, rf1_noisy, false), std::make_tuple(250, rf2_noisy, false) } ) {
    // iterationのTreeの数
    // 木を徐々に増やしていく
    int numTrees = 0;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, 10);
    std::uniform_int_distribution<> distr_1_32(1, 32);
    for_each(irange(0, std::get<0>(tuple)), [&](int k) {
      // 学習
      const int numAdditionalTrees = 1;
      if( std::get<2>(tuple) == true ) {
        std::get<1>(tuple)->train(trainingFeatures, trainingAnswers, numAdditionalTrees, 1);
      } else {
        std::get<1>(tuple)->train_noisy(trainingFeatures, trainingAnswers, numAdditionalTrees, distr(eng), distr_1_32(eng), distr(eng), distr_1_32(eng));
      }
      numTrees += numAdditionalTrees;
    });
  }
  seed->rf_container[0] = rf1;
  seed->rf_container[1] = rf2;
  seed->rf_container[2] = rf1_noisy;
  seed->rf_container[3] = rf2_noisy;
  // 予測と結果表示
  //cout << "-----" << endl;
  //cout << "numTrees=" << numTrees << endl;
  double totalError1 = 0.0;
  double totalError2 = 0.0;
  for (int i = 0; i < numTests; ++i)
  {
    const std::vector<double> myAnswer1 = rf1->predict(testFeatures[i]);
    const std::vector<double> myAnswer2 = rf2->predict(testFeatures[i]);
    const std::vector<double> myAnswer3 = rf1_noisy->predict(testFeatures[i]);
    const std::vector<double> myAnswer4 = rf2_noisy->predict(testFeatures[i]);
    int res = ensemble(myAnswer1, myAnswer2, myAnswer3, myAnswer4);
    if( res != testAnswers[i])
      totalError1 += 1.0;
  }
  for (int i = 0;i<numTrainings; i++)
  {
    std::vector<FeatureType> nextfeat;
    const std::vector<double> myAnswer1 = rf1->predict(trainingFeatures[i]);
    const std::vector<double> myAnswer2 = rf2->predict(trainingFeatures[i]);
    const std::vector<double> myAnswer3 = rf1_noisy->predict(trainingFeatures[i]);
    const std::vector<double> myAnswer4 = rf2_noisy->predict(trainingFeatures[i]);
    for(auto real_feat: trainingFeatures[i]) {
       nextfeat.emplace_back(real_feat);
    }
    for(int i=0; i<NUM_CLASSES; i++) {
       nextfeat[i + NUM_CLASSES*0] = myAnswer1[i];
    }
    for(int i=0; i<NUM_CLASSES; i++) {
       nextfeat[i + NUM_CLASSES*1] = myAnswer2[i];
    }
    for(int i=0; i<NUM_CLASSES; i++) {
       nextfeat[i + NUM_CLASSES*2] = myAnswer3[i];
    }
    for(int i=0; i<NUM_CLASSES; i++) {
       nextfeat[i + NUM_CLASSES*3] = myAnswer4[i];
    }
    for(auto d: nextfeat) {
      cout << d << " ";
    }
    cout << "d2 " << endl;
    const auto real_ans = trainingAnswers[i];
    std::pair<AnswerType, std::vector<FeatureType>> pair;
    pair.first = real_ans;
    pair.second = nextfeat;
    seed->answer_features.emplace_back(pair);
  }
  layer_container.emplace_back(seed);
  cout << "totalError1=" << totalError1 << endl;
  for(int DEEP=0; DEEP < 100; DEEP++) {
    //シード用のRF作成
    auto next = std::shared_ptr<LayerContainer>(new LayerContainer());
    auto last = layer_container[layer_container.size()-1];
    vector<vector<FeatureType>> next_train_feats;
    vector<AnswerType> next_train_ans;
    for( auto answer_feats: last->answer_features ) {
       next_train_feats.emplace_back(answer_feats.second);
       next_train_ans.emplace_back(answer_feats.first);
    }
    // ランダムフォレストを使って予測
    auto rf1 = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
    auto rf2 = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
    for(auto tuple: { std::make_tuple(250, rf1, true), std::make_tuple(250, rf2, true), std::make_tuple(250, rf1_noisy, false), std::make_tuple(250, rf2_noisy, false) } ) {
      // iterationのTreeの数
      // 木を徐々に増やしていく
      int numTrees = 0;
      std::random_device rd;
      std::mt19937 eng(rd());
      std::uniform_int_distribution<> distr(0, 10);
      std::uniform_int_distribution<> distr_1_32(1, 32);
      for_each(irange(0, std::get<0>(tuple)), [&](int k) {
        // 学習
        const int numAdditionalTrees = 1;
        if( std::get<2>(tuple) == true ) {
          std::get<1>(tuple)->train(trainingFeatures, trainingAnswers, numAdditionalTrees, 1);
        } else {
          std::get<1>(tuple)->train_noisy(trainingFeatures, trainingAnswers, numAdditionalTrees, distr(eng), distr_1_32(eng), distr(eng), distr_1_32(eng));
        }
        numTrees += numAdditionalTrees;
      });
    }
    next->rf_container[0] = rf1;
    next->rf_container[1] = rf2;
    next->rf_container[2] = rf1_noisy;
    next->rf_container[3] = rf2_noisy;
    double totalError1 = 0.0;
    double totalError2 = 0.0;
    for (int i = 0; i < numTests; ++i)
    {
      const std::vector<double> myAnswer1 = rf1->predict(testFeatures[i]);
      const std::vector<double> myAnswer2 = rf2->predict(testFeatures[i]);
      const std::vector<double> myAnswer3 = rf1_noisy->predict(testFeatures[i]);
      const std::vector<double> myAnswer4 = rf2_noisy->predict(testFeatures[i]);
      int res = ensemble(myAnswer1, myAnswer2, myAnswer3, myAnswer4);
      if( res != testAnswers[i])
        totalError1 += 1.0;
    }
    for (int i = 0;i<numTrainings; i++)
    {
      std::vector<FeatureType> nextfeat;
      const std::vector<double> myAnswer1 = rf1->predict(trainingFeatures[i]);
      const std::vector<double> myAnswer2 = rf2->predict(trainingFeatures[i]);
      const std::vector<double> myAnswer3 = rf1_noisy->predict(trainingFeatures[i]);
      const std::vector<double> myAnswer4 = rf2_noisy->predict(trainingFeatures[i]);
      for(auto real_feat: trainingFeatures[i]) {
         nextfeat.emplace_back(real_feat);
      }
      for(int i=0; i<NUM_CLASSES; i++) {
        nextfeat[i + NUM_CLASSES*0] = myAnswer1[i];
      }
      for(int i=0; i<NUM_CLASSES; i++) {
        nextfeat[i + NUM_CLASSES*1] = myAnswer2[i];
      }
      for(int i=0; i<NUM_CLASSES; i++) {
        nextfeat[i + NUM_CLASSES*2] = myAnswer3[i];
      }
      for(int i=0; i<NUM_CLASSES; i++) {
        nextfeat[i + NUM_CLASSES*3] = myAnswer3[i];
      }
      const auto real_ans = trainingAnswers[i];
      std::pair<AnswerType, std::vector<FeatureType>> pair;
      pair.first = real_ans;
      pair.second = nextfeat;
      next->answer_features.emplace_back(pair);
    }
    layer_container.emplace_back(next);
    cout << "DEEP iter " << DEEP << endl; 
    cout << "totalError1=" << totalError1 << endl;
  }
 
  return 0;
}
