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
enum class ForestType { Ordinal, Noisy, Odd };  
int ensemble(const vector<double> ins[]) {
  vector<double> tmp(ins[0].size());
  for(int i=0; i< ins[0].size(); i++ ) {
    tmp[i] = (ins[0][i]*1 + ins[1][i]*1 + ins[2][i]*1+ ins[3][i]*1)/4.;
  }
  double max = *std::max_element(tmp.begin(), tmp.end());
  int res = -1;
  for(int i=0; i< ins[0].size(); i++) {
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
 
  std::cin >> numAll >> numTrainings >> numTests >> numFeatures;
  assert(numTrainings+numTests<=numAll);
  std::vector<std::vector<FeatureType>> allFeatures(numAll, std::vector<FeatureType>(numFeatures + NUM_CLASSES*4));
  std::vector<AnswerType> allAnswers(numAll);
 
  for(int i = 0 ; i < numAll; ++i)
  {
    for_each( irange(0, numFeatures),  [&](int k) {
        cin >> allFeatures[i][k+NUM_CLASSES*4];
    } );
    cin >> allAnswers[i];
    assert(allAnswers[i]>=0);
    assert(allAnswers[i]<NUM_CLASSES);
  }
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
  int numTrees = 0;
  for(auto tuple: { std::make_tuple(10, rf1, ForestType::Ordinal), 
  		std::make_tuple(10, rf2, ForestType::Ordinal), 
		std::make_tuple(10, rf1_noisy, ForestType::Noisy), 
		std::make_tuple(10, rf2_noisy, ForestType::Odd) } ) {
    // iterationのTreeの数
    // 木を徐々に増やしていく
    for_each(irange(0, std::get<0>(tuple)), [&](int k) {
      // 学習
      const int numAdditionalTrees = 1;
      if( std::get<2>(tuple) == ForestType::Ordinal ) {
        std::get<1>(tuple)->train(trainingFeatures, trainingAnswers, numAdditionalTrees, 1);
      } else if ( std::get<2>(tuple) == ForestType::Noisy ){
        std::get<1>(tuple)->train_noisy(trainingFeatures, 
		trainingAnswers, 
		numAdditionalTrees, 
		RandXor::rand_range_1_10(), 
		RandXor::rand_range_1_32(), 
		RandXor::rand_range_1_10(), 
		RandXor::rand_range_1_32());
      } else if( std::get<2>(tuple) == ForestType::Odd ) {
        std::get<1>(tuple)->train_noisy(trainingFeatures, 
               trainingAnswers,
               numAdditionalTrees,
			1, 
			16,
			2,
			5 );
      }
      numTrees += numAdditionalTrees;
    });
  }
  seed->rf_container = {rf1, rf2, rf1_noisy, rf2_noisy};
  double totalError1 = 0.0;
  for (int i = 0; i < numTests; ++i)
  {
    const std::vector<double> myAnswers[] = { 
    	rf1->predict(testFeatures[i]), 
	rf2->predict(testFeatures[i]), 
	rf1_noisy->predict(testFeatures[i]), 
	rf2_noisy->predict(testFeatures[i]) };
    int res = ensemble(myAnswers);
    if( res != testAnswers[i])
      totalError1 += 1.0;
  }
  for (int i = 0;i<numTrainings; i++)
  {
    std::vector<FeatureType> nextfeat;
    const std::vector<double> myAnswers[] = {
    	rf1->predict(trainingFeatures[i]),
    	rf2->predict(trainingFeatures[i]),
    	rf1_noisy->predict(trainingFeatures[i]),
    	rf2_noisy->predict(trainingFeatures[i]) } ;
    for(auto real_feat: trainingFeatures[i]) {
       nextfeat.emplace_back(real_feat);
    }
    for(auto index: {0, 1, 2, 3}) { 
      for(int i=0; i<NUM_CLASSES; i++)
        nextfeat[i + NUM_CLASSES*index] = myAnswers[index][i]/100;
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
    auto rf1_noisy = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
    //auto rf2_noisy = std::shared_ptr<RandomForest::RandomForest>(new RandomForest::RandomForest());
    for(auto tuple: {
               std::make_tuple(10, rf1, ForestType::Ordinal), 
    	       std::make_tuple(10, rf2, ForestType::Ordinal), 
	       std::make_tuple(10, rf1_noisy, ForestType::Noisy),
	       std::make_tuple(10, rf2_noisy, ForestType::Noisy) } ) {
      int numTrees = 0;
      for_each(irange(0, std::get<0>(tuple)), [&](int k) {
        const int numAdditionalTrees = 1;
        if( std::get<2>(tuple) == ForestType::Ordinal ) {
          std::get<1>(tuple)->train(trainingFeatures, 
	  		trainingAnswers, 
			numAdditionalTrees, 
			1);
        } else if( std::get<2>(tuple) == ForestType::Noisy ) {
          std::get<1>(tuple)->train_noisy(trainingFeatures, 
	  		trainingAnswers, 
			numAdditionalTrees, 
			RandXor::rand_range_1_10(), 
			RandXor::rand_range_1_32(), 
			RandXor::rand_range_1_10(), 
			RandXor::rand_range_1_32() );
        } else if( std::get<2>(tuple) == ForestType::Odd ) {
          std::get<1>(tuple)->train_noisy(trainingFeatures, 
	                trainingAnswers,
			numAdditionalTrees,
			1, 
			16,
			2,
			5 );
	}
        numTrees += numAdditionalTrees;
      });
    }
    next->rf_container = {rf1, rf2, rf1_noisy, rf2_noisy} ;
    double totalError1 = 0.0;
    for (int i = 0; i < numTests; ++i)
    {
      const std::vector<double> myAnswers[] = {
      	rf1->predict(testFeatures[i]),
      	rf2->predict(testFeatures[i]),
      	rf1_noisy->predict(testFeatures[i]),
	rf2_noisy->predict(testFeatures[i]) } ;
      int res = ensemble(myAnswers);
      if( res != testAnswers[i])
        totalError1 += 1.0;
    }
    for (int i = 0;i<numTrainings; i++)
    {
      std::vector<FeatureType> nextfeat;
      const std::vector<double> myAnswers[] = {
      	rf1->predict(trainingFeatures[i]),
      	rf2->predict(trainingFeatures[i]),
      	rf1_noisy->predict(trainingFeatures[i]),
      	rf2_noisy->predict(trainingFeatures[i]) } ;
      for(auto real_feat: trainingFeatures[i]) {
         nextfeat.emplace_back(real_feat);
      }
      for(auto index: {0, 1, 2, 3} ) {
        for(int i=0; i<NUM_CLASSES; i++) {
          nextfeat[i + NUM_CLASSES*index] = myAnswers[index][i]/100;
        }
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
