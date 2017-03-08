#pragma once
#include <vector>
#include "TreeNode.hpp"
typedef double FeatureType;
typedef int AnswerType;

class Tree {
public:
  Tree() { };
  Tree(const std::vector<std::vector<FeatureType>>& features,
    int minNodeSize,
    int maxLevel,
    int numRandomFeatures,
    int numRandomPositions) { };
  virtual ~Tree() { }; 
  virtual AnswerType estimate(const std::vector<FeatureType>& features) = 0;
private:
  // nodeを葉にして、curNodeを次のノードへ進める
  virtual void setLeaf(TreeNode& node, int& curNode, const std::vector<AnswerType>& answers) = 0 ;
};


