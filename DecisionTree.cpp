#include<vector>
#include "rand_gen.cpp" 
#include "Defines.hpp"
typedef double FeatureType;
typedef int AnswerType;
static RandXor  randxor; 

using namespace std;
enum 
{
  LEFT,
  RIGHT,
  NUM_LR,
};

class TreeNode {
public:
  bool leaf;
  int level;
  int featureID;
  FeatureType value;
  AnswerType answer; 
  vector <int> bags; 
  int left;
  int right;
public:
  TreeNode() {
    leaf = false;
    level = -1;
    featureID = -1;
    value = 0;
    answer = 0;
    left = -1;
    right = -1;
  }
};
class DecisionTree
{
public:
  DecisionTree() { }
  DecisionTree(const vector<vector<FeatureType> >& features,
    const vector<AnswerType>& answers,
    int minNodeSize,
    int maxLevel,
    int numRandomFeatures,
    int numRandomPositions)
  {
    const int numData       = SZ(features);
    const int numFeatures   = SZ(features[0]);
    assert(numData==SZ(answers));
    assert(numData>1);
 
    TreeNode root;  
    root.level = 0;
 
    root.bags.resize(numData);
    for (int i = 0; i < numData; i++)
    {
      root.bags[i] = randxor.random()%numData;
    }
    m_nodes.emplace_back(root);
 
    int curNode = 0;
    while (curNode < SZ(m_nodes))
    {
      TreeNode &node = m_nodes[curNode];
      bool equal = true; 
      for (int i=1;i<SZ(node.bags);i++) 
      {
        if (answers[node.bags[i]] != answers[node.bags[i - 1]])
        {
          equal = false;
          break;
        }
      }
      if (equal || SZ(node.bags) <= minNodeSize || node.level >= maxLevel)
      {
        setLeaf( node, curNode, answers );
        continue;
      }
      int bestFeatureID = -1;
      int bestLeft=0, bestRight=0;
      FeatureType bestValue = 0;
      double bestGini = 1e99; 

      for(int i=0;i<numRandomFeatures;i++)
      { 
        const int featureID = randxor.random()%numFeatures;
        for(int j=0;j<numRandomPositions;j++)   
        {
          const int positionID = randxor.random()%SZ(node.bags);
          const FeatureType splitValue = features[node.bags[positionID]][featureID];
          int total[NUM_LR] = {}; 
          int freq[NUM_LR][NUM_CLASSES]={};
          for (auto p : node.bags)
          {
            int lr = RIGHT;
            if (features[p][featureID] < splitValue)
            {
              lr = LEFT;
            }
            total[lr]++;
            freq[lr][answers[p]]++;
          }
          if (total[LEFT] == 0 || total[RIGHT] == 0) continue;
          double gini = 0;
          for(int lr = 0; lr < NUM_LR; ++lr)
          {
            double tmpGini = 1.0;
            for(int c=0; c<NUM_CLASSES;++c)
            {
              double ratio = (double)freq[lr][c]/total[lr];
              tmpGini -= ratio*ratio;
            }
            gini += tmpGini * total[lr]/SZ(node.bags);
          }
 
          if (gini < bestGini)
          {
            bestGini = gini;
            bestValue = splitValue;
            bestFeatureID = featureID;
            bestLeft = total[LEFT];
            bestRight = total[RIGHT];
          }
        }
      }
 
      if (bestLeft == 0 || bestRight == 0)
      {
        setLeaf( node, curNode, answers );
        continue;
      }
 
      TreeNode left;
      TreeNode right;
 
      left.level = right.level = node.level + 1;
      node.featureID = bestFeatureID;
      node.value = bestValue;
      node.left = SZ(m_nodes);
      node.right = SZ(m_nodes) + 1;
 
      left.bags.resize(bestLeft);
      right.bags.resize(bestRight);
      for (int p : node.bags)
      {
        if (features[p][node.featureID] < node.value)
        {
          left.bags[--bestLeft] = p;
        }
        else
        {
          right.bags[--bestRight] = p;
        }
      }
 
      m_nodes.emplace_back(left);
      m_nodes.emplace_back(right);
      curNode++;
    }
  }
 
  AnswerType estimate(const vector<FeatureType>& features) const
  {
    const TreeNode *pNode = &m_nodes[0];
    while (true)
    {
      if (pNode->leaf) 
      {
        break;
      }
      const int nextNodeID = features[pNode->featureID] < pNode->value ? pNode->left : pNode->right;
      pNode = &m_nodes[nextNodeID];
    }
    return pNode->answer;
  }
  private:
  void setLeaf( TreeNode& node, int& curNode, const vector<AnswerType>& answers ) const
  {
    node.leaf = true;
    int freq[NUM_CLASSES]={};
    for (int p : node.bags) 
    {
      freq[answers[p]]++;
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
    node.answer = bestC;
    curNode++;
  }
  vector<TreeNode> m_nodes;
};
