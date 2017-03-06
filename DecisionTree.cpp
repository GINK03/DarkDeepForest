#include "DecisionTree.hpp"
#include <iostream>
DecisionTree::DecisionTree(){ }

DecisionTree::DecisionTree(const vector <vector <FeatureType> >& features,
  const vector <AnswerType>& answers,
  int minNodeSize,
  int maxLevel,
  int numRandomFeatures,
  int numRandomPositions)
{
  const int numData       = SZ(features);
  const int numFeatures   = SZ(features[0]);
  assert(numData==SZ(answers));
  assert(numData>1);

  TreeNode root;  // ルートのノード
  root.level = 0;

  root.bags.resize(numData);
  for (int i = 0; i < numData; i++)
  {
    // ここで、同じIDが選ばれる可能性があるが、問題なし。
    root.bags[i] = randxor.random()%numData;
  }
  m_nodes.emplace_back(root);

  int curNode = 0;
  // m_nodesに子ノードがどんどん追加されていく幅優先探索
  while (curNode < SZ(m_nodes))
  {
    TreeNode &node = m_nodes[curNode];

    // 現在のノードに入っている目的変数が、すべて同じかどうかを調べる
    // （その場合は、ノードを分ける必要がなくなる）
    bool equal = true; // すべて同じならtrue
    for (int i=1;i<SZ(node.bags);i++) 
    {
      if (answers[node.bags[i]] != answers[node.bags[i - 1]])
      {
        equal = false;
        break;
      }
    }

    // 葉になる条件のチェック
    if (equal || SZ(node.bags) <= minNodeSize || node.level >= maxLevel)
    {
      // 葉にして子ノードは増やさない。
      setLeaf( node, curNode, answers );
      continue;
    }

    // どこで分けるのがベストかを調べる
    int bestFeatureID = -1;
    int bestLeft=0, bestRight=0;
    FeatureType bestValue = 0;
    double bestGini = 1e99;  // ジニ係数

    for(int i=0;i<numRandomFeatures;i++)
    {
      // x0,x1,x2...の、どの軸で分けるかを決める
      const int featureID = randxor.random()%numFeatures;
      for(int j=0;j<numRandomPositions;j++)   // どの位置で分けるか
      {
        const int positionID = randxor.random()%SZ(node.bags);
        const FeatureType splitValue = features[node.bags[positionID]][featureID];

        int total[NUM_LR] = {};  // splitValue未満, splitValue以上の個数
        int freq[NUM_LR][NUM_CLASSES]={};  // [どっち側か][クラス] = 個数

        for (int p : node.bags)
        {
          int lr = RIGHT;
          if (features[p][featureID] < splitValue)
          {
            lr = LEFT;
          }
          total[lr]++;
          freq[lr][answers[p]]++;
        }

        // nodeBagのデータが"未満"か"以上"のどちらかに全部偏ってるので
        // 分け方として意味がないので、すぐやめる。
        if (total[LEFT] == 0 || total[RIGHT] == 0)
          continue;

        // ジニ係数を求める（分類用）
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
        // BEGIN CUT HERE
          cout << " curNode=" << curNode << " gini=" << gini << endl;
        // END CUT HERE


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

    // 左か右にどちらかに偏るような分け方しかできなかった場合は、葉にする
    // （すべての分け方を試すわけではないので、こういうことは起こりえます）
    if (bestLeft == 0 || bestRight == 0)
    {
      setLeaf( node, curNode, answers );
      continue;
    }

    // うまく分けれたので、新しい子ノードを２つ追加する
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

AnswerType DecisionTree::estimate(const vector<FeatureType>& features){
    // ルートからたどるだけ
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

void DecisionTree::setLeaf(TreeNode& node, int& curNode, const vector<AnswerType>& answers) {
    node.leaf = true;
 
    // 分類の場合は、多数決（ここで平均を使う手法もあるよう）
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
