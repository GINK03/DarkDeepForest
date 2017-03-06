#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
 
using namespace std;
#define SZ(a) ((int)(a).size())
 
 
// 乱数は、xorを使ってますが、メルセンヌツイスターの方がよいかも知れません。
class RandXor
{
private:
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
public:
	RandXor()
	{
		init();
	}
 
	void init()
	{
		x=123456789;
		y=362436069;
		z=521288629;
		w= 88675123;
	}
 
	inline unsigned int random()
	{
		unsigned int t;
		t=(x^(x<<11));x=y;y=z;z=w; return( w=(w^(w>>19))^(t^(t>>8)) );
	}
};
 
static RandXor  randxor;    // マルチスレッド対応にするなら、木ごとに乱数用オブジェクトを用意して、シードを変えましょう。
 
 
typedef double FeatureType;
typedef int AnswerType;
static const int NUM_CLASSES = 3; // 分類のときのクラス数
enum 
{
	LEFT,
	RIGHT,
	NUM_LR,
};
 
struct TreeNode {
	bool leaf;          // 葉（=子がない）ならtrue
	int level;          // ノードの深さ。ルートノードは0
	int featureID;      // 説明変数ID。x0, x1, x2... の0,1,2の部分
	FeatureType value;  // 分割する値
	AnswerType answer;  // ノード内（=領域内）の目的変数yの平均値
	vector <int> bags;	// ノード内（=領域内）に含まれるデータのID
	int left;           // 左側の子のノードID
	int right;          // 右側の子のノードID
 
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
 
	// 学習。訓練データをいれて、決定木を作成する。
	// features           説明変数x0,x1,x2...
	// answers            目的変数y
	// minNodeSize        ノード内
	// maxLevel           ノードの深さの最大値
	// numRandomFeatures  領域を分けるときに試す説明変数（グラフでは軸）の数
	// numRandomPositions 領域を分けるときに試すデータ（グラフでは点）の数
	DecisionTree(const vector <vector <FeatureType> >& features,
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
					int freq[NUM_LR][NUM_CLASSES]={};	// [どっち側か][クラス] = 個数
 
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
//					cout << " curNode=" << curNode << " gini=" << gini << endl;
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
 
	// 予測
	// features テスト用の説明変数x0,x1,x2のセット
	// 返り値   目的変数yの予測値
	AnswerType estimate(const vector <FeatureType>& features) const
	{
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
 
private:
 
	// nodeを葉にして、curNodeを次のノードへ進める
	void setLeaf( TreeNode& node, int& curNode, const vector<AnswerType>& answers ) const
	{
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
 
	vector < TreeNode > m_nodes;    // 決定木のノードたち。m_nodes[0]がルート
};
 
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
 
private:
	vector < DecisionTree > m_trees;    // たくさんの決定木
};
 
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
//				cout << "Failure! i=" << i << " myAnswer=" << myAnswer << " testAnswer=" << testAnswers[i] << endl;
				diff = 1;
			}
			totalError += diff;
		}
		cout << "totalError=" << totalError << endl;
	}
 
	delete rf;
 
	return 0; 
}
