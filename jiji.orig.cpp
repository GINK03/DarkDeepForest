#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
 
using namespace std;
#define SZ(a) ((int)(a).size())
 
 
// $BMp?t$O!"(Bxor$B$r;H$C$F$^$9$,!"%a%k%;%s%L%D%$%9%?!<$NJ}$,$h$$$+$bCN$l$^$;$s!#(B
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
 
static RandXor  randxor;    // $B%^%k%A%9%l%C%IBP1~$K$9$k$J$i!"LZ$4$H$KMp?tMQ%*%V%8%'%/%H$rMQ0U$7$F!"%7!<%I$rJQ$($^$7$g$&!#(B
 
 
typedef double FeatureType;
typedef int AnswerType;
static const int NUM_CLASSES = 3; // $BJ,N`$N$H$-$N%/%i%9?t(B
enum 
{
	LEFT,
	RIGHT,
	NUM_LR,
};
 
struct TreeNode {
	bool leaf;          // $BMU!J(B=$B;R$,$J$$!K$J$i(Btrue
	int level;          // $B%N!<%I$N?<$5!#%k!<%H%N!<%I$O(B0
	int featureID;      // $B@bL@JQ?t(BID$B!#(Bx0, x1, x2... $B$N(B0,1,2$B$NItJ,(B
	FeatureType value;  // $BJ,3d$9$kCM(B
	AnswerType answer;  // $B%N!<%IFb!J(B=$BNN0hFb!K$NL\E*JQ?t(By$B$NJ?6QCM(B
	vector <int> bags;	// $B%N!<%IFb!J(B=$BNN0hFb!K$K4^$^$l$k%G!<%?$N(BID
	int left;           // $B:8B&$N;R$N%N!<%I(BID
	int right;          // $B1&B&$N;R$N%N!<%I(BID
 
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
 
	// $B3X=,!#71N}%G!<%?$r$$$l$F!"7hDjLZ$r:n@.$9$k!#(B
	// features           $B@bL@JQ?t(Bx0,x1,x2...
	// answers            $BL\E*JQ?t(By
	// minNodeSize        $B%N!<%IFb(B
	// maxLevel           $B%N!<%I$N?<$5$N:GBgCM(B
	// numRandomFeatures  $BNN0h$rJ,$1$k$H$-$K;n$9@bL@JQ?t!J%0%i%U$G$O<4!K$N?t(B
	// numRandomPositions $BNN0h$rJ,$1$k$H$-$K;n$9%G!<%?!J%0%i%U$G$OE@!K$N?t(B
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
 
		TreeNode root;  // $B%k!<%H$N%N!<%I(B
		root.level = 0;
 
		root.bags.resize(numData);
		for (int i = 0; i < numData; i++)
		{
			// $B$3$3$G!"F1$8(BID$B$,A*$P$l$k2DG=@-$,$"$k$,!"LdBj$J$7!#(B
			root.bags[i] = randxor.random()%numData;
		}
		m_nodes.emplace_back(root);
 
		int curNode = 0;
		// m_nodes$B$K;R%N!<%I$,$I$s$I$sDI2C$5$l$F$$$/I}M%@hC5:w(B
		while (curNode < SZ(m_nodes))
		{
			TreeNode &node = m_nodes[curNode];
 
			// $B8=:_$N%N!<%I$KF~$C$F$$$kL\E*JQ?t$,!"$9$Y$FF1$8$+$I$&$+$rD4$Y$k(B
			// $B!J$=$N>l9g$O!"%N!<%I$rJ,$1$kI,MW$,$J$/$J$k!K(B
			bool equal = true; // $B$9$Y$FF1$8$J$i(Btrue
			for (int i=1;i<SZ(node.bags);i++) 
			{
				if (answers[node.bags[i]] != answers[node.bags[i - 1]])
				{
					equal = false;
					break;
				}
			}
 
			// $BMU$K$J$k>r7o$N%A%'%C%/(B
			if (equal || SZ(node.bags) <= minNodeSize || node.level >= maxLevel)
			{
				// $BMU$K$7$F;R%N!<%I$OA}$d$5$J$$!#(B
				setLeaf( node, curNode, answers );
				continue;
			}
 
			// $B$I$3$GJ,$1$k$N$,%Y%9%H$+$rD4$Y$k(B
			int bestFeatureID = -1;
			int bestLeft=0, bestRight=0;
			FeatureType bestValue = 0;
			double bestGini = 1e99;  // $B%8%K78?t(B
 
			for(int i=0;i<numRandomFeatures;i++)
			{
				// x0,x1,x2...$B$N!"$I$N<4$GJ,$1$k$+$r7h$a$k(B
				const int featureID = randxor.random()%numFeatures;
				for(int j=0;j<numRandomPositions;j++)   // $B$I$N0LCV$GJ,$1$k$+(B
				{
					const int positionID = randxor.random()%SZ(node.bags);
					const FeatureType splitValue = features[node.bags[positionID]][featureID];
 
					int total[NUM_LR] = {};  // splitValue$BL$K~(B, splitValue$B0J>e$N8D?t(B
					int freq[NUM_LR][NUM_CLASSES]={};	// [$B$I$C$AB&$+(B][$B%/%i%9(B] = $B8D?t(B
 
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
 
					// nodeBag$B$N%G!<%?$,(B"$BL$K~(B"$B$+(B"$B0J>e(B"$B$N$I$A$i$+$KA4ItJP$C$F$k$N$G(B
					// $BJ,$1J}$H$7$F0UL#$,$J$$$N$G!"$9$0$d$a$k!#(B
					if (total[LEFT] == 0 || total[RIGHT] == 0)
						continue;
 
					// $B%8%K78?t$r5a$a$k!JJ,N`MQ!K(B
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
 
			// $B:8$+1&$K$I$A$i$+$KJP$k$h$&$JJ,$1J}$7$+$G$-$J$+$C$?>l9g$O!"MU$K$9$k(B
			// $B!J$9$Y$F$NJ,$1J}$r;n$9$o$1$G$O$J$$$N$G!"$3$&$$$&$3$H$O5/$3$j$($^$9!K(B
			if (bestLeft == 0 || bestRight == 0)
			{
				setLeaf( node, curNode, answers );
				continue;
			}
 
			// $B$&$^$/J,$1$l$?$N$G!"?7$7$$;R%N!<%I$r#2$DDI2C$9$k(B
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
 
	// $BM=B,(B
	// features $B%F%9%HMQ$N@bL@JQ?t(Bx0,x1,x2$B$N%;%C%H(B
	// $BJV$jCM(B   $BL\E*JQ?t(By$B$NM=B,CM(B
	AnswerType estimate(const vector <FeatureType>& features) const
	{
		// $B%k!<%H$+$i$?$I$k$@$1(B
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
 
	// node$B$rMU$K$7$F!"(BcurNode$B$r<!$N%N!<%I$X?J$a$k(B
	void setLeaf( TreeNode& node, int& curNode, const vector<AnswerType>& answers ) const
	{
		node.leaf = true;
 
		// $BJ,N`$N>l9g$O!"B??t7h!J$3$3$GJ?6Q$r;H$&<jK!$b$"$k$h$&!K(B
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
 
	vector < TreeNode > m_nodes;    // $B7hDjLZ$N%N!<%I$?$A!#(Bm_nodes[0]$B$,%k!<%H(B
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
 
	// $B71N}(B
	// $B7+$jJV$78F$V$3$H$GLZ$rDI2C$9$k$3$H$b$G$-$k!#(B
	// features           $B@bL@JQ?t(Bx0,x1,x2...$B$N%;%C%H(B
	// answers            $BL\E*JQ?t(By$B$N%;%C%H(B
	// treesNo$B!!!!!!!!!!!!DI2C$9$kLZ$N?t(B
	// minNodeSize        $B%N!<%IFb(B
 
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
 
 
	// $BJ,N`$NM=B,(B
	// features $B%F%9%HMQ$N@bL@JQ?t(Bx0,x1,x2$B$N%;%C%H(B
	// $BJV$jCM(B   $BL\E*JQ?t(By$B$NM=B,CM(B
	AnswerType estimateClassification(vector <FeatureType> &features)
	{
		if (SZ(m_trees) == 0)
		{
			return 0;
		}
 
		// $BB??t7h(B
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
	vector < DecisionTree > m_trees;    // $B$?$/$5$s$N7hDjLZ(B
};
 
int main()
{
	int numAll;         // $BA4%G!<%??t(B
	int numTrainings;   // $B71N}%G!<%??t(B
	int numTests;       // $B%F%9%H%G!<%??t(B
	int numFeatures;    // $B@bL@JQ?t$N?t(B
 
	// y = f(x0,x1,x2,...)
	// x0,x1,x2$B$O@bL@JQ?t$G$9!#%3!<%I>e$G$O(Bfeature$B$HL?L>$7$F$^$9!#(B
	// y$B$OL\E*JQ?t$G$9!#%3!<%I>e$G$O(Banswer$B$H$$$&L?L>$r$7$F$^$9!#(B
 
 
	cin >> numAll >> numTrainings >> numTests >> numFeatures;
	assert(numTrainings+numTests<=numAll);
 
	// $BA4%G!<%?(B
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
 
	// $B%7%c%C%U%kMQ(B
	vector < int > shuffleTable;
	for (int i = 0; i < numTrainings+numTests; ++i)
	{
		shuffleTable.emplace_back(i);
	}
	random_shuffle(shuffleTable.begin(), shuffleTable.end());
 
	// $B71N}%G!<%?(B
	vector < vector <FeatureType> > trainingFeatures(numTrainings, vector <FeatureType>(numFeatures));
	vector < AnswerType >       trainingAnswers(numTrainings);
	for (int i = 0; i < numTrainings; ++i)
	{
		trainingFeatures[i] = allFeatures[shuffleTable[i]];
		trainingAnswers[i]  = allAnswers[shuffleTable[i]];
	}
 
	// $B%F%9%H%G!<%?(B
	vector < vector <FeatureType> > testFeatures(numTests, vector <FeatureType>(numFeatures));
	vector < AnswerType >       testAnswers(numTests);
	for (int i = 0; i < numTests; ++i)
	{
		testFeatures[i] = allFeatures[shuffleTable[numTrainings+i]];
		testAnswers[i]  = allAnswers[shuffleTable[numTrainings+i]];
	}
 
	// $B%i%s%@%`%U%)%l%9%H$r;H$C$FM=B,(B
	RandomForest* rf = new RandomForest();
 
	// $BLZ$r=y!9$KA}$d$7$F$$$/(B
	int numTrees = 0;
	for (int k = 0; k < 20; ++k)
	{
		// $B3X=,(B
		const int numAdditionalTrees = 1;
		rf->train(trainingFeatures, trainingAnswers, numAdditionalTrees, 1);
		numTrees += numAdditionalTrees;
 
		// $BM=B,$H7k2LI=<((B
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
