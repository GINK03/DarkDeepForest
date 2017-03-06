#pragma once
#include <vector>
class TreeNode {
public:
  bool leaf;          // 葉（=子がない）ならtrue
  int level;          // ノードの深さ。ルートノードは0
  int featureID;      // 説明変数ID。x0, x1, x2... の0,1,2の部分
  double value;  // 分割する値
  int answer;  // ノード内（=領域内）の目的変数yの平均値
  std::vector<int> bags;  // ノード内（=領域内）に含まれるデータのID
  int left;           // 左側の子のノードID
  int right;          // 右側の子のノードID
public:
  TreeNode();
};
