LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
export LD_LIBRARY_PATH
rm *.so deepforest
clang++ -std=c++14 RandGen.cpp -shared  -o librandgen.so -fPIC
clang++ -std=c++14 TreeNode.cpp -shared  -o libtreenode.so -fPIC
clang++ -std=c++14 DecisionTree.cpp -shared  -o libdecisiontree.so -fPIC
clang++ -std=c++14 RandomTree.cpp -shared  -o librandomtree.so -fPIC
clang++ -std=c++14 RandomForest.cpp -shared  -o librandomforest.so -fPIC
clang++ DeepForest.cpp -std=c++14 -I./ -I/usr/include/boost -L./ -lrandgen -ltreenode -lrandomtree -ldecisiontree -lrandomforest -o deepforest -v 
cat ./iris.txt | ./deepforest
