clang++ -std=c++1z -shared Defines.cpp -o libdefines.so
clang++ -std=c++1z -I./ -L./ -ldefines jiji.cpp -o jiji_rf -v 
