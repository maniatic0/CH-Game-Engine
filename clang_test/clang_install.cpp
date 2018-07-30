// * Taken from http://blog.johannesmp.com/2015/09/01/installing-clang-on-windows-pt2/
// * Test if stl is properly linked
// * In case of linker warning use clang-cl for windows 
//   https://stackoverflow.com/questions/42545078/clang-version-5-and-lnk4217-warning
//   https://stackoverflow.com/questions/50274547/windows-clang-hello-world-lnk4217
#include <iostream>
#include <vector>

int main()
{
  std::vector<int> vect {1, 2, 3, 4, 5};
  for(auto& el : vect)
    std::cout << " - " << el << std::endl;
  
  return 0;
}