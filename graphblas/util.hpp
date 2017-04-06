#include <vector>
#include <ifstream>

#include <graphblas/types.hpp>

template<typename T>
void readMtx( std::vector<Index>& row_indices,
			  std::vector<Index>& col_indices,
			  std::vector<T>& values,
			  const Index nvals,
              bool& is_weighted,
              std::ifstream& infile )
{
  is_weighted = true;
  int c;
  Index row_ind, col_ind;
  T value;

  int csr_max = 0;
  int csr_current = 0;
  int csr_row = 0;
  int csr_first = 0;

  // Currently checks if there are fewer rows than promised
  // Could add check for edges in diagonal of adjacency matrix
  for( Index i=0; i<nvals; i++ ) {
    if( infile.eof ) {
      printf("Error: not enough rows in mtx file.\n");
      break;
    } else {
      infile >> row_ind;
      infile >> col_ind;

      if( i==0 ) {
        infile.get(c);
        if( c!=32 ) is_weighted = false;
      }

      // If no 3rd column in MTX, set value to 1.0 default
      if( !is_weighted ) {
        h_cooVal[i]=1.0;
      } else {
        infile >> value;
      }

      // Convert 1-based indexing MTX to 0-based indexing C++
      row_indices.push_back(row_ind-1);
      col_indices.push_back(col_ind-1);
      values.push_back(value);

      //std::cout << "The first row is " << row_ind-1 << " " <<  col_ind-1 << std::endl;

      // Finds max csr row.
      if( i!=0 ) {
        if( col_ind-1==0 ) csr_first++;
        if( col_ind-1==col_indices[i-1] )
          csr_current++;
        else {
          csr_current++;
          if( csr_current > csr_max ) {
            csr_max = csr_current;
            csr_current = 0;
            csr_row = h_cooRowInd[j-1];
          } else
            csr_current = 0;
        }
      }
  }}
  std::cout << "The biggest row was " << csr_row << " with " << csr_max << " elements.\n";
  std::cout << "The first row has " << csr_first << " elements.\n";
  if( is_weighted==true ) {
    std::cout << "The graph is weighted.\n";
  } else {
    std::cout << "The graph is unweighted.\n";
}}
