#ifndef GRB_UTIL_HPP
#define GRB_UTIL_HPP

#include <vector>
#include <fstream>
//#include <iostream>
#include <typeinfo>
#include <cstdio>
#include <tuple>
#include <algorithm>
#include <sys/resource.h>
#include <sys/time.h>
#include <string>
#include <libgen.h>

// for commandline arguments
#include <boost/program_options.hpp>

#define CHECK(x) do {                  \
  graphblas::Info err = x;             \
  if (err != graphblas::GrB_SUCCESS) { \
    fprintf(stderr, "Runtime error: %s returned %d at %s:%d\n", #x, err, __FILE__, __LINE__);                          \
    return err;                        \
  } } while (0)

#define CHECKVOID(x) do {              \
  graphblas::Info err = x;             \
  if (err != graphblas::GrB_SUCCESS) { \
    fprintf(stderr, "Runtime error: %s returned %d at %s:%d\n", #x, err, __FILE__, __LINE__);                          \
    return;                        \
  } } while (0)

// Utility functions

namespace po = boost::program_options;

void parseArgs( int argc, char**argv, po::variables_map& vm ) 
{
  // Declare the supported options
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", 
        "produce help message")
    ("ta", po::value<int>()->default_value(32), 
        "threads per A row")
    ("tb", po::value<int>()->default_value(32), 
        "B slab width")
    ("mode", po::value<std::string>()->default_value("fixedrow"), 
        "row or column")
    ("split", po::value<bool>()->default_value(false), 
        "True means split computation when possible e.g. mxm, reduce")

    // General params
    ("niter", po::value<int>()->default_value(10), 
        "Number of iterations to run outer loop after warmup")
    ("max_niter", po::value<int>()->default_value(10000), 
        "Number of iterations to run inner loop to convergence to")
    ("directed", po::value<int>()->default_value(0), 
        "0: follow mtx, 1: force undirected graph to be directed, 2: force directed graph to be undirected")
    ("timing", po::value<int>()->default_value(1),
        "0: outer loop timing, 1: inner loop timing, 2: per graphblas operation timing")
    ("transpose", po::value<bool>()->default_value(false), 
        "True means use transpose graph")
    ("mtxinfo", po::value<bool>()->default_value(true),
        "True means show matrix MTX info")
    ("verbose", po::value<bool>()->default_value(true),
        "0: timing output only, 1: timing output and correctness indicator")

    // mxv params
    ("source", po::value<int>()->default_value(0),
        "Source node traversal is launched from")
    ("mxvmode", po::value<int>()->default_value(1), 
        "0: push-pull, 1: push only, 2: pull only")
    ("switchpoint", po::value<float>()->default_value(0.01),
        "Percentage of nnz needed in order to switch from sparse to dense when mxvmode=0")
    ("dirinfo", po::value<bool>()->default_value(false),
        "True means show mxvmode direction info, and when switches happen")
    ("struconly", po::value<bool>()->default_value(false),
        "True means use implied nonzeroes, False means key-value operations")
    ("opreuse", po::value<bool>()->default_value(false),
        "True means use operand reuse, False means do not use it")

    // mxv (spmspv/push) params
    ("memusage", po::value<float>()->default_value(1.0),
        "Multiple of |E| used to store temporary neighbor list during push phase when using MERGE load-balancing")
    ("endbit", po::value<bool>()->default_value(true),
        "True means do not do radix sort on full 32 bits, False means do it on full 32 bits when using MERGE load-balancing")
    ("sort", po::value<bool>()->default_value(true),
        "True means sort, False means do not sort when using MERGE load-balancing and struconly=1")
    ("atomic", po::value<bool>()->default_value(false),
        "True means use atomics, False means do not use atomics when using SIMPLE or TWC load-balancing")

    // mxv (spmv/pull) params
    ("earlyexit", po::value<bool>()->default_value(true),
        "True means use early exit, False means do not use it when using Boolean LogicalOrAndSemiring")
    ("mask", po::value<bool>()->default_value(true),
        "True means use fused mask in pull direction, False means do not do it")

    // GPU params
    ("nthread", po::value<int>()->default_value(128), 
        "Number of threads per block")
    ("ndevice", po::value<int>()->default_value(0), 
        "GPU device number to use")
    ("debug", po::value<bool>()->default_value(false), 
        "True means show debug messages")
    ("memory", po::value<bool>()->default_value(false), 
        "True means show memory info")
  ;

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Note: No help message is produced if Boost Unittest Framework is also used
  if( vm.count("help") ) {
    std::cout << desc << "\n";
    return;
  }
}

template<typename T>
inline T getEnv(const char *key, T default_val) {
  const char *val = std::getenv(key);
  if (val == NULL) {
    return default_val;
  } else {
    return static_cast<T>(atoi(val));
  }
}

template<typename T>
void setEnv(const char *key, T default_val) {
  std::string s = std::to_string(default_val);
  char const *val = s.c_str();
  setenv(key, val, 0);
}

template<typename T>
bool compare(const std::tuple<graphblas::Index,
                              graphblas::Index,
                              T,
                              graphblas::Index> &lhs, 
             const std::tuple<graphblas::Index,
                              graphblas::Index,
                              T,
                              graphblas::Index> &rhs)
{
  graphblas::Index a = std::get<0>(lhs);
  graphblas::Index b = std::get<0>(rhs);
  graphblas::Index c = std::get<1>(lhs);
  graphblas::Index d = std::get<1>(rhs);
  if( a==b ) return c < d;
  else return a < b;
}

template<typename T>
void customSort( std::vector<graphblas::Index>& row_indices,
                 std::vector<graphblas::Index>& col_indices,
                 std::vector<T>& values )
{
  graphblas::Index nvals = row_indices.size();
  std::vector<std::tuple<graphblas::Index,
                         graphblas::Index,
                         T,
                         graphblas::Index> > my_tuple;

  for(graphblas::Index i=0;i<nvals;++i){
    my_tuple.push_back(std::make_tuple( row_indices[i], col_indices[i], 
                values[i], i));
  }
  std::sort( my_tuple.begin(), my_tuple.end(), compare<T> );
    
  std::vector<graphblas::Index> v1 = row_indices;
  std::vector<graphblas::Index> v2 = col_indices;
  std::vector<T>                v3 = values;

  for(graphblas::Index i=0;i<nvals;++i){
    graphblas::Index index= std::get<3>(my_tuple[i]);
    row_indices[i] = v1[index];
    col_indices[i] = v2[index];
    values[i]      = v3[index];
  }
}

template<typename T, typename mtxT>
void readTuples( std::vector<graphblas::Index>& row_indices,
                 std::vector<graphblas::Index>& col_indices,
                 std::vector<T>& values,
                 const graphblas::Index nvals,
                 FILE* f)
{
  graphblas::Index row_ind, col_ind;
  T value;
  mtxT raw_value;
  char type_str[3];
  type_str[0] = '%';
  if( typeid(mtxT)==typeid(int) )
    type_str[1] = 'd';
  else if( typeid(mtxT)==typeid(float) )
    type_str[1] = 'f';

  // Currently checks if there are fewer rows than promised
  // Could add check for edges in diagonal of adjacency matrix
  for( graphblas::Index i=0; i<nvals; i++ ) {
    if( fscanf(f, "%d", &row_ind)==EOF ) {
      std::cout << "Error: Not enough rows in mtx file!\n";
      return;
    } else {
      int u = fscanf(f, "%d", &col_ind);

      // Convert 1-based indexing MTX to 0-based indexing C++
      row_indices.push_back(row_ind-1);
      col_indices.push_back(col_ind-1);

      u = fscanf(f, type_str, &raw_value);
      value = (T) raw_value;

      values.push_back(value);
  }}
}

template<typename T>
void readTuples( std::vector<graphblas::Index>& row_indices,
                 std::vector<graphblas::Index>& col_indices,
                 std::vector<T>& values,
                 const graphblas::Index nvals,
                 FILE* f)
{
  graphblas::Index row_ind, col_ind;
  T value = (T) 1.0;

  // Currently checks if there are fewer rows than promised
  // Could add check for edges in diagonal of adjacency matrix
  for( graphblas::Index i=0; i<nvals; i++ ) {
    if( fscanf(f, "%d", &row_ind)==EOF ) {
      std::cout << "Error: Not enough rows in mtx file!\n";
      return;
    } else {
      int u = fscanf(f, "%d", &col_ind);

      // Convert 1-based indexing MTX to 0-based indexing C++
      row_indices.push_back(row_ind-1);
      col_indices.push_back(col_ind-1);
      values.push_back(value);
  }}
}

template<typename T>
void makeSymmetric( std::vector<graphblas::Index>& row_indices, 
                    std::vector<graphblas::Index>& col_indices,
                    std::vector<T>& values, 
                    graphblas::Index& nvals )
{
  bool remove_self_loops = getEnv("GRB_UTIL_REMOVE_SELFLOOP", true);

  for( graphblas::Index i=0; i<nvals; i++ ) {
    if( col_indices[i] != row_indices[i] ) {
      row_indices.push_back( col_indices[i] );
      col_indices.push_back( row_indices[i] );
      values.push_back( values[i] );
    }
  }

  nvals = row_indices.size();
  //std::cout << nvals << std::endl;

  // Sort
  customSort<T>( row_indices, col_indices, values );

  graphblas::Index curr = col_indices[0];
  graphblas::Index last;
  graphblas::Index curr_row = row_indices[0];
  graphblas::Index last_row;

  for( graphblas::Index i=1; i<nvals; i++ ) {
    last = curr;
    last_row = curr_row;
    curr = col_indices[i];
    curr_row = row_indices[i];

    // Self-loops (TODO: make self-loops contingent on whether we 
    // are doing graph algorithm or matrix multiplication)
    if( remove_self_loops && curr_row == curr )
      col_indices[i] = -1;

  // Duplicates
    if( curr == last && curr_row == last_row ) {
      col_indices[i] = -1;
  }}

  graphblas::Index shift = 0;

  // Remove self-loops and duplicates marked -1.
  graphblas::Index back = 0;
  for( graphblas::Index i=0; i+shift<nvals; i++ ) {
    if(col_indices[i] == -1) {
      for( ; back<=nvals; shift++ ) {
        back = i+shift;
        if( col_indices[back] != -1 ) {
          col_indices[i] = col_indices[back];
          row_indices[i] = row_indices[back];
          col_indices[back] = -1;
          break;
  }}}}

  nvals = nvals-shift;
  row_indices.resize(nvals);
  col_indices.resize(nvals);
  //std::cout << nvals << std::endl;
  values.resize(nvals);
}

bool exists(const char *fname)
{
  FILE *file;
  if( file = fopen(fname, "r") )
  {
    fclose(file);
    return 1;
  }
  return 0;
}

char* convert( const char* fname, bool is_undirected=true )
{
  char* dat_name = (char*)malloc(256);

  // separate the graph path and the file name
  char *temp1 = strdup(fname);
  char *temp2 = strdup(fname);
  char *file_path = dirname (temp1);
  char *file_name = basename(temp2);

  if (is_undirected)
    sprintf(dat_name, "%s/.%s.ud.%d.%sbin", file_path, file_name, 0,
        ((sizeof(graphblas::Index) == 8) ? "64bVe." : ""));
  else
    sprintf(dat_name, "%s/.%s.d.%d.%sbin", file_path, file_name, 0,
        ((sizeof(graphblas::Index) == 8) ? "64bVe." : ""));

  return dat_name;
}

// Directed controls how matrix is interpreted:
// 0: If it is marked symmetric, then double the edges. Else do nothing.
// 1: Force matrix to be unsymmetric.
// 2: Force matrix to be symmetric.
template<typename T>
int readMtx( const char*                    fname,
             std::vector<graphblas::Index>& row_indices,
             std::vector<graphblas::Index>& col_indices,
             std::vector<T>&                values,
             graphblas::Index&              nrows,
             graphblas::Index&              ncols,
             graphblas::Index&              nvals,
             int                            directed,
             bool                           mtxinfo,
             char**                         dat_name=NULL )
{
  int ret_code;
  MM_typecode matcode;
  FILE *f;

  if ((f = fopen(fname, "r")) == NULL) {
    printf( "File %s not found\n", fname );
    exit(1);
  }

  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Read MTX Size
  if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nvals)) !=0)
    exit(1);

  printf("Undirected due to mtx: %d\n", mm_is_symmetric(matcode) && directed==0);
  printf("Undirected due to cmd: %d\n", directed==2);
  bool is_undirected = (mm_is_symmetric(matcode) && directed==0) || directed==2;
  printf("Undirected: %d\n", is_undirected);
  if (dat_name != NULL)
    *dat_name = convert(fname, is_undirected);

  if( dat_name != NULL && exists(*dat_name) )
  {  
    // The size of the file in bytes is in results.st_size
    // -unserialize vector
    std::ifstream ifs(*dat_name, std::ios::in | std::ios::binary);
    if (ifs.fail())
      std::cout << "Error: Unable to open file for reading!\n";
    else
    {
    // Empty arrays indicate to Matrix::build that binary file exists
      row_indices.clear();
      col_indices.clear();
      values.clear();
    }
  }
  else
  {
    if (mm_is_integer(matcode))
      readTuples<T, int>( row_indices, col_indices, values, nvals, f );
    else if (mm_is_real(matcode))
      readTuples<T, float>( row_indices, col_indices, values, nvals, f );
    else if (mm_is_pattern(matcode))
      readTuples<T>( row_indices, col_indices, values, nvals, f );

    if (is_undirected)
      makeSymmetric<T>( row_indices, col_indices, values, nvals );
    customSort<T>( row_indices, col_indices, values );

    if( mtxinfo ) mm_write_banner(stdout, matcode);
    if( mtxinfo ) mm_write_mtx_crd_size(stdout, nrows, ncols, nvals);
  }

  return ret_code; //TODO: parse ret_code
}

template<typename T>
void printArray( const char* str, const T *array, int length=40, 
                 bool limit=true )
{
  if (limit && length > 40) length=40;
  std::cout << str << ":\n";
  for( int i=0;i<length;i++ )
    std::cout << "[" << i << "]:" << array[i] << " ";
  std::cout << "\n";
}

template<typename T>
void printArray( const char* str, std::vector<T>& array, int length=40,
                 bool limit=true )
{
  if (limit && length > 40) length=40;
  std::cout << str << ":\n";
  for( int i=0;i<length;i++ )
    std::cout << "[" << i << "]:" << array[i] << " ";
  std::cout << "\n";
}

struct CpuTimer {

#if defined(CLOCK_PROCESS_CPUTIME_ID)

  double start;
  double stop;

  void Start()
  {
    static struct timeval tv;
    static struct timezone tz;
    gettimeofday(&tv, &tz);
    start = tv.tv_sec + 1.e-6*tv.tv_usec;
  }

  void Stop()
  {
    static struct timeval tv;
    static struct timezone tz;
    gettimeofday(&tv, &tz);
    stop = tv.tv_sec + 1.e-6*tv.tv_usec;
  }

  double ElapsedMillis()
  {
    return 1000*(stop - start);
  }

#else

  rusage start;
  rusage stop;

  void Start()
  {
    getrusage(RUSAGE_SELF, &start);
  }

  void Stop()
  {
    getrusage(RUSAGE_SELF, &stop);
  }

  float ElapsedMillis()
  {
    float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
    float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

    return (sec * 1000) + (usec /1000);
  }

#endif
};

#endif  // GRB_UTIL_HPP
