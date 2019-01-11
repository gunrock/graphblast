/*!
 *   Matrix Market I/O library for ANSI C
 *
 *   See http://math.nist.gov/MatrixMarket for details.
 *
 *
 */

#ifndef GRAPHBLAS_MMIO_HPP_
#define GRAPHBLAS_MMIO_HPP_

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

int mm_read_banner(FILE *f, MM_typecode *matcode);
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
int mm_read_mtx_array_size(FILE *f, int *M, int *N);

int mm_write_banner(FILE *f, MM_typecode matcode);
int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
int mm_write_mtx_array_size(FILE *f, int M, int N);

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)  ((typecode)[0] == 'M')

#define mm_is_sparse(typecode)  ((typecode)[1] == 'C')
#define mm_is_coordinate(typecode)((typecode)[1] == 'C')
#define mm_is_dense(typecode) ((typecode)[1] == 'A')
#define mm_is_array(typecode) ((typecode)[1] == 'A')

#define mm_is_complex(typecode) ((typecode)[2] == 'C')
#define mm_is_real(typecode)    ((typecode)[2] == 'R')
#define mm_is_pattern(typecode) ((typecode)[2] == 'P')
#define mm_is_integer(typecode) ((typecode)[2] == 'I')

#define mm_is_symmetric(typecode)((typecode)[3] == 'S')
#define mm_is_general(typecode) ((typecode)[3] == 'G')
#define mm_is_skew(typecode)  ((typecode)[3] == 'K')
#define mm_is_hermitian(typecode)((typecode)[3] == 'H')

int mm_is_valid(MM_typecode matcode);   /* too complex for a macro */

/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode) ((*typecode)[0]='M')
#define mm_set_coordinate(typecode) ((*typecode)[1]='C')
#define mm_set_array(typecode)  ((*typecode)[1]='A')
#define mm_set_dense(typecode)  mm_set_array(typecode)
#define mm_set_sparse(typecode) mm_set_coordinate(typecode)

#define mm_set_complex(typecode)((*typecode)[2]='C')
#define mm_set_real(typecode) ((*typecode)[2]='R')
#define mm_set_pattern(typecode)((*typecode)[2]='P')
#define mm_set_integer(typecode)((*typecode)[2]='I')


#define mm_set_symmetric(typecode)((*typecode)[3]='S')
#define mm_set_general(typecode)((*typecode)[3]='G')
#define mm_set_skew(typecode) ((*typecode)[3]='K')
#define mm_set_hermitian(typecode)((*typecode)[3]='H')

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]= \
                  (*typecode)[2]=' ', (*typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)

/********************* Matrix Market error codes ***************************/


#define MM_COULD_NOT_READ_FILE  11
#define MM_PREMATURE_EOF    12
#define MM_NOT_MTX        13
#define MM_NO_HEADER      14
#define MM_UNSUPPORTED_TYPE   15
#define MM_LINE_TOO_LONG    16
#define MM_COULD_NOT_WRITE_FILE 17

/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

            ojbect    sparse/     data        storage 
                  dense       type        scheme

   string position:  [0]        [1]     [2]         [3]

   Matrix typecode:  M(atrix)  C(oord)    R(eal)    G(eneral)
                    A(array)  C(omplex)   H(ermitian)
                      P(attern)   S(ymmetric)
                        I(nteger) K(kew)

 ***********************************************************************/

#define MM_MTX_STR    "matrix"
#define MM_ARRAY_STR  "array"
#define MM_DENSE_STR  "array"
#define MM_COORDINATE_STR "coordinate"
#define MM_SPARSE_STR "coordinate"
#define MM_COMPLEX_STR  "complex"
#define MM_REAL_STR   "real"
#define MM_INT_STR    "integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR   "symmetric"
#define MM_HERM_STR   "hermitian"
#define MM_SKEW_STR   "skew-symmetric"
#define MM_PATTERN_STR  "pattern"

int mm_is_valid(MM_typecode matcode) {
  if (!mm_is_matrix(matcode)) return 0;
  if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
  if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
  if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) ||
      mm_is_skew(matcode))) return 0;
  return 1;
}

int mm_read_banner(FILE *f, MM_typecode *matcode) {
  char line[MM_MAX_LINE_LENGTH];
  char banner[MM_MAX_TOKEN_LENGTH];
  char mtx[MM_MAX_TOKEN_LENGTH];
  char crd[MM_MAX_TOKEN_LENGTH];
  char data_type[MM_MAX_TOKEN_LENGTH];
  char storage_scheme[MM_MAX_TOKEN_LENGTH];
  char *p;

  mm_clear_typecode(matcode);

  if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
    return MM_PREMATURE_EOF;

  if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
      storage_scheme) != 5)
    return MM_PREMATURE_EOF;

  /* convert to lower case */
  for (p = mtx; *p != '\0'; *p = tolower(*p), p++) {}
  for (p = crd; *p != '\0'; *p = tolower(*p), p++) {}
  for (p = data_type; *p != '\0'; *p = tolower(*p), p++) {}
  for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++) {}

  /* check for banner */
  if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
    return MM_NO_HEADER;

  /* first field should be "mtx" */
  if (strcmp(mtx, MM_MTX_STR) != 0)
    return  MM_UNSUPPORTED_TYPE;
  mm_set_matrix(matcode);

  /* second field describes whether this is a sparse matrix (in coordinate
     storage) or a dense array */
  if (strcmp(crd, MM_SPARSE_STR) == 0)
    mm_set_sparse(matcode);
  else if (strcmp(crd, MM_DENSE_STR) == 0)
    mm_set_dense(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  /* third field */
  if (strcmp(data_type, MM_REAL_STR) == 0)
    mm_set_real(matcode);
  else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
    mm_set_complex(matcode);
  else if (strcmp(data_type, MM_PATTERN_STR) == 0)
    mm_set_pattern(matcode);
  else if (strcmp(data_type, MM_INT_STR) == 0)
    mm_set_integer(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  /* fourth field */
  if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
    mm_set_general(matcode);
  else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
    mm_set_symmetric(matcode);
  else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
    mm_set_hermitian(matcode);
  else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
    mm_set_skew(matcode);
  else
    return MM_UNSUPPORTED_TYPE;

  return 0;
}

int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz) {
  if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
    return MM_COULD_NOT_WRITE_FILE;
  else
    return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz) {
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;

  /* set return null parameter values, in case we exit with errors */
  *M = *N = *nz = 0;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MM_PREMATURE_EOF;
  } while (line[0] == '%');

  /* line[] is either blank or has M,N, nz */
  if (sscanf(line, "%d %d %d", M, N, nz) == 3) {
    return 0;
  } else {
    do {
      num_items_read = fscanf(f, "%d %d %d", M, N, nz);
      if (num_items_read == EOF) return MM_PREMATURE_EOF;
    } while (num_items_read != 3);
  }

  return 0;
}

int mm_read_mtx_array_size(FILE *f, int *M, int *N) {
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;
  /* set return null parameter values, in case we exit with errors */
  *M = *N = 0;

  /* now continue scanning until you reach the end-of-comments */
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MM_PREMATURE_EOF;
  }  while (line[0] == '%');

  /* line[] is either blank or has M,N, nz */
  if (sscanf(line, "%d %d", M, N) == 2) {
    return 0;
  } else { /* we have a blank line */
    do {
      num_items_read = fscanf(f, "%d %d", M, N);
      if (num_items_read == EOF) return MM_PREMATURE_EOF;
    } while (num_items_read != 2);
  }

  return 0;
}

int mm_write_mtx_array_size(FILE *f, int M, int N) {
  if (fprintf(f, "%d %d\n", M, N) != 2)
    return MM_COULD_NOT_WRITE_FILE;
  else
    return 0;
}

int mm_write_banner(FILE *f, MM_typecode matcode) {
  char *str = mm_typecode_to_str(matcode);
  int ret_code;

  ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
  free(str);
  if (ret_code != 2)
    return MM_COULD_NOT_WRITE_FILE;
  else
    return 0;
}

/**
*  Create a new copy of a string s.  mm_strdup() is a common routine, but
*  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
*
*/
char *mm_strdup(const char *s) {
  int len = strlen(s);
  char *s2 = reinterpret_cast<char *>(malloc((len+1)*sizeof(char)));
  // return strcpy(s2, s);
  snprintf(s2, len+1, "%s", s);
  return s2;
}

char *mm_typecode_to_str(MM_typecode matcode) {
  char buffer[MM_MAX_LINE_LENGTH];
  char const *types[4];
  char *mm_strdup(const char *);

  /* check for MTX type */
  if (mm_is_matrix(matcode))
    types[0] = MM_MTX_STR;

  /* check for CRD or ARR matrix */
  if (mm_is_sparse(matcode))
    types[1] = MM_SPARSE_STR;
  else if (mm_is_dense(matcode))
    types[1] = MM_DENSE_STR;
  else
    return NULL;

  /* check for element data type */
  if (mm_is_real(matcode))
    types[2] = MM_REAL_STR;
  else if (mm_is_complex(matcode))
    types[2] = MM_COMPLEX_STR;
  else if (mm_is_pattern(matcode))
    types[2] = MM_PATTERN_STR;
  else if (mm_is_integer(matcode))
    types[2] = MM_INT_STR;
  else
    return NULL;

  /* check for symmetry type */
  if (mm_is_general(matcode))
    types[3] = MM_GENERAL_STR;
  else if (mm_is_symmetric(matcode))
    types[3] = MM_SYMM_STR;
  else if (mm_is_hermitian(matcode))
    types[3] = MM_HERM_STR;
  else if (mm_is_skew(matcode))
    types[3] = MM_SKEW_STR;
  else
    return NULL;

  snprintf(buffer, MM_MAX_LINE_LENGTH, "%s %s %s %s", types[0], types[1],
      types[2], types[3]);
  return mm_strdup(buffer);
}

#endif  // GRAPHBLAS_MMIO_HPP_
