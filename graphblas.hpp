namespace GraphBLAS
{

  typedef uint64_t Index;

  // Sparse (CSC format with single value instead of colptr) by default
  // Temporarily keeping member variables public to avoid having to use get
  template<typename T>
  //using Vector = std::vector<T>;
  class Vector {
    public:
      Index                 num;
      std::vector<Index> rowind;
      std::vector<T>        val;
  };

  // CSC format by default
  // Temporarily keeping member variables public to avoid having to use get
  template<typename T>
  class Matrix {
    public:
      std::vector<Index> rowind;
      std::vector<Index> colptr;
      std::vector<T>        val;
      Index               nrows; // m in math spec
      Index               ncols; // n in math spec

      Matrix( Index m, Index n ) : nrows(m), ncols(n){};
  };

  // Matrix constructor with dimensions
  //template<typename T>
  //Matrix<T>::Matrix( Index m, Index n ) {
  //  nrows = m;
  //  ncols = n;
  //}

  template<typename T>
  class Tuple {
    public:
      std::vector<Index> I;
      std::vector<Index> J;
      std::vector<T>   val;
  };

  // Input argument preprocessing functions.
  enum Transform {
    TRANSFORM_NULL  =   0,
    TRANSFORM_NEG   =   1,
    TRANSFORM_T     =   2,
    TRANSFORM_NEG_T =   3,
    TRANSFORM_NOT_T =   4
  };

// Next is the relevant number of assignment operators.  Since Boolean data is
// of significant interest, I have added the stAnd and stOr ops for now
  enum Assign {
    ASSIGN_NOOP   =     0,      /* Simple assignment */
    ASSIGN_ADDOP  =     1       /* Store with Circle plus */
  };

// List of ops that can be used in map/reduce operations.
  enum BinaryOp {
    BINARY_MULT   =     0,
    BINARY_ADD    =     1,
    BINARY_AND    =     2,
    BINARY_OR     =     3,
    BINARY_MIN    =     4
  };

// List of additive identities that can be used
  enum AdditiveId {
    IDENTITY_ADD =         0,
    IDENTITY_MIN =         1,
    IDENTITY_MAX =         2
  };

  class fnCallDesc {
      Assign assignDesc;
      Transform arg1Desc;
      Transform arg2Desc;
      Transform maskDesc;
      int32_t dim;                      // dimension for reduction operation on matrices
      BinaryOp addOp;
      BinaryOp multOp;
      AdditiveId addId;
    public:
      fnCallDesc( const std::string& semiring = "Matrix Multiply" ):
        assignDesc(ASSIGN_NOOP),
        arg1Desc(TRANSFORM_NULL),
        arg2Desc(TRANSFORM_NULL),
        maskDesc(TRANSFORM_NULL),
        dim(1),
        addOp(BINARY_ADD),
        multOp(BINARY_MULT),
        addId(IDENTITY_ADD)
      { }
      Assign getAssign() const {
        return assignDesc; }
      void setAssign(Assign state) {
        assignDesc = state; }
      Transform getTransformArg1() const {
        return arg1Desc; }
      void setTransformArg1(Transform state) {
        arg1Desc = state; }
      Transform getTransformArg2() const {
        return arg2Desc; }
      void setTransformArg2(Transform state) {
        arg2Desc = state; }
      BinaryOp getAddOp() const {
        return addOp; }
      void setAddOp( BinaryOp operation ) {
        addOp = operation; }
      BinaryOp getMultOp() const {
        return multOp; }
      void setMultOp( BinaryOp operation ) {
        multOp = operation; }
  };

  template<typename Scalar>
  void buildMatrix( Matrix<Scalar>&, std::vector<Index>&, std::vector<Index>&, std::vector<Scalar>& );

  template<typename Scalar>
  void extractTuples( std::vector<Index>&, std::vector<Index>&, std::vector<Scalar>&, Matrix<Scalar>& ); 

  template<typename Scalar>
  void ewiseMult( Scalar, Vector<Scalar>&, Vector<Scalar>&, fnCallDesc& );

  template<typename Scalar>
  void ewiseMult( Vector<Scalar>&, Vector<Scalar>&, Vector<Scalar>&, fnCallDesc& );

  template<typename Scalar>
  void ewiseAdd( Vector<Scalar>&, Vector<Scalar>&, Vector<Scalar>&, fnCallDesc& );

  template<typename Scalar>
  void mXv( Vector<Scalar>&, Matrix<Scalar>&, Vector<Scalar>&, fnCallDesc& );

  template<typename Scalar>
  void mXm( Matrix<Scalar>&, Matrix<Scalar>&, Matrix<Scalar>&, fnCallDesc& );

  namespace app {
    template<typename MatrixT, typename VectorT>
    void bfsMasked( Matrix<MatrixT>&, Vector<VectorT>&, Vector<Index>& );
  }

}
