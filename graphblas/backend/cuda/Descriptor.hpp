#ifndef GRB_BACKEND_CUDA_DESCRIPTOR_HPP
#define GRB_BACKEND_CUDA_DESCRIPTOR_HPP

#include <vector>

#include <moderngpu.cuh>

#include "graphblas/backend/cuda/util.hpp"

namespace graphblas
{
namespace backend
{
  class Descriptor
  {
    public:
    // Descriptions of these default settings are in "graphblas/types.hpp"
    Descriptor() : desc_{ GrB_DEFAULT, GrB_DEFAULT, GrB_DEFAULT, GrB_DEFAULT, 
      GrB_FIXEDROW, GrB_32, GrB_32, GrB_128, GrB_PUSHPULL, GrB_16, GrB_CUDA},
      //d_buffer_(NULL), d_buffer_size_(0), d_temp_(NULL), d_temp_size_(0),
      d_context_(mgpu::CreateCudaDevice(0)), ta_(0), tb_(0), mode_(""), 
      split_(0), enable_split_(0), niter_(0), max_niter_(0), directed_(0), 
      timing_(0), transpose_(0), mtxinfo_(0), verbose_(0), mxvmode_(0),
      switchpoint_(0), lastmxv_(GrB_PUSHONLY), dirinfo_(0), struconly_(0),
      opreuse_(0), memusage_(0), endbit_(0), sort_(0), atomic_(0),
      earlyexit_(0), fusedmask_(0), nthread_(0), ndevice_(0), debug_(0),
      memory_(0) 
    {
      // Preallocate d_buffer_size
      d_buffer_size_ = 183551;
      CUDA_CALL( cudaMalloc(&d_buffer_, d_buffer_size_) );

      // Preallocate d_temp_size
      d_temp_size_ = 183551;
      CUDA_CALL( cudaMalloc(&d_temp_, d_temp_size_) );
    }

    // Default Destructor
    ~Descriptor();

    // C API Methods
    Info set( Desc_field field, Desc_value  value );
    Info get( Desc_field field, Desc_value* value ) const;

    // Useful methods
    Info toggle( Desc_field field );
    Info loadArgs( const po::variables_map& vm );

    // TODO: make this static so printMemory can use it
    inline bool debug()  { return debug_;  }
    inline bool memory() { return memory_; }

    // TODO: use this in lieu of GrB_BOOL detector for now
    inline bool struconly()      { return struconly_; }
    inline bool split()          { return split_ && enable_split_; }
    inline bool dirinfo()        { return dirinfo_; }
    inline bool earlyexit()      { return earlyexit_; }
    inline bool opreuse()        { return opreuse_; }
    inline bool endbit()         { return endbit_; }
    inline bool sort()           { return sort_; }
    inline bool fusedmask()      { return fusedmask_; }
    inline bool atomic()         { return atomic_; }
    inline float switchpoint()   { return switchpoint_; }
    inline float memusage()      { return memusage_; }

    private:
    Info resize( size_t target, std::string field );
    Info clear( std::string field );

    private:
    Desc_value desc_[GrB_NDESCFIELD];

    // Workspace memory
    void*       d_buffer_;      // Used for internal graphblas calls
    size_t      d_buffer_size_;
    void*       d_temp_;        // Used for CUB calls
    size_t      d_temp_size_;

    // MGPU context
    mgpu::ContextPtr d_context_;

    // Algorithm specific params
    int         ta_;
    int         tb_;
    std::string mode_;
    bool        split_;
    bool        enable_split_;

    // General params
    int         niter_;
    int         max_niter_;
    int         directed_;
    int         timing_;
    bool        transpose_;
    bool        mtxinfo_;
    bool        verbose_;

    // mxv params
    int         mxvmode_;
    Desc_value  lastmxv_;
    float       switchpoint_;
    bool        dirinfo_;
    bool        struconly_;
    bool        opreuse_;

    // mxv (spmspv/push) params
    float       memusage_;
    bool        endbit_;
    bool        sort_;
    bool        atomic_;

    // mxv (spmv/pull) params
    bool        earlyexit_;
    bool        fusedmask_;

    // GPU params
    int         nthread_;
    int         ndevice_;
    bool        debug_;
    bool        memory_;
  };

  Descriptor::~Descriptor()
  {
    if( d_buffer_!=NULL ) CUDA_CALL( cudaFree(d_buffer_) );
    if( d_temp_  !=NULL ) CUDA_CALL( cudaFree(d_temp_)   );
  }

  Info Descriptor::set( Desc_field field, Desc_value value )
  {
    desc_[field] = value;
    return GrB_SUCCESS;
  }

  Info Descriptor::get( Desc_field field, Desc_value* value ) const
  {
    *value = desc_[field];
    return GrB_SUCCESS;
  }

  // Toggles transpose vs. non-transpose
  Info Descriptor::toggle( Desc_field field )
  {
    int my_field = static_cast<int>(field);
    if( my_field<4 )
    {
      if(  desc_[field]!=GrB_DEFAULT ) desc_[field] = GrB_DEFAULT;
      else
      {
        if( my_field>2 )
          desc_[field] = GrB_TRAN;
        else
          desc_[field] = static_cast<Desc_value>(my_field);
      }
    }
    return GrB_SUCCESS;
  }

  Info Descriptor::resize( size_t target, std::string field )
  {
    void*   d_temp_buffer;
    size_t* d_size;
    if( field=="buffer" ) 
    {
      d_temp_buffer =  d_buffer_;
      d_size        = &d_buffer_size_;
    }
    else if( field=="temp" )
    {
      d_temp_buffer =  d_temp_;
      d_size        = &d_temp_size_;
    }

    if( target>*d_size )
    {
      if( GrB_MEMORY )
      {
        std::cout << "Resizing "+field+" from " << *d_size << " to " << 
            target << "!\n";
      }
      if( field=="buffer" ) 
      {
        CUDA_CALL( cudaMalloc(&d_buffer_, target) );
        if( GrB_MEMORY )
        {
          printMemory( "desc_buffer" );
        }
        if( d_temp_buffer!=NULL )
          CUDA_CALL( cudaMemcpy(d_buffer_, d_temp_buffer, *d_size, 
              cudaMemcpyDeviceToDevice) );
      }
      else if( field=="temp" ) 
      {
        CUDA_CALL( cudaMalloc(&d_temp_, target) );
        if( GrB_MEMORY )
        {
          printMemory( "desc_temp" );
        }
        if( d_temp_buffer!=NULL )
          CUDA_CALL( cudaMemcpy(d_temp_, d_temp_buffer, *d_size, 
              cudaMemcpyDeviceToDevice) );
      }
      *d_size = target;

      CUDA_CALL( cudaFree(d_temp_buffer) );
    }
    return GrB_SUCCESS;
  }

  Info Descriptor::clear( std::string field )
  {
    if( d_buffer_size_>0 )
    { 
      if( field=="buffer" )
        CUDA_CALL( cudaMemset(d_buffer_, 0, d_buffer_size_) );
        //CUDA_CALL( cudaMemsetAsync(d_buffer_, 0, d_buffer_size_) );
      else if( field=="temp" )
        CUDA_CALL( cudaMemset(d_temp_,   0, d_temp_size_) );
        //CUDA_CALL( cudaMemsetAsync(d_temp_,   0, d_temp_size_) );
    }

    return GrB_SUCCESS;
  }

  Info Descriptor::loadArgs( const po::variables_map& vm )
  {
    // Algorithm specific params
    ta_             = vm["ta"            ].as<int>();
    tb_             = vm["tb"            ].as<int>();
    mode_           = vm["mode"          ].as<std::string>();
    split_          = vm["split"         ].as<bool>();

    // General params
    niter_          = vm["niter"         ].as<int>();
    max_niter_      = vm["max_niter"     ].as<int>();
    directed_       = vm["directed"      ].as<int>();
    timing_         = vm["timing"        ].as<int>();
    transpose_      = vm["transpose"     ].as<bool>();
    mtxinfo_        = vm["mtxinfo"       ].as<bool>();
    verbose_        = vm["verbose"       ].as<bool>();

    // mxv params
    mxvmode_        = vm["mxvmode"       ].as<int>();
    switchpoint_    = vm["switchpoint"   ].as<float>();
    dirinfo_        = vm["dirinfo"       ].as<bool>();
    struconly_      = vm["struconly"     ].as<bool>();
    opreuse_        = vm["opreuse"       ].as<bool>();

    // mxv (spmspv/push) params
    memusage_       = vm["memusage"      ].as<float>();
    endbit_         = vm["endbit"        ].as<bool>();
    sort_           = vm["sort"          ].as<bool>();
    atomic_         = vm["atomic"        ].as<bool>();

    // mxv (spmv/pull) params
    earlyexit_      = vm["earlyexit"     ].as<bool>();
    fusedmask_      = vm["fusedmask"     ].as<bool>();

    // GPU params
    nthread_        = vm["nthread"       ].as<int>();
    ndevice_        = vm["ndevice"       ].as<int>();
    debug_          = vm["debug"         ].as<bool>();
    memory_         = vm["memory"        ].as<bool>();

    switch( mxvmode_ )
    {
      case 0:
        CHECK( set(GrB_MXVMODE, GrB_PUSHPULL) );
        break;
      case 1:
        CHECK( set(GrB_MXVMODE, GrB_PUSHONLY) );
        break;
      case 2:
        CHECK( set(GrB_MXVMODE, GrB_PULLONLY) );
        break;
      default:
        std::cout << "Error: incorrect mxvmode selection!\n";
    }

    switch( nthread_ )
    {
      case 32:
        CHECK( set(GrB_NT, GrB_32) );
        break;
      case 64:
        CHECK( set(GrB_NT, GrB_64) );
        break;
      case 128:
        CHECK( set(GrB_NT, GrB_128) );
        break;
      case 256:
        CHECK( set(GrB_NT, GrB_256) );
        break;
      case 512:
        CHECK( set(GrB_NT, GrB_512) );
        break;
      case 1024:
        CHECK( set(GrB_NT, GrB_1024) );
        break;
      default:
        std::cout << "Error: incorrect nthread selection!\n";
    }

    // TODO: Enable ndevice_
    //if( ndevice_!=0 )

    return GrB_SUCCESS; 
  }

}  // backend
}  // graphblas

#endif  // GRB_DESCRIPTOR_HPP
