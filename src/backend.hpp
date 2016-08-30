#ifdef GRB_USE_SEQUENTIAL
#define __GRB_BACKEND_ROOT sequential
#elif GRB_USE_APSPIE
#define __GRB_BACKEND_ROOT apspie
#else
#pragma message "Error: No GraphBLAS library specified!"
#endif
