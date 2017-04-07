#ifndef GRB_BACKEND_HPP
#define GRB_BACKEND_HPP

#ifdef GRB_USE_SEQUENTIAL
#define __GRB_BACKEND_ROOT sequential
#endif
#ifdef GRB_USE_APSPIE
#define __GRB_BACKEND_ROOT apspie
#else
#pragma message "Error: No GraphBLAS library specified!"
#endif

#endif  // GRB_BACKEND_HPP
