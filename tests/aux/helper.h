#ifndef _HELPER_H_
#define _HELPER_H_

#if REAL_SIZE == 8

    typedef double real;
    typedef unsigned long bytes;

#   define MKL_GEMM( Layout, TA, TB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ) \
    cblas_dgemm(Layout, transa, transb, \
                m, n, k, alpha, A , lda, B, ldb, beta, C , ldc)
#   define TOLERANCE 19
#   define EPS 1e-10


#elif REAL_SIZE == 4

    typedef float real;
    typedef unsigned bytes;

#   define MKL_GEMM( Layout, TA, TB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ) \
    cblas_sgemm(Layout, transa, transb, \
                m, n, k, alpha, A , lda, B, ldb, beta, C , ldc)

#   define TOLERANCE 19
#   define EPS 1e-5

#else
#   error REAL_SIZE not supported.
#endif

void fillWithStuff(real* matrix, const unsigned arr_size);
void fillWithStuff(real* matrix, unsigned arr_size, const real defualt_value);
const unsigned count_incorrect_bits(real ref, real computed);


#endif