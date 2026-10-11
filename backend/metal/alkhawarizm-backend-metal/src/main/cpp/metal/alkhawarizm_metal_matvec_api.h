/**
 * alkhawarizm_metal_matvec_api.h — exported Metal matvec C API.
 */

#ifndef ALKHAWARIZM_METAL_MATVEC_API_H
#define ALKHAWARIZM_METAL_MATVEC_API_H

#ifdef __cplusplus
extern "C" {
#endif

int alkhawarizm_metal_matvec_tb_half(void *C, const void *A, const void *B,
                                     int K, int N);

int alkhawarizm_metal_matvec_tb_half_mps(void *C, const void *A, const void *B,
                                         int K, int N);

int alkhawarizm_metal_matvec_t_half(void *C, const void *A, const void *B,
                                    int K, int N);

int alkhawarizm_metal_matvec_tb_bf16(void *C, const void *A, const void *B,
                                     int K, int N);

int alkhawarizm_metal_matvec_tb_int4(void *C, const void *A,
                                     const void *B_packed, const void *scales,
                                     int K, int N, int blockSize);

// GGML-format quantized kernels (block formats from GGUF)
int alkhawarizm_metal_matvec_tb_q4_k(void *C, const void *A, const void *B,
                                     int K, int N);
int alkhawarizm_metal_matvec_tb_q6_k(void *C, const void *A, const void *B,
                                     int K, int N);
int alkhawarizm_metal_matvec_tb_q8_0(void *C, const void *A, const void *B,
                                     int K, int N);

int alkhawarizm_metal_matvec_tb_nf4(void *C, const void *A,
                                    const void *B_packed, const void *absmax,
                                    int K, int N, int blockSize);

int alkhawarizm_metal_matvec_tb_half_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int K,
                                          int N);

int alkhawarizm_metal_matvec_tb_bf16_pair(void *C0, void *C1, const void *A,
                                          const void *B0, const void *B1, int K,
                                          int N);

int alkhawarizm_metal_matvec_tb_half_triple_mixed(void *C0, void *C1, void *C2,
                                                  const void *A, const void *B0,
                                                  const void *B1,
                                                  const void *B2, int K, int N0,
                                                  int N1, int N2);

int alkhawarizm_metal_matvec_tb_bf16_triple_mixed(void *C0, void *C1, void *C2,
                                                  const void *A, const void *B0,
                                                  const void *B1,
                                                  const void *B2, int K, int N0,
                                                  int N1, int N2);

#ifdef __cplusplus
}
#endif

#endif
