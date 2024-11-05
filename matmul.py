from collections.abc import Callable
import mlx.core as mx
import time

def matmul_kernel(a: mx.array, b: mx.array):
    # Matrix multiplication kernel
    source = '''
        const int row = thread_position_in_grid.y;
        const int col = thread_position_in_grid.x;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    '''

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible matrix dimensions"
    assert M % 16 == 0 and N % 16 == 0, "Dims should be multiple of 16"

    kernel = mx.fast.metal_kernel(
        name="matmul",
        input_names=["a", "b", "M", "N", "K"],
        output_names=["c"],
        source=source
    )

    outputs = kernel(
        inputs=[a, b, M, N, K],
        grid=(N, M, 1),  # 2D grid for matrix multiplication
        threadgroup=(16, 16, 1),  # 2D threadgroup
        output_shapes=[(M, N)],
        output_dtypes=[a.dtype],
        verbose=True,
    )

    return outputs[0]

# def simdgroup_matmul_kernel(a: mx.array, b: mx.array):
#     M, K = a.shape
#     K2, N = b.shape
#     assert K == K2, "Incompatible matrix dimensions"

#     TILE_SIZE = 8
#     source = f'''
#         // a = (M, K)
#         // b = (K, N)
#         const int row = threadgroup_position_in_grid.x;
#         const int col = threadgroup_position_in_grid.y;

#         simdgroup_float8x8 sgMatA;
#         simdgroup_float8x8 sgMatB;
#         simdgroup_float8x8 sgMatR = simdgroup_float8x8(0.0f);

#         for (int i = 0; i < {N}/{TILE_SIZE}; i++) {{
#             simdgroup_load(sgMatA, a + (row * {TILE_SIZE} * N) + (i * {TILE_SIZE}), N);
#             simdgroup_load(sgMatB, b + (i * {TILE_SIZE} * N) + (col * {TILE_SIZE}), N);

#             simdgroup_multiply_accumulate(sgMatR, sgMatA, sgMatB, sgMatR);
#         }}

#         simdgroup_store(sgMatR, c + (row * {TILE_SIZE} * N) + (col * {TILE_SIZE}), N);
#     '''

#     kernel = mx.fast.metal_kernel(
#         name="matmul",
#         input_names=["a", "b", "M", "N", "K"],
#         output_names=["c"],
#         source=source,
#         header="#include <metal_simdgroup_matrix>\n"
#     )

#     outputs = kernel(
#         inputs=[a, b, M, N, K],
#         grid=(M, N, 1),  # 2D grid for matrix multiplication
#         threadgroup=(8, 8, 1),  # 2D threadgroup
#         output_shapes=[(M, N)],
#         output_dtypes=[a.dtype],
#         verbose=False,
#     )

#     return outputs[0]

def simdgroup_4x4_matmul_kernel(a: mx.array, b: mx.array):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible matrix dimensions"

    TILE_SIZE = 8
    TILEGROUP_SIZE = 4
    source = f'''
        // a = (M, K)
        // b = (K, N)
        const int row = threadgroup_position_in_grid.x;
        const int col = threadgroup_position_in_grid.y;

        simdgroup_float8x8 sgMatA[{TILEGROUP_SIZE}];
        simdgroup_float8x8 sgMatB[{TILEGROUP_SIZE}];
        simdgroup_float8x8 sgMatR[{TILEGROUP_SIZE}][{TILEGROUP_SIZE}];
        for (int i = 0; i < {TILEGROUP_SIZE}; i++) {{
            for (int j = 0; j < {TILEGROUP_SIZE}; j++) {{
                sgMatR[i][j] = simdgroup_float8x8(0.0f);
            }}
        }}

        for (int i = 0; i < {N}/{TILE_SIZE}; i++) {{
            for (int j = 0; j < {TILEGROUP_SIZE}; j++) {{
                simdgroup_load(sgMatA[j], a + ((row * {TILEGROUP_SIZE} + j) * {TILE_SIZE} * N) + (i * {TILE_SIZE}), N);
                simdgroup_load(sgMatB[j], b + (i * {TILE_SIZE} * N) + ((col * {TILEGROUP_SIZE} + j) * {TILE_SIZE}), N);
            }}

            // metal compiler doesn't inline :/
            //for (int j = 0; j < {TILEGROUP_SIZE}; j++) {{
            //    for (int k = 0; k < {TILEGROUP_SIZE}; k++) {{
            //        simdgroup_multiply_accumulate(sgMatR[j][k], sgMatA[j], sgMatB[k], sgMatR[j][k]);
            //    }}
            //}}

            simdgroup_multiply_accumulate(sgMatR[0][0], sgMatA[0], sgMatB[0], sgMatR[0][0]);
            simdgroup_multiply_accumulate(sgMatR[0][1], sgMatA[0], sgMatB[1], sgMatR[0][1]);
            simdgroup_multiply_accumulate(sgMatR[0][2], sgMatA[0], sgMatB[2], sgMatR[0][2]);
            simdgroup_multiply_accumulate(sgMatR[0][3], sgMatA[0], sgMatB[3], sgMatR[0][3]);
            simdgroup_multiply_accumulate(sgMatR[1][0], sgMatA[1], sgMatB[0], sgMatR[1][0]);
            simdgroup_multiply_accumulate(sgMatR[1][1], sgMatA[1], sgMatB[1], sgMatR[1][1]);
            simdgroup_multiply_accumulate(sgMatR[1][2], sgMatA[1], sgMatB[2], sgMatR[1][2]);
            simdgroup_multiply_accumulate(sgMatR[1][3], sgMatA[1], sgMatB[3], sgMatR[1][3]);
            simdgroup_multiply_accumulate(sgMatR[2][0], sgMatA[2], sgMatB[0], sgMatR[2][0]);
            simdgroup_multiply_accumulate(sgMatR[2][1], sgMatA[2], sgMatB[1], sgMatR[2][1]);
            simdgroup_multiply_accumulate(sgMatR[2][2], sgMatA[2], sgMatB[2], sgMatR[2][2]);
            simdgroup_multiply_accumulate(sgMatR[2][3], sgMatA[2], sgMatB[3], sgMatR[2][3]);
            simdgroup_multiply_accumulate(sgMatR[3][0], sgMatA[3], sgMatB[0], sgMatR[3][0]);
            simdgroup_multiply_accumulate(sgMatR[3][1], sgMatA[3], sgMatB[1], sgMatR[3][1]);
            simdgroup_multiply_accumulate(sgMatR[3][2], sgMatA[3], sgMatB[2], sgMatR[3][2]);
            simdgroup_multiply_accumulate(sgMatR[3][3], sgMatA[3], sgMatB[3], sgMatR[3][3]);
        }}

        for (int i = 0; i < {TILEGROUP_SIZE}; i++) {{
            for (int j = 0; j < {TILEGROUP_SIZE}; j++) {{
                simdgroup_store(sgMatR[i][j], c + ((row*{TILEGROUP_SIZE} + i) * {TILE_SIZE} * N) + ((col*{TILEGROUP_SIZE} + j) * {TILE_SIZE}), N);
            }}
        }}
    '''

    kernel = mx.fast.metal_kernel(
        name="matmul",
        input_names=["a", "b", "M", "N", "K"],
        output_names=["c"],
        source=source,
        header="#include <metal_simdgroup_matrix>\n"
    )

    outputs = kernel(
        inputs=[a, b, M, N, K],
        grid=(M//4, N//4, 1),  # 2D grid for matrix multiplication
        threadgroup=(8, 8, 1),  # 2D threadgroup
        output_shapes=[(M, N)],
        output_dtypes=[a.dtype],
        verbose=False,
    )

    return outputs[0]


def tinygrad_matmul_kernel(a: mx.array, b: mx.array):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible matrix dimensions"

    LID = 2
    # Matrix multiplication kernel
    source = f'''
        uint3 gid = threadgroup_position_in_grid;
        uint3 lid = thread_position_in_threadgroup;

        a += (gid.x * 32 * {N}) + (gid.y * {LID} + lid.y) * 32;
        data1 += gid.x * 32 * {N};
        data2 += (gid.y * {LID} + lid.y) * 32;

        simdgroup_float8x8 acc[4][4];
        for (uint i = 0; i < 4; i++) {{
            for (uint j = 0; j < 4; j++) {{
                acc[i][j] = simdgroup_float8x8(0);
            }}
        }}

        simdgroup_float8x8 A[4];
        simdgroup_float8x8 B[4];
        for (uint k = 0; k < {N}; k+=8) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(A[0], data1+k+{0*N}, {N}, ulong2(0, 0));
            simdgroup_load(A[1], data1+k+{8*N}, {N}, ulong2(0, 0));
            simdgroup_load(A[2], data1+k+{16*N}, {N}, ulong2(0, 0));
            simdgroup_load(A[3], data1+k+{24*N}, {N}, ulong2(0, 0));
            simdgroup_load(B[0], data2+0+k*{N}, {N}, ulong2(0, 0));
            simdgroup_load(B[1], data2+8+k*{N}, {N}, ulong2(0, 0));
            simdgroup_load(B[2], data2+16+k*{N}, {N}, ulong2(0, 0));
            simdgroup_load(B[3], data2+24+k*{N}, {N}, ulong2(0, 0));

            simdgroup_multiply_accumulate(acc[0][0], A[0], B[0], acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], A[1], B[0], acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], A[2], B[0], acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], A[3], B[0], acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], A[0], B[1], acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], A[1], B[1], acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], A[2], B[1], acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], A[3], B[1], acc[1][3]);
            simdgroup_multiply_accumulate(acc[2][0], A[0], B[2], acc[2][0]);
            simdgroup_multiply_accumulate(acc[2][1], A[1], B[2], acc[2][1]);
            simdgroup_multiply_accumulate(acc[2][2], A[2], B[2], acc[2][2]);
            simdgroup_multiply_accumulate(acc[2][3], A[3], B[2], acc[2][3]);
            simdgroup_multiply_accumulate(acc[3][0], A[0], B[3], acc[3][0]);
            simdgroup_multiply_accumulate(acc[3][1], A[1], B[3], acc[3][1]);
            simdgroup_multiply_accumulate(acc[3][2], A[2], B[3], acc[3][2]);
            simdgroup_multiply_accumulate(acc[3][3], A[3], B[3], acc[3][3]);
        }}
        simdgroup_store(acc[0][0], a+{0+0*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[1][0], a+{8+0*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[2][0], a+{16+0*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[3][0], a+{24+0*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[0][1], a+{0+8*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[1][1], a+{8+8*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[2][1], a+{16+8*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[3][1], a+{24+8*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[0][2], a+{0+16*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[1][2], a+{8+16*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[2][2], a+{16+16*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[3][2], a+{24+16*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[0][3], a+{0+24*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[1][3], a+{8+24*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[2][3], a+{16+24*N}, {N}, ulong2(0, 0));
        simdgroup_store(acc[3][3], a+{24+24*N}, {N}, ulong2(0, 0));
    '''

    kernel = mx.fast.metal_kernel(
        name="matmul",
        input_names=["data1", "data2", "M", "N", "K"],
        output_names=["a"],
        source=source,
        header="#include <metal_simdgroup_matrix>\n"
    )

    outputs = kernel(
        inputs=[a, b, M, N, K],
        grid=(M, N//32, 1),  # 2D grid for matrix multiplication
        threadgroup=(32, LID, 1),  # 2D threadgroup
        output_shapes=[(M, N)],
        output_dtypes=[a.dtype],
        verbose=False,
    )

    return outputs[0]

def modded_tinygrad_matmul_kernel(a: mx.array, b: mx.array):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible matrix dimensions"

    LID = 2
    # Matrix multiplication kernel
    source = f'''
        uint3 gid = threadgroup_position_in_grid;
        uint3 lid = thread_position_in_threadgroup;

        a += (gid.x * 32 * {N}) + (gid.y * {LID} + lid.y) * 32;
        data1 += gid.x * 32 * {N};
        data2 += (gid.y * {LID} + lid.y) * 32;

        simdgroup_float8x8 acc[4][4];
        for (uint i = 0; i < 4; i++) {{
            for (uint j = 0; j < 4; j++) {{
                acc[i][j] = simdgroup_float8x8(0);
            }}
        }}

        simdgroup_float8x8 A[4];
        simdgroup_float8x8 B[4];
        for (uint k = 0; k < {N}; k+=8) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = 0; i < 4; i++) {{
                simdgroup_load(A[i], data1+k+(i*8*{N}), {N}, ulong2(0, 0));
                simdgroup_load(B[i], data2+(i*8)+k*{N}, {N}, ulong2(0, 0));
            }}

            simdgroup_multiply_accumulate(acc[0][0], A[0], B[0], acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], A[1], B[0], acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], A[2], B[0], acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], A[3], B[0], acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], A[0], B[1], acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], A[1], B[1], acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], A[2], B[1], acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], A[3], B[1], acc[1][3]);
            simdgroup_multiply_accumulate(acc[2][0], A[0], B[2], acc[2][0]);
            simdgroup_multiply_accumulate(acc[2][1], A[1], B[2], acc[2][1]);
            simdgroup_multiply_accumulate(acc[2][2], A[2], B[2], acc[2][2]);
            simdgroup_multiply_accumulate(acc[2][3], A[3], B[2], acc[2][3]);
            simdgroup_multiply_accumulate(acc[3][0], A[0], B[3], acc[3][0]);
            simdgroup_multiply_accumulate(acc[3][1], A[1], B[3], acc[3][1]);
            simdgroup_multiply_accumulate(acc[3][2], A[2], B[3], acc[3][2]);
            simdgroup_multiply_accumulate(acc[3][3], A[3], B[3], acc[3][3]);
        }}

        for (int i = 0; i < 4; i++) {{
            for (int j = 0; j < 4; j++) {{
                simdgroup_store(acc[j][i], a+(j*8)+(i*8*{N}), {N}, ulong2(0, 0));
            }}
        }}
    '''

    kernel = mx.fast.metal_kernel(
        name="matmul",
        input_names=["data1", "data2", "M", "N", "K"],
        output_names=["a"],
        source=source,
        header="#include <metal_simdgroup_matrix>\n"
    )

    outputs = kernel(
        inputs=[a, b, M, N, K],
        grid=(M, N//32, 1),  # 2D grid for matrix multiplication
        threadgroup=(32, LID, 1),  # 2D threadgroup
        output_shapes=[(M, N)],
        output_dtypes=[a.dtype],
        verbose=False,
    )

    return outputs[0]


def bench(matmul_impl: Callable[[mx.array, mx.array], mx.array], N: int = 1024):
    M, K, N = N, N, N
    a = mx.random.normal(shape=(M, K)).astype(mx.float32)
    b = mx.random.normal(shape=(K, N)).astype(mx.float32)

    # warmup
    tmp = matmul_impl(a, b)
    mx.eval(tmp)

    tic = time.perf_counter()
    TIMES = 10
    for _ in range(TIMES):
        c_kernel = matmul_impl(a, b)
        mx.eval(c_kernel)
    toc = time.perf_counter()
    s = (toc - tic) / float(TIMES)
    gflop = (2.0 * N * N * N) * 1e-9
    print(f"custom: {gflop/s} (GFLOP/S)")

    c_mx = mx.matmul(a, b)
    print("Kernel result:")
    print(tmp)
    print("\nMLX result:")
    print(c_mx)
    print("\nClose?", mx.allclose(tmp, c_mx, atol=1e-4).item())


if __name__ == "__main__":
    bench(simdgroup_4x4_matmul_kernel, N=1024)
