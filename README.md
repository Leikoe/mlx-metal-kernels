# mlx-metal-kernels

Observations:

Metal compiler (atleast when using mlx.fast.metal_kernel) doesn't unroll loops and ignores pragmas unroll.
Metal has private hidden async operations which can be found here: https://github.com/dougallj/applegpu/issues/28
