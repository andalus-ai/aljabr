#include <iostream>
#include <mlx/mlx.h>
#include <numeric>
#include <vector>

extern "C" {

// Initialize MLX and print device info
void aljabr_mlx_init() {
  auto &device = mlx::core::default_device();
  std::cout << "[MLX] Initialized on device: " << device << std::endl;
}

// Create an MLX array from a float pointer
void *aljabr_mlx_array_from_float(const float *data, const int64_t *shape,
                                       int ndim) {
  mlx::core::Shape shape_vec;
  for (int i = 0; i < ndim; ++i) {
    shape_vec.push_back(static_cast<int>(shape[i]));
  }

  // Create an array by copying the data
  // The template constructor takes (pointer, shape, dtype)
  auto arr = mlx::core::array(data, shape_vec, mlx::core::float32);
  return new mlx::core::array(arr);
}

// Perform matmul: out = a @ b
void *aljabr_mlx_matmul(void *a_ptr, void *b_ptr) {
  auto &a = *static_cast<mlx::core::array *>(a_ptr);
  auto &b = *static_cast<mlx::core::array *>(b_ptr);

  auto res = mlx::core::matmul(a, b);
  return new mlx::core::array(res);
}

// Evaluate an array (MLX is lazy)
void aljabr_mlx_eval(void *ptr) {
  auto &arr = *static_cast<mlx::core::array *>(ptr);
  mlx::core::eval(arr);
}

// Copy data back to a float pointer
void aljabr_mlx_array_get_data(void *ptr, float *out_data) {
  auto &arr = *static_cast<mlx::core::array *>(ptr);
  // Ensure it's evaluated
  mlx::core::eval(arr);

  const float *src = arr.data<float>();
  std::copy(src, src + arr.size(), out_data);
}

// Free an array pointer
void aljabr_mlx_array_free(void *ptr) {
  delete static_cast<mlx::core::array *>(ptr);
}
}
