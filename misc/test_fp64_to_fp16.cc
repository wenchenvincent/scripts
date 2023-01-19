#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

const std::vector<float> normal_values = {
  
  /* 0 */   0x0.2p-6, 0x0.4p-6, 0x0.6p-6, 0x0.8p-6, 0x0.ap-6, 0x0.cp-6, 0x0.ep-6,
  
  0x1.0p-6, 0x1.2p-6, 0x1.4p-6, 0x1.6p-6, 0x1.8p-6, 0x1.ap-6, 0x1.cp-6, 0x1.ep-6,
  0x1.0p-5, 0x1.2p-5, 0x1.4p-5, 0x1.6p-5, 0x1.8p-5, 0x1.ap-5, 0x1.cp-5, 0x1.ep-5,
  0x1.0p-4, 0x1.2p-4, 0x1.4p-4, 0x1.6p-4, 0x1.8p-4, 0x1.ap-4, 0x1.cp-4, 0x1.ep-4,
  0x1.0p-3, 0x1.2p-3, 0x1.4p-3, 0x1.6p-3, 0x1.8p-3, 0x1.ap-3, 0x1.cp-3, 0x1.ep-3,
  0x1.0p-2, 0x1.2p-2, 0x1.4p-2, 0x1.6p-2, 0x1.8p-2, 0x1.ap-2, 0x1.cp-2, 0x1.ep-2,
  0x1.0p-1, 0x1.2p-1, 0x1.4p-1, 0x1.6p-1, 0x1.8p-1, 0x1.ap-1, 0x1.cp-1, 0x1.ep-1,
  0x1.0p0,  0x1.2p0,  0x1.4p0,  0x1.6p0,  0x1.8p0,  0x1.ap0,  0x1.cp0,  0x1.ep0,
  0x1.0p1,  0x1.2p1,  0x1.4p1,  0x1.6p1,  0x1.8p1,  0x1.ap1,  0x1.cp1,  0x1.ep1,
  0x1.0p2,  0x1.2p2,  0x1.4p2,  0x1.6p2,  0x1.8p2,  0x1.ap2,  0x1.cp2,  0x1.ep2,
  0x1.0p3,  0x1.2p3,  0x1.4p3,  0x1.6p3,  0x1.8p3,  0x1.ap3,  0x1.cp3,  0x1.ep3,
  0x1.0p4,  0x1.2p4,  0x1.4p4,  0x1.6p4,  0x1.8p4,  0x1.ap4,  0x1.cp4,  0x1.ep4,
  0x1.0p5,  0x1.2p5,  0x1.4p5,  0x1.6p5,  0x1.8p5,  0x1.ap5,  0x1.cp5,  0x1.ep5,
  0x1.0p6,  0x1.2p6,  0x1.4p6,  0x1.6p6,  0x1.8p6,  0x1.ap6,  0x1.cp6,  0x1.ep6,
  0x1.0p7,  0x1.2p7,  0x1.4p7,  0x1.6p7,  0x1.8p7,  0x1.ap7,  0x1.cp7,  0x1.ep7,
};

template<typename T>
void test_rounding_standard(const std::vector<float>& normal_values) {
  int N = (normal_values.size() - 1) * 4;

  T* host_input_T = (T*) malloc (N * sizeof(T));
  // initialize the inputs
  for (int i=0, j=0; i<N; i+=4, j++) {
    float value = normal_values[j];
    float delta = normal_values[j+1] - normal_values[j];
    printf("value = %f, value.data = 0x%x, delta = %f, delta.data = 0x%x\n", value, reinterpret_cast<uint32_t&>(value), delta, reinterpret_cast<uint32_t&>(delta));
    host_input_T[i+0] = static_cast<T>(value + (delta * 0.2));
    host_input_T[i+1] = static_cast<T>(value + (delta * 0.4));
    host_input_T[i+2] = static_cast<T>(value + (delta * 0.6));
    host_input_T[i+3] = static_cast<T>(value + (delta * 0.8));
    printf("fp32_value = %f, host_input_T[%d]=%f, host_input_T[%d].data=0x%x\n", (value + (delta * 0.2)), i+0, (float)host_input_T[i+0], i+0, reinterpret_cast<uint16_t&>(host_input_T[i+0]));
    printf("fp32_value = %f, host_input_T[%d]=%f, host_input_T[%d].data=0x%x\n", (value + (delta * 0.4)), i+1, (float)host_input_T[i+1], i+1, reinterpret_cast<uint16_t&>(host_input_T[i+1]));
    printf("fp32_value = %f, host_input_T[%d]=%f, host_input_T[%d].data=0x%x\n", (value + (delta * 0.6)), i+2, (float)host_input_T[i+2], i+2, reinterpret_cast<uint16_t&>(host_input_T[i+2]));
    printf("fp32_value = %f, host_input_T[%d]=%f, host_input_T[%d].data=0x%x\n", (value + (delta * 0.8)), i+3, (float)host_input_T[i+3], i+3, reinterpret_cast<uint16_t&>(host_input_T[i+3]));
  }
  free(host_input_T);
}

int main() {
  test_rounding_standard<__half>(normal_values);
  return 0;
}
