#include <stdio.h>

int float_to_bits(float number) {
    float* float_addr = &number;
    printf("%p\n", float_addr);
    return 0;
}

int main(void) {
    float_to_bits(0.1);
    return 0;
}

