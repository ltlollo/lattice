#include <const.h>
#include <ntt1024.h>
#include <stdio.h>
#include <string.h>

int
main(void) {
	unsigned long long q = 0x20008001;
    unsigned long long vals[] = {
        1, 2, 10, 0x20008000, 0x2583dc, 0x20000000,
    };

    alignas(32) int src[1024], dst[1024];
    unsigned i, j;

    for (j = 0; j < sizeof(vals) / sizeof(vals[0]); ++j) {
        for (int i = 0; i < 1024; ++i) {
            src[i] = (int)((unsigned long long)phi[i] * vals[j] % q);
        }
        mulvec(dst, src, iphi);
        for (i = 0; i < 1024; ++i) {
            src[i] = (int)vals[j];
        }
        if (memcmp(dst, src, sizeof(dst)) != 0) {
            return 1;
        }
    }
    return 0;
}

