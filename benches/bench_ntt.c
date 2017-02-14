#include <ntt1024.h>
#include <stdalign.h>
#include <stdio.h>
#include <string.h>

int
main(void) {
    alignas(32) int src[1024] = { 0 }, dst[1024] = { 0 };
	int i;

    for (i = 0; i < 1024 / 2; ++i) {
        src[i] = i;
    }
    for (i = 0; i < 1 << 19; ++i) {
        nttdif32x1024mulphibitrev(dst, src);
    }
	return 0;
}

