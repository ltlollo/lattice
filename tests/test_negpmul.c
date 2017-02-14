#include <const.h>
#include <ntt1024.h>
#include <stdalign.h>
#include <stdio.h>
#include <string.h>

int
main(void) {
    alignas(32) int src[1024] = { 0 }, one[1024] = {[512] = 1 },
                    p1[1024] = { 0 }, p2[1024] = { 0 };
	int i;

    for (i = 0; i < 1024; ++i) {
        src[i] = i;
    }
    ntt32x1024mulphi(p1, src);
    ntt32x1024mulphi(p2, one);
    mulvec(p1, p1, p2);
    intt32x1024muliphi(p2, p1);
    if (memcmp(src, p2 + 512, sizeof(src) / 2) != 0) {
        return 1;
    }
	return 0;
}

