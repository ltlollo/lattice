#include <const.h>
#include <ntt1024.h>
#include <stdalign.h>
#include <stdio.h>
#include <string.h>

int
main(void) {
    alignas(32) int src[1024] = { 0 }, one[1024] = {[512] = 1 },
                    p1[1024] = { 0 };
	int i;

    for (i = 0; i < 1024; ++i) {
        src[i] = i;
        p1[i] = i;
    }
    nttdif32x1024mulphibitrev(p1, p1);
    nttdif32x1024mulphibitrev(one, one);
    mulvec(p1, p1, one);
    bitrevinttdit32x1024muliphi(p1, p1);

    if (memcmp(src, p1 + 512, sizeof(src) / 2) != 0) {
        return 1;
    }
	return 0;
}

