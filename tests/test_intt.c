#include <ntt1024.h>
#include <stdalign.h>
#include <stdio.h>
#include <string.h>

int
main(void) {
    alignas(32) int src[1024] = { 0 }, fw[1024] = { 0 }, bk[1024] = { 0 };
	int i;

    for (i = 0; i < 1024 / 2; ++i) {
        src[i] = i;
    }
    ntt32x1024(fw, src);
    intt32x1024(bk, fw);
	if (memcmp(src, bk, sizeof(src)) != 0) {
		return 1;
	}
	return 0;
}

