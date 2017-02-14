#include <const.h>
#include <ntt1024.h>
#include <string.h>

int
main(void) {
    alignas(32) int src[1024], dst[1024];
    int i;

    for (i = 0; i < 1024; ++i) {
        src[i] = i;
	}
    bitrev32x1024(dst, src);
	return memcmp(dst, off, sizeof(dst));
}

