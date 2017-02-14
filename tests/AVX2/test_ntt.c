#include <const.h>
#include <ntt1024.h>
#include <stdio.h>
#include <string.h>

static long long q = 0x20008001;
static alignas(32) int src[1024] = { 0 }, dst[1024] = { 0 };
static alignas(32) int srctest[1024] = { 0 }, dsttest[1024] = { 0 };


int mod(long long x) {
	if (x >= 0) {
        return (int)(x % q);
    }
	while (x < 0) {
		x += q;
	}
	return (int)x;
}

int
equal(int *d, int *s) {
    if (memcmp(s, d, sizeof(dst)) != 0) {
        return 0;
    }
    return 1;
}

void
cpy(int *d, int *s) {
    memcpy(d, s, sizeof(dst));
}

void
print(int im, int *f, int *s) {
    printf("im : %4d\n", im);
    for (int i = 0; i < 1024; ++i) {
        printf("%10d, %10d\n", f[i], s[i]);
    }
}

void
prepare_input() {
    int i;

    for (i = 0; i < 512; ++i) {
        src[i] = i;
        srctest[i] = i;
    }
    bitrev32x1024(dst, src);
    cpy(src, dst);
}

void
fire_stage(int im) {
    switch (im) {
    case 512:
        nttditstage0(dsttest, srctest);
        cpy(srctest, dsttest);
        break;
    case 256:
        nttditstage1(srctest, 0x20000000);
        break;
    case 128:
        nttditstage2(srctest, 0xb754980, 0x20000000, 0x6b56455);
        break;
    case 64:
        nttditstage(srctest, wvec256, im, 1);
        break;
    case 32:
        nttditstage(srctest, wvec256 + 8, im, 2);
        break;
    case 16:
        nttditstage(srctest, wvec256 + 8 + 16, im, 4);
        break;
    case 8:
        nttditstage(srctest, wvec256 + 8 + 16 + 32, im, 8);
        break;
    case 4:
        nttditstage(srctest, wvec256 + 8 + 16 + 32 + 64, im, 16);
        break;
    case 2:
        nttditstage(srctest, wvec256 + 8 + 16 + 32 + 64 + 128, im, 32);
        break;
    case 1:
        nttditstage(srctest, wvec256 + 8 + 16 + 32 + 64 + 128 + 256, im, 64);
        break;
    }
    sub0x20008001(srctest, srctest);
    cpy(dsttest, srctest);
}

int
main(void) {
    const int *wt = wtest;
    int im = 512, jm = 1, f, s;
    long long fp, sp, fo, so;
    int i, j;

    prepare_input();
    while (im) {
        fire_stage(im);
        for (i = 0; i < im; ++i) {
            for (j = 0; j < jm; ++j) {
                f = (2 * i + 0) * jm + j;
                s = (2 * i + 1) * jm + j;
                fp = src[f];
                sp = src[s];
                sp = mod(wt[j] * sp);

                fo = mod(fp + sp);
                so = mod(fp - sp);

                dst[f] = (int)fo;
                dst[s] = (int)so;
            }
        }
        if (equal(dst, dsttest) == 0) {
            print(im, dst, dsttest);
            return 1;
        }
        wt += jm;
        im >>= 1;
        jm <<= 1;

        cpy(src, dst);
        cpy(srctest, dsttest);
    }
    return 0;
}

