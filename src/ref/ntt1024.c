#include <const.h>
#include <ntt1024.h>
#include <string.h>

static void stages(int *restrict, const int *restrict, const int *);
static void stagesmulphi(int *restrict, const int *restrict, const int *);

DECL void
ntt32x1024(int *restrict dst, const int *restrict src) {
    stages(dst, src, wtest);
}

DECL void
intt32x1024(int *restrict dst, const int *restrict src) {
    stages(dst, src, iwtest);
    mulconst(dst, dst, 0x1ff87fe1);
}

DECL void
intt32x1024muliphi(int *restrict dst, const int *restrict src) {
    stages(dst, src, iwtest);
    mulvec(dst, dst, inphi);
}

DECL void
ntt32x1024mulphi(int *restrict dst, const int *restrict src) {
    stagesmulphi(dst, src, wtest);
}

INTER void
bitrev32x1024(int *dst, int const *src) {
    for (int i = 0; i < 1024; ++i) {
        dst[i] = src[off[i]];
    }
}

INTER void
bitrev32x1024mulphi(int *dst, int const *src) {
    long long imm, q = 0x20008001;

    for (int i = 0; i < 1024; ++i) {
        imm = src[off[i]];
        dst[i] = (int)(imm * phirev[i] % q);
    }
}

static void
stages(int *restrict dst, const int *restrict src, const int *wt) {
    long long fp, sp, fo, so, q = 0x20008001;
    int im = 512, jm = 1, f, s;

    bitrev32x1024(dst, src);
    while (im) {
        for (int i = 0; i < im; ++i) {
            for (int j = 0; j < jm; ++j) {
                f = (2 * i + 0) * jm + j;
                s = (2 * i + 1) * jm + j;
                fp = dst[f];
                sp = dst[s];
                sp = (wt[j] * sp) % q;

                fo = (fp + sp) % q;
                so = (q + fp - sp) % q;

                dst[f] = (int)fo;
                dst[s] = (int)so;
            }
        }
        wt += jm;
        im >>= 1;
        jm <<= 1;
    }
}

INTER void
nttditstage(int *restrict dst, int const *restrict wt, int im, int jm) {
    long long fp, sp, fo, so, q = 0x20008001;
    int f, s;

    for (int i = 0; i < im; ++i) {
        for (int j = 0; j < jm; ++j) {
            f = (2 * i + 0) * jm + j;
            s = (2 * i + 1) * jm + j;
            fp = dst[f];
            sp = dst[s];
            sp = (wt[j] * sp) % q;

            fo = (fp + sp) % q;
            so = (q + fp - sp) % q;

            dst[f] = (int)fo;
            dst[s] = (int)so;
        }
    }
}

INTER void
nttdifstage(int *restrict dst, int const *restrict wt, int im, int jm) {
    long long fp, sp, fo, so, q = 0x20008001;
    int f, s;

    for (int i = 0; i < im; ++i) {
        for (int j = 0; j < jm; ++j) {
            f = (2 * i + 0) * jm + j;
            s = (2 * i + 1) * jm + j;
            fp = dst[f];
            sp = dst[s];

            fo = (fp + sp) % q;
            so = (q + fp - sp) % q;

            so = (wt[j] * so) % q;

            dst[f] = (int)fo;
            dst[s] = (int)so;
        }
    }
}

DECL void
nttdif32x1024mulphibitrev(int *dst, const int *src) {
    mulvec(dst, src, phi);
    nttdifstage(dst, difwvec + 0x000, 0x001, 0x200);
    nttdifstage(dst, difwvec + 0x200, 0x002, 0x100);
    nttdifstage(dst, difwvec + 0x300, 0x004, 0x080);
    nttdifstage(dst, difwvec + 0x380, 0x008, 0x040);
    nttdifstage(dst, difwvec + 0x3c0, 0x010, 0x020);
    nttdifstage(dst, difwvec + 0x3e0, 0x020, 0x010);
    nttdifstage(dst, difwvec + 0x3f0, 0x040, 0x008);
    nttdifstage(dst, difwvec + 0x3f8, 0x080, 0x004);
    nttdifstage(dst, difwvec + 0x3fc, 0x100, 0x002);
    nttdifstage(dst, difwvec + 0x3fe, 0x200, 0x001);
}

DECL void
bitrevinttdit32x1024muliphi(int *dst, const int *src) {
    memcpy(dst, src, 1024 * sizeof(int));
    nttditstage(dst, iwtest + 0x000, 0x200, 0x001);
    nttditstage(dst, iwtest + 0x001, 0x100, 0x002);
    nttditstage(dst, iwtest + 0x003, 0x080, 0x004);
    nttditstage(dst, iwtest + 0x007, 0x040, 0x008);
    nttditstage(dst, iwtest + 0x00f, 0x020, 0x010);
    nttditstage(dst, iwtest + 0x01f, 0x010, 0x020);
    nttditstage(dst, iwtest + 0x03f, 0x008, 0x040);
    nttditstage(dst, iwtest + 0x07f, 0x004, 0x080);
    nttditstage(dst, iwtest + 0x0ff, 0x002, 0x100);
    nttditstage(dst, iwtest + 0x1ff, 0x001, 0x200);
    mulvec(dst, dst, inphi);
}

static void
stagesmulphi(int *restrict dst, const int *restrict src, const int *wt) {
    long long fp, sp, fo, so, q = 0x20008001;
    int im = 512, jm = 1, f, s;

    bitrev32x1024mulphi(dst, src);
    while (im) {
        for (int i = 0; i < im; ++i) {
            for (int j = 0; j < jm; ++j) {
                f = (2 * i + 0) * jm + j;
                s = (2 * i + 1) * jm + j;
                fp = dst[f];
                sp = dst[s];
                sp = (wt[j] * sp) % q;

                fo = (fp + sp) % q;
                so = (q + fp - sp) % q;

                dst[f] = (int)fo;
                dst[s] = (int)so;
            }
        }
        wt += jm;
        im >>= 1;
        jm <<= 1;
    }
}

DECL void
mulvec(int *dst, const int *src, const int *m) {
    long long t, u, q = 0x20008001;

    for (int i = 0; i < 1024; ++i) {
        t = src[i], u = m[i];
        t = (u * t) % q;
        dst[i] = t;
    }
}

INTER void
mulconst(int *dst, const int *src, const int m) {
    long long t, q = 0x20008001;

    for (int i = 0; i < 1024; ++i) {
        t = src[i];
        t = (m * t) % q;
        dst[i] = t;
    }
}

