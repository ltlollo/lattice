#include <immintrin.h>
#include <string.h>

#include <const.h>
#include <ntt1024.h>

typedef int i32;
typedef __m128i i128;

static inline i128 mulmod0x20008001u(i128, i128);

DECL void
ntt32x1024(i32 *restrict dst, const i32 *restrict src) {
    nttditstage0(dst, src);
    nttditstage1(dst, 0x20000000);
    nttditstagen(dst, wvec128);
}

DECL void
ntt32x1024mulphi(i32 *restrict dst, const i32 *restrict src) {
    bitrev32x1024mulphi(dst, src);
    bitrevnttditstage0(dst, dst);
    nttditstage1(dst, 0x20000000);
    nttditstagen(dst, wvec128);
}

DECL void
intt32x1024(i32 *restrict dst, const i32 *restrict src) {
    nttditstage0(dst, src);
    nttditstage1(dst, 0x8001);
    nttditstagen(dst, iwvec128);
    mulconst(dst, dst, 0x1ff87fe1);
}

DECL void
intt32x1024muliphi(i32 *restrict dst, const i32 *restrict src) {
    nttditstage0(dst, src);
    nttditstage1(dst, 0x8001);
    nttditstagen(dst, iwvec128);
    mulvec(dst, dst, inphi);
}

DECL void
nttdif32x1024mulphibitrev(i32 * dst, const i32 *src) {
    mulvec(dst, src, phi);
    nttdifstagen(dst, difwvec128);
    nttdifstage1(dst, 0x20000000);
    nttdifstage0(dst);
}

DECL void
bitrevinttdit32x1024muliphi(i32 *dst, const i32 *src) {
    bitrevnttditstage0(dst, src);
    nttditstage1(dst, 0x8001);
    nttditstagen(dst, iwvec128);
    mulvec(dst, dst, inphi);
}

static inline i128
mulmod0x20008001u(i128 v, i128 mv) {
    // v, mv must be positive
    i128 hn1m, hn1lo, hn1hi, hp2m, hp2lo, hp2hi, hn2, hi, hp1, lo, lp1, ln1m,
            ln1lo, ln1hi, lp2m, lp2lo, lp2hi, ln2, p1, n1hi, p2hi, n1lo, p2lo,
            n2, gq, mq, mvhi, mvlo;

    hi = _mm_srli_epi64(v, 32);
    mvhi = _mm_srli_epi64(mv, 32);
    lo = _mm_and_si128(v, _mm_set1_epi64x(0xffffffff));
    mvlo = _mm_and_si128(mv, _mm_set1_epi64x(0xffffffff));

    hi = _mm_mul_epu32(hi, mvhi);
    hp1 = _mm_and_si128(hi, _mm_set1_epi64x(0x1fffffff));
    hp1 = _mm_slli_epi64(hp1, 32);
    hn1m = _mm_and_si128(hi, _mm_set1_epi64x(0x7ffe0000000));
    hn1lo = _mm_slli_epi64(hn1m, 3);
    hn1hi = _mm_slli_epi64(hn1m, 18);
    hp2m = _mm_and_si128(hi, _mm_set1_epi64x(0x3fff80000000000));
    hp2lo = _mm_srli_epi64(hp2m, 11);
    hp2hi = _mm_slli_epi64(hp2m, 3);
    hn2 = _mm_srli_epi64(hi, 26);

    lo = _mm_mul_epu32(lo, mvlo);
    lp1 = _mm_and_si128(lo, _mm_set1_epi64x(0x1fffffff));
    ln1m = _mm_and_si128(lo, _mm_set1_epi64x(0x7ffe0000000));
    ln1lo = _mm_srli_epi64(ln1m, 29);
    ln1hi = _mm_srli_epi64(ln1m, 14);
    lp2m = _mm_and_si128(lo, _mm_set1_epi64x(0x3fff80000000000));
    lp2lo = _mm_srli_epi64(lp2m, 43);
    lp2hi = _mm_srli_epi64(lp2m, 29);
    ln2 = _mm_srli_epi64(lo, 58);

    p1 = _mm_blend_epi32(lp1, hp1, 170);
    n1hi = _mm_blend_epi32(ln1hi, hn1hi, 170);
    p2hi = _mm_blend_epi32(lp2hi, hp2hi, 170);
    n1lo = _mm_blend_epi32(ln1lo, hn1lo, 170);
    p2lo = _mm_blend_epi32(lp2lo, hp2lo, 170);
    n2 = _mm_blend_epi32(ln2, hn2, 170);

    p1 = _mm_sub_epi32(p1, n1hi);
    p1 = _mm_add_epi32(p1, p2hi);
    p1 = _mm_sub_epi32(p1, n1lo);
    p1 = _mm_add_epi32(p1, p2lo);
    p1 = _mm_sub_epi32(p1, n2);

    // p1:[01000000000000000011111111111110, -00011111111111111100000000111110]

    gq = _mm_cmpgt_epi32(p1, _mm_set1_epi32(0x20008000));
    mq = _mm_and_si128(gq, _mm_set1_epi32(0x20008001));
    p1 = _mm_sub_epi32(p1, mq);

    gq = _mm_cmpgt_epi32(_mm_set1_epi32(0), p1);
    mq = _mm_and_si128(gq, _mm_set1_epi32(0x20008001));
    p1 = _mm_add_epi32(p1, mq);
    return p1;
}

INTER void
nttditstage0(i32 *dst, const i32 *src) {
    i32 i = 0;
    i128 i0, i1, i2, i3, o0, o1, mask1, q = _mm_set1_epi32(0x20008001);
    i128 maxf = _mm_set1_epi32(0x20008000);
    i128 mask0;
    i128 minf = _mm_set1_epi32(0);

    bitrev32x1024(dst, src);

    i0 = _mm_load_si128((i128 *)(dst + i + 0));
    i1 = _mm_load_si128((i128 *)(dst + i + 4));
    o0 = _mm_hadd_epi32(i0, i1);
    i2 = _mm_load_si128((i128 *)(dst + i + 8));
    i3 = _mm_load_si128((i128 *)(dst + i + 12));
    o1 = _mm_hsub_epi32(i0, i1);

    mask0 = _mm_cmpgt_epi32(o0, maxf);
    mask1 = _mm_cmpgt_epi32(minf, o1);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    o0 = _mm_sub_epi32(o0, mask0);
    o1 = _mm_add_epi32(o1, mask1);
    i0 = _mm_unpacklo_epi32(o0, o1);
    i1 = _mm_unpackhi_epi32(o0, o1);

    for (; i < 1024 - 16; i += 16) {
        _mm_store_si128((i128 *)(dst + i + 0), i0);
        o0 = _mm_hadd_epi32(i2, i3);
        i0 = _mm_load_si128((i128 *)(dst + i + 16));
        _mm_store_si128((i128 *)(dst + i + 4), i1);
        o1 = _mm_hsub_epi32(i2, i3);
        i1 = _mm_load_si128((i128 *)(dst + i + 16 + 4));

        mask0 = _mm_cmpgt_epi32(o0, maxf);
        mask1 = _mm_cmpgt_epi32(minf, o1);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        o0 = _mm_sub_epi32(o0, mask0);
        o1 = _mm_add_epi32(o1, mask1);
        i2 = _mm_unpacklo_epi32(o0, o1);
        i3 = _mm_unpackhi_epi32(o0, o1);

        _mm_store_si128((i128 *)(dst + i + 8), i2);
        o0 = _mm_hadd_epi32(i0, i1);
        i2 = _mm_load_si128((i128 *)(dst + i + 16 + 8));

        _mm_store_si128((i128 *)(dst + i + 12), i3);
        o1 = _mm_hsub_epi32(i0, i1);
        i3 = _mm_load_si128((i128 *)(dst + i + 16 + 12));

        mask0 = _mm_cmpgt_epi32(o0, maxf);
        mask1 = _mm_cmpgt_epi32(minf, o1);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        o0 = _mm_sub_epi32(o0, mask0);
        o1 = _mm_add_epi32(o1, mask1);
        i0 = _mm_unpacklo_epi32(o0, o1);
        i1 = _mm_unpackhi_epi32(o0, o1);
    }
    _mm_store_si128((i128 *)(dst + i + 0), i0);
    o0 = _mm_hadd_epi32(i2, i3);

    _mm_store_si128((i128 *)(dst + i + 4), i1);
    o1 = _mm_hsub_epi32(i2, i3);

    mask0 = _mm_cmpgt_epi32(o0, maxf);
    mask1 = _mm_cmpgt_epi32(minf, o1);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    o0 = _mm_sub_epi32(o0, mask0);
    o1 = _mm_add_epi32(o1, mask1);
    i2 = _mm_unpacklo_epi32(o0, o1);
    i3 = _mm_unpackhi_epi32(o0, o1);

    _mm_store_si128((i128 *)(dst + i + 8), i2);
    _mm_store_si128((i128 *)(dst + i + 12), i3);
}

INTER void
nttditstage0mulphi(i32 *dst, const i32 *src) {
    bitrev32x1024mulphi(dst, src);
    bitrevnttditstage0(dst, dst);
}

INTER void
nttditstage1(i32 *dst, i32 w256) {
    i32 i = 0;
    i128 i0, i1, i2, i3, o0, o1, mask1, m = _mm_set_epi32(w256, 1, w256, 1),
                                        // intel likes his args little fermat
            q = _mm_set1_epi32(0x20008001), minf = _mm_set1_epi32(0);
    i128 maxf = _mm_set1_epi32(0x20008000);
    i128 mask0;

    i0 = _mm_load_si128((i128 *)(dst + i + 0));
    i1 = _mm_load_si128((i128 *)(dst + i + 4));

    o0 = _mm_unpacklo_epi64(i0, i1);
    o1 = _mm_unpackhi_epi64(i0, i1);
    o1 = mulmod0x20008001u(o1, m);

    i2 = _mm_load_si128((i128 *)(dst + i + 8));
    i3 = _mm_load_si128((i128 *)(dst + i + 12));

    i0 = _mm_add_epi32(o0, o1);
    i1 = _mm_sub_epi32(o0, o1);
    mask0 = _mm_cmpgt_epi32(i0, maxf);
    mask1 = _mm_cmpgt_epi32(minf, i1);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    i0 = _mm_sub_epi32(i0, mask0);
    i1 = _mm_add_epi32(i1, mask1);
    o0 = _mm_unpacklo_epi64(i0, i1);
    o1 = _mm_unpackhi_epi64(i0, i1);

    for (; i < 1024 - 16; i += 16) {
        _mm_store_si128((i128 *)(dst + i + 0), o0);
        i0 = _mm_load_si128((i128 *)(dst + i + 16));
        o0 = _mm_unpacklo_epi64(i2, i3);

        _mm_store_si128((i128 *)(dst + i + 4), o1);
        i1 = _mm_load_si128((i128 *)(dst + i + 16 + 4));
        o1 = _mm_unpackhi_epi64(i2, i3);
        o1 = mulmod0x20008001u(o1, m);

        i2 = _mm_add_epi32(o0, o1);
        i3 = _mm_sub_epi32(o0, o1);
        mask0 = _mm_cmpgt_epi32(i2, maxf);
        mask1 = _mm_cmpgt_epi32(minf, i3);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        i2 = _mm_sub_epi32(i2, mask0);
        i3 = _mm_add_epi32(i3, mask1);
        o0 = _mm_unpacklo_epi64(i2, i3);
        o1 = _mm_unpackhi_epi64(i2, i3);

        _mm_store_si128((i128 *)(dst + i + 8), o0);
        i2 = _mm_load_si128((i128 *)(dst + i + 16 + 8));
        o0 = _mm_unpacklo_epi64(i0, i1);

        _mm_store_si128((i128 *)(dst + i + 12), o1);
        i3 = _mm_load_si128((i128 *)(dst + i + 16 + 12));
        o1 = _mm_unpackhi_epi64(i0, i1);
        o1 = mulmod0x20008001u(o1, m);

        i0 = _mm_add_epi32(o0, o1);
        i1 = _mm_sub_epi32(o0, o1);
        mask0 = _mm_cmpgt_epi32(i0, maxf);
        mask1 = _mm_cmpgt_epi32(minf, i1);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        i0 = _mm_sub_epi32(i0, mask0);
        i1 = _mm_add_epi32(i1, mask1);
        o0 = _mm_unpacklo_epi64(i0, i1);
        o1 = _mm_unpackhi_epi64(i0, i1);
    }
    _mm_store_si128((i128 *)(dst + i + 0), o0);
    o0 = _mm_unpacklo_epi64(i2, i3);

    _mm_store_si128((i128 *)(dst + i + 4), o1);
    o1 = _mm_unpackhi_epi64(i2, i3);
    o1 = mulmod0x20008001u(o1, m);

    i2 = _mm_add_epi32(o0, o1);
    i3 = _mm_sub_epi32(o0, o1);
    mask0 = _mm_cmpgt_epi32(i2, maxf);
    mask1 = _mm_cmpgt_epi32(minf, i3);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    i2 = _mm_sub_epi32(i2, mask0);
    i3 = _mm_add_epi32(i3, mask1);
    o0 = _mm_unpacklo_epi64(i2, i3);
    o1 = _mm_unpackhi_epi64(i2, i3);

    _mm_store_si128((i128 *)(dst + i + 8), o0);
    _mm_store_si128((i128 *)(dst + i + 12), o1);
}

INTER void
nttditstagen(i32 *restrict dst, const i32 *restrict wvec) {
    nttditstage(dst, wvec + 0x000, 0x80, 0x01);
    nttditstage(dst, wvec + 0x004, 0x40, 0x02);
    nttditstage(dst, wvec + 0x00c, 0x20, 0x04);
    nttditstage(dst, wvec + 0x01c, 0x10, 0x08);
    nttditstage(dst, wvec + 0x03c, 0x08, 0x10);
    nttditstage(dst, wvec + 0x07c, 0x04, 0x20);
    nttditstage(dst, wvec + 0x0fc, 0x02, 0x40);
    nttditstage(dst, wvec + 0x1fc, 0x01, 0x80);
}

INTER void
nttdifstagen(i32 *restrict dst, const i32 *restrict wvec) {
    nttdifstage(dst, wvec + 0x000, 0x01, 0x80);
    nttdifstage(dst, wvec + 0x200, 0x02, 0x40);
    nttdifstage(dst, wvec + 0x300, 0x04, 0x20);
    nttdifstage(dst, wvec + 0x380, 0x08, 0x10);
    nttdifstage(dst, wvec + 0x3c0, 0x10, 0x08);
    nttdifstage(dst, wvec + 0x3e0, 0x20, 0x04);
    nttdifstage(dst, wvec + 0x3f0, 0x40, 0x02);
    nttdifstage(dst, wvec + 0x3f8, 0x80, 0x01);
}

INTER void
nttditstage(i32 *restrict dst, i32 const *restrict wvec, i32 im, i32 jm) {
    i32 f, s, i, j;
    const i32 *vw = wvec;
    i128 m, i0, i1, o0, o1, mask1, q = _mm_set1_epi32(0x20008001),
                                   minf = _mm_set1_epi32(0);
    i128 maxf = _mm_set1_epi32(0x20008000);
    i128 mask0;

    for (i = 0; i < im; ++i) {
        for (j = 0; j < jm; ++j) {
            f = 4 * ((2 * i + 0) * jm + j);
            s = 4 * ((2 * i + 1) * jm + j);
            m = _mm_load_si128((i128 *)(vw + 4 * j));
            o0 = _mm_load_si128((i128 *)(dst + f));
            o1 = _mm_load_si128((i128 *)(dst + s));
            o1 = mulmod0x20008001u(o1, m);

            i0 = _mm_add_epi32(o0, o1);
            i1 = _mm_sub_epi32(o0, o1);

            mask0 = _mm_cmpgt_epi32(i0, maxf);
            mask1 = _mm_cmpgt_epi32(minf, i1);
            mask0 = _mm_and_si128(mask0, q);
            mask1 = _mm_and_si128(mask1, q);
            i0 = _mm_sub_epi32(i0, mask0);
            i1 = _mm_add_epi32(i1, mask1);
            _mm_store_si128((i128 *)(dst + f), i0);
            _mm_store_si128((i128 *)(dst + s), i1);
        }
    }
}

DECL void
mulvec(i32 *dst, const i32 *f, const i32 *s) {
    i32 i = 0;
    i128 i0, i1, i2, i3, i4, i5, o0, o1, o2;

    i0 = _mm_load_si128((i128 *)(f + i + 0));
    i2 = _mm_load_si128((i128 *)(f + i + 4));
    i1 = _mm_load_si128((i128 *)(s + i + 0));
    i3 = _mm_load_si128((i128 *)(s + i + 4));
    o0 = mulmod0x20008001u(i0, i1);
    i4 = _mm_load_si128((i128 *)(f + i + 8));
    i5 = _mm_load_si128((i128 *)(s + i + 8));
    for (i = 12; i < 1020; i += 12) {
        i0 = _mm_load_si128((i128 *)(f + i + 0));
        i1 = _mm_load_si128((i128 *)(s + i + 0));
        o1 = mulmod0x20008001u(i2, i3);
        _mm_store_si128((i128 *)(dst + i - 12), o0);
        i2 = _mm_load_si128((i128 *)(f + i + 4));
        i3 = _mm_load_si128((i128 *)(s + i + 4));
        o2 = mulmod0x20008001u(i4, i5);
        _mm_store_si128((i128 *)(dst + i - 8), o1);
        i4 = _mm_load_si128((i128 *)(f + i + 8));
        i5 = _mm_load_si128((i128 *)(s + i + 8));
        o0 = mulmod0x20008001u(i0, i1);
        _mm_store_si128((i128 *)(dst + i - 4), o2);
    }
    i0 = _mm_load_si128((i128 *)(f + i + 0));
    i1 = _mm_load_si128((i128 *)(s + i + 0));
    o1 = mulmod0x20008001u(i2, i3);
    _mm_store_si128((i128 *)(dst + i - 12), o0);
    o2 = mulmod0x20008001u(i4, i5);
    _mm_store_si128((i128 *)(dst + i - 8), o1);
    o0 = mulmod0x20008001u(i0, i1);
    _mm_store_si128((i128 *)(dst + i - 4), o2);
    _mm_store_si128((i128 *)(dst + i + 0), o0);
}

INTER void
mulconst(i32 *dst, const i32 *f, i32 c) {
    i32 i = 0;
    i128 i0, i2, i4, vm, o0, o1, o2;
    i0 = _mm_load_si128((i128 *)(f + i + 0));
    i2 = _mm_load_si128((i128 *)(f + i + 4));
    vm = _mm_set1_epi32(c);
    o0 = mulmod0x20008001u(i0, vm);
    i4 = _mm_load_si128((i128 *)(f + i + 8));
    for (i = 12; i < 1020; i += 12) {
        i0 = _mm_load_si128((i128 *)(f + i + 0));
        o1 = mulmod0x20008001u(i2, vm);
        _mm_store_si128((i128 *)(dst + i - 12), o0);
        i2 = _mm_load_si128((i128 *)(f + i + 4));
        o2 = mulmod0x20008001u(i4, vm);
        _mm_store_si128((i128 *)(dst + i - 8), o1);
        i4 = _mm_load_si128((i128 *)(f + i + 8));
        o0 = mulmod0x20008001u(i0, vm);
        _mm_store_si128((i128 *)(dst + i - 4), o2);
    }
    i0 = _mm_load_si128((i128 *)(f + i + 0));
    o1 = mulmod0x20008001u(i2, vm);
    _mm_store_si128((i128 *)(dst + i - 12), o0);
    o2 = mulmod0x20008001u(i4, vm);
    _mm_store_si128((i128 *)(dst + i - 8), o1);
    o0 = mulmod0x20008001u(i0, vm);
    _mm_store_si128((i128 *)(dst + i - 4), o2);
    _mm_store_si128((i128 *)(dst + i + 0), o0);
}

INTER void
sub0x20008001(i32 *dst, const i32 *f) {
    i32 i = 0;
    i128 i0, i2, i4, o0, o1, o2, q = _mm_set1_epi32(0x20008001),
                                 maxf = _mm_set1_epi32(0x20008000);
#define subq(dst, src)                                                        \
    dst = _mm_cmpgt_epi32(src, maxf);                                         \
    dst = _mm_and_si128(dst, q);                                              \
    dst = _mm_sub_epi32(src, dst);

    i0 = _mm_load_si128((i128 *)(f + i + 0));
    i2 = _mm_load_si128((i128 *)(f + i + 4));
    subq(o0, i0);

    i4 = _mm_load_si128((i128 *)(f + i + 8));
    for (i = 12; i < 1020; i += 12) {
        i0 = _mm_load_si128((i128 *)(f + i + 0));
        subq(o1, i2);
        _mm_store_si128((i128 *)(dst + i - 12), o0);
        i2 = _mm_load_si128((i128 *)(f + i + 4));
        subq(o2, i4);
        _mm_store_si128((i128 *)(dst + i - 8), o1);
        i4 = _mm_load_si128((i128 *)(f + i + 8));
        subq(o0, i0);
        _mm_store_si128((i128 *)(dst + i - 4), o2);
    }
    i0 = _mm_load_si128((i128 *)(f + i + 0));
    subq(o1, i2);
    _mm_store_si128((i128 *)(dst + i - 12), o0);
    subq(o2, i4);
    _mm_store_si128((i128 *)(dst + i - 8), o1);
    subq(o0, i0);
    _mm_store_si128((i128 *)(dst + i - 4), o2);
    _mm_store_si128((i128 *)(dst + i + 0), o0);
#undef subq
}

DECL void
cleanregs(void) {
    __asm__ __volatile__("vpxor\t%%xmm0,\t%%xmm0,\t%%xmm0\n\t"
                         "vpxor\t%%xmm1,\t%%xmm1,\t%%xmm1\n\t"
                         "vpxor\t%%xmm2,\t%%xmm2,\t%%xmm2\n\t"
                         "vpxor\t%%xmm3,\t%%xmm3,\t%%xmm3\n\t"
                         "vpxor\t%%xmm4,\t%%xmm4,\t%%xmm4\n\t"
                         "vpxor\t%%xmm5,\t%%xmm5,\t%%xmm5\n\t"
                         "vpxor\t%%xmm6,\t%%xmm6,\t%%xmm6\n\t"
                         "vpxor\t%%xmm7,\t%%xmm7,\t%%xmm7"
                         :
                         :
                         : "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4",
                           "%xmm5", "%xmm6", "%xmm7");
}

INTER void
bitrev32x1024(i32 *restrict dst, const i32 *restrict src) {
    i32 i = 0;
    i128 vindex0, v0, vindex1, v1, vindex2, v2, vindex3, v3;

    vindex0 = _mm_load_si128((i128 *)(off + i + 0));
    v0 = _mm_i32gather_epi32(src, vindex0, 4);

    vindex1 = _mm_load_si128((i128 *)(off + i + 4));
    v1 = _mm_i32gather_epi32(src, vindex1, 4);

    vindex2 = _mm_load_si128((i128 *)(off + i + 8));
    v2 = _mm_i32gather_epi32(src, vindex2, 4);

    vindex3 = _mm_load_si128((i128 *)(off + i + 12));
    v3 = _mm_i32gather_epi32(src, vindex3, 4);

    for (; i < 1024 - 16; i += 16) {
        _mm_store_si128((i128 *)(dst + i + 0), v0);
        vindex0 = _mm_load_si128((i128 *)(off + i + 0 + 16));
        v0 = _mm_i32gather_epi32(src, vindex0, 4);

        _mm_store_si128((i128 *)(dst + i + 4), v1);
        vindex1 = _mm_load_si128((i128 *)(off + i + 4 + 16));
        v1 = _mm_i32gather_epi32(src, vindex1, 4);

        _mm_store_si128((i128 *)(dst + i + 8), v2);
        vindex2 = _mm_load_si128((i128 *)(off + i + 8 + 16));
        v2 = _mm_i32gather_epi32(src, vindex2, 4);

        _mm_store_si128((i128 *)(dst + i + 12), v3);
        vindex3 = _mm_load_si128((i128 *)(off + i + 12 + 16));
        v3 = _mm_i32gather_epi32(src, vindex3, 4);
    }
    _mm_store_si128((i128 *)(dst + i + 0), v0);
    _mm_store_si128((i128 *)(dst + i + 4), v1);
    _mm_store_si128((i128 *)(dst + i + 8), v2);
    _mm_store_si128((i128 *)(dst + i + 12), v3);
}

INTER void
bitrev32x1024mulphi(i32 *restrict dst, const i32 *restrict src) {
    i32 i = 0;
    i128 vindex0, v0, vindex1, v1, o0, o1, vm0, vm1;

    vm0 = _mm_load_si128((i128 *)(phirev + i + 0));
    vm1 = _mm_load_si128((i128 *)(phirev + i + 4));

    vindex0 = _mm_load_si128((i128 *)(off + i + 0));
    v0 = _mm_i32gather_epi32(src, vindex0, 4);
    o0 = mulmod0x20008001u(v0, vm0);

    vindex1 = _mm_load_si128((i128 *)(off + i + 4));
    v1 = _mm_i32gather_epi32(src, vindex1, 4);

    o1 = mulmod0x20008001u(v1, vm1);

    for (; i < 1024 - 8; i += 8) {
        vindex0 = _mm_load_si128((i128 *)(off + i + 0 + 8));
        _mm_store_si128((i128 *)(dst + i + 0), o0);
        v0 = _mm_i32gather_epi32(src, vindex0, 4);
        vm0 = _mm_load_si128((i128 *)(phirev + i + 0 + 8));
        o0 = mulmod0x20008001u(v0, vm0);

        vindex1 = _mm_load_si128((i128 *)(off + i + 4 + 8));
        _mm_store_si128((i128 *)(dst + i + 4), o1);
        v1 = _mm_i32gather_epi32(src, vindex1, 4);
        vm1 = _mm_load_si128((i128 *)(phirev + i + 4 + 8));
        o1 = mulmod0x20008001u(v1, vm1);
    }
    _mm_store_si128((i128 *)(dst + i + 0), o0);
    _mm_store_si128((i128 *)(dst + i + 4), o1);
}

INTER void
nttdifstage0(i32 *dst) {
    i32 i = 0;
    i128 i0, i1, i2, i3, o0, o1, mask1, q = _mm_set1_epi32(0x20008001);
    i128 maxf = _mm_set1_epi32(0x20008000);
    i128 mask0;
    i128 minf = _mm_set1_epi32(0);

    i0 = _mm_load_si128((i128 *)(dst + i + 0));
    i1 = _mm_load_si128((i128 *)(dst + i + 4));
    o0 = _mm_hadd_epi32(i0, i1);
    i2 = _mm_load_si128((i128 *)(dst + i + 8));
    i3 = _mm_load_si128((i128 *)(dst + i + 12));
    o1 = _mm_hsub_epi32(i0, i1);

    mask0 = _mm_cmpgt_epi32(o0, maxf);
    mask1 = _mm_cmpgt_epi32(minf, o1);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    o0 = _mm_sub_epi32(o0, mask0);
    o1 = _mm_add_epi32(o1, mask1);
    i0 = _mm_unpacklo_epi32(o0, o1);
    i1 = _mm_unpackhi_epi32(o0, o1);

    for (; i < 1024 - 16; i += 16) {
        _mm_store_si128((i128 *)(dst + i + 0), i0);
        o0 = _mm_hadd_epi32(i2, i3);
        i0 = _mm_load_si128((i128 *)(dst + i + 16));
        _mm_store_si128((i128 *)(dst + i + 4), i1);
        o1 = _mm_hsub_epi32(i2, i3);
        i1 = _mm_load_si128((i128 *)(dst + i + 16 + 4));

        mask0 = _mm_cmpgt_epi32(o0, maxf);
        mask1 = _mm_cmpgt_epi32(minf, o1);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        o0 = _mm_sub_epi32(o0, mask0);
        o1 = _mm_add_epi32(o1, mask1);
        i2 = _mm_unpacklo_epi32(o0, o1);
        i3 = _mm_unpackhi_epi32(o0, o1);

        _mm_store_si128((i128 *)(dst + i + 8), i2);
        o0 = _mm_hadd_epi32(i0, i1);
        i2 = _mm_load_si128((i128 *)(dst + i + 16 + 8));

        _mm_store_si128((i128 *)(dst + i + 12), i3);
        o1 = _mm_hsub_epi32(i0, i1);
        i3 = _mm_load_si128((i128 *)(dst + i + 16 + 12));

        mask0 = _mm_cmpgt_epi32(o0, maxf);
        mask1 = _mm_cmpgt_epi32(minf, o1);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        o0 = _mm_sub_epi32(o0, mask0);
        o1 = _mm_add_epi32(o1, mask1);
        i0 = _mm_unpacklo_epi32(o0, o1);
        i1 = _mm_unpackhi_epi32(o0, o1);
    }
    _mm_store_si128((i128 *)(dst + i + 0), i0);
    o0 = _mm_hadd_epi32(i2, i3);

    _mm_store_si128((i128 *)(dst + i + 4), i1);
    o1 = _mm_hsub_epi32(i2, i3);

    mask0 = _mm_cmpgt_epi32(o0, maxf);
    mask1 = _mm_cmpgt_epi32(minf, o1);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    o0 = _mm_sub_epi32(o0, mask0);
    o1 = _mm_add_epi32(o1, mask1);
    i2 = _mm_unpacklo_epi32(o0, o1);
    i3 = _mm_unpackhi_epi32(o0, o1);

    _mm_store_si128((i128 *)(dst + i + 8), i2);
    _mm_store_si128((i128 *)(dst + i + 12), i3);
}

INTER void
nttdifstage1(i32 *dst, i32 w256) {
    i32 i = 0;
    i128 i0, i1, i2, i3, o0, o1, mask1, m = _mm_set_epi32(w256, 1, w256, 1),
                                        // intel likes his args little fermat
            q = _mm_set1_epi32(0x20008001), minf = _mm_set1_epi32(0);
    i128 maxf = _mm_set1_epi32(0x20008000);
    i128 mask0;

    i0 = _mm_load_si128((i128 *)(dst + i + 0));
    i1 = _mm_load_si128((i128 *)(dst + i + 4));

    o0 = _mm_unpacklo_epi64(i0, i1);
    o1 = _mm_unpackhi_epi64(i0, i1);

    i2 = _mm_load_si128((i128 *)(dst + i + 8));
    i3 = _mm_load_si128((i128 *)(dst + i + 12));

    i0 = _mm_add_epi32(o0, o1);
    i1 = _mm_sub_epi32(o0, o1);
    mask1 = _mm_cmpgt_epi32(minf, i1);
    mask0 = _mm_cmpgt_epi32(i0, maxf);
    mask1 = _mm_and_si128(mask1, q);
    mask0 = _mm_and_si128(mask0, q);
    i1 = _mm_add_epi32(i1, mask1);
    i0 = _mm_sub_epi32(i0, mask0);

    i1 = mulmod0x20008001u(i1, m);

    o0 = _mm_unpacklo_epi64(i0, i1);
    o1 = _mm_unpackhi_epi64(i0, i1);

    for (; i < 1024 - 16; i += 16) {
        _mm_store_si128((i128 *)(dst + i + 0), o0);
        i0 = _mm_load_si128((i128 *)(dst + i + 16));
        o0 = _mm_unpacklo_epi64(i2, i3);

        _mm_store_si128((i128 *)(dst + i + 4), o1);
        i1 = _mm_load_si128((i128 *)(dst + i + 16 + 4));
        o1 = _mm_unpackhi_epi64(i2, i3);

        i2 = _mm_add_epi32(o0, o1);
        i3 = _mm_sub_epi32(o0, o1);
        mask1 = _mm_cmpgt_epi32(minf, i3);
        mask0 = _mm_cmpgt_epi32(i2, maxf);
        mask1 = _mm_and_si128(mask1, q);
        mask0 = _mm_and_si128(mask0, q);
        i3 = _mm_add_epi32(i3, mask1);
        i2 = _mm_sub_epi32(i2, mask0);
        i3 = mulmod0x20008001u(i3, m);
        o0 = _mm_unpacklo_epi64(i2, i3);
        o1 = _mm_unpackhi_epi64(i2, i3);

        _mm_store_si128((i128 *)(dst + i + 8), o0);
        i2 = _mm_load_si128((i128 *)(dst + i + 16 + 8));
        o0 = _mm_unpacklo_epi64(i0, i1);

        _mm_store_si128((i128 *)(dst + i + 12), o1);
        i3 = _mm_load_si128((i128 *)(dst + i + 16 + 12));
        o1 = _mm_unpackhi_epi64(i0, i1);

        i0 = _mm_add_epi32(o0, o1);
        i1 = _mm_sub_epi32(o0, o1);
        mask1 = _mm_cmpgt_epi32(minf, i1);
        mask0 = _mm_cmpgt_epi32(i0, maxf);
        mask1 = _mm_and_si128(mask1, q);
        mask0 = _mm_and_si128(mask0, q);
        i1 = _mm_add_epi32(i1, mask1);
        i0 = _mm_sub_epi32(i0, mask0);
        i1 = mulmod0x20008001u(i1, m);
        o0 = _mm_unpacklo_epi64(i0, i1);
        o1 = _mm_unpackhi_epi64(i0, i1);
    }
    _mm_store_si128((i128 *)(dst + i + 0), o0);
    o0 = _mm_unpacklo_epi64(i2, i3);

    _mm_store_si128((i128 *)(dst + i + 4), o1);
    o1 = _mm_unpackhi_epi64(i2, i3);

    i2 = _mm_add_epi32(o0, o1);
    i3 = _mm_sub_epi32(o0, o1);
    mask1 = _mm_cmpgt_epi32(minf, i3);
    mask0 = _mm_cmpgt_epi32(i2, maxf);
    mask1 = _mm_and_si128(mask1, q);
    mask0 = _mm_and_si128(mask0, q);
    i3 = _mm_add_epi32(i3, mask1);
    i2 = _mm_sub_epi32(i2, mask0);
    i3 = mulmod0x20008001u(i3, m);

    o0 = _mm_unpacklo_epi64(i2, i3);
    o1 = _mm_unpackhi_epi64(i2, i3);

    _mm_store_si128((i128 *)(dst + i + 8), o0);
    _mm_store_si128((i128 *)(dst + i + 12), o1);
}

INTER void
nttdifstage(i32 *restrict dst, i32 const *restrict wvec, i32 im, i32 jm) {
    i32 f, s, i, j;
    const i32 *vw = wvec;
    i128 m, i0, i1, o0, o1, mask1, q = _mm_set1_epi32(0x20008001),
                                   minf = _mm_set1_epi32(0);
    i128 maxf = _mm_set1_epi32(0x20008000);
    i128 mask0;

    for (i = 0; i < im; ++i) {
        for (j = 0; j < jm; ++j) {
            f = 4 * ((2 * i + 0) * jm + j);
            s = 4 * ((2 * i + 1) * jm + j);
            m = _mm_load_si128((i128 *)(vw + 4 * j));
            o0 = _mm_load_si128((i128 *)(dst + f));
            o1 = _mm_load_si128((i128 *)(dst + s));

            i0 = _mm_add_epi32(o0, o1);
            i1 = _mm_sub_epi32(o0, o1);

            mask1 = _mm_cmpgt_epi32(minf, i1);
            mask0 = _mm_cmpgt_epi32(i0, maxf);
            mask1 = _mm_and_si128(mask1, q);
            mask0 = _mm_and_si128(mask0, q);
            i1 = _mm_add_epi32(i1, mask1);
            i0 = _mm_sub_epi32(i0, mask0);
            i1 = mulmod0x20008001u(i1, m);

            _mm_store_si128((i128 *)(dst + f), i0);
            _mm_store_si128((i128 *)(dst + s), i1);
        }
    }
}

INTER void
bitrevnttditstage0(i32 *dst, const i32 *src) {
    i32 i = 0;
    i128 i0, i1, i2, i3, o0, o1, mask1, q = _mm_set1_epi32(0x20008001);
    i128 maxf = _mm_set1_epi32(0x20008000);
    i128 mask0;
    i128 minf = _mm_set1_epi32(0);

    if (src != dst) {
        memcpy(dst, src, 1024 * sizeof(i32));
    }

    i0 = _mm_load_si128((i128 *)(dst + i + 0));
    i1 = _mm_load_si128((i128 *)(dst + i + 4));
    o0 = _mm_hadd_epi32(i0, i1);
    i2 = _mm_load_si128((i128 *)(dst + i + 8));
    i3 = _mm_load_si128((i128 *)(dst + i + 12));
    o1 = _mm_hsub_epi32(i0, i1);

    mask0 = _mm_cmpgt_epi32(o0, maxf);
    mask1 = _mm_cmpgt_epi32(minf, o1);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    o0 = _mm_sub_epi32(o0, mask0);
    o1 = _mm_add_epi32(o1, mask1);
    i0 = _mm_unpacklo_epi32(o0, o1);
    i1 = _mm_unpackhi_epi32(o0, o1);

    for (; i < 1024 - 16; i += 16) {
        _mm_store_si128((i128 *)(dst + i + 0), i0);
        o0 = _mm_hadd_epi32(i2, i3);
        i0 = _mm_load_si128((i128 *)(dst + i + 16));
        _mm_store_si128((i128 *)(dst + i + 4), i1);
        o1 = _mm_hsub_epi32(i2, i3);
        i1 = _mm_load_si128((i128 *)(dst + i + 16 + 4));

        mask0 = _mm_cmpgt_epi32(o0, maxf);
        mask1 = _mm_cmpgt_epi32(minf, o1);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        o0 = _mm_sub_epi32(o0, mask0);
        o1 = _mm_add_epi32(o1, mask1);
        i2 = _mm_unpacklo_epi32(o0, o1);
        i3 = _mm_unpackhi_epi32(o0, o1);

        _mm_store_si128((i128 *)(dst + i + 8), i2);
        o0 = _mm_hadd_epi32(i0, i1);
        i2 = _mm_load_si128((i128 *)(dst + i + 16 + 8));

        _mm_store_si128((i128 *)(dst + i + 12), i3);
        o1 = _mm_hsub_epi32(i0, i1);
        i3 = _mm_load_si128((i128 *)(dst + i + 16 + 12));

        mask0 = _mm_cmpgt_epi32(o0, maxf);
        mask1 = _mm_cmpgt_epi32(minf, o1);
        mask0 = _mm_and_si128(mask0, q);
        mask1 = _mm_and_si128(mask1, q);
        o0 = _mm_sub_epi32(o0, mask0);
        o1 = _mm_add_epi32(o1, mask1);
        i0 = _mm_unpacklo_epi32(o0, o1);
        i1 = _mm_unpackhi_epi32(o0, o1);
    }
    _mm_store_si128((i128 *)(dst + i + 0), i0);
    o0 = _mm_hadd_epi32(i2, i3);

    _mm_store_si128((i128 *)(dst + i + 4), i1);
    o1 = _mm_hsub_epi32(i2, i3);

    mask0 = _mm_cmpgt_epi32(o0, maxf);
    mask1 = _mm_cmpgt_epi32(minf, o1);
    mask0 = _mm_and_si128(mask0, q);
    mask1 = _mm_and_si128(mask1, q);
    o0 = _mm_sub_epi32(o0, mask0);
    o1 = _mm_add_epi32(o1, mask1);
    i2 = _mm_unpacklo_epi32(o0, o1);
    i3 = _mm_unpackhi_epi32(o0, o1);

    _mm_store_si128((i128 *)(dst + i + 8), i2);
    _mm_store_si128((i128 *)(dst + i + 12), i3);
}

