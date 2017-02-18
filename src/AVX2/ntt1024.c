#include <immintrin.h>
#include <string.h>

#include <const.h>
#include <ntt1024.h>

typedef int i32;
typedef __m256i i256;

static inline i256 mulmod0x20008001u(i256, i256);

DECL void
ntt32x1024(i32 *restrict dst, const i32 *restrict src) {
    nttditstage0(dst, src);
    nttditstage1(dst, 0x20000000);
    nttditstage2(dst, 0xb754980, 0x20000000, 0x6b56455);
    nttditstagen(dst, wvec256);
}

DECL void
ntt32x1024mulphi(i32 *restrict dst, const i32 *restrict src) {
    nttditstage0mulphi(dst, src);
    nttditstage1(dst, 0x20000000);
    nttditstage2(dst, 0xb754980, 0x20000000, 0x6b56455);
    nttditstagen(dst, wvec256);
}

DECL void
intt32x1024(i32 *restrict dst, const i32 *restrict src) {
    nttditstage0(dst, src);
    nttditstage1(dst, 0x8001);
    nttditstage2(dst, 0x194b1bac, 0x8001, 0x148b3681);
    nttditstagen(dst, iwvec256);
    mulconst(dst, dst, 0x1ff87fe1);
}

DECL void
intt32x1024muliphi(i32 *restrict dst, const i32 *restrict src) {
    nttditstage0(dst, src);
    nttditstage1(dst, 0x8001);
    nttditstage2(dst, 0x194b1bac, 0x8001, 0x148b3681);
    nttditstagen(dst, iwvec256);
    mulvec(dst, dst, inphi);
}

DECL void
nttdif32x1024mulphibitrev(i32 * dst, const i32 * src) {
    mulvec(dst, src, phi);
    nttdifstagen(dst, difwvec256);
    nttdifstage2(dst, 0xb754980, 0x20000000, 0x6b56455);
    nttdifstage1(dst, 0x20000000);
    nttdifstage0(dst);
}

DECL void
bitrevinttdit32x1024muliphi(i32 * dst, const i32 * src) {
    bitrevnttditstage0(dst, src);
    nttditstage1(dst, 0x8001);
    nttditstage2(dst, 0x194b1bac, 0x8001, 0x148b3681);
    nttditstagen(dst, iwvec256);
    mulvec(dst, dst, inphi);
}

static inline i256
mulmod0x20008001u(i256 v, i256 mv) {
    // v, mv must be positive
    i256 hn1m, hn1lo, hn1hi, hp2m, hp2lo, hp2hi, hn2, hi, hp1, lo, lp1, ln1m,
            ln1lo, ln1hi, lp2m, lp2lo, lp2hi, ln2, p1, n1hi, p2hi, n1lo, p2lo,
            n2, gq, mq, mvhi, mvlo;

    hi = _mm256_srli_epi64(v, 32);
    mvhi = _mm256_srli_epi64(mv, 32);
    lo = _mm256_and_si256(v, _mm256_set1_epi64x(0xffffffff));
    mvlo = _mm256_and_si256(mv, _mm256_set1_epi64x(0xffffffff));

    hi = _mm256_mul_epu32(hi, mvhi);
    hp1 = _mm256_and_si256(hi, _mm256_set1_epi64x(0x1fffffff));
    hp1 = _mm256_slli_epi64(hp1, 32);
    hn1m = _mm256_and_si256(hi, _mm256_set1_epi64x(0x7ffe0000000));
    hn1lo = _mm256_slli_epi64(hn1m, 3);
    hn1hi = _mm256_slli_epi64(hn1m, 18);
    hp2m = _mm256_and_si256(hi, _mm256_set1_epi64x(0x3fff80000000000));
    hp2lo = _mm256_srli_epi64(hp2m, 11);
    hp2hi = _mm256_slli_epi64(hp2m, 3);
    hn2 = _mm256_srli_epi64(hi, 26);

    lo = _mm256_mul_epu32(lo, mvlo);
    lp1 = _mm256_and_si256(lo, _mm256_set1_epi64x(0x1fffffff));
    ln1m = _mm256_and_si256(lo, _mm256_set1_epi64x(0x7ffe0000000));
    ln1lo = _mm256_srli_epi64(ln1m, 29);
    ln1hi = _mm256_srli_epi64(ln1m, 14);
    lp2m = _mm256_and_si256(lo, _mm256_set1_epi64x(0x3fff80000000000));
    lp2lo = _mm256_srli_epi64(lp2m, 43);
    lp2hi = _mm256_srli_epi64(lp2m, 29);
    ln2 = _mm256_srli_epi64(lo, 58);

    p1 = _mm256_blend_epi32(lp1, hp1, 170);
    n1hi = _mm256_blend_epi32(ln1hi, hn1hi, 170);
    p2hi = _mm256_blend_epi32(lp2hi, hp2hi, 170);
    n1lo = _mm256_blend_epi32(ln1lo, hn1lo, 170);
    p2lo = _mm256_blend_epi32(lp2lo, hp2lo, 170);
    n2 = _mm256_blend_epi32(ln2, hn2, 170);

    p1 = _mm256_sub_epi32(p1, n1hi);
    p1 = _mm256_add_epi32(p1, p2hi);
    p1 = _mm256_sub_epi32(p1, n1lo);
    p1 = _mm256_add_epi32(p1, p2lo);
    p1 = _mm256_sub_epi32(p1, n2);

    // p1:[01000000000000000011111111111110, -00011111111111111100000000111110]

    gq = _mm256_cmpgt_epi32(p1, _mm256_set1_epi32(0x20008000));
    mq = _mm256_and_si256(gq, _mm256_set1_epi32(0x20008001));
    p1 = _mm256_sub_epi32(p1, mq);

    gq = _mm256_cmpgt_epi32(_mm256_set1_epi32(0), p1);
    mq = _mm256_and_si256(gq, _mm256_set1_epi32(0x20008001));
    p1 = _mm256_add_epi32(p1, mq);
    return p1;
}

INTER void
nttditstage0(i32 *restrict dst, const i32 *restrict src) {
    i32 i = 0;
    i256 i0, i1, i2, i3, o0, o1, mask1, q = _mm256_set1_epi32(0x20008001);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;
    i256 minf = _mm256_set1_epi32(0);

    bitrev32x1024(dst, src);

    i0 = _mm256_load_si256((i256 *)(dst + i + 0));
    i1 = _mm256_load_si256((i256 *)(dst + i + 8));
    o0 = _mm256_hadd_epi32(i0, i1);
    i2 = _mm256_load_si256((i256 *)(dst + i + 16));
    i3 = _mm256_load_si256((i256 *)(dst + i + 24));
    o1 = _mm256_hsub_epi32(i0, i1);

    mask0 = _mm256_cmpgt_epi32(o0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, o1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    o0 = _mm256_sub_epi32(o0, mask0);
    o1 = _mm256_add_epi32(o1, mask1);
    i0 = _mm256_unpacklo_epi32(o0, o1);
    i1 = _mm256_unpackhi_epi32(o0, o1);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), i0);
        o0 = _mm256_hadd_epi32(i2, i3);
        i0 = _mm256_load_si256((i256 *)(dst + i + 32));
        _mm256_store_si256((i256 *)(dst + i + 8), i1);
        o1 = _mm256_hsub_epi32(i2, i3);
        i1 = _mm256_load_si256((i256 *)(dst + i + 32 + 8));

        mask0 = _mm256_cmpgt_epi32(o0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, o1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        o0 = _mm256_sub_epi32(o0, mask0);
        o1 = _mm256_add_epi32(o1, mask1);
        i2 = _mm256_unpacklo_epi32(o0, o1);
        i3 = _mm256_unpackhi_epi32(o0, o1);

        _mm256_store_si256((i256 *)(dst + i + 16), i2);
        o0 = _mm256_hadd_epi32(i0, i1);
        i2 = _mm256_load_si256((i256 *)(dst + i + 32 + 16));

        _mm256_store_si256((i256 *)(dst + i + 24), i3);
        o1 = _mm256_hsub_epi32(i0, i1);
        i3 = _mm256_load_si256((i256 *)(dst + i + 32 + 24));

        mask0 = _mm256_cmpgt_epi32(o0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, o1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        o0 = _mm256_sub_epi32(o0, mask0);
        o1 = _mm256_add_epi32(o1, mask1);
        i0 = _mm256_unpacklo_epi32(o0, o1);
        i1 = _mm256_unpackhi_epi32(o0, o1);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), i0);
    o0 = _mm256_hadd_epi32(i2, i3);

    _mm256_store_si256((i256 *)(dst + i + 8), i1);
    o1 = _mm256_hsub_epi32(i2, i3);

    mask0 = _mm256_cmpgt_epi32(o0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, o1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    o0 = _mm256_sub_epi32(o0, mask0);
    o1 = _mm256_add_epi32(o1, mask1);
    i2 = _mm256_unpacklo_epi32(o0, o1);
    i3 = _mm256_unpackhi_epi32(o0, o1);

    _mm256_store_si256((i256 *)(dst + i + 16), i2);
    _mm256_store_si256((i256 *)(dst + i + 24), i3);
}

INTER void
nttditstage0mulphi(i32 *restrict dst, const i32 *restrict src) {
    bitrev32x1024mulphi(dst, src);
    bitrevnttditstage0(dst, dst);
}

INTER void
nttditstage1(i32 *dst, i32 w256) {
    i32 i = 0;
    i256 i0, i1, i2, i3, o0, o1, mask1,
            m = _mm256_set_epi32(w256, 1, w256, 1, w256, 1, w256, 1),
            // intel likes his args little fermat
            q = _mm256_set1_epi32(0x20008001), minf = _mm256_set1_epi32(0);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;

    i0 = _mm256_load_si256((i256 *)(dst + i + 0));
    i1 = _mm256_load_si256((i256 *)(dst + i + 8));

    o0 = _mm256_unpacklo_epi64(i0, i1);
    o1 = _mm256_unpackhi_epi64(i0, i1);
    o1 = mulmod0x20008001u(o1, m);

    i2 = _mm256_load_si256((i256 *)(dst + i + 16));
    i3 = _mm256_load_si256((i256 *)(dst + i + 24));

    i0 = _mm256_add_epi32(o0, o1);
    i1 = _mm256_sub_epi32(o0, o1);
    mask0 = _mm256_cmpgt_epi32(i0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, i1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    i0 = _mm256_sub_epi32(i0, mask0);
    i1 = _mm256_add_epi32(i1, mask1);
    o0 = _mm256_unpacklo_epi64(i0, i1);
    o1 = _mm256_unpackhi_epi64(i0, i1);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), o0);
        i0 = _mm256_load_si256((i256 *)(dst + i + 32));
        o0 = _mm256_unpacklo_epi64(i2, i3);

        _mm256_store_si256((i256 *)(dst + i + 8), o1);
        i1 = _mm256_load_si256((i256 *)(dst + i + 32 + 8));
        o1 = _mm256_unpackhi_epi64(i2, i3);
        o1 = mulmod0x20008001u(o1, m);

        i2 = _mm256_add_epi32(o0, o1);
        i3 = _mm256_sub_epi32(o0, o1);
        mask0 = _mm256_cmpgt_epi32(i2, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, i3);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        i2 = _mm256_sub_epi32(i2, mask0);
        i3 = _mm256_add_epi32(i3, mask1);
        o0 = _mm256_unpacklo_epi64(i2, i3);
        o1 = _mm256_unpackhi_epi64(i2, i3);

        _mm256_store_si256((i256 *)(dst + i + 16), o0);
        i2 = _mm256_load_si256((i256 *)(dst + i + 32 + 16));
        o0 = _mm256_unpacklo_epi64(i0, i1);

        _mm256_store_si256((i256 *)(dst + i + 24), o1);
        i3 = _mm256_load_si256((i256 *)(dst + i + 32 + 24));
        o1 = _mm256_unpackhi_epi64(i0, i1);
        o1 = mulmod0x20008001u(o1, m);

        i0 = _mm256_add_epi32(o0, o1);
        i1 = _mm256_sub_epi32(o0, o1);
        mask0 = _mm256_cmpgt_epi32(i0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, i1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        i0 = _mm256_sub_epi32(i0, mask0);
        i1 = _mm256_add_epi32(i1, mask1);
        o0 = _mm256_unpacklo_epi64(i0, i1);
        o1 = _mm256_unpackhi_epi64(i0, i1);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    o0 = _mm256_unpacklo_epi64(i2, i3);

    _mm256_store_si256((i256 *)(dst + i + 8), o1);
    o1 = _mm256_unpackhi_epi64(i2, i3);
    o1 = mulmod0x20008001u(o1, m);

    i2 = _mm256_add_epi32(o0, o1);
    i3 = _mm256_sub_epi32(o0, o1);
    mask0 = _mm256_cmpgt_epi32(i2, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, i3);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    i2 = _mm256_sub_epi32(i2, mask0);
    i3 = _mm256_add_epi32(i3, mask1);
    o0 = _mm256_unpacklo_epi64(i2, i3);
    o1 = _mm256_unpackhi_epi64(i2, i3);

    _mm256_store_si256((i256 *)(dst + i + 16), o0);
    _mm256_store_si256((i256 *)(dst + i + 24), o1);
}

INTER void
nttditstage2(i32 *dst, i32 w128, i32 w256, i32 w384) {
    i32 i = 0;
    i256 i0, i1, i2, i3, o0, o1, mask1,
            m = _mm256_set_epi32(w384, w256, w128, 1, w384, w256, w128, 1),
            q = _mm256_set1_epi32(0x20008001), minf = _mm256_set1_epi32(0);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;

    i0 = _mm256_load_si256((i256 *)(dst + i + 0));
    i1 = _mm256_load_si256((i256 *)(dst + i + 8));

    o0 = _mm256_permute2f128_si256(i0, i1, 32);
    o1 = _mm256_permute2f128_si256(i0, i1, 49);
    o1 = mulmod0x20008001u(o1, m);

    i2 = _mm256_load_si256((i256 *)(dst + i + 16));
    i3 = _mm256_load_si256((i256 *)(dst + i + 24));

    i0 = _mm256_add_epi32(o0, o1);
    i1 = _mm256_sub_epi32(o0, o1);
    mask0 = _mm256_cmpgt_epi32(i0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, i1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    i0 = _mm256_sub_epi32(i0, mask0);
    i1 = _mm256_add_epi32(i1, mask1);
    o0 = _mm256_permute2f128_si256(i0, i1, 32);
    o1 = _mm256_permute2f128_si256(i0, i1, 49);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), o0);
        i0 = _mm256_load_si256((i256 *)(dst + i + 32));
        o0 = _mm256_unpacklo_epi64(i2, i3);
        o0 = _mm256_permute2f128_si256(i2, i3, 32);

        _mm256_store_si256((i256 *)(dst + i + 8), o1);
        i1 = _mm256_load_si256((i256 *)(dst + i + 32 + 8));
        o1 = _mm256_permute2f128_si256(i2, i3, 49);
        o1 = mulmod0x20008001u(o1, m);

        i2 = _mm256_add_epi32(o0, o1);
        i3 = _mm256_sub_epi32(o0, o1);
        mask0 = _mm256_cmpgt_epi32(i2, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, i3);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        i2 = _mm256_sub_epi32(i2, mask0);
        i3 = _mm256_add_epi32(i3, mask1);
        o0 = _mm256_permute2f128_si256(i2, i3, 32);
        o1 = _mm256_permute2f128_si256(i2, i3, 49);

        _mm256_store_si256((i256 *)(dst + i + 16), o0);
        i2 = _mm256_load_si256((i256 *)(dst + i + 32 + 16));
        o0 = _mm256_permute2f128_si256(i0, i1, 32);

        _mm256_store_si256((i256 *)(dst + i + 24), o1);
        i3 = _mm256_load_si256((i256 *)(dst + i + 32 + 24));
        o1 = _mm256_permute2f128_si256(i0, i1, 49);
        o1 = mulmod0x20008001u(o1, m);

        i0 = _mm256_add_epi32(o0, o1);
        i1 = _mm256_sub_epi32(o0, o1);
        mask0 = _mm256_cmpgt_epi32(i0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, i1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        i0 = _mm256_sub_epi32(i0, mask0);
        i1 = _mm256_add_epi32(i1, mask1);
        o0 = _mm256_permute2f128_si256(i0, i1, 32);
        o1 = _mm256_permute2f128_si256(i0, i1, 49);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    o0 = _mm256_permute2f128_si256(i2, i3, 32);

    _mm256_store_si256((i256 *)(dst + i + 8), o1);
    o1 = _mm256_permute2f128_si256(i2, i3, 49);
    o1 = mulmod0x20008001u(o1, m);

    i2 = _mm256_add_epi32(o0, o1);
    i3 = _mm256_sub_epi32(o0, o1);
    mask0 = _mm256_cmpgt_epi32(i2, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, i3);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    i2 = _mm256_sub_epi32(i2, mask0);
    i3 = _mm256_add_epi32(i3, mask1);
    o0 = _mm256_permute2f128_si256(i2, i3, 32);
    o1 = _mm256_permute2f128_si256(i2, i3, 49);

    _mm256_store_si256((i256 *)(dst + i + 16), o0);
    _mm256_store_si256((i256 *)(dst + i + 24), o1);
}

INTER void
nttditstagen(i32 *restrict dst, const i32 *restrict wvec) {
    nttditstage(dst, wvec + 0x000, 0x40, 0x01);
    nttditstage(dst, wvec + 0x008, 0x20, 0x02);
    nttditstage(dst, wvec + 0x018, 0x10, 0x04);
    nttditstage(dst, wvec + 0x038, 0x08, 0x08);
    nttditstage(dst, wvec + 0x078, 0x04, 0x10);
    nttditstage(dst, wvec + 0x0f8, 0x02, 0x20);
    nttditstage(dst, wvec + 0x1f8, 0x01, 0x40);
}

INTER void
nttdifstagen(i32 *restrict dst, const i32 *restrict wvec) {
    nttdifstage(dst, wvec + 0x000, 0x01, 0x40);
    nttdifstage(dst, wvec + 0x200, 0x02, 0x20);
    nttdifstage(dst, wvec + 0x300, 0x04, 0x10);
    nttdifstage(dst, wvec + 0x380, 0x08, 0x08);
    nttdifstage(dst, wvec + 0x3c0, 0x10, 0x04);
    nttdifstage(dst, wvec + 0x3e0, 0x20, 0x02);
    nttdifstage(dst, wvec + 0x3f0, 0x40, 0x01);
}

INTER void
nttditstage(i32 *restrict dst, i32 const *restrict wvec, i32 im, i32 jm) {
    i32 f, s, i, j;
    const i32 *vw = wvec;
    i256 m, i0, i1, o0, o1, mask1, q = _mm256_set1_epi32(0x20008001),
                                   minf = _mm256_set1_epi32(0);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;

    for (i = 0; i < im; ++i) {
        for (j = 0; j < jm; ++j) {
            f = 8 * ((2 * i + 0) * jm + j);
            s = 8 * ((2 * i + 1) * jm + j);
            m = _mm256_load_si256((i256 *)(vw + 8 * j));
            o0 = _mm256_load_si256((i256 *)(dst + f));
            o1 = _mm256_load_si256((i256 *)(dst + s));
            o1 = mulmod0x20008001u(o1, m);

            i0 = _mm256_add_epi32(o0, o1);
            i1 = _mm256_sub_epi32(o0, o1);

            mask0 = _mm256_cmpgt_epi32(i0, maxf);
            mask1 = _mm256_cmpgt_epi32(minf, i1);
            mask0 = _mm256_and_si256(mask0, q);
            mask1 = _mm256_and_si256(mask1, q);
            i0 = _mm256_sub_epi32(i0, mask0);
            i1 = _mm256_add_epi32(i1, mask1);
            _mm256_store_si256((i256 *)(dst + f), i0);
            _mm256_store_si256((i256 *)(dst + s), i1);
        }
    }
}

DECL void
mulvec(i32 *dst, const i32 *f, const i32 *s) {
    i32 i = 0;
    i256 i0, i1, i2, i3, i4, i5, o0, o1, o2;

    i0 = _mm256_load_si256((i256 *)(f + i + 0));
    i2 = _mm256_load_si256((i256 *)(f + i + 8));
    i1 = _mm256_load_si256((i256 *)(s + i + 0));
    i3 = _mm256_load_si256((i256 *)(s + i + 8));
    o0 = mulmod0x20008001u(i0, i1);
    i4 = _mm256_load_si256((i256 *)(f + i + 16));
    i5 = _mm256_load_si256((i256 *)(s + i + 16));

    for (i = 24; i < 1008; i += 24) {
        i0 = _mm256_load_si256((i256 *)(f + i + 0));
        i1 = _mm256_load_si256((i256 *)(s + i + 0));
        o1 = mulmod0x20008001u(i2, i3);
        _mm256_store_si256((i256 *)(dst + i - 24), o0);
        i2 = _mm256_load_si256((i256 *)(f + i + 8));
        i3 = _mm256_load_si256((i256 *)(s + i + 8));
        o2 = mulmod0x20008001u(i4, i5);
        _mm256_store_si256((i256 *)(dst + i - 16), o1);
        i4 = _mm256_load_si256((i256 *)(f + i + 16));
        i5 = _mm256_load_si256((i256 *)(s + i + 16));
        o0 = mulmod0x20008001u(i0, i1);
        _mm256_store_si256((i256 *)(dst + i - 8), o2);
    }
    i0 = _mm256_load_si256((i256 *)(f + i + 0));
    i1 = _mm256_load_si256((i256 *)(s + i + 0));
    o1 = mulmod0x20008001u(i2, i3);
    _mm256_store_si256((i256 *)(dst + i - 24), o0);
    i2 = _mm256_load_si256((i256 *)(f + i + 8));
    i3 = _mm256_load_si256((i256 *)(s + i + 8));
    o2 = mulmod0x20008001u(i4, i5);
    _mm256_store_si256((i256 *)(dst + i - 16), o1);
    o0 = mulmod0x20008001u(i0, i1);
    _mm256_store_si256((i256 *)(dst + i - 8), o2);
    o1 = mulmod0x20008001u(i2, i3);
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    _mm256_store_si256((i256 *)(dst + i + 8), o1);
}

INTER void
mulconst(i32 *dst, const i32 *f, i32 c) {
    i32 i = 0;
    i256 i0, i2, i4, vm, o0, o1, o2;
    i0 = _mm256_load_si256((i256 *)(f + i + 0));
    i2 = _mm256_load_si256((i256 *)(f + i + 8));
    vm = _mm256_set1_epi32(c);
    o0 = mulmod0x20008001u(i0, vm);
    i4 = _mm256_load_si256((i256 *)(f + i + 16));
    for (i = 24; i < 1008; i += 24) {
        i0 = _mm256_load_si256((i256 *)(f + i + 0));
        o1 = mulmod0x20008001u(i2, vm);
        _mm256_store_si256((i256 *)(dst + i - 24), o0);
        i2 = _mm256_load_si256((i256 *)(f + i + 8));
        o2 = mulmod0x20008001u(i4, vm);
        _mm256_store_si256((i256 *)(dst + i - 16), o1);
        i4 = _mm256_load_si256((i256 *)(f + i + 16));
        o0 = mulmod0x20008001u(i0, vm);
        _mm256_store_si256((i256 *)(dst + i - 8), o2);
    }
    i0 = _mm256_load_si256((i256 *)(f + i + 0));
    o1 = mulmod0x20008001u(i2, vm);
    _mm256_store_si256((i256 *)(dst + i - 24), o0);
    i2 = _mm256_load_si256((i256 *)(f + i + 8));
    o2 = mulmod0x20008001u(i4, vm);
    _mm256_store_si256((i256 *)(dst + i - 16), o1);
    o0 = mulmod0x20008001u(i0, vm);
    _mm256_store_si256((i256 *)(dst + i - 8), o2);
    o1 = mulmod0x20008001u(i2, vm);
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    _mm256_store_si256((i256 *)(dst + i + 8), o1);
}

INTER void
sub0x20008001(i32 *dst, const i32 *f) {
    i32 i = 0;
    i256 i0, i2, i4, o0, o1, o2, q = _mm256_set1_epi32(0x20008001),
                                 maxf = _mm256_set1_epi32(0x20008000);
#define subq(dst, src)                                                        \
    dst = _mm256_cmpgt_epi32(src, maxf);                                      \
    dst = _mm256_and_si256(dst, q);                                           \
    dst = _mm256_sub_epi32(src, dst);

    i0 = _mm256_load_si256((i256 *)(f + i + 0));
    i2 = _mm256_load_si256((i256 *)(f + i + 8));
    subq(o0, i0);

    i4 = _mm256_load_si256((i256 *)(f + i + 16));
    for (i = 24; i < 1008; i += 24) {
        i0 = _mm256_load_si256((i256 *)(f + i + 0));
        subq(o1, i2);
        _mm256_store_si256((i256 *)(dst + i - 24), o0);
        i2 = _mm256_load_si256((i256 *)(f + i + 8));
        subq(o2, i4);
        _mm256_store_si256((i256 *)(dst + i - 16), o1);
        i4 = _mm256_load_si256((i256 *)(f + i + 16));
        subq(o0, i0);
        _mm256_store_si256((i256 *)(dst + i - 8), o2);
    }
    i0 = _mm256_load_si256((i256 *)(f + i + 0));
    subq(o1, i2);
    _mm256_store_si256((i256 *)(dst + i - 24), o0);
    i2 = _mm256_load_si256((i256 *)(f + i + 8));
    subq(o2, i4);
    _mm256_store_si256((i256 *)(dst + i - 16), o1);
    subq(o0, i0);
    _mm256_store_si256((i256 *)(dst + i - 8), o2);
    subq(o1, i2);
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    _mm256_store_si256((i256 *)(dst + i + 8), o1);
#undef subq
}

DECL void
cleanregs(void) {
    __asm__ __volatile__("vpxor\t%%ymm0,\t%%ymm0,\t%%ymm0\n\t"
                         "vpxor\t%%ymm1,\t%%ymm1,\t%%ymm1\n\t"
                         "vpxor\t%%ymm2,\t%%ymm2,\t%%ymm2\n\t"
                         "vpxor\t%%ymm3,\t%%ymm3,\t%%ymm3\n\t"
                         "vpxor\t%%ymm4,\t%%ymm4,\t%%ymm4\n\t"
                         "vpxor\t%%ymm5,\t%%ymm5,\t%%ymm5\n\t"
                         "vpxor\t%%ymm6,\t%%ymm6,\t%%ymm6\n\t"
                         "vpxor\t%%ymm7,\t%%ymm7,\t%%ymm7\n\t"
                         "vpxor\t%%ymm8,\t%%ymm8,\t%%ymm8\n\t"
                         "vpxor\t%%ymm9,\t%%ymm9,\t%%ymm9\n\t"
                         "vpxor\t%%ymm10,\t%%ymm10,\t%%ymm10\n\t"
                         "vpxor\t%%ymm11,\t%%ymm11,\t%%ymm11\n\t"
                         "vpxor\t%%ymm12,\t%%ymm12,\t%%ymm12\n\t"
                         "vpxor\t%%ymm13,\t%%ymm13,\t%%ymm13\n\t"
                         "vpxor\t%%ymm14,\t%%ymm14,\t%%ymm14\n\t"
                         "vpxor\t%%ymm15,\t%%ymm15,\t%%ymm15"
                         :
                         :
                         : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                           "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",
                           "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                           "%ymm15");
}

INTER void
bitrev32x1024(i32 *restrict dst, const i32 *restrict src) {
    i32 i = 0;
    i256 vindex0, v0, vindex1, v1, vindex2, v2, vindex3, v3;

    vindex0 = _mm256_load_si256((i256 *)(off + i + 0));
    v0 = _mm256_i32gather_epi32(src, vindex0, 4);

    vindex1 = _mm256_load_si256((i256 *)(off + i + 8));
    v1 = _mm256_i32gather_epi32(src, vindex1, 4);

    vindex2 = _mm256_load_si256((i256 *)(off + i + 16));
    v2 = _mm256_i32gather_epi32(src, vindex2, 4);

    vindex3 = _mm256_load_si256((i256 *)(off + i + 24));
    v3 = _mm256_i32gather_epi32(src, vindex3, 4);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), v0);
        vindex0 = _mm256_load_si256((i256 *)(off + i + 0 + 32));
        v0 = _mm256_i32gather_epi32(src, vindex0, 4);

        _mm256_store_si256((i256 *)(dst + i + 8), v1);
        vindex1 = _mm256_load_si256((i256 *)(off + i + 8 + 32));
        v1 = _mm256_i32gather_epi32(src, vindex1, 4);

        _mm256_store_si256((i256 *)(dst + i + 16), v2);
        vindex2 = _mm256_load_si256((i256 *)(off + i + 16 + 32));
        v2 = _mm256_i32gather_epi32(src, vindex2, 4);

        _mm256_store_si256((i256 *)(dst + i + 24), v3);
        vindex3 = _mm256_load_si256((i256 *)(off + i + 24 + 32));
        v3 = _mm256_i32gather_epi32(src, vindex3, 4);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), v0);
    _mm256_store_si256((i256 *)(dst + i + 8), v1);
    _mm256_store_si256((i256 *)(dst + i + 16), v2);
    _mm256_store_si256((i256 *)(dst + i + 24), v3);
}

INTER void
bitrev32x1024mulphi(i32 *restrict dst, const i32 *restrict src) {
    i32 i = 0;
    i256 vindex0, v0, vindex1, v1, o0, o1, vm0, vm1;

    vm0 = _mm256_load_si256((i256 *)(phirev + i + 0));
    vm1 = _mm256_load_si256((i256 *)(phirev + i + 8));
    vindex0 = _mm256_load_si256((i256 *)(off + i + 0));
    v0 = _mm256_i32gather_epi32(src, vindex0, 4);
    o0 = mulmod0x20008001u(v0, vm0);

    vindex1 = _mm256_load_si256((i256 *)(off + i + 8));
    v1 = _mm256_i32gather_epi32(src, vindex1, 4);
    o1 = mulmod0x20008001u(v1, vm1);

    for (; i < 1024 - 16; i += 16) {
        vindex0 = _mm256_load_si256((i256 *)(off + i + 0 + 16));
        v0 = _mm256_i32gather_epi32(src, vindex0, 4);
        _mm256_store_si256((i256 *)(dst + i + 0), o0);
        vm0 = _mm256_load_si256((i256 *)(phirev + i + 0 + 16));
        o0 = mulmod0x20008001u(v0, vm0);

        vindex1 = _mm256_load_si256((i256 *)(off + i + 8 + 16));
        v1 = _mm256_i32gather_epi32(src, vindex1, 4);
        _mm256_store_si256((i256 *)(dst + i + 8), o1);
        vm1 = _mm256_load_si256((i256 *)(phirev + i + 8 + 16));
        o1 = mulmod0x20008001u(v1, vm1);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    _mm256_store_si256((i256 *)(dst + i + 8), o1);
}

INTER void
nttdifstage0(i32 *restrict dst) {
    i32 i = 0;
    i256 i0, i1, i2, i3, o0, o1, mask1, q = _mm256_set1_epi32(0x20008001);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;
    i256 minf = _mm256_set1_epi32(0);

    i0 = _mm256_load_si256((i256 *)(dst + i + 0));
    i1 = _mm256_load_si256((i256 *)(dst + i + 8));
    o0 = _mm256_hadd_epi32(i0, i1);
    i2 = _mm256_load_si256((i256 *)(dst + i + 16));
    i3 = _mm256_load_si256((i256 *)(dst + i + 24));
    o1 = _mm256_hsub_epi32(i0, i1);

    mask0 = _mm256_cmpgt_epi32(o0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, o1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    o0 = _mm256_sub_epi32(o0, mask0);
    o1 = _mm256_add_epi32(o1, mask1);
    i0 = _mm256_unpacklo_epi32(o0, o1);
    i1 = _mm256_unpackhi_epi32(o0, o1);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), i0);
        o0 = _mm256_hadd_epi32(i2, i3);
        i0 = _mm256_load_si256((i256 *)(dst + i + 32));
        _mm256_store_si256((i256 *)(dst + i + 8), i1);
        o1 = _mm256_hsub_epi32(i2, i3);
        i1 = _mm256_load_si256((i256 *)(dst + i + 32 + 8));

        mask0 = _mm256_cmpgt_epi32(o0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, o1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        o0 = _mm256_sub_epi32(o0, mask0);
        o1 = _mm256_add_epi32(o1, mask1);
        i2 = _mm256_unpacklo_epi32(o0, o1);
        i3 = _mm256_unpackhi_epi32(o0, o1);

        _mm256_store_si256((i256 *)(dst + i + 16), i2);
        o0 = _mm256_hadd_epi32(i0, i1);
        i2 = _mm256_load_si256((i256 *)(dst + i + 32 + 16));

        _mm256_store_si256((i256 *)(dst + i + 24), i3);
        o1 = _mm256_hsub_epi32(i0, i1);
        i3 = _mm256_load_si256((i256 *)(dst + i + 32 + 24));

        mask0 = _mm256_cmpgt_epi32(o0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, o1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        o0 = _mm256_sub_epi32(o0, mask0);
        o1 = _mm256_add_epi32(o1, mask1);
        i0 = _mm256_unpacklo_epi32(o0, o1);
        i1 = _mm256_unpackhi_epi32(o0, o1);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), i0);
    o0 = _mm256_hadd_epi32(i2, i3);

    _mm256_store_si256((i256 *)(dst + i + 8), i1);
    o1 = _mm256_hsub_epi32(i2, i3);

    mask0 = _mm256_cmpgt_epi32(o0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, o1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    o0 = _mm256_sub_epi32(o0, mask0);
    o1 = _mm256_add_epi32(o1, mask1);
    i2 = _mm256_unpacklo_epi32(o0, o1);
    i3 = _mm256_unpackhi_epi32(o0, o1);

    _mm256_store_si256((i256 *)(dst + i + 16), i2);
    _mm256_store_si256((i256 *)(dst + i + 24), i3);
}

INTER void
nttdifstage1(i32 *dst, i32 w256) {
    i32 i = 0;
    i256 i0, i1, i2, i3, o0, o1, mask1,
            m = _mm256_set_epi32(w256, 1, w256, 1, w256, 1, w256, 1),
            // intel likes his args little fermat
            q = _mm256_set1_epi32(0x20008001), minf = _mm256_set1_epi32(0);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;

    i0 = _mm256_load_si256((i256 *)(dst + i + 0));
    i1 = _mm256_load_si256((i256 *)(dst + i + 8));

    o0 = _mm256_unpacklo_epi64(i0, i1);
    o1 = _mm256_unpackhi_epi64(i0, i1);

    i2 = _mm256_load_si256((i256 *)(dst + i + 16));
    i3 = _mm256_load_si256((i256 *)(dst + i + 24));

    i0 = _mm256_add_epi32(o0, o1);
    i1 = _mm256_sub_epi32(o0, o1);
    mask1 = _mm256_cmpgt_epi32(minf, i1);
    mask0 = _mm256_cmpgt_epi32(i0, maxf);
    mask1 = _mm256_and_si256(mask1, q);
    mask0 = _mm256_and_si256(mask0, q);
    i1 = _mm256_add_epi32(i1, mask1);
    i0 = _mm256_sub_epi32(i0, mask0);

    i1 = mulmod0x20008001u(i1, m);

    o0 = _mm256_unpacklo_epi64(i0, i1);
    o1 = _mm256_unpackhi_epi64(i0, i1);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), o0);
        i0 = _mm256_load_si256((i256 *)(dst + i + 32));
        o0 = _mm256_unpacklo_epi64(i2, i3);

        _mm256_store_si256((i256 *)(dst + i + 8), o1);
        i1 = _mm256_load_si256((i256 *)(dst + i + 32 + 8));
        o1 = _mm256_unpackhi_epi64(i2, i3);

        i2 = _mm256_add_epi32(o0, o1);
        i3 = _mm256_sub_epi32(o0, o1);
        mask1 = _mm256_cmpgt_epi32(minf, i3);
        mask0 = _mm256_cmpgt_epi32(i2, maxf);
        mask1 = _mm256_and_si256(mask1, q);
        mask0 = _mm256_and_si256(mask0, q);
        i3 = _mm256_add_epi32(i3, mask1);
        i2 = _mm256_sub_epi32(i2, mask0);
        i3 = mulmod0x20008001u(i3, m);
        o0 = _mm256_unpacklo_epi64(i2, i3);
        o1 = _mm256_unpackhi_epi64(i2, i3);

        _mm256_store_si256((i256 *)(dst + i + 16), o0);
        i2 = _mm256_load_si256((i256 *)(dst + i + 32 + 16));
        o0 = _mm256_unpacklo_epi64(i0, i1);

        _mm256_store_si256((i256 *)(dst + i + 24), o1);
        i3 = _mm256_load_si256((i256 *)(dst + i + 32 + 24));
        o1 = _mm256_unpackhi_epi64(i0, i1);

        i0 = _mm256_add_epi32(o0, o1);
        i1 = _mm256_sub_epi32(o0, o1);
        mask1 = _mm256_cmpgt_epi32(minf, i1);
        mask0 = _mm256_cmpgt_epi32(i0, maxf);
        mask1 = _mm256_and_si256(mask1, q);
        mask0 = _mm256_and_si256(mask0, q);
        i1 = _mm256_add_epi32(i1, mask1);
        i0 = _mm256_sub_epi32(i0, mask0);
        i1 = mulmod0x20008001u(i1, m);
        o0 = _mm256_unpacklo_epi64(i0, i1);
        o1 = _mm256_unpackhi_epi64(i0, i1);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    o0 = _mm256_unpacklo_epi64(i2, i3);

    _mm256_store_si256((i256 *)(dst + i + 8), o1);
    o1 = _mm256_unpackhi_epi64(i2, i3);

    i2 = _mm256_add_epi32(o0, o1);
    i3 = _mm256_sub_epi32(o0, o1);
    mask1 = _mm256_cmpgt_epi32(minf, i3);
    mask0 = _mm256_cmpgt_epi32(i2, maxf);
    mask1 = _mm256_and_si256(mask1, q);
    mask0 = _mm256_and_si256(mask0, q);
    i3 = _mm256_add_epi32(i3, mask1);
    i2 = _mm256_sub_epi32(i2, mask0);
    i3 = mulmod0x20008001u(i3, m);

    o0 = _mm256_unpacklo_epi64(i2, i3);
    o1 = _mm256_unpackhi_epi64(i2, i3);

    _mm256_store_si256((i256 *)(dst + i + 16), o0);
    _mm256_store_si256((i256 *)(dst + i + 24), o1);
}

INTER void
nttdifstage2(i32 *dst, i32 w128, i32 w256, i32 w384) {
    i32 i = 0;
    i256 i0, i1, i2, i3, o0, o1, mask1,
            m = _mm256_set_epi32(w384, w256, w128, 1, w384, w256, w128, 1),
            q = _mm256_set1_epi32(0x20008001), minf = _mm256_set1_epi32(0);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;

    i0 = _mm256_load_si256((i256 *)(dst + i + 0));
    i1 = _mm256_load_si256((i256 *)(dst + i + 8));

    o0 = _mm256_permute2f128_si256(i0, i1, 32);
    o1 = _mm256_permute2f128_si256(i0, i1, 49);

    i2 = _mm256_load_si256((i256 *)(dst + i + 16));
    i3 = _mm256_load_si256((i256 *)(dst + i + 24));

    i0 = _mm256_add_epi32(o0, o1);
    i1 = _mm256_sub_epi32(o0, o1);
    mask1 = _mm256_cmpgt_epi32(minf, i1);
    mask0 = _mm256_cmpgt_epi32(i0, maxf);
    mask1 = _mm256_and_si256(mask1, q);
    mask0 = _mm256_and_si256(mask0, q);
    i1 = _mm256_add_epi32(i1, mask1);
    i0 = _mm256_sub_epi32(i0, mask0);
    i1 = mulmod0x20008001u(i1, m);
    o0 = _mm256_permute2f128_si256(i0, i1, 32);
    o1 = _mm256_permute2f128_si256(i0, i1, 49);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), o0);
        i0 = _mm256_load_si256((i256 *)(dst + i + 32));
        o0 = _mm256_unpacklo_epi64(i2, i3);
        o0 = _mm256_permute2f128_si256(i2, i3, 32);

        _mm256_store_si256((i256 *)(dst + i + 8), o1);
        i1 = _mm256_load_si256((i256 *)(dst + i + 32 + 8));
        o1 = _mm256_permute2f128_si256(i2, i3, 49);

        i2 = _mm256_add_epi32(o0, o1);
        i3 = _mm256_sub_epi32(o0, o1);
        mask1 = _mm256_cmpgt_epi32(minf, i3);
        mask0 = _mm256_cmpgt_epi32(i2, maxf);
        mask1 = _mm256_and_si256(mask1, q);
        mask0 = _mm256_and_si256(mask0, q);
        i3 = _mm256_add_epi32(i3, mask1);
        i2 = _mm256_sub_epi32(i2, mask0);
        i3 = mulmod0x20008001u(i3, m);

        o0 = _mm256_permute2f128_si256(i2, i3, 32);
        o1 = _mm256_permute2f128_si256(i2, i3, 49);

        _mm256_store_si256((i256 *)(dst + i + 16), o0);
        i2 = _mm256_load_si256((i256 *)(dst + i + 32 + 16));
        o0 = _mm256_permute2f128_si256(i0, i1, 32);

        _mm256_store_si256((i256 *)(dst + i + 24), o1);
        i3 = _mm256_load_si256((i256 *)(dst + i + 32 + 24));
        o1 = _mm256_permute2f128_si256(i0, i1, 49);

        i0 = _mm256_add_epi32(o0, o1);
        i1 = _mm256_sub_epi32(o0, o1);
        mask1 = _mm256_cmpgt_epi32(minf, i1);
        mask0 = _mm256_cmpgt_epi32(i0, maxf);
        mask1 = _mm256_and_si256(mask1, q);
        mask0 = _mm256_and_si256(mask0, q);
        i1 = _mm256_add_epi32(i1, mask1);
        i0 = _mm256_sub_epi32(i0, mask0);
        i1 = mulmod0x20008001u(i1, m);

        o0 = _mm256_permute2f128_si256(i0, i1, 32);
        o1 = _mm256_permute2f128_si256(i0, i1, 49);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), o0);
    o0 = _mm256_permute2f128_si256(i2, i3, 32);

    _mm256_store_si256((i256 *)(dst + i + 8), o1);
    o1 = _mm256_permute2f128_si256(i2, i3, 49);

    i2 = _mm256_add_epi32(o0, o1);
    i3 = _mm256_sub_epi32(o0, o1);
    mask1 = _mm256_cmpgt_epi32(minf, i3);
    mask0 = _mm256_cmpgt_epi32(i2, maxf);
    mask1 = _mm256_and_si256(mask1, q);
    mask0 = _mm256_and_si256(mask0, q);
    i3 = _mm256_add_epi32(i3, mask1);
    i2 = _mm256_sub_epi32(i2, mask0);
    i3 = mulmod0x20008001u(i3, m);

    o0 = _mm256_permute2f128_si256(i2, i3, 32);
    o1 = _mm256_permute2f128_si256(i2, i3, 49);

    _mm256_store_si256((i256 *)(dst + i + 16), o0);
    _mm256_store_si256((i256 *)(dst + i + 24), o1);
}

INTER void
nttdifstage(i32 *restrict dst, i32 const *restrict wvec, i32 im, i32 jm) {
    i32 f, s, i, j;
    const i32 *vw = wvec;
    i256 m, i0, i1, o0, o1, mask1, q = _mm256_set1_epi32(0x20008001),
                                   minf = _mm256_set1_epi32(0);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;

    for (i = 0; i < im; ++i) {
        for (j = 0; j < jm; ++j) {
            f = 8 * ((2 * i + 0) * jm + j);
            s = 8 * ((2 * i + 1) * jm + j);
            m = _mm256_load_si256((i256 *)(vw + 8 * j));
            o0 = _mm256_load_si256((i256 *)(dst + f));
            o1 = _mm256_load_si256((i256 *)(dst + s));

            i0 = _mm256_add_epi32(o0, o1);
            i1 = _mm256_sub_epi32(o0, o1);

            mask1 = _mm256_cmpgt_epi32(minf, i1);
            mask0 = _mm256_cmpgt_epi32(i0, maxf);
            mask1 = _mm256_and_si256(mask1, q);
            mask0 = _mm256_and_si256(mask0, q);
            i1 = _mm256_add_epi32(i1, mask1);
            i0 = _mm256_sub_epi32(i0, mask0);
            i1 = mulmod0x20008001u(i1, m);

            _mm256_store_si256((i256 *)(dst + f), i0);
            _mm256_store_si256((i256 *)(dst + s), i1);
        }
    }
}

INTER void
bitrevnttditstage0(i32 *restrict dst, const i32 *restrict src) {
    i32 i = 0;
    i256 i0, i1, i2, i3, o0, o1, mask1, q = _mm256_set1_epi32(0x20008001);
    i256 maxf = _mm256_set1_epi32(0x20008000);
    i256 mask0;
    i256 minf = _mm256_set1_epi32(0);

    if (src != dst) {
        memcpy(dst, src, 1024 * sizeof(i32));
    }

    i0 = _mm256_load_si256((i256 *)(dst + i + 0));
    i1 = _mm256_load_si256((i256 *)(dst + i + 8));
    o0 = _mm256_hadd_epi32(i0, i1);
    i2 = _mm256_load_si256((i256 *)(dst + i + 16));
    i3 = _mm256_load_si256((i256 *)(dst + i + 24));
    o1 = _mm256_hsub_epi32(i0, i1);

    mask0 = _mm256_cmpgt_epi32(o0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, o1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    o0 = _mm256_sub_epi32(o0, mask0);
    o1 = _mm256_add_epi32(o1, mask1);
    i0 = _mm256_unpacklo_epi32(o0, o1);
    i1 = _mm256_unpackhi_epi32(o0, o1);

    for (; i < 1024 - 32; i += 32) {
        _mm256_store_si256((i256 *)(dst + i + 0), i0);
        o0 = _mm256_hadd_epi32(i2, i3);
        i0 = _mm256_load_si256((i256 *)(dst + i + 32));
        _mm256_store_si256((i256 *)(dst + i + 8), i1);
        o1 = _mm256_hsub_epi32(i2, i3);
        i1 = _mm256_load_si256((i256 *)(dst + i + 32 + 8));

        mask0 = _mm256_cmpgt_epi32(o0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, o1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        o0 = _mm256_sub_epi32(o0, mask0);
        o1 = _mm256_add_epi32(o1, mask1);
        i2 = _mm256_unpacklo_epi32(o0, o1);
        i3 = _mm256_unpackhi_epi32(o0, o1);

        _mm256_store_si256((i256 *)(dst + i + 16), i2);
        o0 = _mm256_hadd_epi32(i0, i1);
        i2 = _mm256_load_si256((i256 *)(dst + i + 32 + 16));

        _mm256_store_si256((i256 *)(dst + i + 24), i3);
        o1 = _mm256_hsub_epi32(i0, i1);
        i3 = _mm256_load_si256((i256 *)(dst + i + 32 + 24));

        mask0 = _mm256_cmpgt_epi32(o0, maxf);
        mask1 = _mm256_cmpgt_epi32(minf, o1);
        mask0 = _mm256_and_si256(mask0, q);
        mask1 = _mm256_and_si256(mask1, q);
        o0 = _mm256_sub_epi32(o0, mask0);
        o1 = _mm256_add_epi32(o1, mask1);
        i0 = _mm256_unpacklo_epi32(o0, o1);
        i1 = _mm256_unpackhi_epi32(o0, o1);
    }
    _mm256_store_si256((i256 *)(dst + i + 0), i0);
    o0 = _mm256_hadd_epi32(i2, i3);

    _mm256_store_si256((i256 *)(dst + i + 8), i1);
    o1 = _mm256_hsub_epi32(i2, i3);

    mask0 = _mm256_cmpgt_epi32(o0, maxf);
    mask1 = _mm256_cmpgt_epi32(minf, o1);
    mask0 = _mm256_and_si256(mask0, q);
    mask1 = _mm256_and_si256(mask1, q);
    o0 = _mm256_sub_epi32(o0, mask0);
    o1 = _mm256_add_epi32(o1, mask1);
    i2 = _mm256_unpacklo_epi32(o0, o1);
    i3 = _mm256_unpackhi_epi32(o0, o1);

    _mm256_store_si256((i256 *)(dst + i + 16), i2);
    _mm256_store_si256((i256 *)(dst + i + 24), i3);
}

