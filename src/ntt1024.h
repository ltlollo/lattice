#ifndef NTT1024_H
#define NTT1024_H

#ifndef DECL
#define DECL
#endif

#ifndef INTER
#define INTER
#endif

#ifdef __cplusplus
#define restrict
#endif

DECL void ntt32x1024(int *restrict, const int *restrict);
DECL void ntt32x1024mulphi(int *restrict, const int *restrict);
DECL void intt32x1024(int *restrict, const int *restrict);
DECL void intt32x1024muliphi(int *restrict, const int *restrict);
DECL void mulvec(int *, const int *, const int *);
DECL void cleanregs(void);

DECL void bitrevinttdit32x1024muliphi(int *, const int *);
DECL void nttdif32x1024mulphibitrev(int *, const int *);

INTER void bitrevnttditstage0(int *restrict, const int *restrict);
INTER void nttditstage0(int *restrict, const int *restrict);
INTER void nttditstage0mulphi(int *restrict, const int *restrict);
INTER void nttditstage1(int *, int);
INTER void nttditstage2(int *, int , int , int);
INTER void nttditstagen(int *restrict, const int *restrict);
INTER void nttditstage(int *restrict, const int *restrict, int, int);

INTER void nttdifstage0(int *restrict);
INTER void nttdifstage1(int *, int);
INTER void nttdifstage2(int *, int, int, int);
INTER void nttdifstagen(int *restrict, const int *restrict);
INTER void nttdifstage(int *restrict, int const *restrict, int, int);

INTER void mulconst(int *, const int *, int);
INTER void sub0x20008001(int *, const int *);
INTER void bitrev32x1024(int *restrict, const int *restrict);
INTER void bitrev32x1024mulphi(int *restrict, const int *restrict);



#endif // NTT1024_H

