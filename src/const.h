#ifndef CONST_H
#define CONST_H

#include <stdalign.h>

#include "bitrev.h"
#include "phiconst.h"
#include "wconst.h"

// q				= 0x20008001 | 536903681
// n				= 0x00000400 |      1024
// exp(n, -1)		= 0x1ff87fe1 | 536379361
// w				= 0x002583dc |   2458588
// exp(w, -1)		= 0x08936511 | 143877393
// phi = sqrt(w)	= 0x0b7a5273 | 192565875
// exp(phi, -1)		= 0x1f5ff22a | 526381610

__attribute__((unused)) static const int *wvec128 = wvec + 4;
__attribute__((unused)) static const int *iwvec128 = iwvec + 4;
__attribute__((unused)) static const int *wvec256 = wvec + 8;
__attribute__((unused)) static const int *iwvec256 = iwvec + 8;
__attribute__((unused)) static const int *wtest = wvec + 1;
__attribute__((unused)) static const int *iwtest = iwvec + 1;

__attribute__((unused)) static const int *difwvec256 = difwvec;
__attribute__((unused)) static const int *difwvec128 = difwvec;

#endif // CONST_H

