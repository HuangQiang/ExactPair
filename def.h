#ifndef __DEF_H
#define __DEF_H

// -----------------------------------------------------------------------------
//  Macros
// -----------------------------------------------------------------------------
#define min(x, y) 			((x) < (y) ? (x) : (y))
#define max(x, y) 			((x) < (y) ? (y) : (x))
#define min3(x, y, z) 		(min(x, min(y, z)))
#define argmin3(x, y, z) 	(((x) < (y) && (x) < (z)) ? 0 : ((y) < (z) ? 1 : 2))
#define dist(x, y) 			(((x) - (y)) * ((x) - (y)))
#define sgn(x) 				((x) < 0 ? -1 : 1)
#define abs(x) 				((x)*sgn(x))
#define unused(x) 			((void) x)

// -----------------------------------------------------------------------------
//  Constants
// -----------------------------------------------------------------------------
const int   MAXK    = 100;			// max top-k value
const float MAXREAL = 3.402823466e+38F;
const float MINREAL = -MAXREAL;		// min real value

#endif // __DEF_H