/*	Random numbers for the benzene tensor library. 

This file provides an implementation of the PCG random number generator (https://www.pcg-random.org/), along with some commonly used distributions.
 
C++ already provides some RNGs - mt19937 and the like, but the promise of PCG to be just as statistically performant with a fraction of the memory use (https://www.pcg-random.org/#:~:text=At%2Da%2DGlance%20Summary) and complexity made it a compelling choice. 
 
The interface provided here is meant to be as close to the original (https://github.com/imneme/pcg-cpp/blob/master/include/pcg_random.hpp). The original code has implementations at multiple levels of state and statistical power, this file only provides 32 bit input and 32 bit output, with a single technique for random streams (LCG).
 */
#pragma once
#include <cstdint> 
typedef uint32_t u32;
typedef uint64_t u64;
typedef float 	 f32;
typedef double 	 f64;

namespace bz{

class PCG32{
	u64 state;
	u64 increment;
	
public: 
	PCG32(u64 seed = 42, u64 stream = 1): state(0), increment((stream << 1) | 1){
		next();
		state += seed; 
		next();
	}
	u32 next(){
		u64 old = state; 
		state = old * 6364136223846793005ULL + increment;
		u32 xorshifted = ((old >> 18u) ^ old) >> 27u;
		u32 rot = old >> 59u;
		return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
	}
	float next_float(){
		return (float)next() / (float)0xFFFFFFFF;
	}
};

}



