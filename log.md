# Development Log

## Oct 2nd, 2025
I ran Zwart's C code and find his precision is higher than me, although the speed is still his shortage. However, the difference of precision i.e. the reconstruction quality is not observable with naked eyes.

I find Zwart's C code use a polynominal function, which seems like a fit of the optimal kernel, and that kernel outperforms the kernel in his paper. This is why my reproduction is worse than his original C implementation.

So I have to complete with another baseline. I decide to choose the implementation from sigpy. That implementation use a conventional Kaiser kernel as used in Pipe's work, and the grid convolution proposed by Zwarts. (which means the only difference is about kernel function)