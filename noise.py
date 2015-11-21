""" Various noise functions.
        ValueNoise is an improvement over CSCI 480 Assignment 1
        WorleyNoise is also known as Cellular Noise.
    WorleyNoise is also planned to be implemented in pbrt, but the
    ideas will be reinforced in python first.
        The primary method of all Noise objects is
                    noise: R^d -> [0,1]
    where d is the dimension of the input and R is all real numbers.

    CS 580 - Fall 2015 - Geof Matthews
    Author: Sam Pollard
    Last Modified: October 24, 2015
"""

import numpy as np
import random
import math

class Noise():
    def __init__(self, dim=2):
        self.dim = dim

    def noise(x):
        pass

class ValueNoise(Noise):
    """ ValueNoise here is latticeNoise with cubic interpolation.
    """
    def __init__(self, hashSz=256, dim=2, octaves=6,
                 latticeSize=64, period=None, noiseType=None):
        """ hashSz is the size of the hash table. Dimension is the size
            of the input, octaves specifies the maximum power of 2
            wavelength add for the fractal noise. latticeSize specifies
            where the lattice points are located. Period determines
            whether the noise can tile. To get tiling along the image
            boundaries, the image dimension should be a multiple of
            period*latticeSize.
        """
        Noise.__init__(self, dim)
        self.octaves = octaves
        self.latticeSize = latticeSize
        # Create the random hash table
        self.n = hashSz
        self.hashTable = range(self.n)
        random.shuffle(self.hashTable)
        self.hashTable = np.array(self.hashTable)
        # Create lattice points
        self.noiseTable = np.linspace(0, 1, hashSz)  # Evenly spaced in [0,1]
        # Check for tileability
        if period is None or period <= self.n:
            self.period = period
        else:
            raise ValueError("period ({}) must be <= hashSz ({})".format(
                    hashSz, period))
        if not period is None and period * 2**(octaves - 1) > hashSz:
            raise ValueError("Error: hash table < period * 2^(octaves-1)")

        if self.dim == 2:
            if noiseType == "turbulence":
                self.noise = self.turbulence2d
            else:
                self.noise = self.noise2d
        elif self.dim == 1:
            self.noise = self.noise1d
        elif self.dim == 3:
            if noiseType == "turbulence":
                self.noise = self.turbulence3d
            else:
                self.noise = self.noise3d
        else:
            raise ValueError("Noise not implemented for dim > 3")


    def smerp(self, x, y1, y2):
        """ Smoothly interpolate using a cubic between points x1 and
            x2 with noise values y1, y2. Assumes x in [0,1]. Returns
            a value in [0,1].
        """
        if x == 1.0:
            return y2
        elif x > 1.0 or x < 0.0:
            raise ValueError("x is {} but must be in [0,1]".format(x))
        else:
            x1 = math.floor(x / self.n)
            x2 = self.noiseTable[x1 + 1]
        return 2*(y1 - y2)*x*x*x + 3*(y2 - y1)*x*x + y1

    def latticeNoise(self, point, period):
        """Get a lattice point from an integer or tuple of integers."""
        n = self.n
        if not period:
            if type(point) == int:
                return self.noiseTable[self.hashTable[point % n]]
            else:
                entry = point[0]
                for x in point[1:]:
                    entry = self.hashTable[(entry + self.hashTable[x % n]) % n]
                return self.noiseTable[entry]
        else:
            if type(point) == int:
                return self.noiseTable[self.hashTable[point % period]]
            else:
                entry = point[0]
                for x in point[1:]:
                    entry = self.hashTable[(entry + self.hashTable[x % period])
                            % period]
                return self.noiseTable[entry]

    def noise1d(self, p):
        """ Create 1-dimensional noise. p should be the pixel value.
        """
        amplitude = 1.0
        cumu_noise = 0.0
        for i in range(self.octaves):
            amplitude /= 2
            scaled_p = p * 2**i
            y1 = self.latticeNoise(scaled_p)
            y2 = self.latticeNoise(scaled_p + 1)
            x = scaled_p % self.n / float(self.n)
            cumu_noise += amplitude * self.smerp(x, y1, y2)
        return cumu_noise

    def noise2dHelper(self, x, y, period):
        """ Indexing into the hashTable using the integer part of
            x and y modulo the chunk size (c), then smerp-ing
            between them.
        """
        # x and y as the number of chunks
        chunk_x = x / float(self.latticeSize)
        chunk_y = y / float(self.latticeSize)
        intx = math.floor(chunk_x)
        inty = math.floor(chunk_y)
        pctx = chunk_x - intx
        pcty = chunk_y - inty
        aa = self.latticeNoise((intx,inty), period)
        ab = self.latticeNoise((intx,inty+1), period)
        ba = self.latticeNoise((intx+1,inty), period)
        bb = self.latticeNoise((intx+1,inty+1), period)
        xa = self.smerp(pctx, aa, ba)
        xb = self.smerp(pctx, ab, bb)
        return self.smerp(pcty, xa, xb)

    def noise2d(self, pt):
        """ Create 2-dimensional noise. x and y are real numbers."""
        amplitude = 1.0
        cumu_noise = 0.0
        x = pt[0]# % self.n
        y = pt[1]# % self.n
        p = self.period if self.period else 0
        for i in range(self.octaves):
            amplitude /= 2
            cumu_noise += amplitude*self.noise2dHelper(
                    x*1.99**i, y*1.99**i, p*2**i)
        return cumu_noise

    def noise3dHelper(self, x, y, z, period):
        chunkxyz = tuple(map(lambda a: a / float(self.latticeSize), (x,y,z)))
        (intx, inty, intz) = tuple(map(math.floor, chunkxyz))
        (pctx, pcty, pctz) = tuple(map(lambda a,b: a - b,
                                       chunkxyz,
                                       (intx, inty, intz)))
        # Interpolate the bottom part of the cube
        aab = self.latticeNoise((intx,inty,intz), period)
        abb = self.latticeNoise((intx,inty+1,intz), period)
        bab = self.latticeNoise((intx+1,inty,intz), period)
        bbb = self.latticeNoise((intx+1,inty+1,intz), period)
        xab = self.smerp(pctx, aab, bab)
        xbb = self.smerp(pctx, abb, bbb)
        bot = self.smerp(pcty, xab, xbb)
        # Interpolate the top part of the cube
        aat = self.latticeNoise((intx,inty,intz+1), period)
        abt = self.latticeNoise((intx,inty+1,intz+1), period)
        bat = self.latticeNoise((intx+1,inty,intz+1), period)
        bbt = self.latticeNoise((intx+1,inty+1,intz+1), period)
        xat = self.smerp(pctx, aat, bat)
        xbt = self.smerp(pctx, abt, bbt)
        top = self.smerp(pcty, xat, xbt)
        return self.smerp(pctz, bot, top)

    def noise3d(self, pt):
        """ Create 3-dimensional noise. x, y, z are real numbers."""
        amplitude = 1.0
        cumu_noise = 0.0
        x = pt[0]# % self.n
        y = pt[1]# % self.n
        z = pt[2]# % self.n
        p = self.period if self.period else 0
        for i in range(self.octaves):
            amplitude /= 2
            cumu_noise += amplitude*self.noise3dHelper(
                    x*2**i, y*2**i, z*2**i, p*2**i)
        return cumu_noise

    def turbulence3d(self, pt):
        """ Create turbulent 3-dimensional noise. The first derivative
            discontinuities are achieved by transforming noise to [-1,1]
            then summing the absolute values.
        """
        amplitude = 2.0
        cumu_noise = 0.0
        (x, y, z) = tuple(pt)
        p = self.period if self.period else 0
        for i in range(self.octaves):
            amplitude /= 2
            cumu_noise += amplitude * abs(
                    self.noise3dHelper(x*2**i, y*2**i, z*2**i, p*2**i) - 0.5)
        return cumu_noise

    def turbulence2d(self, pt):
        """Create turbulent 2-dimensional noise."""
        amplitude = 2.0
        cumu_noise = 0.0
        (x,y) = tuple(pt)
        p = self.period if self.period else 0
        for i in range(self.octaves):
            amplitude /= 2
            cumu_noise += amplitude * abs(
                    self.noise2dHelper(x*2**i, y*2**i, p*2**i) - 0.5)
        return cumu_noise


class WorleyNoise(Noise):
    def __init__(self, hashSz=256, dim=2, octaves=6,
                 latticeSize=64, period=None):
        pass