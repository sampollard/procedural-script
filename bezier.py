""" Procedurally generated handwriting using bezier curves.
    The primary method is evaluate which takes as input two textures
    and returns the color at one or the other for each point (u,v)
"""

import inspect
import numpy as np
import random

class Texture():
    def __init__(self, eval_func=None):
        self.evaluate = eval_func
    
    def evaluate(self, u, v):
        """Determine the color given u,v coordinates"""
        pass


class Handwriting(Texture):
    def __init__(self, texture1=Texture(lambda u,v: (0,0,0)),
                       texture2=Texture(lambda u,v: (1,1,1)),
                       thickness=lambda t: 0.1,
                       scale=1,
                       strokes=30,
                       unique_chars=16,
                       connect_p=1.0):
                       # XXX: Curviness \in [0,infty)
                       # where a value of 1 is default random numbers
                       # between [0,1) from random(), 0 is collinear,
                       # and numbers larger than 1 scale random to be
                       # in the range [0,curviness).
        self.tex1 = texture1     # Inside bezier curve
        self.tex2 = texture2     # Outside bezier curve
        self.scale = scale
        # strokes is the number of unique strokes. May increase when
        # connect_p > 0.
        self.strokes = strokes
        self.octaves = 9         # Accurate to dimension / 2^(octaves+1) pixels
        self.eps = lambda t: thickness(t) / self.scale

        # Create a random hash table to which and how many bezier curves
        # are used for each character.
        random.seed(0)
        hash_sz = 257  # Choose a prime number for the hash size
        self.hash_table = list(range(hash_sz))
        random.shuffle(self.hash_table)

        # Create the control points. 4 control points = 1 bezier curve
        self.control_points = [
                [[random.random(), random.random()] for cp in range(4)]
                for stroke in range(strokes)]
        # Create the characters which contain between [min_s, max_s] strokes
        min_s = 1
        max_s = 3

        # Create each character. self.characters indexing goes
        # [character][bezier curve][control point][uv coordinate]
        # We keep track of which characters have been modified so
        # strokes don't get changed twice
        self.characters = []
        used_strokes = {}
        for i in range(unique_chars):
            char = []
            for stroke in range(random.randint(min_s, max_s)):
                char.append(self.hash_table[(max_s*i) + stroke] % strokes)
                used_strokes[char[-1]] = True
            self.characters.append(char)
        
        # Connect the strokes at the character edges
        cp = self.control_points
        for i in range(unique_chars):
            if random.random() < connect_p:
                first_stroke = self.characters[i][-1]
                second_stroke = self.characters[(i+1) % int(unique_chars)][0]
                print(first_stroke, second_stroke)
                # Connect the first control point of the first stroke with
                # the first control point of the second stroke.
                if used_strokes[first_stroke]:
                    cp.append(cp[first_stroke])
                    self.characters[i][-1] = len(cp) - 1
                    self.strokes += 1
                    used_strokes[len(cp) - 1] = True
                    cp[-1][0] = [1.0, cp[second_stroke][0][1]]
                if used_strokes[second_stroke]:
                    cp.append(cp[second_stroke])
                    self.characters[(i+1) % int(unique_chars)][0] = len(cp)-1
                    self.strokes += 1
                    used_strokes[len(cp) - 1] = True
                    cp[-1][0] = [0.0, cp[second_stroke][0][1]]
                    
                print(self.control_points[-2])
                print(self.control_points[-1])
                
                
                
                #self.control_points[self.characters[i][-1]][3][0] = 1.0
                #self.control_points[self.characters[(i+1) % int(unique_chars)
                        #][0]][0] = (0.0, prev_uv[1])
                
                #if i == 0:
                    #print(self.control_points[self.characters[i][-1]])


    def __repr__(self):
        cp_str = "[\n"
        for cp in self.control_points:
            for p in cp:
                cp_str += " ({:.4f}, {:.4f})".format(p[0], p[1])
            cp_str += "\n"
        cp_str += "]"
        return "control_points: {}\ncharacters: {}\neps: {}".format(
                cp_str, self.characters, inspect.getsource(self.eps).strip())

    def evaluate(self, u, v):
        """ The primary function. Determine whether the point is
            inside any of the bezier curves.
        """
        if self.inside_curve(u,v):
            return self.tex1.evaluate(u,v)
        else:
            return self.tex2.evaluate(u,v)

    def B(self, t, i):
        """Evaluate the ith bezier curve at t."""
        p = self.control_points[i]

        # Get the coefficients using coeffs(collect(B),t) in Matlab
        t2 = t*t
        t3 = t2*t
        return np.array((
                # u
                p[0][0] +
                3*(p[1][0] - p[0][0])*t +
                3*(p[0][0] - 2*p[1][0] + p[2][0])*t2 +
                (3*p[1][0] - p[0][0] - 3*p[2][0] + p[3][0])*t3,
                # v
                p[0][1] +
                3*(p[1][1] - p[0][1])*t +
                3*(p[0][1] - 2*p[1][1] + p[2][1])*t2 +
                (3*p[1][1] - p[0][1] - 3*p[2][1] + p[3][1])*t3
            ))

    def inside_curve(self, u, v):
        """ Determine if the point (u,v) is inside of the bezier region.
            via binary search
        """
        def norm2(x):
            """Computes the inner product of a 2d vector x with itself"""
            return x[0]*x[0] + x[1]*x[1]

        def in_curve_exact(uv, i):
            """ Finding the real roots of the polynomial
                dB_udt * (B_u - u) + dB_vdt*(B_v - v).
                This was generated using the following Matlab code:
            syms t u v p0u p1u p2u p3u p0v p1v p2v p3v
            B_u(t) = (1-t)^3*p0u + 3*(1-t)^2*t*p1u + 3*(1-t)*t^2*p2u + t^3*p3u
            B_v(t) = (1-t)^3*p0v + 3*(1-t)^2*t*p1v + 3*(1-t)*t^2*p2v + t^3*p3v
            dB_udt = diff(B_u,t)
            %  dB_udt(t) = 3*p1*(t - 1)^2 - 3*p0*(t - 1)^2 - 3*p2*t^2 +
            %              3*p3*t^2 + 3*p1*t*(2*t - 2) - 2*p2*t*(3*t - 3)
            %  And likewise for B_v. Then we solve the polynomial
            poly = dB_udt*(B_u - u) + dB_vdt*(B_v - v)
            %  We get the roots (and thus the nastly np.array) with
            collect(poly, t)
            """
            # Set variables to be in the form that Matlab outputs
            (u,v) = uv
            ((p0u,p0v),(p1u,p1v),(p2u,p2v),(p3u,p3v)) = self.control_points[i]
            roots = np.roots(np.array((
                    ((p0u - 3*p1u + 3*p2u - p3u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (p0v - 3*p1v + 3*p2v - p3v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)),
                    (- (6*p0u - 12*p1u + 6*p2u)*(p0u - 3*p1u + 3*p2u - p3u) - (6*p0v - 12*p1v + 6*p2v)*(p0v - 3*p1v + 3*p2v - p3v) - (3*p0u - 6*p1u + 3*p2u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (3*p0v - 6*p1v + 3*p2v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)),
                    ((3*p0u - 3*p1u)*(p0u - 3*p1u + 3*p2u - p3u) + (3*p0v - 3*p1v)*(p0v - 3*p1v + 3*p2v - p3v) + (3*p0u - 3*p1u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (3*p0u - 6*p1u + 3*p2u)*(6*p0u - 12*p1u + 6*p2u) + (3*p0v - 3*p1v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v) + (3*p0v - 6*p1v + 3*p2v)*(6*p0v - 12*p1v + 6*p2v)),
                    (- (3*p0u - 3*p1u)*(3*p0u - 6*p1u + 3*p2u) - (3*p0u - 3*p1u)*(6*p0u - 12*p1u + 6*p2u) - (3*p0v - 3*p1v)*(3*p0v - 6*p1v + 3*p2v) - (3*p0v - 3*p1v)*(6*p0v - 12*p1v + 6*p2v) - (p0u - u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (p0v - v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)),
                    ((3*p0u - 3*p1u)**2 + (3*p0v - 3*p1v)**2 + (p0u - u)*(6*p0u - 12*p1u + 6*p2u) + (p0v - v)*(6*p0v - 12*p1v + 6*p2v)),
                     - (p0u - u)*(3*p0u - 3*p1u) - (p0v - v)*(3*p0v - 3*p1v))))
            for r in roots:
                if r.imag == 0.0:
                    t = r.real
                    if np.linalg.norm(uv - self.B(t, i)) < self.eps(t) and (
                            t < 1 + self.eps(t) or t > -self.eps(t)):
                        return True
            return False
 

        def in_curve_bs(uv, i):
            """ Determines if the point (u,v) is inside curve i with
                binary search. May not find all points.
            """
            t = 0.5
            sf = 0.25  # How much to change t for the binary search
            for o in range(self.octaves):
                # If ||(uv - B(t+sf)||^2 < ||(uv - B(t-sf)||^2
                if norm2(uv-self.B(t+sf, i)) < norm2(uv-self.B(t-sf, i)):
                    t = t + sf
                else:
                    t = t - sf
                sf /= 2
            if np.linalg.norm(uv - self.B(t, i)) < self.eps(t) and (
                    t < 1 + self.eps(t) or t > -self.eps(t)):
                return True
            else:
                return False

        scaled_u = self.scale * u
        scaled_v = self.scale * v
        pctu = (scaled_u) % 1
        pctv = (scaled_v) % 1
        # Move left to right in row-major order through the bezier curves
        # XXX: Alternatively, _i_ could index into the hash table to get
        #      an index into the character array
        i = (self.scale*int(scaled_u) + int(scaled_v)) % len(self.characters)
        uv = np.array((pctu,pctv))
        # Search through each bezier curve used in this box
        return any([in_curve_exact(uv, val) for val in self.characters[i]])
