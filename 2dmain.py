"""
CSCI 580 Project
Author: Sam Pollard (pollars at students dot wwu dot edu)
Last Modified: November 23, 2015
Code taken from Geof Matthews (github.com/geofmatthews/csci480)
"""

import os
import math
import numpy as np
import random
import pygame
import noise
import bezier
import time
from pygame.locals import *

if __name__ == "__main__":
    main_dir = os.getcwd() 
else:
    main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')

def handleInput(screen):
    #Handle Keyboard Events
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return True
            elif event.key == K_s:
                pygame.event.set_blocked(KEYDOWN|KEYUP)
                fname = raw_input("File name?  ")
                pygame.event.set_blocked(0)
                pygame.image.save(screen,fname)
    return False

def clamp(x, low, high):
    x = [max(xi,  low) for xi in x]
    x = [min(high, xi) for xi in x]
    return np.array(x)

def drawImage(title, colorAt, noiseGen):
    """ Given noise and a colorAt function, draw the picture."""
    #Initialize Everything
    screen = pygame.display.set_mode((640,640))
    pygame.display.set_caption(title)

    #Create The Backgound
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((64, 128, 255))

    #Display The Background
    screen.blit(background, (0, 0))
    pygame.display.flip()

    #Prepare Game Objects
    clock = pygame.time.Clock()
    going = True
    pixelsize = 256 # power of 2
    width, height = screen.get_size()
    
    # main loop
    start = time.clock()
    while going:
        going = not(handleInput(screen))
        # start drawing loop
        while pixelsize > 0:
            print(pixelsize)
            # clock.tick(1)
            for x in range(0,width,pixelsize):
                xx = x/float(width)
                for y in range(0,height,pixelsize):
                    #clock.tick(2)
                    yy = y/float(height)
                    # draw into background surface
                    # noise = noiseGen.noise((x,y))
                    noise = 1
                    color = 255*clamp(noise*np.array(colorAt(xx, yy)), 0, 1)
                    background.fill(color, ((x,y),(pixelsize,pixelsize)))
                    if handleInput(screen):
                        return
                #draw background into screen
                screen.blit(background, (0,0))
                pygame.display.flip()
            pixelsize /= 2
            if pixelsize == 0:
                print("Finished drawing. Elapsed time = {}s".format(
                        time.clock() - start))

def main():
    pygame.init()
    print("Press s to save, esc to exit")
    noiseGen = noise.ValueNoise(noiseType="turbulence")
    stroke = bezier.Handwriting(scale=4)#, thickness=lambda t: 0.1*(1-t))
    print(stroke)  # Print some information about the texture (TEST)
    drawImage("Handwriting!", stroke.evaluate, noiseGen)

if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()
