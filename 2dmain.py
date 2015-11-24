"""
CSCI 580 Project
Author: Sam Pollard (pollars at students dot wwu dot edu)
Last Modified: November 23, 2015
Code taken from Geof Matthews (github.com/geofmatthews/csci480)
"""

import os
import numpy as np
import pygame
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

def screen_init():
    """Initializes the pygame surface. Returns a screen and a background."""
    screen = pygame.display.set_mode((640,640))

    #Create The Backgound
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((64, 128, 255))

    #Display The Background
    screen.blit(background, (0, 0))
    pygame.display.flip()
    return screen, background


def draw_image_explicit(colorAt):
    """Given a colorAt function, draw the picture by sampling at each pixel."""
    screen, background = screen_init()
    
    #Prepare Game Objects
    going = True
    pixelsize = 256 # power of 2
    width, height = screen.get_size()
    
    clock = pygame.time.Clock()
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
                    color = 255*clamp(colorAt(xx, yy), 0, 1)
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

#def draw_image_implicit(representation):
    #""" Given an implicit representation of an image, run through the
        #parameterized value to draw each object. Assumes a single parameter
        #with acceptable values in [0,1].
    #"""
    #screen, background = screen_init()
    
    ##Prepare Game Objects
    #going = True
    #width, height = screen.get_size()
    #granularity = 2  # Doubles each iteration
    
    #start = time.clock()
    #clock = pygame.time.Clock()
    #while going:
        #going = not(handleInput(screen))
        ## start drawing loop
        #while granularity < max(width, height):
            #print(granularity)
            #for t in np.linspace(0.0,1.0, granularity + 1):
                #background.fill(representation(t))
            ## clock.tick(1)
            #background.fill(color, ((x,y),(pixelsize,pixelsize)))
            ##draw background into screen
            #screen.blit(background, (0,0))
            #pygame.display.flip()
            #if granularity > max_time:
                #print("Finished drawing. Elapsed time = {}s".format(
                        #time.clock() - start))
    
def main():
    pygame.init()
    print("Press s to save, esc to exit")
    pygame.display.set_caption("Handwriting!")
    stroke = bezier.Handwriting(scale=4)#, brush=lambda t: 0.1*(1-t))
    print(stroke)  # Print some information about the texture (TEST)
    draw_image_explicit(stroke.evaluate)

if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()
