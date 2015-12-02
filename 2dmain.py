"""
CSCI 580 Project - Fall 2015 - Western Washington University
Author: Sam Pollard (pollars at students dot wwu dot edu)
Last Modified: December 2, 2015
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

xres,yres = (640,640)

def handleInput(screen):
    # Handle Keyboard and Mouse Events
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
        elif event.type == MOUSEBUTTONDOWN:
            x,y = pygame.mouse.get_pos()
            print("clicked at " + str(float(x)/xres) + "," + str(float(y)/yres))
    return False

def clamp(x, low, high):
    x = [max(xi,  low) for xi in x]
    x = [min(high, xi) for xi in x]
    return np.array(x)

def screen_init():
    """Initializes the pygame surface. Returns a screen and a background."""
    screen = pygame.display.set_mode((xres,yres))

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
    width, height = xres, yres
    
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
    
def main():
    pygame.init()
    print("Press s to save, esc to exit")
    pygame.display.set_caption("Handwriting!")
    stroke = bezier.Handwriting(
            scale=4, brush=lambda t: 0.1*(1-t), min_s=2, max_s=5)
    print(stroke)  # Print some information about the texture
    draw_image_explicit(stroke.evaluate)

if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()
