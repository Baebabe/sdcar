#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Dynamic Weather:

Connect to a CARLA Simulator instance and control the weather. Change Sun
position smoothly with time and generate storms occasionally.
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pygame
import time
import math
import random

# Constants for the mini-map size and position
MINIMAP_WIDTH = 300
MINIMAP_HEIGHT = 300
MINIMAP_POSITION = (800, 50)  # Top-right corner of the screen
CLICK_RADIUS = 2.0  # Threshold for clicking accuracy

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1280, 720))  # Adjust screen size
pygame.display.set_caption("CARLA Mini-Map")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


def draw_minimap(world, player, surface):
    """Draw a simple mini-map of the environment."""
    vehicles = world.get_actors().filter('vehicle.*')  # Get all vehicles
    player_location = player.get_location()
    player_x, player_y = player_location.x, player_location.y

    # Center the minimap on the player
    surface.fill(BLACK)
    for vehicle in vehicles:
        loc = vehicle.get_location()
        relative_x = loc.x - player_x
        relative_y = loc.y - player_y
        minimap_x = MINIMAP_WIDTH / 2 + relative_x / 10
        minimap_y = MINIMAP_HEIGHT / 2 - relative_y / 10

        color = GREEN if vehicle.id != player.id else RED
        pygame.draw.circle(surface, color, (int(minimap_x), int(minimap_y)), 5)


def get_clicked_position(mouse_pos, player, world):
    """Convert mouse position on the minimap to world coordinates."""
    mouse_x, mouse_y = mouse_pos
    minimap_x, minimap_y = MINIMAP_POSITION
    rel_x = (mouse_x - minimap_x - MINIMAP_WIDTH / 2) * 10
    rel_y = -(mouse_y - minimap_y - MINIMAP_HEIGHT / 2) * 10
    player_loc = player.get_location()
    target_x = player_loc.x + rel_x
    target_y = player_loc.y + rel_y
    return carla.Location(x=target_x, y=target_y, z=player_loc.z)


def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # Get the first vehicle as the player's car
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    player = world.spawn_actor(random.choice(blueprint_library.filter('vehicle.*')), random.choice(spawn_points))

    clock = pygame.time.Clock()
    running = True

    minimap_surface = pygame.Surface((MINIMAP_WIDTH, MINIMAP_HEIGHT))
    teleport = False  # Flag for teleporting the vehicle

    while running:
        # Pygame events (e.g., check for clicks)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if (MINIMAP_POSITION[0] < mouse_pos[0] < MINIMAP_POSITION[0] + MINIMAP_WIDTH and
                        MINIMAP_POSITION[1] < mouse_pos[1] < MINIMAP_POSITION[1] + MINIMAP_HEIGHT):
                    target_loc = get_clicked_position(mouse_pos, player, world)
                    player.set_transform(carla.Transform(target_loc, player.get_transform().rotation))
                    teleport = True

        # Update the CARLA world
        world.tick()
        clock.tick_busy_loop(30)

        # Render the mini-map
        draw_minimap(world, player, minimap_surface)
        screen.fill(WHITE)
        screen.blit(minimap_surface, MINIMAP_POSITION)

        # Display a message if teleport happened
        if teleport:
            font = pygame.font.Font(None, 36)
            text = font.render("Teleported!", True, RED)
            screen.blit(text, (50, 50))
            teleport = False

        # Refresh the display
        pygame.display.flip()

    # Cleanup
    player.destroy()
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
