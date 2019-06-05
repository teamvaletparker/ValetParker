import sys,os,time
import pygame
from math import tan, radians, degrees, copysign, cos ,sin
from pygame.math import Vector2
from pygame.locals import *

class Car:
    def __init__(self, x, y, angle=0.0, width = 1.1, length=4, max_steering=30, max_acceleration=5.0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20
        self.brake_deceleration = 10
        self.free_deceleration = 2
        self.width = width;

        self.acceleration = 0.0
        self.steering = 0.0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt

class Block:
    def __init__(self, x, y):
        self.position = Vector2(x, y)

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60 # 초당 60 프레임
        self.exit = False
        self.count = 0   
        now = time.localtime()
        self.year = now.tm_year
        self.month = now.tm_mon
        self.day = now.tm_mday
        self.hour = now.tm_hour 
        self.min = now.tm_min
        self.sec = now.tm_sec

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        car = Car(3, 3)
#        ppu = 32
        image_path2 = os.path.join(current_dir, "block2.png")
        block_image = pygame.image.load(image_path2)
        block = Block(10 , 10)
        ppu = 32
        #create folder 
        Dir = "D:\\ORG\\WorkSpace\\그래픽캡쳐모듈\\record history\\"+str(self.year)+"-"+str(self.month)+"-"+str(self.day)+"_"+str(self.hour)+"."+str(self.min)+"."+str(self.sec)
        if not os.path.isdir(Dir):
            os.mkdir(Dir)
        #set frame for game capture
        pygame.time.set_timer(USEREVENT+1, 33) #1초(1000밀리초)/60frame 구하기 =16.6...FPS

        while not self.exit:
            dt = 0.1

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
                elif event.type == USEREVENT+1: #Capture : Game capture and save files
                    name = str(self.count)+".jpg"
                    pygame.image.save(self.screen,Dir+"\\"+name)         
                    self.count = self.count+16
            # User input
            pressed = pygame.key.get_pressed()

            if pressed[pygame.K_UP]:
                if car.velocity.x < 0:
                    car.acceleration = car.brake_deceleration
                else:
                    car.acceleration += 1
            elif pressed[pygame.K_DOWN]:
                if car.velocity.x > 0:
                    car.acceleration = -car.brake_deceleration
                else:
                    car.acceleration -= 1 
            elif pressed[pygame.K_SPACE]:
                if abs(car.velocity.x) > dt * car.brake_deceleration:
                    car.acceleration = -copysign(car.brake_deceleration, car.velocity.x)
                else:
                    car.acceleration = -car.velocity.x / dt
            else:
                if abs(car.velocity.x) > dt * car.free_deceleration:
                    car.acceleration = -copysign(car.free_deceleration, car.velocity.x)
                else:
                    if dt != 0:
                        car.acceleration = -car.velocity.x / dt
            car.acceleration = max(-car.max_acceleration, min(car.acceleration, car.max_acceleration))

            if pressed[pygame.K_RIGHT]:
                car.steering -= 3
            elif pressed[pygame.K_LEFT]:
                car.steering += 3
            else:
                car.steering = 0
            car.steering = max(-car.max_steering, min(car.steering, car.max_steering))
            
            if car.position.x > 40 : 
                car.position.x = 40
            if car.position.x < 0 : 
                car.position.x = 0
            if car.position.y >21 : 
                car.position.y = 21
            if car.position.y < 0 : 
                car.position.y = 0
            # Logic
            car.update(dt)

            # Drawing
            self.screen.fill((0, 0, 0))
            rotated = pygame.transform.rotate(car_image, car.angle)
            rect = rotated.get_rect()
            rect2 = block_image.get_rect() #좌표값얻음
            self.screen.blit(block_image, block.position)
            self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))
            pygame.display.flip()

            self.clock.tick(self.ticks) # 60 FPS(frames per sec)설정
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
