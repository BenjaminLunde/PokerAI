import random
import os
import pygame
from poker import Card

import tensorflow as tf

import PokerHandClassifier as PHC

from PokerRL.cfr.VanillaCFR import VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase


WIDTH = 645
HEIGHT = 480
RED = (0, 0, 0)
OTHER = (255, 255, 255)


current_path = os.path.dirname(__file__) # Where your .py file is located
resource_path = os.path.join(current_path, 'resources') # The resource folder path


# Card deck found here: https://acbl.mybigcommerce.com/52-playing-cards/

test = list(Card)
print("test: ", test)

# Trail of tears
all_images = [
    "images/2C.png",
    "images/2D.png",
    "images/2H.png",
    "images/2S.png",
    "images/3C.png",
    "images/3D.png",
    "images/3H.png",
    "images/3S.png",
    "images/4C.png",
    "images/4D.png",
    "images/4H.png",
    "images/4S.png",
    "images/5C.png",
    "images/5D.png",
    "images/5H.png",
    "images/5S.png",
    "images/6C.png",
    "images/6D.png",
    "images/6H.png",
    "images/6S.png",
    "images/7C.png",
    "images/7D.png",
    "images/7H.png",
    "images/7S.png",
    "images/8C.png",
    "images/8D.png",
    "images/8H.png",
    "images/8S.png",
    "images/9C.png",
    "images/9D.png",
    "images/9H.png",
    "images/9S.png",
    "images/10C.png",
    "images/10D.png",
    "images/10H.png",
    "images/10S.png",
    "images/JC.png",
    "images/JD.png",
    "images/JH.png",
    "images/JS.png",
    "images/QC.png",
    "images/QD.png",
    "images/QH.png",
    "images/QS.png",
    "images/KC.png",
    "images/KD.png",
    "images/KH.png",
    "images/KS.png",
    "images/AC.png",
    "images/AD.png",
    "images/AH.png",
    "images/AS.png",
]

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("LukyLuke")
        #Intutuion model
        # Create a new model instance
        self.model = PHC.PokerHandClassifier()
        self.model.compile(optimizer= tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.model.load_weights('./checkpoints/my_checkpoint')


        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.background_image = pygame.image.load(os.path.join(current_path, "table.png")).convert() # https://www.pngjoy.com/preview/f7c8m2r4q7s0s7_poker-table-poker-table-online-transparent-png/
        self.clock = pygame.time.Clock()
        self.own_hand = []
        self.oponents_hand = []
        self.deck = []
        self.running_round = True
        self.cash_money = 100000
        self.betted_money = 0
        self.rounds = 0
        self.cards_on_board = []
        self.go_on = False
        self.card_to_image = dict(zip(test, all_images))
        
        
    def text_to_screen(self, pos_x, pos_y, what_to_render, screen_text):
        font = pygame.font.Font(pygame.font.get_default_font(), 25)
        text = font.render(screen_text + str(what_to_render) + "$", True, OTHER)
        self.screen.blit(text, (pos_x, pos_y))
    
    def handle_board(self):
        if self.go_on: 
            if self.rounds == 0:
                # Add 3 cards
                # self.rounds += 1
                flop = [self.deck.pop() for i in range(3)]
                self.cards_on_board.append(flop)
                
            if self.rounds == 1:
                # Add 1 more card to board
                print(self.deck)
                turn = self.deck.pop(-1)
                self.cards_on_board.append(turn)
            
            if self.rounds == 2:
                # Add last card
                river = self.deck.pop()
                self.cards_on_board.append(river)
            
            if self.rounds > 2:
                # Starting over again
                self.rounds = 0   
                self.cards_on_board = [] 
                self.deck = list(Card)
                random.shuffle(self.deck)
                self.own_hand = [self.deck.pop() for i in range(2)]
                self.oponents_hand = [self.deck.pop() for i in range(2)]
                self.cards_on_board = [self.deck.pop() for i in range(3)]
                
            self.go_on = False
    
    def event_handler(self, running):
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
                    pygame.quit()
                    exit()
                
                elif event.key == pygame.K_a:
                    # All in
                    self.betted_money += self.cash_money
                    self.cash_money = 0
                
                elif event.key == pygame.K_f:
                    # fold
                    self.rounds = 0   
                    self.betted_money = 0
                    self.cards_on_board = [] 
                    self.deck = list(Card)
                    random.shuffle(self.deck)
                    self.own_hand = [self.deck.pop() for i in range(2)]
                    self.cards_on_board = [self.deck.pop() for i in range(3)]
                    
                elif event.key == pygame.K_c:
                    # Call
                    self.rounds += 1
                    
                    
                elif event.key == pygame.K_r:
                    # raise
                    self.cash_money -= 10
                    self.betted_money += 10
                
                elif event.key == pygame.K_q:
                    self.cash_money = 100000
                    self.betted_money = 0
                    self.rounds = 0
                    self.cards_on_board = [self.deck.pop() for i in range(3)]
                    self.deck = list(Card)
                    random.shuffle(self.deck)
                
                elif event.key == pygame.K_d:
                    # For debugging purposes
                    self.rounds += 1
                    self.go_on = True
                    self.cash_money -= 10
                    self.betted_money += 10
                    print("self.deck: ", self.deck)
                    print("self.own_hand: ", self.own_hand)
                    print("self.oponents_hand", self.oponents_hand)
                    print("self.running_round: ", self.running_round)
                    print("self.cash_money: ", self.cash_money)
                    print("self.rounds", self.rounds)
                    print("self.cards_on_board", self.cards_on_board)
                    
                    
            elif event.type == pygame.QUIT:
                run = False
                pygame.quit()
                exit()

    def lost(self):
        if self.cash_money <= 0:
            self.text_to_screen(WIDTH/2, HEIGHT/2, "You lost, press q to start again", "")
        
    def display_cards_on_deck(self):
        if self.cards_on_board:
            first_pos_x = WIDTH/2 - (75*4)
            for i in self.cards_on_board:
                first_pos_x += 75
                self.screen.blit(pygame.transform.scale(pygame.image.load(os.path.join(current_path, self.card_to_image[i])), (75, 100)), [first_pos_x, 130])     
    
    def display_oponents_cards(self):
        if self.oponents_hand:
            first_pos_x = WIDTH/2 - 75
            for i in self.oponents_hand:
                first_pos_x += 75
                self.screen.blit(pygame.transform.scale(pygame.image.load(os.path.join(current_path, self.card_to_image[i])), (75, 100)), [first_pos_x, 0])
    
    def display_my_own_cards(self):
        if self.own_hand:
            first_pos_x = WIDTH/2 - 75
            for i in self.own_hand:
                first_pos_x += 75
                self.screen.blit(pygame.transform.scale(pygame.image.load(os.path.join(current_path, self.card_to_image[i])), (75, 100)), [first_pos_x, HEIGHT - 130])
    
    def start(self):
        running = True
        self.deck = list(Card)
        random.shuffle(self.deck)
        self.own_hand = [self.deck.pop() for i in range(2)]
        self.oponents_hand = [self.deck.pop() for i in range(2)]
        
        # Initial flop
        self.cards_on_board = [self.deck.pop() for i in range(3)]

        while running:
            self.screen.fill(RED)
            self.event_handler(running)
            self.handle_board()
            self.lost()
            self.text_to_screen(WIDTH - 125, HEIGHT - 25, self.cash_money, "")
            self.text_to_screen(0, HEIGHT - 25, self.betted_money, "Money on table: ")
            self.screen.blit(self.background_image, [0, 0])
            self.display_cards_on_deck()
            self.display_my_own_cards()
            self.display_oponents_cards()
            pygame.display.flip()
            self.clock.tick(20)
            pygame.display.update()

        
if __name__ == "__main__":

    from PokerRL._.CrayonWrapper import CrayonWrapper

    n_iterations = 150
    name = "CFR_EXAMPLE"

    # Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
    chief = ChiefBase(t_prof=None)
    crayon = CrayonWrapper(name=name,
                           path_log_storage=None,
                           chief_handle=chief,
                           runs_distributed=False,
                           runs_cluster=False,
                           )
    cfr = VanillaCFR(name=name,
                     game_cls=DiscretizedNLLeduc,
                     agent_bet_set=bet_sets.POT_ONLY,
                     chief_handle=chief)

    for iter_id in range(n_iterations):
        print("Iteration: ", iter_id)
        cfr.iteration()
        crayon.update_from_log_buffer()
        crayon.export_all(iter_nr=iter_id)
        
    Game().start()