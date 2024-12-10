# -*- coding: utf-8 -*-
7
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.
import time
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from test_controller import TestController
from scott_dick_controller import ScottDickController
from graphics_both import GraphicsBoth
from FuzzyController import FuzzyController
import random

def run_test(seed):
    my_test_scenario = Scenario(name='Test Scenario',
                                num_asteroids=20,
                                ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3},
                                {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 2, 'mines_remaining': 3},
                                ],
                                map_size=(1000, 800),
                                seed = random.seed(17),
                                # num_asteroids=20,
                                # ship_states=[
                                # {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 5},
                                # {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 2, 'mines_remaining': 5},
                                # ],
                                # map_size=(1000, 800),
                                # seed = random.seed(1),
                                time_limit=60,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)

    game_settings = {'perf_tracker': True,
                    'graphics_type': GraphicsType.Tkinter,
                    'realtime_multiplier': 1,
                    'graphics_obj': None}
    game = KesslerGame(settings=game_settings) # Use this to visualize the game scenario
    # game = TrainerEnvironment(settings=game_settings) # Use this for max-speed, no-graphics simulation
    pre = time.perf_counter()

    fuzzy_controller = FuzzyController(False)
    score, perf_data = game.run(scenario=my_test_scenario, controllers = [fuzzy_controller, TestController()])
    print(f"Run {seed}")
    print('Scenario eval time: '+ str(time.perf_counter()-pre))
    print(score.stop_reason)
    print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    print('Deaths: ' + str([team.deaths for team in score.teams]))
    print('Accuracy: ' + str([team.accuracy for team in score.teams]))
    print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    print('Evaluated frames: ' + str([controller.eval_frames for controller in score.final_controllers]))
    print()
    return score.teams[0].asteroids_hit

if __name__ == "__main__":
    # run_test(0)
    p1_scores = []
    # p2_scores = []
    for i in range(1):
        p1_score = run_test(i)
        p1_scores.append(p1_score)
        # p2_scores.append(p2_score)

    print(f"Average P1 {sum(p1_scores)/len(p1_scores)}")
    # print(f"Average P2 {sum(p2_scores)/len(p2_scores)}")