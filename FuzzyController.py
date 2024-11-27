import time
from kesslergame import GraphicsType, KesslerController, Scenario, TrainerEnvironment # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
# from EasyGA import *
from test_controller import TestController

class FuzzyController(KesslerController):
    def __init__(self, run_genetics, chromosome):
        if run_genetics:
            if chromosome is None:
                self.chromosome = get_best_chromosome().gene_value_list[0]
            else:
                self.chromosome = chromosome.gene_value_list[0]
        else:
            self.chromosome = chromosome # Pass in chromosome

        self.targeting_control, self.thrust_control = self.setup_fuzzy_controller()

        
    def setup_fuzzy_controller(self):
        self.eval_frames = 0 #What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'])) 
        
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
     
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        targeting_control = ctrl.ControlSystem()
        targeting_control.addrule(rule1)
        targeting_control.addrule(rule2)
        targeting_control.addrule(rule3)
        # targeting_control.addrule(rule4)
        targeting_control.addrule(rule5)
        targeting_control.addrule(rule6)
        targeting_control.addrule(rule7)
        targeting_control.addrule(rule8)
        targeting_control.addrule(rule9)
        targeting_control.addrule(rule10)
        # targeting_control.addrule(rule11)
        targeting_control.addrule(rule12)
        targeting_control.addrule(rule13)
        targeting_control.addrule(rule14)
        targeting_control.addrule(rule15)
        targeting_control.addrule(rule16)
        targeting_control.addrule(rule17)
        # targeting_control.addrule(rule18)
        targeting_control.addrule(rule19)
        targeting_control.addrule(rule20)
        targeting_control.addrule(rule21)

        # Create fuzzy variables
        num_asteroid_approach = ctrl.Antecedent(np.arange(0, 11, 1), 'num_asteroid_approach')
        time_to_collision = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'time_to_collision')
        angle_to_safe_direction = ctrl.Antecedent(np.arange(0, 181, 1), 'angle_to_safe_direction')

        ship_thrust = ctrl.Consequent(np.arange(-4, 5, 1), 'ship_thrust')
        ship_turn = ctrl.Consequent(np.arange(-90, 91, 1), 'ship_turn')

        # Define membership functions for inputs
        num_asteroid_approach['low'] = fuzz.trimf(num_asteroid_approach.universe, [0, 0, 3])
        num_asteroid_approach['medium'] = fuzz.trimf(num_asteroid_approach.universe, [2, 5, 8])
        num_asteroid_approach['high'] = fuzz.trimf(num_asteroid_approach.universe, [7, 10, 10])

        time_to_collision['far'] = fuzz.trimf(time_to_collision.universe, [5, 10, 10])
        time_to_collision['medium'] = fuzz.trimf(time_to_collision.universe, [2, 5, 8])
        time_to_collision['near'] = fuzz.trimf(time_to_collision.universe, [0, 0, 3])

        angle_to_safe_direction['small'] = fuzz.trimf(angle_to_safe_direction.universe, [0, 0, 45])
        angle_to_safe_direction['moderate'] = fuzz.trimf(angle_to_safe_direction.universe, [30, 90, 150])
        angle_to_safe_direction['large'] = fuzz.trimf(angle_to_safe_direction.universe, [135, 180, 180])

        # Define membership functions for outputs
        ship_thrust['reverse'] = fuzz.trimf(ship_thrust.universe, [-4, -4, -2])
        ship_thrust['stop'] = fuzz.trimf(ship_thrust.universe, [-1, 0, 1])
        ship_thrust['forward'] = fuzz.trimf(ship_thrust.universe, [2, 4, 4])

        ship_turn['left_hard'] = fuzz.trimf(ship_turn.universe, [-90, -90, -45])
        ship_turn['left'] = fuzz.trimf(ship_turn.universe, [-45, -30, 0])
        ship_turn['none'] = fuzz.trimf(ship_turn.universe, [-5, 0, 5])
        ship_turn['right'] = fuzz.trimf(ship_turn.universe, [0, 30, 45])
        ship_turn['right_hard'] = fuzz.trimf(ship_turn.universe, [45, 90, 90])

        # Define fuzzy rules
        rules = [
            ctrl.Rule(num_asteroid_approach['high'] & time_to_collision['near'], ship_thrust['reverse']),
            ctrl.Rule(num_asteroid_approach['medium'] & time_to_collision['medium'], ship_thrust['stop']),
            ctrl.Rule(num_asteroid_approach['low'] & time_to_collision['far'], ship_thrust['forward']),
            ctrl.Rule(angle_to_safe_direction['small'], ship_turn['none']),
            ctrl.Rule(angle_to_safe_direction['moderate'], ship_turn['left']),
            ctrl.Rule(angle_to_safe_direction['large'], ship_turn['left_hard'])
        ]

        # Create the control system
        thrust_control = ctrl.ControlSystem(rules)

        return targeting_control, thrust_control

    def count_colliding_asteroids(self, ship_state, game_state):
        ship_pos = np.array(ship_state["position"])
        ship_vel = np.array(ship_state["velocity"])
        collision_count = 0
        asteroids = []
        asteroid_angles = []
        
        for asteroid in game_state["asteroids"]:
            asteroid_pos = np.array(asteroid["position"])
            asteroid_vel = np.array(asteroid["velocity"])

            # Relative position and velocity
            r = asteroid_pos - ship_pos
            v = asteroid_vel - ship_vel

            # Time of closest approach
            v_norm_sq = np.dot(v, v)
            if v_norm_sq == 0:
                # Static relative motion thus skip this asteroid
                continue
            t_ca = -np.dot(r, v) / v_norm_sq

            if t_ca < 0:
                # Asteroid is moving away
                continue

            # Minimum distance
            d_min = np.linalg.norm(r + t_ca * v)

            # Check for collision
            if d_min <= asteroid['radius']+20:
                collision_count += 1

                asteroid_angle = np.arctan2(r[1], r[0])  # Compute the angle
                asteroids.append(asteroid)
                asteroid_angles.append(asteroid_angle*180/np.pi)
        return asteroids, collision_count, asteroid_angles

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        
        asteroids, collision_count, asteroid_angles = self.count_colliding_asteroids(ship_state, game_state)
        if collision_count == 0:
            asteroids = game_state["asteroids"]
        else:
            print(asteroid_angles)
        for a in asteroids:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)

        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t + bullet_t*15 + 1/8)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t + bullet_t*15 + 1/8)

        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
        
        drop_mine = False
        thrust = 0
        self.eval_frames +=1
        
        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire, drop_mine
        
    @property
    def name(self) -> str:
        return "Fuzzy Controller"

def fitness(chromosome):
    my_test_scenario = Scenario(name='Test Scenario',
                        num_asteroids=10,
                        ship_states=[
                        {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1},
                        {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 2},
                        ],
                        map_size=(1000, 800),
                        time_limit=60,
                        ammo_limit_multiplier=0,
                        stop_if_no_ammo=False)
    game_settings = {'perf_tracker': True,
                    'graphics_type': GraphicsType.Tkinter,
                    'realtime_multiplier': 1,
                    'graphics_obj': None}
    game = TrainerEnvironment(settings=game_settings) # Use this for max-speed, no-graphics simulation
    _ = time.perf_counter()
    fuzzy_controller = FuzzyController(True, chromosome)
    score, _ = game.run(scenario=my_test_scenario, controllers = [TestController(), fuzzy_controller])
    
    return score.teams[1].asteroids_hit - 50*score.teams[1].deaths
        
def generate_chromosome():
    # Thrust MF
    thrust_nm = np.random.randint(-150, -99)
    thrust_ns = np.random.randint(-50, 1)
    thrust_z = np.random.randint(-50, 51)
    thrust_ps = np.random.randint(50, 151)
    thrust_pm = np.random.randint(100, 151)
    thrust_nm = [-200, -200, thrust_nm]
    thrust_ns = [-200, thrust_ns, 0]
    thrust_z = [-100, thrust_z, 100]
    thrust_ps = [0, thrust_ps, 200]
    thrust_pm = [thrust_pm, 200, 200]

    chromosome = [
        thrust_nm,
        thrust_ns,
        thrust_z,
        thrust_ps,
        thrust_pm,
    ]
    return chromosome

def get_best_chromosome():        
    ga = EasyGA.GA()
    ga.gene_impl = generate_chromosome
    ga.chromosome_length = 1
    ga.population_size = 5
    ga.target_fitness_type = 'max'
    ga.generation_goal = 1
    ga.fitness_function_impl = fitness
    ga.evolve()
    ga.print_best_chromosome()

    return ga.population[0]