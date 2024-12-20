import time
from kesslergame import GraphicsType, KesslerController, Scenario, TrainerEnvironment # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import pygad
from test_controller import TestController

BEST_CHROMOSOME = [-200, -200, -120, -200, -22, 0, -100, 0, 100, 0, 107, 200, 125, 200, 200]
def generate_chromosome():
    """
    Generate a chromosome representing the thrust membership functions for ship_thrust.
    Each membership function is represented by 3 values (start, mid, end).
    We have 5 sets: NM, NS, Z, PS, PM.
    That totals 5*3 = 15 genes.

    After generating these arrays, we'll flatten them into a 1D list of length 15.
    """
    thrust_nm_val = np.random.randint(-150, -99)
    thrust_ns_val = np.random.randint(-50, 1)
    thrust_z_val = np.random.randint(-50, 51)
    thrust_ps_val = np.random.randint(50, 151)
    thrust_pm_val = np.random.randint(100, 151)

    thrust_nm = [-200, -200, thrust_nm_val]
    thrust_ns = [-200, thrust_ns_val, 0]
    thrust_z = [-100, thrust_z_val, 100]
    thrust_ps = [0, thrust_ps_val, 200]
    thrust_pm = [thrust_pm_val, 200, 200]

    chromosome = thrust_nm + thrust_ns + thrust_z + thrust_ps + thrust_pm
    return chromosome

def fitness(ga_instance, solution, solution_idx):
    """
    Runs the Kessler Game scenario using the given chromosome for the FuzzyController,
    and returns a fitness value. The fitness is defined as:
    score.teams[1].asteroids_hit - 50*score.teams[1].deaths

    We want to maximize this fitness. Higher is better.
    """
    chromosome = solution
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
    game = TrainerEnvironment(settings=game_settings)
    _ = time.perf_counter()
    fuzzy_controller = FuzzyController(run_genetics=False, chromosome=chromosome)
    score, _ = game.run(scenario=my_test_scenario, controllers=[TestController(), fuzzy_controller])

    # Fitness: maximize asteroids hit, minimize deaths
    return score.teams[1].asteroids_hit - 50*score.teams[1].deaths

def get_best_chromosome():
    """
    Use pygad to evolve a population of chromosomes and find the best one.
    """
    num_genes = 15
    sol_per_pop = 5
    num_generations = 5
    num_parents_mating = 2

    # Generate initial population
    initial_population = []
    for _ in range(sol_per_pop):
        c = generate_chromosome() # Returns a list of length 15
        initial_population.append(c)
    initial_population = np.array(initial_population)

    ga_instance = pygad.GA(
        fitness_func=fitness,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        initial_population=initial_population,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        mutation_probability=0.1,
        crossover_probability=0.8,
        parent_selection_type="sss", # steady-state selection
        crossover_type="single_point",
        mutation_type="random",
        keep_parents=1
    )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best solution fitness:", solution_fitness)
    print("Best solution:", solution)
    return solution
class FuzzyController(KesslerController):
    def __init__(self, run_genetics, chromosome=None):
        # BEST_CHROMOSOME is defined at top of file
        # To find best chromosome, pass in run_genetics=True and chromosome=None
        # Or to use a specific chromosome, pass in run_genetics=False and chromosome=CHROMOSOME
        if run_genetics:
            if chromosome is None:
                self.chromosome = get_best_chromosome()
            else:
                self.chromosome = chromosome
        else:
            self.chromosome = BEST_CHROMOSOME # Pass in chromosome

        self.targeting_control = self.setup_fuzzy_controller()
        self.escaping_mine_frames = 0  # Tracks the number of frames in escape mode
        self.mine_cooldown_frames = 0  # Tracks cooldown period after dropping a mine
        
    def setup_fuzzy_controller(self):
        self.eval_frames = 0

        # self.chromosome is a 1D array of length 15
        # Extract sets for ship_thrust
        # NM: [0:3], NS: [3:6], Z: [6:9], PS: [9:12], PM: [12:15]
        NM = self.chromosome[0:3]
        NS = self.chromosome[3:6]
        Z =  self.chromosome[6:9]
        PS = self.chromosome[9:12]
        PM = self.chromosome[12:15]

        # Sort each triple before applying trimf
        NM = np.sort(NM)
        NS = np.sort(NS)
        Z = np.sort(Z)
        PS = np.sort(PS)
        PM = np.sort(PM)

        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        ship_thrust = ctrl.Consequent(np.arange(-200, 200, 25), 'ship_thrust')

        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe,[0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)

        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])

        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])

        # Use the chromosome segments for ship_thrust
        ship_thrust['NM'] = fuzz.trimf(ship_thrust.universe, NM)
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, NS)
        ship_thrust['Z'] = fuzz.trimf(ship_thrust.universe, Z)
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, PS)
        ship_thrust['PM'] = fuzz.trimf(ship_thrust.universe, PM)

        asteroid_distance = ctrl.Antecedent(np.arange(0, 1000, 50), 'asteroid_distance')
        asteroid_distance['Close'] = fuzz.trimf(asteroid_distance.universe, [0, 0, 100])
        asteroid_distance['Medium'] = fuzz.trimf(asteroid_distance.universe, [200, 500, 800])
        asteroid_distance['Far'] = fuzz.trimf(asteroid_distance.universe, [600, 1000, 1000])

        mine_danger = ctrl.Antecedent(np.arange(0, 2, 1), 'mine_danger')
        mine_danger['Safe'] = fuzz.trimf(mine_danger.universe, [0, 0, 1])
        mine_danger['Danger'] = fuzz.trimf(mine_danger.universe, [0, 1, 1])

        deploy_mine = ctrl.Consequent(np.arange(-1, 2, 1), 'deploy_mine')
        deploy_mine['Hold'] = fuzz.trimf(deploy_mine.universe, [-1, -1, 0])
        deploy_mine['Deploy'] = fuzz.trimf(deploy_mine.universe, [0, 1, 1])

        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Z']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['Z']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['Z']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_thrust['Z']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['Z']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['Z']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Z']))

        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['PS']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['PS']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['PS']))
        rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_thrust['NS']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['PS']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['PS']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['PS']))

        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Z']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['NS']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['NM']))
        rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_thrust['NM']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['NM']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['NS']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Z']))

        rule_mine_close = ctrl.Rule(asteroid_distance['Close'] & bullet_time['S'], deploy_mine['Deploy'])
        rule_mine_medium = ctrl.Rule(asteroid_distance['Medium'] & bullet_time['L'], deploy_mine['Hold'])
        rule_mine_far = ctrl.Rule(asteroid_distance['Far'], deploy_mine['Hold'])
        rule_22 = ctrl.Rule(mine_danger['Safe'], ship_thrust['Z'])
        rule_23 = ctrl.Rule(mine_danger['Danger'], ship_thrust['NM'])

        targeting_control = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
        rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17,
        rule18, rule19, rule20, rule21, rule_mine_close, rule_mine_medium, rule_mine_far,
        rule_22, rule_23
        ])

        return targeting_control

    def get_mines(self, ship_state, game_state):
        for m in game_state["mines"]:
            ship_mine_dist = math.sqrt((ship_state["position"][0] - m["position"][0])**2 + (ship_state["position"][1] - m["position"][1])**2)
            if (ship_mine_dist <= 175):
                return 1
        return 0

    def find_colliding_asteroids(self, ship_state, game_state):
        ship_pos = np.array(ship_state["position"])
        ship_vel = np.array(ship_state["velocity"])
        collision_count = 0
        asteroids = []
        asteroid_angles = []
        time_collision = []
        
        # Game boundaries
        game_width = 1000
        game_height = 800
        closest_asteroid_distance = float('inf')
        for asteroid in game_state["asteroids"]:
            curr_dist = math.sqrt((ship_state["position"][0] - asteroid["position"][0])**2 + (ship_state["position"][1] - asteroid["position"][1])**2)
            closest_asteroid_distance = min(closest_asteroid_distance, curr_dist)
            
            asteroid_pos = np.array(asteroid["position"])
            asteroid_vel = np.array(asteroid["velocity"])

            # Relative position and velocity
            r = asteroid_pos - ship_pos
            v = asteroid_vel - ship_vel

            # Adjust for wrapping boundaries
            r_wrapped = r.copy()
            if r[0] > game_width / 2:
                r_wrapped[0] -= game_width
            elif r[0] < -game_width / 2:
                r_wrapped[0] += game_width

            if r[1] > game_height / 2:
                r_wrapped[1] -= game_height
            elif r[1] < -game_height / 2:
                r_wrapped[1] += game_height

            # Time of closest approach
            v_norm_sq = np.dot(v, v)
            if v_norm_sq == 0:
                # Static relative motion thus skip this asteroid
                continue
            t_ca = -np.dot(r_wrapped, v) / v_norm_sq

            if t_ca < 0:
                # Asteroid is moving away
                continue

            # Minimum distance
            d_min = np.linalg.norm(r_wrapped + t_ca * v)

            # Check for collision
            if d_min <= asteroid['radius'] + 20:  # +20 compensates for the size of ship
                collision_count += 1

                asteroid_angle = np.arctan2(r_wrapped[1], r_wrapped[0])  # Compute the angle
                asteroid_angle = asteroid_angle * 180 / np.pi
                if asteroid_angle < 0:
                    asteroid_angle = 360 + asteroid_angle
                asteroids.append(asteroid)
                asteroid_angles.append(asteroid_angle)
                time_collision.append(t_ca)
                
        return asteroids, collision_count, asteroid_angles, time_collision, closest_asteroid_distance


    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
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
        
        asteroids, collision_count, asteroid_angles, time_collision, closest_asteroid_distance = self.find_colliding_asteroids(ship_state, game_state)

        if collision_count == 0 or self.escaping_mine_frames > 0:
            asteroids = game_state["asteroids"]


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
        if targ_det < 0:
        # No valid intercept, return default actions
            return 0, 0, False, False
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
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+bullet_t*25)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+bullet_t*25)

        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        mine_danger = self.get_mines(ship_state, game_state)
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.input['asteroid_distance'] = closest_asteroid["dist"]
        shooting.input['mine_danger'] = mine_danger
        
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        deploy_mine_output = shooting.output.get('deploy_mine', 0)
        if self.mine_cooldown_frames > 0:
            self.mine_cooldown_frames -= 1

        if self.escaping_mine_frames > 0:
            self.escaping_mine_frames -= 1

        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
        drop_mine = False

        if self.mine_cooldown_frames == 0 and deploy_mine_output > 0:
            drop_mine = True
            self.mine_cooldown_frames = 900  # Cooldown duration (e.g., 90 frames)
            self.escaping_mine_frames = 30  # Escape duration (e.g., 30 frames)

        # Scale thrust by 5
        thrust = 5 * shooting.output['ship_thrust']
        
        self.eval_frames +=1
        
        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)

        return thrust, turn_rate, fire, drop_mine
        
    @property
    def name(self) -> str:
        return "Fuzzy Controller"