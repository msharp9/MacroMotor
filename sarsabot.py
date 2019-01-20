# C:\Users\Matt\AppData\Roaming\Python\Python36\site-packages\sc2

import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY,\
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY,\
    ZEALOT
from sc2.ids.buff_id import BuffId
from sc2.ids.ability_id import AbilityId

from examples.protoss.cannon_rush import CannonRushBot
from examples.protoss.warpgate_push import WarpGateBot
from examples.zerg.zerg_rush import ZergRushBot

import math
import random
import time
import os

import cv2
import numpy as np
import pandas as pd
# import keras
# import tensorflow as tf

HEADLESS = True

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.InteractiveSession(config=config)

# print(Difficulty)
print(dir(Difficulty))

class SarsaBot(sc2.BotAI):
    def __init__(self, action_space=None, learning_rate=0.01, reward_decay=0.9,
        e_greedy=0.9, model=None, title=1, record="sarsa-record.txt"):
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.model = model
        self.title = title
        self.record = record
        self.scouts_and_spots = {}

        self.probes_per_nexus = random.randint(12,20)
        self.supply_ratio = random.uniform(0,1)
        self.critical_army_size = random.randint(50,190)

        # ADDED THE CHOICES #
        self.choices = {0: self.build_scout,
                        1: self.build_zealot,
                        2: self.build_gateway,
                        3: self.build_voidray,
                        4: self.build_stalker,
                        5: self.build_worker,
                        6: self.build_assimilator,
                        7: self.build_stargate,
                        8: self.build_pylon,
                        9: self.defend_nexus,
                        10: self.defend_main,
                        11: self.attack_known_enemy_structure,
                        12: self.expand,
                        13: self.do_nothing,
                        14: self.group_up,
                        }
        if action_space:
            self.actions = action_space
        else:
            self.actions = [str(c) for c in self.choices]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.observation = None
        self.action = None
        self.reward = 0

        self.friendly_units = set()
        self.friendly_buildings = set()
        self.enemy_units = set()
        self.enemy_buildings = set()

        if self.model and os.path.isfile(self.model):
            self.q_table = pd.read_csv(self.model, index_col=0)
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # Random action
            action = np.random.choice(self.actions)

            # Weighted Random action
            # expand_weight = 1
            # worker_weight = 1
            # if len(self.units(NEXUS))*self.probes_per_nexus > len(self.units(PROBE)):
            #     worker_weight = 10
            # else:
            #     expand_weight = 10
            #
            # if self.supply_left/self.supply_cap < self.supply_ratio:
            #     pylon_weight = 10
            # else:
            #     pylon_weight = 1
            # stargate_weight = 1
            # gateway_weight = 1
            #
            # zealot_weight = 1
            # voidray_weight = 1
            # stalker_weight = 1
            #
            # defend_weight = 1
            # attack_weight = 1
            # if(self.units(NEXUS)):
            #     if self.known_enemy_units.closer_than(20, random.choice(self.units(NEXUS))):
            #         defend_weight = 2
            #     if self.supply_used > self.critical_army_size:
            #         attack_weight = 20
            #     else:
            #         defend_weight = defend_weight * 10
            #         zealot_weight = 5
            #         stalker_weight = 20
            #         voidray_weight = 30
            #
            # choice_weights = 1*[0]+zealot_weight*[1]+gateway_weight*[2]+voidray_weight*[3]+stalker_weight*[4]+worker_weight*[5]+1*[6]+stargate_weight*[7]+pylon_weight*[8]+defend_weight*[9]+1*[10]+attack_weight*[11]+expand_weight*[12]+1*[13]+1*[14]
            # # print(choice_weights)
            # choice = random.choice(choice_weights)
            # action = str(choice)
        return action

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, str(a)]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, str(a_)]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, str(a)] += self.lr * (q_target - q_predict)  # update

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.model)

        if(game_result == Result.Defeat):
            self.learn(self.observation,self.action,-10000,'terminal',0)
        else:
            self.learn(self.observation,self.action,10000,'terminal',0)

        with open(self.record,"a") as f:
            f.write("Model {} - Result {} - Time {}\n".format(self.model, game_result, int(time.time())))

        if self.model:
            self.q_table.to_csv(self.model)
        else:
            self.q_table.to_csv('model/sarsamodel.csv')

    async def on_unit_destroyed(self, unit_tag):
        if unit_tag in self.enemy_buildings:
            self.reward += 100
        if unit_tag in self.enemy_units:
            self.reward += 1
        if unit_tag in self.friendly_buildings:
            self.reward -= 100
        if unit_tag in self.friendly_units:
            self.reward -= 1

    async def log_enemy_units(self):
        for unit in self.known_enemy_units:
            if unit.is_structure:
                self.enemy_buildings.add(unit.tag)
            else:
                self.enemy_units.add(unit.tag)

        for unit in self.units():
            if unit.is_structure:
                self.friendly_buildings.add(unit.tag)
            else:
                self.friendly_units.add(unit.tag)

    async def on_step(self, iteration):
        # self.time = (self.state.game_loop/22.4) / 60
        #print('Time:',self.time)
        await self.distribute_workers()
        await self.scout()
        await self.log_enemy_units()
        await self.chronoboost()
        _observation = self.intel()
        _action = self.choose_action(_observation)
        if self.action is not None:
            self.learn(self.observation, self.action, self.reward, _observation, _action)
            self.reward = 0
        self.observation = _observation
        self.action = _action
        try:
            await self.choices[int(self.action)]()
        except Exception as e:
            # print(str(e))
            pass

    def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit in self.units():
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius), (255, 255, 255), math.ceil(int(unit.radius*0.5)))

        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            population_ratio = self.supply_left / (self.supply_cap + 0.0001)
            if population_ratio > 1.0:
                population_ratio = 1.0

            plausible_supply = self.supply_cap / 200.0

            worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            # print(str(e))
            pass

        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow(str(self.title), resized)
            cv2.waitKey(1)

        resized = cv2.resize(self.flipped, dsize=None, fx=0.1, fy=0.1)
        return ''.join([str(_) for _ in resized.flatten()])

    async def chronoboost(self):
        for nexus in self.units(NEXUS):
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not nexus.noqueue:
                abilities = await self.get_available_abilities(nexus)
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        #  FIXED THIS
        x += random.randrange(-15,15)
        y += random.randrange(-15,15)

        if x < 0:
            print("x below")
            x = 0
        if y < 0:
            print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            print("y above")
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))

        return go_to

    async def scout(self):
        self.expand_dis_dir = {}

        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            #print(distance_to_enemy_start)
            self.expand_dis_dir[distance_to_enemy_start] = el

        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        # removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0 and self.time < 160:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = self.expand_dis_dir[dist] #next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue

                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            # print(len(self.units(OBSERVER)), self.time/3)
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
                break
        if len(self.units(ROBOTICSFACILITY)) == 0:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)

    async def build_worker(self):
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            if self.can_afford(PROBE):
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready
        if gateways.exists:
            if self.can_afford(ZEALOT):
                gateway = random.choice(gateways)
                if gateway.noqueue:
                    await self.do(gateway.train(ZEALOT))

    async def build_gateway(self):
        pylon = self.units(PYLON).ready.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon)

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                stargate = random.choice(stargates)
                if stargate.noqueue:
                    await self.do(stargate.train(VOIDRAY))

    async def build_stalker(self):
        pylon = self.units(PYLON).ready.random
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                gateway = random.choice(gateways)
                if gateway.noqueue:
                    await self.do(gateway.train(STALKER))

        if not cybernetics_cores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def build_stargate(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

    async def build_pylon(self):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

    async def expand(self):
        try:
            if self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.time + wait

    async def defend_nexus(self):
        if len(self.known_enemy_units) > 0:
            close_enemy_units = self.known_enemy_units.closer_than(20, random.choice(self.units(NEXUS)))
            target = close_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            target = target.position
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.attack(target))
            for u in self.units(OBSERVER).idle if u.tag not in self.scouts_and_spots:
                await self.do(u.attack(target))

    async def defend_main(self):
            # print(dir(self.main_base_ramp))
            target = self.main_base_ramp.barracks_correct_placement
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.attack(target))
            for u in self.units(OBSERVER).idle if u.tag not in self.scouts_and_spots:
                await self.do(u.attack(target))

    async def group_up(self):
            target = random.choice(self.units({VOIDRAY, STALKER, ZEALOT}))
            target = target.position
            for u in self.units(VOIDRAY):
                await self.do(u.attack(target))
            for u in self.units(STALKER):
                await self.do(u.attack(target))
            for u in self.units(ZEALOT):
                await self.do(u.attack(target))
            for u in self.units(OBSERVER).idle if u.tag not in self.scouts_and_spots:
                await self.do(u.attack(target))

    async def attack_known_enemy_structure(self):
        if len(self.known_enemy_structures) > 0:
            target = random.choice(self.known_enemy_structures)
            target = target.position
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.attack(target))
            for u in self.units(OBSERVER).idle if u.tag not in self.scouts_and_spots:
                await self.do(u.attack(target))

if __name__ == "__main__":
    for episode in range(10):
        print('Episode: '+str(episode))
        run_game(maps.get("AbyssalReefLE"), [
            Bot(Race.Protoss, SarsaBot(model='model/sarsamodel_medium.csv',
                record="sarsa-record-vs-easy.txt",
                learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, title=1)),
            # Bot(Race.Protoss, SarsaBot(model='model/sarsamodel.csv', learning_rate=0.01, e_greedy=0.9*(episode/100), title=1)),
            # Bot(Race.Protoss, SarsaBot(model=None, title=2)),
            # Human(Race.Terran),
            Computer(Race.Protoss, Difficulty.Easy),
            # Computer(Race.Protoss, Difficulty.Medium),
            # Bot(Race.Protoss, WarpGateBot()),
            # Bot(Race.Zerg, ZergRushBot()),
            ], realtime=False)
        time.sleep(5)
    time.sleep(10)
