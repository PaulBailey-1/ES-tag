from src.agent import Agent

class SimpleAgent(Agent):
        
    def policy(self, agentData, playersData, powerUpsData):

        left = False
        right = False
        up = False
        down = False

        if len(playersData) > 0:

            if (len(powerUpsData) > 0):

                powerUpData = [0]
                minDist = 10000
                for powerUp in powerUpsData:
                    dist = abs(powerUp['x'] - agentData['x'])
                    if dist < minDist:
                        minDist = dist
                        powerUpData = powerUp

                if powerUpData['x'] + 10 < agentData['x']:
                    left = True
                elif powerUpData['x'] > agentData['x'] + 10:
                    right = True
                down = True
            
            elif (agentData['color'] == 'red'):
                evaderData = next(iter(playersData.values()))

                if evaderData['x'] + 30 < agentData['x']:
                    left = True
                elif evaderData['x'] > agentData['x'] + 30:
                    right = True
                if evaderData['y'] + 30 < agentData['y']:
                    up = True
                elif evaderData['y'] > agentData['y'] + 30:
                    down = True

            elif (agentData['color'] == 'green'):
                taggerData = {}
                for player in playersData:
                    if playersData[player]['color'] == 'red':
                        taggerData = playersData[player]

                if (len(taggerData) > 0):
                    if taggerData['x'] > agentData['x']:
                        left = True
                    elif taggerData['x'] < agentData['x']:
                        right = True
                    if taggerData['y'] > agentData['y']:
                        up = True
                    elif taggerData['y'] < agentData['y']:
                        down = True
                    up = True
        
            self.conn.move([up, down, left, right])