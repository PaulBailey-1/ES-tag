from src.game_connection import GameConnection

class Agent:
    def __init__(self, gameUrl, config=None):
        self.frameLag = 2
        if config:
            self.frameLag = config['frameLag']
        self.conn = GameConnection(gameUrl)
        self.reset()
        
    def run(self, forceRed=None):

        if self.conn.updated:

            self.conn.updated = False

            agentData  = self.conn.state["playerData"][self.conn.socket.get_sid()]
            playersData = self.conn.state["playerData"]
            playersData.pop(self.conn.socket.get_sid())
            powerUpsData = self.conn.state["powerUpData"]

            if (self.ignoreCount > 0):
                self.ignoreCount -= 1
            else:
                self.frameLog.append((agentData, playersData, powerUpsData))

            if (len(self.frameLog) > self.frameLag):
                data = self.frameLog.pop(0)
                agentData = data[0]
                playersData = data[1]
                powerUpsData = data[2]

                self.x = agentData['x']
                self.y = agentData['y']

                self.isRed = agentData['color'] == 'red'
                if forceRed:
                    self.isRed = forceRed
                self.score = agentData['score']

                action = self.policy(agentData, playersData, powerUpsData)
                if (action):
                    self.conn.move(action)

    def policy(self, agentData, playersData, powerUpsData):
        return None
    
    def reset(self):

        self.frameLog = []
        self.ignoreCount = 3

        self.isRed = True
        self.score = -1

        self.x = 0
        self.y = 0