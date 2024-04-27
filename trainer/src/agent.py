from src.game_connection import GameConnection

FRAME_LAG = 2
class Agent:
    def __init__(self, gameUrl):
        self.conn = GameConnection(gameUrl)
        self.reset()
        
    def run(self):

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

            if (len(self.frameLog) > FRAME_LAG):
                data = self.frameLog.pop(0)
                agentData = data[0]
                playersData = data[1]
                powerUpsData = data[2]

                self.isRed = agentData['color'] == 'red'
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