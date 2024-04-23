from src.game_connection import GameConnection

FRAME_LAG = 2
class Agent:
    def __init__(self, gameUrl):
        self.conn = GameConnection(gameUrl)
        self.frameLog = []
        
    def run(self):

        if self.conn.updated:

            self.conn.updated = False

            agentData  = self.conn.state["playerData"][self.conn.socket.get_sid()]
            playersData = self.conn.state["playerData"]
            playersData.pop(self.conn.socket.get_sid())
            powerUpsData = self.conn.state["powerUpData"]

            self.frameLog.append((agentData, playersData, powerUpsData))
            if (len(self.frameLog) > FRAME_LAG):
                data = self.frameLog.pop(0)
                agentData = data[0]
                playersData = data[1]
                powerUpsData = data[2]

                action = self.policy(agentData, playersData, powerUpsData)
                if (action):
                    self.conn.move(action)

    def policy(self, agentData, playersData, powerUpsData):
        return None
