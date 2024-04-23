from src.game_connection import GameConnection

class TaggerBot:
    def __init__(self):
        self.conn = GameConnection("http://localhost:5000")

        self.frameLog = []
        
    def run(self):

        if self.conn.updated:

            self.conn.updated = False

            agentData  = self.conn.state["playerData"][self.conn.socket.get_sid()]

            playersData = self.conn.state["playerData"]
            playersData.pop(self.conn.socket.get_sid())

            left = False
            right = False
            up = False
            down = False

            if len(playersData) > 0:
                evaderData = next(iter(playersData.values()))

                self.frameLog.append(evaderData)
                if (len(self.frameLog) > 1):
                    evaderData = self.frameLog.pop(0)

                    if evaderData['x'] + 30 < agentData['x']:
                        left = True
                    elif evaderData['x'] > agentData['x'] + 30:
                        right = True
                    if evaderData['y'] + 30 < agentData['y']:
                        up = True
                    elif evaderData['y'] > agentData['y'] + 30:
                        down = True
            
                self.conn.move([up, down, left, right])
