import socketio

class GameConnection:

    def __init__(self, url):
        self.socket = socketio.Client()

        self.socket.on("connect", lambda : print("Connected"))
        self.socket.on("disconnect", lambda : print("Disconnected"))
        self.socket.on("state", self._updateState)

        self.socket.connect(url)
        self.socket.emit("new player")

        self.state = {}
        self.updated = False

    def _updateState(self, data):
        self.state = data
        # if (self.conn.socket.get_sid() in self.state["playerData"]):
        self.updated = True
    
    def move(self, movement):
        packet = {
            "up": movement[0],
            "down": movement[1],
            "left": movement[2],
            "right": movement[3]
        }
        self.socket.emit("movement", packet)