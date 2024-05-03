import socketio

class GameConnection:

    def __init__(self, url):
        self.socket = socketio.Client()

        self.socket.on("connect", lambda : print("Connected"))
        self.socket.on("disconnect", lambda : print("Disconnected"))
        self.socket.on("state", self._updateState)

        print(f"Connecting to {url} ... ", end=None)
        self.socket.connect(url)
        self.socket.emit("new player")

        self.state = {}
        self.updated = False

    def _updateState(self, data):
        self.state = data
        # if (self.conn.socket.get_sid() in self.state["playerData"]):
        self.updated = True
    
    def move(self, movement):
        assert len(movement) == 4
        packet = {
            "up": movement[0],
            "down": movement[1],
            "left": movement[2],
            "right": movement[3]
        }
        self.socket.emit("movement", packet)

    def joinGame(self):
        self.socket.emit("join")

    def restartGame(self):
        self.socket.emit("restart")

    def startGame(self, tagger=False):
        if (tagger):
            self.socket.emit("start tagger")
            # print("Started game as tagger")
        else:
            self.socket.emit("start")
            # print("Started game")