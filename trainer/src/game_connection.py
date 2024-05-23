from src.logger import log
import socketio

class GameConnection:

    def __init__(self, url, gameTag=None):
        self.socket = socketio.Client()

        self.socket.on("connect", lambda : log("Connected"))
        self.socket.on("disconnect", lambda : log("Disconnected"))
        self.socket.on("state", self._updateState)

        log(f"Connecting to {url} ... ", end="")
        self.socket.connect(url)
        
        if gameTag != None:
            self.socket.emit("new player force", gameTag)
        else:
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

    def joinGame(self, gameTag=None):
        if gameTag:
            log("Force join ", gameTag)
            self.socket.emit("force join", gameTag)
        else:
            self.socket.emit("join")

    def restartGame(self):
        self.socket.emit("restart")

    def startGame(self, tagger=False):
        if (tagger):
            self.socket.emit("start tagger")
            # log("Started game as tagger")
        else:
            self.socket.emit("start")
            # log("Started game")