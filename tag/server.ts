import { isNull, last } from "lodash";
import { Socket } from "socket.io";

// Dependencies
let express = require('express');
let http = require('http');
let path = require('path');
let socketIO = require('socket.io');
let _ = require('lodash');
let app = express();
let server = http.Server(app);
let io = socketIO(server);

let port = 5000
if (process.argv.length > 2) {
    port = parseInt(process.argv[2]);
}

app.set('port', port);
app.use('/static', express.static(__dirname + '/static'));

// Routing
app.get('/', function (request, response) {
    response.sendFile(path.join(__dirname, '/static/index.html'));
});
let lastWatchTag = 0;
app.get('/view', function (request, response) {
    if ('tag' in request.query) {
        lastWatchTag = request.query.tag;
    } else {
        lastWatchTag = -1;
    }
    response.sendFile(path.join(__dirname, '/static/view.html'));
});

interface ClientData {
    up: boolean,
    down: boolean,
    left: boolean,
    right: boolean,
}

interface Connection {
    socket: Socket,
    game: number
}

const width = 1000;
const height = 500;

const playerXSpeed = 8 / 17;
const playerYSpeed = 12 / 17;

const gravity = 0.03 / 17.0;
const initScore = 60;
const powerUpTime = 25;
const powerUpValue = 5;

const playerSize = 30;
const platformWidth = 300;
const platformHeight = 5;
const powerUpSize = 20;

const platforms = [];

let games: Game[] = [];
let connections: Connection[] = [];

function joinGame(socketId: string, watch: boolean = false, gameTag = -1) {

    if (games[connections[socketId].game] !== undefined && games[connections[socketId].game].players[socketId] !== undefined) {
        games[connections[socketId].game].removePlayer(socketId);
        if (games[connections[socketId].game].playerCount == 0) {
            games[connections[socketId].game].restart();
        }
    }
    let gameId  = -1;
    let highestPlayerCount = -1;
    for (let i in games) {
        if (!games[i].running) {
            if (games[i].playerCount > highestPlayerCount) {
                gameId = games[i].id;
                highestPlayerCount = games[i].playerCount;
            }
        }
    }
    if (watch) {
        gameId = 0;
    }
    if (gameTag != -1) {
        console.log("Connecting to tagged game: " + gameTag);
        gameId = -1;
        for (let i = 0; i < games.length; i++) {
            if (games[i].tag == gameTag) {
                gameId = i;
            }
        }
    }
    if (gameId === -1) {
        gameId = games.length;
        games.push(new Game(games.length));
        games[gameId].run();
        if (gameTag != -1) {
            games[gameId].tag = gameTag;
            console.log("Created tagged game: " + gameTag);
        }
    }
    connections[socketId].game = gameId;
    if (!watch) {
        games[gameId].addPlayer(socketId);
        console.log('Player ' + socketId + ' connected to game ' + gameId);
    } else {
        console.log('Watcher connected to game ' + gameId);
    }
    connections[socketId].socket.emit('new game');
    connections[socketId].socket.join(String(gameId));
}

// Starts the server.
server.listen(port, function () {
    console.log('Starting server on port ' + port.toString());
});

io.on('connection', function (socket: Socket) {

    socket.on('new player', function () {
        connections[socket.id] = {
            socket: socket
        };
        joinGame(socket.id);
    });

    socket.on('new player force', function (gameTag) {
        connections[socket.id] = {
            socket: socket
        };
        joinGame(socket.id, false, gameTag);
    });
    
    socket.on('new watcher', function() {
        connections[socket.id] = {
            socket: socket
        };
        joinGame(socket.id, true, lastWatchTag);
    });

    socket.on('start', function () {
        games[connections[socket.id].game].startGame();
    });

    socket.on('start tagger', function () {
        games[connections[socket.id].game].startGame(socket.id);
    });

    socket.on('restart', function () {
        console.log("Restarting game " + connections[socket.id].game);
        games[connections[socket.id].game].restart();
    });

    socket.on('join', function () {
        socket.leave(String(connections[socket.id].game));
        joinGame(socket.id);
    });

    socket.on('force join', function (gameTag) {
        socket.leave(String(connections[socket.id].game));
        joinGame(socket.id, false, gameTag);
    });

    socket.on('movement', function (data: ClientData) {
        if (connections[socket.id] !== undefined) {
            games[connections[socket.id].game].movePlayer(socket.id, data);
        }
    });

    socket.on('disconnect', function () {
        if (connections[socket.id] !== undefined) {
            if (games[connections[socket.id].game] != null) {
                games[connections[socket.id].game].removePlayer(socket.id);
            }
            if (games[connections[socket.id].game].playerCount == 0) {
                games[connections[socket.id].game].restart();
            }
            console.log('Player Disconnected');
        }
    });
});

class Game {

    public players = [];
    private powerUps = [];

    public id: number;
    public tag: number;
    public running: boolean;
    public runTime: number;
    
    public playerCount: number;
    private lastScoreTime: number;
    private powerUpCounter: number;
    private lastUpdateTime: number;
    private startTime: number;

    constructor(id: number) {
        this.id = id;
        this.tag = -1;
        this.running = false;
        this.playerCount = 0;
        this.lastScoreTime = 0.0;
        this.powerUpCounter = powerUpTime;
        this.lastUpdateTime = 0.0;
    }

    addPlayer(id: string) {
        this.playerCount++;
        let highest = 0;
        for (let id in this.players) {
            if (this.players[id].num > highest) {
                highest = this.players[id].num;
            }
        }
        this.players[id] = new Player(highest + 1, id);
    }

    removePlayer(id: string) {
        if (this.players[id] != undefined && this.players[id].status) {
            this.players[id].dead = true;
            this.updateTagger();
        }
        delete this.players[id];
        this.playerCount--;
    }

    movePlayer(id: string, data: ClientData) {
        let player: Player = this.players[id];
        if (player !== undefined) {
            player.move(data)
        }
    }

    run() {
        let time = (new Date()).getTime();
        this.lastScoreTime = time;
        this.lastUpdateTime = time;
        setInterval((game: Game) => {

            let time = (new Date()).getTime();
            let dt = time - this.lastUpdateTime;
            this.lastUpdateTime = time;

            game.update(dt);

            io.to(String(this.id)).emit('state', game.getData());
        }, 1000 / 60, this);

    }

    getData() {
        let data = { runTime: this.runTime, playerData: {}, powerUpData: [], platformData: []};
        for (let id in this.players) {
            let player: Player = this.players[id];
            data.playerData[id] = player.toData();
        }
        for (let i = 0; i < this.powerUps.length; i++) {
            data.powerUpData[i] = this.powerUps[i].toData();
        }

        for (let i = 0; i < platforms.length; i++) {
            data.platformData[i] = platforms[i].toData();
        }
        return data;
    }

    updateTagger() {
        let playersAlive = 0;
        for (let id2 in this.players) {
            if (!this.players[id2].dead) {
                playersAlive++;
            }
        }
        let nextIt = _.random(0, playersAlive - 1)
        let count = 0;
        for (let id in this.players) {
            if (!this.players[id].dead) {
                if (count === nextIt) {
                    this.players[id].status = true;
                    break;
                }
                count++;
            }
        }

        let i = 0
        let winner: Player;
        for (let id2 in this.players) {
            if (!this.players[id2].dead) {
                i++;
                winner = this.players[id2];
            }
        }
        if (i == 1) {
            winner.win = true;
            setTimeout(() => this.restart(), 3000);
        }
    }
    
    startGame(taggerId = null) {
        this.running = true;
        this.startTime = (new Date()).getTime();
        if (taggerId == null) {
            this.updateTagger();
        } else {
            this.players[taggerId].status = true;
        }
        console.log("Started game " + this.id);
    }

    update(dt: number) {
        let currentTime = (new Date()).getTime();
        this.runTime = currentTime - this.startTime;

        for (let id in this.players) {
            this.players[id].update(dt);
        }

        if (this.running) {
            for (let i in this.powerUps) {
                this.powerUps[i].update(dt);
                if (this.powerUps[i].y >= height) {
                    this.powerUps.splice(Number(i), 1);
                }
            }
        
            // Tag check
            for (let firstId in this.players) {
                for (let secondId in this.players) {
                    if (firstId != secondId) {
                        let player1: Player = this.players[firstId];
                        let player2: Player = this.players[secondId];
        
                        if (player1.dead || player2.dead) {
                            continue;
                        }
        
                        player1.tag(player2);
                    }
                }
            }
    
            // Powerup check
            for (let i in this.players) {
                for (let j in this.powerUps) {
                    if (!this.players[i].dead && this.players[i].collide(this.powerUps[j])) {
                        this.players[i].score += powerUpValue;
                        this.powerUps.splice(Number(j), 1);
                    }
                }
            }
        
            // Scoring
            let timeDifference = currentTime - this.lastScoreTime;
            if (timeDifference >= 1000) {
                for (let id in this.players) {
                    let player = this.players[id];
                    if (player.scoring(this.playerCount)) {
                        this.updateTagger();
                    }
                }
                
                // Power ups
                this.powerUpCounter--;
                if (_.random(0, this.powerUpCounter) == 0) {
                    this.powerUps.push(new PowerUp());
                    this.powerUpCounter = powerUpTime;
                }
    
                this.lastScoreTime = currentTime;
            }
        }
    }

    restart() {
        
        let playerIds = [];
        for (let i in this.players) {
            playerIds[i] = this.players[i].id;
        }
        this.players = [];
        this.playerCount = 0;
        this.running = false;
        this.powerUps = [];
        this.powerUpCounter = powerUpTime;
        
        for (let id in playerIds) {
            if (this.tag === -1) {
                connections[id].socket.leave(String(this.id));
                joinGame(id);
            } else {
                this.addPlayer(id);
                connections[id].socket.emit('new game');
            }
        }
    }
}

class Graphic {
    public x: number;
    public y: number;
    public width: number;
    public height: number;

    toData() {
        return {
            x: this.x,
            y: this.y,
        }
    }
}

class Player extends Graphic {

    public id;

    private xspeed: number;
    private yspeed: number;
    private num: number;
    private score: number;

    private collision: boolean;
    private grounded: boolean;
    private platformDown: boolean;
    private wall: boolean;
    private wallTimeout: boolean;
    public dead: boolean;
    public win: boolean;
    private status: boolean;

    constructor(num, id) {
        super();
        this.id = id;
        this.x = _.random(100, 900),
        this.y = 500 - playerSize;
        this.xspeed = 0;
        this.yspeed = 0;
        this.width = playerSize;
        this.height = playerSize;
        this.num = num;
        this.collision = false;
        this.grounded = true;
        this.platformDown = false;
        this.wall = false;
        this.wallTimeout = false;
        this.score = initScore;
        this.dead = false;
        this.win = false
        this.status = false;
    }
    

    update(dt: number) {

        // Boundries
        if (this.y > 500 - this.height) {
            this.yspeed = 0;
            this.y = 500 - this.height;
        }
        if (this.y < 0) {
            this.yspeed = 0;
            this.y = 0;
        }
        if (this.x > 1000 - this.width) {
            this.xspeed = 0;
            this.x = 1000 - this.width;
        }
        if (this.x < 0) {
            this.xspeed = 0;
            this.x = 0;
        }


        if ((this.x == 0 || this.x == 1000 - this.width) && !this.wallTimeout) {
            this.wall = true;
        } else {
            this.wall = false;
        }

        // gravity
        if (!this.grounded) {
            this.yspeed += gravity * dt;
        }

        this.x += this.xspeed * dt;
        this.y += this.yspeed * dt;


        // grounded
        this.grounded = false;
        if (this.y >= 500 - this.height) {
            this.grounded = true;
            this.wallTimeout = false;
        }
        platforms.forEach((platform, i) => {
            let left1 = this.x;
            let right1 = this.x + (this.width);
            // let top1 = this.y;
            let bottom1 = this.y + (this.height);
            let left2 = platform.x;
            let right2 = platform.x + (platform.width);
            let top2 = platform.y;
            let bottom2 = platform.y + (platform.height);

            if (((bottom1 < bottom2 + Math.abs(this.yspeed * dt)) && (bottom1 >= top2) && (right1 > left2) && (left1 < right2))) {
                if (this.platformDown) {
                    this.yspeed = playerYSpeed;
                } else {
                    this.yspeed = (top2 - bottom1) / dt / 2;
                    this.grounded = true;
                    this.wallTimeout = false;
                }
            }
        });
    }

    move(data: ClientData) {
        if (data.left && this.x > 0) {
            this.xspeed = -playerXSpeed;
        } else {
            this.xspeed = 0;
        }
        if (data.up && this.y > 0 && this.grounded) {
            this.yspeed = -playerYSpeed;
        } else if (data.up && this.y > 0 && this.wall) {
            this.yspeed = -playerYSpeed;
            this.wallTimeout = true;
        }
        if (data.right && this.x < 1000 - this.width) {
            this.xspeed = playerXSpeed;
        }
        if (data.down && this.y < 500 - this.height) {
            this.platformDown = true;
        } else {
            this.platformDown = false;
        }
    }

    collide(other: Graphic) {

        let left1 = this.x;
        let right1 = this.x + (this.width);
        let top1 = this.y;
        let bottom1 = this.y + (this.height);
        let left2 = other.x;
        let right2 = other.x + (other.width);
        let top2 = other.y;
        let bottom2 = other.y + (other.height);

        return top1 < bottom2 && bottom1 > top2 && left1 < right2 && right1 > left2;
    }

    tag(other: Player) {
        if (this.collide(other)) {
            if (this.status && !this.collision) {
                this.status = false;
                other.status = true;
                this.collision = true;
                other.collision = true;
            } else if (other.status && !other.collision) {
                other.status = false;
                this.status = true;
                other.collision = true;
                this.collision = true;
            }
        } else if (this.collision && other.collision) {
            this.collision = false;
            other.collision = false;
        }
    }

    scoring(playerCount: number) {
        if (this.status && playerCount > 1 && !this.dead && !this.win) {
            this.score--;
            if (this.score == 0) {
                this.dead = true;
                this.status = false;
                return true;
            }
        }
    }

    toData() {
        let color = this.status ? "red" : "green";
        return {
            x: this.x,
            y: this.y,
            color: color,
            num: this.num,
            score: this.score,
            dead: this.dead,
            win: this.win
        }
    }
}

class Platform extends Graphic {

    constructor(x, y, width, height) {
        super();
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
}

class PowerUp extends Graphic {

    private yspeed: number;

    constructor() {
        super();
        this.x = _.random(100, width - 100);
        this.y = -powerUpSize;
        this.width = powerUpSize;
        this.height = powerUpSize;
        this.yspeed = 0.0;
    }

    update(dt: number) {
        this.yspeed += gravity * dt / 4;
        this.y += this.yspeed * dt;
    }
}

platforms.push(new Platform(100, 400, platformWidth, platformHeight));
platforms.push(new Platform(600, 400, platformWidth, platformHeight));
platforms.push(new Platform(350, 300, platformWidth, platformHeight));
platforms.push(new Platform(100, 200, platformWidth, platformHeight));
platforms.push(new Platform(600, 200, platformWidth, platformHeight));