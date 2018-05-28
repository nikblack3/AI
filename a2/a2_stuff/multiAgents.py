# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # print "now ghost is in:", gameState.getGhostPosition(1)
        # print "start choosing"
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best, choose the besy along some options

        "Add more of your code here if you want to"
        # print "go from",gameState.getPacmanPosition(),"make a move:", legalMoves[chosenIndex]

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        oldFood = currentGameState.getFood()
        old_food_list = oldFood.asList()
        distance_list = []
        for x in old_food_list:
            distance_list.append(manhattanDistance(x,newPos))
        bonus = 0
        if newPos in old_food_list:
            bonus = 10000
        possible = []
        for ghost in successorGameState.getGhostPositions():
            possible.append(ghost)
            possible.append((ghost[0],ghost[1] + 1))
            possible.append((ghost[0], ghost[1] - 1))
            possible.append((ghost[0] + 1, ghost[1]))
            possible.append((ghost[0] - 1, ghost[1]))
        if newPos in possible:
            return -float("inf")
        else:
            return -(min(distance_list)) + bonus


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.minimax(gameState,0,0)[0]

    def minimax(self,gameState,index,depth):
        best_move = None
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return best_move, self.evaluationFunction(gameState)
        if index == gameState.getNumAgents() - 1:
            depth += 1
        if index == 0:
            # if it's max
            value = -float("inf")
        if index != 0:
            # if it's min
            value = float("inf")
        for action in gameState.getLegalActions(index):
            next_pos = gameState.generateSuccessor(index, action)
            next_move, next_val = self.minimax(next_pos,(index + 1) % gameState.getNumAgents(),depth)
            if index == 0 and value < next_val:
                value, best_move = next_val, action
            elif index != 0 and value > next_val:
                value, best_move = next_val, action
        return best_move, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(-float("inf"),float("inf"),gameState,0,0)[0]


    def alphabeta(self,alpha,beta,gameState,index,depth):
        best_move = None
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return best_move, self.evaluationFunction(gameState)
        if index == gameState.getNumAgents() - 1:
            depth += 1
        if index == 0:
            # if it's max
            value = -float("inf")
        if index != 0:
            # if it's min
            value = float("inf")
        for action in gameState.getLegalActions(index):
            next_pos = gameState.generateSuccessor(index, action)
            next_move, next_val = self.alphabeta(alpha,beta,next_pos,(index + 1) % gameState.getNumAgents(),depth)
            if index == 0:
                if value < next_val:
                    best_move, value = action, next_val
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            elif index != 0:
                if value > next_val:
                    best_move, value = action, next_val
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)
        return best_move, value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState,0,0)[0]

    def expectimax(self,gameState,index,depth):
        best_move = None
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return best_move, self.evaluationFunction(gameState)
        if index == gameState.getNumAgents() - 1:
            depth += 1
        if index == 0:
            # if it's max
            value = -float("inf")
        if index != 0:
            # if it's min
            value = 0
        for action in gameState.getLegalActions(index):
            next_pos = gameState.generateSuccessor(index, action)
            next_move, next_val = self.expectimax(next_pos,(index + 1) % gameState.getNumAgents(),depth)
            if index == 0 and value < next_val:
                value, best_move = next_val, action
            elif index != 0:
                value +=  float((float(1) / float(len(gameState.getLegalActions(index)))) * float(next_val))
        return best_move, value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    PacPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    food_list = Food.asList()
    distance_ghost = 0
    distance_ghost_list = []
    for ghost in GhostStates:
        distance_ghost += manhattanDistance(PacPos, ghost.getPosition())
        distance_ghost_list.append(manhattanDistance(PacPos, ghost.getPosition()))
    distance_food_sum = 0
    distance_food_list = []
    for food in food_list:
        distance_food_sum += manhattanDistance(PacPos, food)
        distance_food_list.append(manhattanDistance(PacPos, food))
    if len(food_list) != 0:
        return distance_ghost - len(food_list) - float(1 / 4) * min(distance_food_list)
    else:
        return distance_ghost - len(food_list)
    # PacPos = currentGameState.getPacmanPosition()
    # Food = currentGameState.getFood()
    # GhostStates = currentGameState.getGhostStates()
    # distance_ghost = 0
    # distance_ghost_list = []
    # for ghost in GhostStates:
    #     distance_ghost += manhattanDistance(PacPos, ghost.getPosition())
    #     distance_ghost_list.append(manhattanDistance(PacPos, ghost.getPosition()))
    # for distance in distance_ghost_list:
    #     if distance < 3:
    #         return -float("inf")
    # # for action in currentGameState.getLegalActions(0):
    # #     next_pos = currentGameState.generateSuccessor(0, action)
    # #     if next_pos.getPacmanPosition() in Food.asList():
    # #         return float("inf")
    # print "search start"
    # cost = breadthFirstSearch(currentGameState)
    # print "cost is:", cost
    # print "search is done"
    # print "function's value is:", -cost - 100 * len(Food.asList())
    # return -cost - 100 * len(Food.asList())
    #

def breadthFirstSearch(gameState):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    open = Queue()
    start = {"path of states": [gameState], "actions": [], "food?": ""}
    open.push(start)
    current_food = gameState.getFood()
    current_food_list = current_food.asList()
    if len(current_food_list) == 0:
        return -100000
    while not open.isEmpty():
        dic = open.pop()
        if dic["food?"] == "Yes":
            print dic["actions"]
            return len(dic["actions"])
        path = dic["path of states"]
        for action in path[-1].getLegalActions(0):
            next_pos = path[-1].generateSuccessor(0, action)
            if next_pos not in path:
                new_path, new_actions = dic["path of states"][:], dic["actions"][:]
                new_path.append(next_pos)
                new_actions.append(action)
                indicator = "No"
                if next_pos.getPacmanPosition() in current_food_list:
                    indicator = "Yes"
                open.push({"path of states": new_path, "actions": new_actions, "food?": indicator})
    return -1000000





# Abbreviation
better = betterEvaluationFunction

