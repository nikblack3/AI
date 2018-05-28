# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    from util import Stack
    from util import Queue
    from game import Directions
    startstate = problem.getStartState()
    open = Stack()
    start = {"path": [startstate], "actions": []}
    open.push(start)
    while not open.isEmpty():
        dic = open.pop()
        path = dic["path"]
        if problem.isGoalState(path[-1]) is True:
            return dic["actions"]
        for x in problem.getSuccessors(path[-1]):
            if not x[0] in path:
                new_path, new_actions = dic["path"][:], dic["actions"][:]
                new_path.append(x[0])
                new_actions.append(x[1])
                open.push({"path": new_path, "actions": new_actions})
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    startstate = problem.getStartState()
    open = Queue()
    start = {"path of states": [startstate], "actions": []}
    open.push(start)
    visited = []
    count = 0
    while not open.isEmpty():
        dic = open.pop()
        path = dic["path of states"]
        if problem.isGoalState(path[-1]) is True:
            return dic["actions"]
        for s in path:
            if not s in visited:
                visited.append(s)
        for x in problem.getSuccessors(path[-1]):
            if not x[0] in visited:
                new_path, new_actions = dic["path of states"][:], dic["actions"][:]
                new_path.append(x[0])
                new_actions.append(x[1])
                open.push({"path of states": new_path, "actions": new_actions})
                visited.append(x[0])
                count += 1


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    startstate = problem.getStartState()
    open = PriorityQueue()
    start = {"path": [startstate], "actions": []}
    open.push(start,problem.getCostOfActions(start['actions']))
    visited = {startstate : 0}
    while not open.isEmpty():
        dic = open.pop()
        path = dic["path"]
        if problem.isGoalState(path[-1]) is True:
            return dic["actions"]
        for x in problem.getSuccessors(path[-1]):
            new_actions = dic["actions"][:]
            new_actions.append(x[1])
            new_cost = problem.getCostOfActions(new_actions)
            if not x[0] in visited or new_cost < visited[x[0]]:
                new_path, new_actions = dic["path"][:], dic["actions"][:]
                new_path.append(x[0])
                new_actions.append(x[1])
                open.update({"path": new_path, "actions": new_actions},problem.getCostOfActions(new_actions))
                #print "what's generated:" ,{"path": new_path, "actions": new_actions}
                visited[x[0]] = new_cost


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    from util import PriorityQueue
    from searchAgents import CornersProblem
    startstate = problem.getStartState()
    if type(startstate) != dict:
        startstate = problem.getStartState()
        open = PriorityQueue()
        start = {"path": [startstate], "actions": []}
        open.push(start, heuristic(startstate, problem))
        visited = {startstate:heuristic(startstate, problem)}
        while not open.isEmpty():
            dic = open.pop()
            path = dic["path"]
            if problem.isGoalState(path[-1]) is True:
                return dic["actions"]
            for x in problem.getSuccessors(path[-1]):
                new_actions = dic["actions"][:]
                new_actions.append(x[1])
                new_cost = problem.getCostOfActions(new_actions) + heuristic(x[0], problem)
                if x[0] not in visited or new_cost < visited[x[0]]:
                    new_path, new_actions = dic["path"][:], dic["actions"][:]
                    new_path.append(x[0])
                    new_actions.append(x[1])
                    open.push({"path": new_path, "actions": new_actions}, new_cost)
                    visited[x[0]] = new_cost



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
