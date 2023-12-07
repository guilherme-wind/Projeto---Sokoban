from planningPlus import *
from logic import *
from utils import *
from search import *

def planning(puzzle):
    m = puzzle.split('\n')[:-1]
    n = len(m)
    grid = []
    for i in range(n):
        grid.append([obj for obj in m[i]])

    sx, sy = None, None         # Posição do sokoban
    objetivos = []              # Posições dos objetivos
    for i in range(n):
        for j in range(len(grid[i])):
            if grid[i][j] == '@':
                sx, sy = i, j
            elif grid[i][j] == 'o':
                objetivos.append((i, j))

    estado = []
    for i in range(n):
        for j in range(len(grid[i])):
            if grid[i][j] == '@':
                estado.append(expr('Sokoban({})'.format((i, j))))
            elif grid[i][j] == '.':
                estado.append(expr('Livre({})'.format((i, j))))
            elif grid[i][j] == 'o':
                estado.append(expr('Objetivo({})'.format((i, j))))
    print(estado)

    goal = []
    
    pp = PlanningProblem(initial=estado, goals=goal, actions=[], domain=[])

    return 






linha1 = "@..\n"
linha2 = "..o\n"
grelha=linha1+linha2
try:
    p=planning(grelha)
    travel_sol = breadth_first_graph_search_plus(p)
    if travel_sol:
        print('Solução em',len(travel_sol.solution()),'passos')
    else:
        print('No way!')
except Exception as e:
    print(repr(e))

"""
estado = [
    expr('Sobre(D1,D2)'),
    expr('Sobre(D2,D3)'),
    expr('Sobre(D3,A)'),
    expr('Livre(D1)'),
    expr('Livre(B)'),
    expr('Livre(C)')
]

dominio = 'Disco(D1) & Disco(D2) & Disco(D3) & Pino(A) & Pino(B) & Pino(C) & Menor(D1,D2) & ' + \
            'Menor(D2,D3) & Menor(D1,D3) & Menor(D1,A) & Menor(D1,B) & Menor(D1,C) & ' + \
            'Menor(D2,A) & Menor(D2,B) & Menor(D2,C) & Menor(D3,A) & Menor(D3,B) & Menor(D3,C)'

goal = 'Sobre(D3,C) & Sobre(D2,D3) & Sobre(D1,D2)'

acoes = [Action('Move(d,x,y)', 
               precond='Livre(d) & Sobre(d,x) & Livre(y)',
               effect='Sobre(d,y) & Livre(x) & ~Sobre(d,x) & ~Livre(y)',
               domain='Disco(d) & Menor(d,x) & Menor(d,y)')]

th = PlanningProblem(initial=estado, goals=goal, actions=acoes, domain=dominio)
p = ForwardPlan(th)

try:
    travel_sol = breadth_first_graph_search_plus(p)
    if travel_sol:
        print('Solução em',len(travel_sol.solution()),'passos')
        print(travel_sol.solution())
    else:
        print('No way!')
except Exception as e:
    print(repr(e))
"""
