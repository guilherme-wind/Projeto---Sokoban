from planningPlus import *
from logic import *
from utils import *
from search import *

def sokoban(puzzle):

    # Converte a a string "puzzle" em uma matriz de chars
    m = puzzle.split('\n')[:-1]
    n = len(m)
    grid = []
    for i in range(n):
        grid.append([obj for obj in m[i]])

    sx, sy = None, None         # Posição do sokoban
    caixas = []                 # Posições das caixas
    objetivos = []              # Posições dos objetivos
    for i in range(n):
        for j in range(len(grid[i])):
            if grid[i][j] in ['@', '+']:
                sx, sy = i, j
            if grid[i][j] in ['$', '*']:
                caixas.append((i, j))
            if grid[i][j] in ['o', '+', '*']:
                objetivos.append((i, j))

    estado = []                 # Estado inicial
    estado.append(expr('Sokoban({})'.format((sx, sy))))
    for i in range(len(caixas)):
        estado.append(expr('Caixa({})'.format(caixas[i])))

    objetivo = []               # Objetivo
    for i in range(len(objetivos)):
        objetivo.append(expr('Caixa({})'.format(objetivos[i])))

    acoes = []
    # Gera as ações
    for caixa in caixas:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= caixa[0] + dx < n and 0 <= caixa[1] + dy < len(grid[0]) and grid[caixa[0] + dx][caixa[1] + dy] != '#':
                a = Action('Move(C,{},{})'.format(caixa[0] + dx, caixa[1] + dy),
                             precond=expr('Caixa({})'.format((caixa[0], caixa[1]))),
                             effect=expr('Caixa({})'.format((caixa[0] + dx, caixa[1] + dy))))
                acoes.append(a)

    # Constrói a instância do problema
    pp = PlanningProblem(initial=estado, goals=objetivo, actions=[], domain=[])         # parece-me bem até aqui
    # as acoes é que estao provavelmente mal, nao sei como fazer
    fp = ForwardPlan(pp)     # isto crasha o programa, provavelmente por causa das acoes
    # fp = ForwardPlan(pp, True, True)
    fp.expanded_actions = acoes

    return fp





linha1= "##########\n"
linha2= "#........#\n"
linha3= "#..$..+..#\n"
linha4= "#........#\n"
linha5= "##########\n"
grelha=linha1+linha2+linha3+linha4+linha5
try:
    p=sokoban(grelha)
    travel_sol = breadth_first_graph_search_plus(p)
    if travel_sol:
        print('Solução em',len(travel_sol.solution()),'passos')
    else:
        print('No way!')
except Exception as e:
    print(repr(e))