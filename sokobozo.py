from planningPlus import *
from logic import *
from utils import *
from search import *

def sokoban(puzzle):

    # Converte a grelha em uma matriz de objetos
    m = puzzle.split('\n')[:-1]
    n = len(m)
    grid = []
    print(m)
    for i in range(n):
        grid.append([obj for obj in m[i]])

    # Obtém a posição do sokoban
    sx, sy = None, None
    for i in range(n):
        for j in range(len(grid[i])):
            if grid[i][j] in ['@', '+']:
                sx, sy = i, j
                break

    # Obtém as posições das caixas
    caixas = []
    for i in range(n):
        for j in range(len(grid[i])):
            if grid[i][j] in ['$', '*']:
                caixas.append((i, j))

    # Obtém as posições dos objectivos
    objetivos = []
    for i in range(n):
        for j in range(len(grid[i])):
            if grid[i][j] in ['o', '+', '*']:
                objetivos.append((i, j))

    # Constrói o estado inicial
    estado = []
    for i in range(n):
        estado.append([expr('No(s)') if grid[i][j] == '@' else expr('Livre({})'.format(grid[i][j])) for j in range(len(grid[i]))])

    # Constrói o objetivo
    objetivo = []
    for i in range(n):
        objetivo.append(expr('No({})'.format(grid[i][j])) for j in range(len(grid[i])))

    # Constrói a instância do problema
    p = PlanningProblem(initial=estado, goals=objetivo, actions=[], domain=[])

    # Gera as ações
    for caixa in caixas:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= caixa[0] + dx < n and 0 <= caixa[1] + dy < len(grid[0]) and grid[caixa[0] + dx][caixa[1] + dy] == '.':
                # Ação para mover a caixa para cima
                a = Action('Move(C,{},{})'.format(caixa[0], caixa[1]),
                             precond=expr('No({}) & No(C) & No(C,{},{})'.format(caixa[0] + dx, caixa[1] + dy, caixa[0], caixa[1])),
                             effect=expr('No(C,{}) & No({}) & No({}) & No(C) & Sobre(C,{},{})'.format(caixa[0], caixa[1], caixa[0] + dx, caixa[1] + dy, caixa[0] + dx, caixa[1] + dy)))
                p.add_action(a)

    # Guarda as ações expandidas
    p.expanded_actions = p.actions

    return p





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