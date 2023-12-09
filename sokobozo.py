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
    estado.append(expr('Sobre(Sokoban, ({}, {}))'.format(sx, sy)))
    for i in range(len(caixas)):
        l, c = caixas[i][0], caixas[i][1]
        estado.append(expr('Caixa(L{},C{})'.format(l, c)))

    objetivo = []               # Objetivo
    for i in range(len(objetivos)):
        l, c = objetivos[i][0], objetivos[i][1]
        objetivo.append(expr('Objetivo(L{},C{})'.format(l, c)))
    
    acoes = [Action('Move(Sokoban, x, y)',
                precond='Sobre(Sokoban, x) & Livre(b)',
                effect='Sobre(b, Mesa) & Livre(x) & ~Sobre(b, x)',
                domain='Bloco(b) & Bloco(x)')]

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