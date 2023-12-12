from planningPlus import *
from logic import *
from utils import *
from search import *

def sokoban(puzzle):
    
    # valid pos = not out of bounds
    def valid_pos(i, j):
        return (0 <= i < n and 0 <= j < len(grid[0]))

    def is_wall(i, j):
        return valid_pos(i, j) and grid[i][j] in ['#', ' ']
    
    def is_objetivo(i, j):
        return valid_pos(i, j) and grid[i][j] in ['o', '+', '*']

    def linha_fronteira(i):
        if i == 1 or i == n - 2:
            for j in range(len(grid[i])):
                if is_objetivo(i, j):
                    return True
            return False
        return True
    
    def coluna_fronteira(j):
        if j == 1 or j == len(grid[0]) - 2:
            for i in range(n):
                if is_objetivo(i, j):
                    return True
            return False
        return True

    def corner_check(i, j, action):
        if action == 'MoverCimaCaixa':
            return is_objetivo(i-2, j) or not (is_wall(i-3, j) and (is_wall(i-2, j-1) or is_wall(i-2, j+1)))
        elif action == 'MoverBaixoCaixa':
            return is_objetivo(i+2, j) or not (is_wall(i+3, j) and (is_wall(i+2, j-1) or is_wall(i+2, j+1)))
        elif action == 'MoverEsquerdaCaixa':
            return is_objetivo(i, j-2) or not (is_wall(i, j-3) and (is_wall(i-1, j-2) or is_wall(i+1, j-2)))
        elif action == 'MoverDireitaCaixa':
            return is_objetivo(i, j+2) or not (is_wall(i, j+3) and (is_wall(i-1, j+2) or is_wall(i+1, j+2)))
        
    def optimize(i, j, action):
        if action == 'MoverCimaCaixa':
            return linha_fronteira(i - 2) and corner_check(i, j, action)
        elif action == 'MoverBaixoCaixa':
            return linha_fronteira(i + 2) and corner_check(i, j, action)
        elif action == 'MoverEsquerdaCaixa':
            return coluna_fronteira(j - 2) and corner_check(i, j, action)
        elif action == 'MoverDireitaCaixa':
            return coluna_fronteira(j + 2) and corner_check(i, j, action)

    # Converte a a string "puzzle" em uma matriz de chars
    m = puzzle.split('\n')[:-1]
    n = len(m)
    grid = []
    for i in range(n):
        grid.append([obj for obj in m[i]])
    
    estado = []                 # Estado inicial
    objetivo = []               # Objetivo
    acoes = []                  # Acoes
    for i in range(n):
        for j in range(len(grid[i])):
            # print(grid[i][j])
            if grid[i][j] == '.':
                estado.append(expr('~Sobre(Sokoban, L{}C{})'.format(i, j)))
                estado.append(expr('~Sobre(Caixa, L{}C{})'.format(i, j)))
            elif grid[i][j] == 'o':
                estado.append(expr('~Sobre(Sokoban, L{}C{})'.format(i, j)))
                estado.append(expr('~Sobre(Caixa, L{}C{})'.format(i, j)))
                objetivo.append(expr('Sobre(Caixa, L{}C{})'.format(i, j)))
            elif grid[i][j] == '@':
                estado.append(expr('Sobre(Sokoban, L{}C{})'.format(i, j)))
                estado.append(expr('~Sobre(Caixa, L{}C{})'.format(i, j)))
            elif grid[i][j] == '+':
                estado.append(expr('Sobre(Sokoban, L{}C{})'.format(i, j)))
                estado.append(expr('~Sobre(Caixa, L{}C{})'.format(i, j)))
                objetivo.append(expr('Sobre(Caixa, L{}C{})'.format(i, j)))
            elif grid[i][j] == '@':
                estado.append(expr('Sobre(Sokoban, L{}C{})'.format(i, j)))
                estado.append(expr('~Sobre(Caixa, L{}C{})'.format(i, j)))
            elif grid[i][j] == '$':
                estado.append(expr('~Sobre(Sokoban, L{}C{})'.format(i, j)))
                estado.append(expr('Sobre(Caixa, L{}C{})'.format(i, j)))
            elif grid[i][j] == '*':
                estado.append(expr('~Sobre(Sokoban, L{}C{})'.format(i, j)))
                estado.append(expr('Sobre(Caixa, L{}C{})'.format(i, j)))
                objetivo.append(expr('Sobre(Caixa, L{}C{})'.format(i, j)))
                
            curr_acoes_p = []
            curr_acoes_b = []
            
            for dx, dy, action in [(-1, 0, 'MoverCima'), (1, 0, 'MoverBaixo'), (0, -1, 'MoverEsquerda'), (0, 1, 'MoverDireita')]:
                if (not is_wall(i, j) and not is_wall(i + dx, j + dy)):
                    acao = Action(expr(action + f'(L{i}C{j})'),
                                  precond=f'Sobre(Sokoban, L{i}C{j}) & ~Sobre(Caixa, L{i + dx}C{j + dy})',
                                  effect=f'~Sobre(Sokoban, L{i}C{j}) & Sobre(Sokoban, L{i + dx}C{j + dy})')
                    acoes.append(acao)
                    curr_acoes_p.append(acao)

            for dx, dy, action in [(-1, 0, 'MoverCimaCaixa'), (1, 0, 'MoverBaixoCaixa'), (0, -1, 'MoverEsquerdaCaixa'), (0, 1, 'MoverDireitaCaixa')]:
                if (not is_wall(i, j) and not is_wall(i + dx, j + dy) and not is_wall(i + 2*dx, j + 2*dy) and optimize(i, j, action)):
                    acao = Action(expr(action + f'(L{i}C{j})'),
                                  precond=f'Sobre(Sokoban, L{i}C{j}) & Sobre(Caixa, L{i + dx}C{j + dy}) & ~Sobre(Caixa, L{i + 2*dx}C{j + 2*dy})',
                                  effect=f'~Sobre(Sokoban, L{i}C{j}) & Sobre(Sokoban, L{i + dx}C{j + dy}) & ~Sobre(Caixa, L{i + dx}C{j + dy}) & Sobre(Caixa, L{i + 2*dx}C{j + 2*dy})')
                    acoes.append(acao)
                    curr_acoes_b.append(acao)
                    
            # print(f"L{i}C{j} = {grid[i][j]}")
            # print(f"ACOES SOKOBAN = {curr_acoes_p}")
            # print(f"ACOES CAIXA = {curr_acoes_b}")
            # input()

    pp = PlanningProblem(initial=estado, goals=objetivo, actions=[], domain=[])

    fp = ForwardPlan(pp)
    # fp = ForwardPlan(pp, True, True)
    fp.expanded_actions = acoes

    return fp





linha1= "  ##### \n"
linha2= "###...# \n"
linha3= "#o@$..# \n"
linha4= "###.$o# \n"
linha5= "#o##..# \n"
linha6= "#.#...##\n"
linha7= "#$.*.$o#\n"
linha8= "#......#\n"
linha9= "########\n"
grelha=linha1+linha2+linha3+linha4+linha5+linha6+linha7+linha8+linha9
try:
    p=sokoban(grelha)
    travel_sol = breadth_first_graph_search_plus(p)
    if travel_sol:
        print('Solução em',len(travel_sol.solution()),'passos')
    else:
        print('No way!')
except Exception as e:
    print(repr(e))