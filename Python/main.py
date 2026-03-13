import random
import numpy as np
import matplotlib.pyplot as plt
import time
from docplex.mp.model import Model
import sys

# --- 1. GERAÇÃO DA INSTÂNCIA DO PROBLEMA ---
def gerar_instancia(num_clientes, capacidade_veiculo, seed=None):
    """Gera uma instância aleatória e reproduzível para o CVRP."""
    if seed is not None:
        random.seed(seed)

    depot = (50, 50)
    clientes_coords = {0: depot}
    demandas = {0: 0}

    # Gera clientes aleatórios
    for i in range(1, num_clientes + 1):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        clientes_coords[i] = (x, y)
        demandas[i] = random.randint(1, 10)

    # Calcula a matriz de distâncias euclidianas
    num_nos = num_clientes + 1
    matriz_distancias = np.zeros((num_nos, num_nos))
    for i in range(num_nos):
        for j in range(num_nos):
            dist = np.hypot(clientes_coords[i][0] - clientes_coords[j][0], clientes_coords[i][1] - clientes_coords[j][1])
            matriz_distancias[i, j] = dist

    return clientes_coords, demandas, matriz_distancias, capacidade_veiculo

# --- 2. SOLUÇÃO EXATA COM CPLEX ---
def resolver_cvrp_cplex(matriz_distancias, demandas, capacidade, num_veiculos_frota, time_limit=120):
    """Resolve o CVRP usando o CPLEX para obter a solução ótima ou a melhor possível no tempo limite."""
    print(f"\n--- Iniciando Solver CPLEX (Limite de Tempo: {time_limit}s) ---")
    start_time = time.time()

    num_nos = len(demandas)
    nos = list(range(num_nos))
    arcos = [(i, j) for i in nos for j in nos if i != j]

    try:
        mdl = Model('CVRP')

        # Variáveis de decisão
        x = mdl.binary_var_dict(arcos, name='x')
        u = mdl.continuous_var_dict([i for i in nos if i != 0], ub=capacidade, name='u')

        # Função Objetivo: Minimizar a distância total
        mdl.minimize(mdl.sum(matriz_distancias[i, j] * x[i, j] for i, j in arcos))

        # Restrições
        for j in nos:
            if j != 0:
                mdl.add_constraint(mdl.sum(x[i, j] for i in nos if i != j) == 1, ctname=f'in_{j}')

        for k in nos:
             mdl.add_constraint(mdl.sum(x[k,j] for j in nos if k!=j) - mdl.sum(x[i,k] for i in nos if i!=k) == 0)

        mdl.add_constraint(mdl.sum(x[0, j] for j in nos if j != 0) <= num_veiculos_frota)

        for i, j in arcos:
            if i != 0 and j != 0:
                mdl.add_indicator(x[i, j], u[i] + demandas[j] == u[j], name=f'subtour_{i}_{j}')

        for i in nos:
            if i != 0:
                mdl.add_constraint(u[i] >= demandas[i], ctname=f'demand_lb_{i}')
                mdl.add_constraint(u[i] <= capacidade, ctname=f'demand_ub_{i}')

        mdl.parameters.timelimit = time_limit
        solucao = mdl.solve(log_output=False) # log_output=False para um output mais limpo

        end_time = time.time()

        if solucao:
            distancia_total = solucao.get_objective_value()
            rotas = []
            arcos_ativos = [a for a in arcos if x[a].solution_value > 0.9]

            nos_partida = [j for i, j in arcos_ativos if i == 0]
            for start_node in nos_partida:
                rota_atual = [0, start_node]
                no_atual = start_node
                while no_atual != 0:
                    proximo_no = next((j for i, j in arcos_ativos if i == no_atual), 0)
                    rota_atual.append(proximo_no)
                    no_atual = proximo_no
                rotas.append(rota_atual)

            print(f"Solução do CPLEX encontrada em {end_time - start_time:.2f}s")
            return distancia_total, len(rotas), rotas, end_time - start_time
        else:
            print(f"Nenhuma solução encontrada pelo CPLEX em {time_limit}s.")
            return float('inf'), float('inf'), [], time_limit

    except Exception as e:
        print(f"Erro ao executar o CPLEX: {e}")
        return float('inf'), float('inf'), [], time_limit

# --- 3. SOLUÇÃO COM META-HEURÍSTICA AG-VNS ---
class AG_VNS_Solver:
    def __init__(self, matriz_distancias, demandas, capacidade, **kwargs):
        self.matriz_distancias = matriz_distancias
        self.demandas = demandas
        self.capacidade = capacidade
        self.num_clientes = len(demandas) - 1

        # Parâmetros do algoritmo
        self.pop_size = kwargs.get('pop_size', 100)
        self.generations = kwargs.get('generations', 500) # Aumentado para instâncias maiores
        self.elite_size = kwargs.get('elite_size', 10)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.penalty = 10000 # Penalidade alta por veículo extra

    def solve(self):
        print("\n--- Iniciando Meta-heurística AG-VNS ---")
        start_time = time.time()

        populacao = self._criar_populacao_inicial()
        melhor_solucao_global = min(populacao, key=lambda s: s['fitness'])

        for gen in range(self.generations):
            populacao = sorted(populacao, key=lambda s: s['fitness'])

            if populacao[0]['fitness'] < melhor_solucao_global['fitness']:
                melhor_solucao_global = populacao[0]

            nova_populacao = populacao[:self.elite_size]

            while len(nova_populacao) < self.pop_size:
                p1 = self._selecao_torneio(populacao)
                p2 = self._selecao_torneio(populacao)

                filho_cromossomo = self._crossover_ox(p1['cromossomo'], p2['cromossomo'])
                filho_cromossomo = self._mutacao_swap(filho_cromossomo)

                filho_solucao = self._avaliar_cromossomo(filho_cromossomo)

                # Hibridização: Aplica VNS (busca local) no novo filho
                filho_melhorado = self._vns_local_search(filho_solucao)

                nova_populacao.append(filho_melhorado)

            populacao = nova_populacao

            if gen % 100 == 0: # Imprime a cada 100 gerações
                print(f"Geração {gen}: Melhor Fitness = {melhor_solucao_global['fitness']:.2f} "
                      f"(Veículos: {melhor_solucao_global['num_veiculos']}, Dist: {melhor_solucao_global['distancia']:.2f})")

        end_time = time.time()
        print(f"\nSolução da Meta-heurística AG-VNS encontrada em {end_time - start_time:.2f}s")
        return melhor_solucao_global['distancia'], melhor_solucao_global['num_veiculos'], melhor_solucao_global['rotas'], end_time - start_time

    def _avaliar_cromossomo(self, cromossomo):
        rotas, distancia, num_veiculos = self._decodificar(cromossomo)
        fitness = (num_veiculos * self.penalty) + distancia
        return {'cromossomo': cromossomo, 'rotas': rotas, 'distancia': distancia, 'num_veiculos': num_veiculos, 'fitness': fitness}

    def _decodificar(self, cromossomo):
        rotas = []
        rota_atual = [0]
        carga_atual = 0
        dist_total = 0

        for cliente in cromossomo:
            if carga_atual + self.demandas[cliente] <= self.capacidade:
                rota_atual.append(cliente)
                carga_atual += self.demandas[cliente]
            else:
                rota_atual.append(0)
                rotas.append(rota_atual)
                dist_total += self._distancia_rota(rota_atual)
                rota_atual = [0, cliente]
                carga_atual = self.demandas[cliente]

        rota_atual.append(0)
        rotas.append(rota_atual)
        dist_total += self._distancia_rota(rota_atual)

        return rotas, dist_total, len(rotas)

    def _distancia_rota(self, rota):
        dist = 0
        for i in range(len(rota) - 1):
            dist += self.matriz_distancias[rota[i], rota[i+1]]
        return dist

    def _criar_populacao_inicial(self):
        pop = []
        clientes = list(range(1, self.num_clientes + 1))
        for _ in range(self.pop_size):
            cromossomo = random.sample(clientes, len(clientes))
            pop.append(self._avaliar_cromossomo(cromossomo))
        return pop

    def _selecao_torneio(self, populacao, k=3):
        return min(random.sample(populacao, k), key=lambda s: s['fitness'])

    def _crossover_ox(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        filho = [None] * size
        filho[a:b+1] = p1[a:b+1]

        p2_genes = [gene for gene in p2 if gene not in filho]

        idx_p2 = 0
        for i in range(size):
            if filho[i] is None:
                filho[i] = p2_genes[idx_p2]
                idx_p2 += 1
        return filho

    def _mutacao_swap(self, cromossomo):
        if random.random() < self.mutation_rate:
            a, b = random.sample(range(len(cromossomo)), 2)
            cromossomo[a], cromossomo[b] = cromossomo[b], cromossomo[a]
        return cromossomo

    def _vns_local_search(self, solucao):
        melhor_solucao = solucao
        for _ in range(3): # Tenta melhorar 3 vezes
            cromossomo_shaked = self._mutacao_swap(melhor_solucao['cromossomo'][:]) # Shake simples
            solucao_shaked = self._avaliar_cromossomo(cromossomo_shaked)
            solucao_melhorada = self._two_opt(solucao_shaked)
            if solucao_melhorada['fitness'] < melhor_solucao['fitness']:
                melhor_solucao = solucao_melhorada
        return melhor_solucao

    def _two_opt(self, solucao):
        melhor_solucao = solucao
        melhorou = True
        while melhorou:
            melhorou = False
            cromossomo = melhor_solucao['cromossomo']
            for i in range(len(cromossomo) - 1):
                for j in range(i + 2, len(cromossomo)):
                    novo_cromossomo = cromossomo[:i] + cromossomo[i:j][::-1] + cromossomo[j:]
                    nova_solucao = self._avaliar_cromossomo(novo_cromossomo)
                    if nova_solucao['fitness'] < melhor_solucao['fitness']:
                        melhor_solucao = nova_solucao
                        melhorou = True
                        break
                if melhorou:
                    break
        return melhor_solucao

# --- 4. VISUALIZAÇÃO ---
def visualizar_rotas(clientes_coords, rotas, titulo, num_clientes):
    plt.figure(figsize=(12, 12))
    for i, (x, y) in clientes_coords.items():
        if i == 0:
            plt.scatter(x, y, c='red', marker='*', s=300, zorder=5, label='Depósito')
        else:
            plt.scatter(x, y, c='blue', s=60, zorder=5)
            plt.text(x + 1.5, y + 1.5, str(i), fontsize=12)

    cores = plt.cm.gist_rainbow(np.linspace(0, 1, len(rotas)))
    for i, rota in enumerate(rotas):
        cor = cores[i]
        for j in range(len(rota) - 1):
            p1 = clientes_coords[rota[j]]
            p2 = clientes_coords[rota[j+1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=cor, linewidth=2.5, label=f'Veículo {i+1}' if j==0 else "")

    plt.title(titulo, fontsize=16)
    plt.xlabel("Coordenada X", fontsize=12)
    plt.ylabel("Coordenada Y", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Salva a figura com nome padronizado
    safe_title = titulo.replace(" ", "_").replace("-", "").lower()
    plt.savefig(f"rotas_{safe_title}.png")
    plt.show()

# --- 5. EXECUÇÃO E COMPARAÇÃO ---
if __name__ == "__main__":
    # Parâmetros dos Problemas
    # AUMENTADO PARA 10 INSTÂNCIAS
    PROBLEM_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    CAPACIDADE_VEICULO = 50
    NUM_VEICULOS_FROTA_MAX = 15 # Um limite superior para o CPLEX
    CPLEX_TIME_LIMIT = 180 # Limite de tempo em segundos para o CPLEX

    resultados_finais = []

    for num_clientes in PROBLEM_SIZES:
        print("\n" + "="*70)
        print(f"EXECUTANDO PARA {num_clientes} CLIENTES")
        print("="*70)

        clientes_coords, demandas, matriz_distancias, capacidade = gerar_instancia(num_clientes, CAPACIDADE_VEICULO, seed=42)

        # --- Resolve com CPLEX ---
        dist_cplex, veiculos_cplex, rotas_cplex, tempo_cplex = resolver_cvrp_cplex(matriz_distancias, demandas, capacidade, NUM_VEICULOS_FROTA_MAX, time_limit=CPLEX_TIME_LIMIT)
        # Visualiza apenas se for uma instância pequena para não poluir a saída
        if rotas_cplex and num_clientes <= 30:
            visualizar_rotas(clientes_coords, rotas_cplex, f"CPLEX - {num_clientes} Clientes", num_clientes)

        # --- Resolve com a Meta-heurística AG-VNS ---
        solver_ag_vns = AG_VNS_Solver(matriz_distancias, demandas, capacidade, pop_size=100, generations=500)
        dist_agvns, veiculos_agvns, rotas_agvns, tempo_agvns = solver_ag_vns.solve()
        if rotas_agvns and num_clientes <= 30:
             visualizar_rotas(clientes_coords, rotas_agvns, f"AG-VNS - {num_clientes} Clientes", num_clientes)

        resultados_finais.append({
            "Clientes": num_clientes,
            "Dist CPLEX": dist_cplex,
            "Veículos CPLEX": veiculos_cplex,
            "Tempo CPLEX (s)": tempo_cplex,
            "Dist AG-VNS": dist_agvns,
            "Veículos AG-VNS": veiculos_agvns,
            "Tempo AG-VNS (s)": tempo_agvns
        })

    # --- Tabela de Resumo Final ---
    print("\n\n" + "="*80)
    print("RESUMO COMPARATIVO FINAL")
    print("="*80)
    print(f"| {'Clientes':<10} | {'Dist. CPLEX':<15} | {'Dist. AG-VNS':<15} | {'Tempo CPLEX (s)':<18} | {'Tempo AG-VNS (s)':<18} |")
    print("-"*80)
    for res in resultados_finais:
        print(f"| {res['Clientes']:<10} | {res['Dist CPLEX']:<15.2f} | {res['Dist AG-VNS']:<15.2f} | {res['Tempo CPLEX (s)']:<18.2f} | {res['Tempo AG-VNS (s)']:<18.2f} |")
    print("-"*80)

    # --- Gráfico de Performance Final ---
    clientes = [r['Clientes'] for r in resultados_finais]
    tempos_cplex = [r['Tempo CPLEX (s)'] for r in resultados_finais]
    tempos_agvns = [r['Tempo AG-VNS (s)'] for r in resultados_finais]

    plt.figure(figsize=(12, 7))
    plt.plot(clientes, tempos_cplex, marker='o', linestyle='-', label='CPLEX (Exato)')
    plt.plot(clientes, tempos_agvns, marker='s', linestyle='--', label='AG-VNS (Meta-heurística)')
    plt.title('Comparação de Tempo de Execução vs. Tamanho do Problema', fontsize=16)
    plt.xlabel('Número de Clientes', fontsize=12)
    plt.ylabel('Tempo de Execução (segundos)', fontsize=12)
    plt.xticks(clientes)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.savefig("grafico_performance_tempo_10_instancias.png")
    plt.show()
