#include <ilcplex/ilocplex.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;

// ==================== ESTRUTURAS DE DADOS ====================

struct CVRPInstance {
    int num_clientes;
    int capacidade_veiculo;
    vector<pair<double, double>> coordenadas;
    vector<int> demandas;
    vector<vector<double>> matriz_distancias;
};

struct CVRPSolution {
    double distancia_total;
    int num_veiculos;
    vector<vector<int>> rotas;
    double tempo_execucao;
    bool encontrada;

    CVRPSolution() : distancia_total(INFINITY), num_veiculos(0),
        tempo_execucao(0.0), encontrada(false) {}
};

struct Solucao {
    vector<int> cromossomo;
    vector<vector<int>> rotas;
    double distancia;
    int num_veiculos;
    double fitness;
};

// ==================== GERAÇÃO DA INSTÂNCIA ====================

CVRPInstance gerarInstancia(int num_clientes, int capacidade_veiculo, int seed = 42) {
    CVRPInstance inst;
    inst.num_clientes = num_clientes;
    inst.capacidade_veiculo = capacidade_veiculo;

    mt19937 gen(seed);
    uniform_int_distribution<> dis_coord(0, 100);
    uniform_int_distribution<> dis_demand(1, 10);

    // Depósito na posição (50, 50)
    inst.coordenadas.push_back({ 50.0, 50.0 });
    inst.demandas.push_back(0);

    // Gerar clientes aleatórios
    for (int i = 1; i <= num_clientes; i++) {
        double x = dis_coord(gen);
        double y = dis_coord(gen);
        inst.coordenadas.push_back({ x, y });
        inst.demandas.push_back(dis_demand(gen));
    }

    // Calcular matriz de distâncias euclidianas
    int n = num_clientes + 1;
    inst.matriz_distancias.resize(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dx = inst.coordenadas[i].first - inst.coordenadas[j].first;
            double dy = inst.coordenadas[i].second - inst.coordenadas[j].second;
            inst.matriz_distancias[i][j] = sqrt(dx * dx + dy * dy);
        }
    }

    return inst;
}

// ==================== SOLUÇÃO COM CPLEX ====================

CVRPSolution resolverCVRP_CPLEX(const CVRPInstance& inst, int num_veiculos_max, double time_limit = 120.0) {
    cout << "\n--- Iniciando Solver CPLEX (Limite de Tempo: " << time_limit << "s) ---" << endl;

    auto start_time = chrono::high_resolution_clock::now();
    CVRPSolution sol;

    int num_nos = inst.num_clientes + 1;

    try {
        IloEnv env;

        try {
            IloModel model(env);

            // Variáveis de decisão x[i][j] - binária se arco (i,j) é usado
            IloArray<IloNumVarArray> x(env, num_nos);
            for (int i = 0; i < num_nos; i++) {
                x[i] = IloNumVarArray(env, num_nos, 0, 1, ILOBOOL);
            }

            // Variáveis u[i] para eliminação de subtours (MTZ)
            IloNumVarArray u(env, num_nos);
            for (int i = 1; i < num_nos; i++) {
                u[i] = IloNumVar(env, inst.demandas[i], inst.capacidade_veiculo, ILOFLOAT);
            }

            // Função objetivo: minimizar distância total
            IloExpr objExpr(env);
            for (int i = 0; i < num_nos; i++) {
                for (int j = 0; j < num_nos; j++) {
                    if (i != j) {
                        objExpr += inst.matriz_distancias[i][j] * x[i][j];
                    }
                }
            }
            model.add(IloMinimize(env, objExpr));
            objExpr.end();

            // Restrição 1: Cada cliente deve ser visitado exatamente uma vez
            for (int j = 1; j < num_nos; j++) {
                IloExpr inFlow(env);
                for (int i = 0; i < num_nos; i++) {
                    if (i != j) {
                        inFlow += x[i][j];
                    }
                }
                model.add(inFlow == 1);
                inFlow.end();
            }

            // Restrição 2: Conservação de fluxo
            for (int k = 0; k < num_nos; k++) {
                IloExpr outFlow(env);
                IloExpr inFlow(env);

                for (int j = 0; j < num_nos; j++) {
                    if (k != j) outFlow += x[k][j];
                }
                for (int i = 0; i < num_nos; i++) {
                    if (i != k) inFlow += x[i][k];
                }

                model.add(outFlow - inFlow == 0);
                outFlow.end();
                inFlow.end();
            }

            // Restrição 3: Limite de veículos
            IloExpr numVeiculos(env);
            for (int j = 1; j < num_nos; j++) {
                numVeiculos += x[0][j];
            }
            model.add(numVeiculos <= num_veiculos_max);
            numVeiculos.end();

            // Restrição 4: Eliminação de subtours (MTZ com capacidade)
            for (int i = 1; i < num_nos; i++) {
                for (int j = 1; j < num_nos; j++) {
                    if (i != j) {
                        model.add(u[i] + inst.demandas[j] - u[j] <=
                            inst.capacidade_veiculo * (1 - x[i][j]));
                    }
                }
            }

            // Criar objeto CPLEX
            IloCplex cplex(model);

            // Configurar parâmetros
            cplex.setParam(IloCplex::Param::TimeLimit, time_limit);
            cplex.setParam(IloCplex::Param::Threads, 1);
            cplex.setOut(env.getNullStream());

            // Resolver
            if (cplex.solve()) {
                auto end_time = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = end_time - start_time;

                sol.encontrada = true;
                sol.distancia_total = cplex.getObjValue();
                sol.tempo_execucao = elapsed.count();

                // Extrair rotas
                vector<vector<int>> arcos_ativos;
                for (int i = 0; i < num_nos; i++) {
                    for (int j = 0; j < num_nos; j++) {
                        if (i != j && cplex.getValue(x[i][j]) > 0.5) {
                            arcos_ativos.push_back({ i, j });
                        }
                    }
                }

                // Reconstruir rotas
                for (int j = 1; j < num_nos; j++) {
                    bool is_start = false;
                    for (const auto& arco : arcos_ativos) {
                        if (arco[0] == 0 && arco[1] == j) {
                            is_start = true;
                            break;
                        }
                    }

                    if (is_start) {
                        vector<int> rota = { 0, j };
                        int no_atual = j;

                        while (no_atual != 0) {
                            int proximo = -1;
                            for (const auto& arco : arcos_ativos) {
                                if (arco[0] == no_atual) {
                                    proximo = arco[1];
                                    break;
                                }
                            }
                            if (proximo == -1) break;
                            rota.push_back(proximo);
                            no_atual = proximo;
                        }
                        sol.rotas.push_back(rota);
                    }
                }

                sol.num_veiculos = sol.rotas.size();

                cout << "Solucao do CPLEX encontrada em " << fixed << setprecision(2)
                    << sol.tempo_execucao << "s" << endl;

            }
            else {
                auto end_time = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = end_time - start_time;
                sol.tempo_execucao = elapsed.count();
                cout << "Nenhuma solucao encontrada pelo CPLEX em " << time_limit << "s." << endl;
            }

            cplex.end();
        }
        catch (IloException& e) {
            cerr << "Erro CPLEX: " << e << endl;
        }

        env.end();

    }
    catch (exception& e) {
        cerr << "Erro: " << e.what() << endl;
    }

    return sol;
}

// ==================== CLASSE AG-VNS SOLVER ====================

class AG_VNS_Solver {
private:
    const CVRPInstance& inst;
    int pop_size;
    int generations;
    int elite_size;
    double mutation_rate;
    double penalty;
    mt19937 gen;

    double distanciaRota(const vector<int>& rota) {
        double dist = 0.0;
        for (size_t i = 0; i < rota.size() - 1; i++) {
            dist += inst.matriz_distancias[rota[i]][rota[i + 1]];
        }
        return dist;
    }

    void decodificar(const vector<int>& cromossomo, vector<vector<int>>& rotas,
        double& dist_total, int& num_veiculos) {
        rotas.clear();
        vector<int> rota_atual = { 0 };
        int carga_atual = 0;
        dist_total = 0.0;

        for (int cliente : cromossomo) {
            if (carga_atual + inst.demandas[cliente] <= inst.capacidade_veiculo) {
                rota_atual.push_back(cliente);
                carga_atual += inst.demandas[cliente];
            }
            else {
                rota_atual.push_back(0);
                rotas.push_back(rota_atual);
                dist_total += distanciaRota(rota_atual);
                rota_atual = { 0, cliente };
                carga_atual = inst.demandas[cliente];
            }
        }

        rota_atual.push_back(0);
        rotas.push_back(rota_atual);
        dist_total += distanciaRota(rota_atual);
        num_veiculos = rotas.size();
    }

    Solucao avaliarCromossomo(const vector<int>& cromossomo) {
        Solucao sol;
        sol.cromossomo = cromossomo;
        decodificar(cromossomo, sol.rotas, sol.distancia, sol.num_veiculos);
        sol.fitness = (sol.num_veiculos * penalty) + sol.distancia;
        return sol;
    }

    vector<Solucao> criarPopulacaoInicial() {
        vector<Solucao> populacao;
        vector<int> clientes(inst.num_clientes);
        for (int i = 0; i < inst.num_clientes; i++) {
            clientes[i] = i + 1;
        }

        for (int i = 0; i < pop_size; i++) {
            shuffle(clientes.begin(), clientes.end(), gen);
            populacao.push_back(avaliarCromossomo(clientes));
        }

        return populacao;
    }

    Solucao selecaoTorneio(const vector<Solucao>& populacao, int k = 3) {
        uniform_int_distribution<> dis(0, populacao.size() - 1);
        Solucao melhor = populacao[dis(gen)];

        for (int i = 1; i < k; i++) {
            Solucao candidato = populacao[dis(gen)];
            if (candidato.fitness < melhor.fitness) {
                melhor = candidato;
            }
        }
        return melhor;
    }

    vector<int> crossoverOX(const vector<int>& p1, const vector<int>& p2) {
        int size = p1.size();
        uniform_int_distribution<> dis(0, size - 1);

        int a = dis(gen);
        int b = dis(gen);
        if (a > b) swap(a, b);

        vector<int> filho(size, -1);
        vector<bool> presente(inst.num_clientes + 1, false);

        for (int i = a; i <= b; i++) {
            filho[i] = p1[i];
            presente[p1[i]] = true;
        }

        int idx_p2 = 0;
        for (int i = 0; i < size; i++) {
            if (filho[i] == -1) {
                while (presente[p2[idx_p2]]) {
                    idx_p2++;
                }
                filho[i] = p2[idx_p2];
                presente[p2[idx_p2]] = true;
                idx_p2++;
            }
        }

        return filho;
    }

    vector<int> mutacaoSwap(vector<int> cromossomo) {
        uniform_real_distribution<> dis_prob(0.0, 1.0);

        if (dis_prob(gen) < mutation_rate) {
            uniform_int_distribution<> dis_idx(0, cromossomo.size() - 1);
            int a = dis_idx(gen);
            int b = dis_idx(gen);
            swap(cromossomo[a], cromossomo[b]);
        }

        return cromossomo;
    }

    Solucao twoOpt(const Solucao& sol) {
        Solucao melhor_sol = sol;
        bool melhorou = true;

        while (melhorou) {
            melhorou = false;
            vector<int> cromossomo = melhor_sol.cromossomo;

            for (size_t i = 0; i < cromossomo.size() - 1 && !melhorou; i++) {
                for (size_t j = i + 2; j < cromossomo.size() && !melhorou; j++) {
                    vector<int> novo_cromossomo = cromossomo;
                    reverse(novo_cromossomo.begin() + i, novo_cromossomo.begin() + j + 1);

                    Solucao nova_sol = avaliarCromossomo(novo_cromossomo);

                    if (nova_sol.fitness < melhor_sol.fitness) {
                        melhor_sol = nova_sol;
                        melhorou = true;
                    }
                }
            }
        }

        return melhor_sol;
    }

    Solucao vnsLocalSearch(const Solucao& sol) {
        Solucao melhor_sol = sol;

        for (int tentativa = 0; tentativa < 3; tentativa++) {
            vector<int> cromossomo_shaked = mutacaoSwap(melhor_sol.cromossomo);
            Solucao sol_shaked = avaliarCromossomo(cromossomo_shaked);
            Solucao sol_melhorada = twoOpt(sol_shaked);

            if (sol_melhorada.fitness < melhor_sol.fitness) {
                melhor_sol = sol_melhorada;
            }
        }

        return melhor_sol;
    }

public:
    AG_VNS_Solver(const CVRPInstance& instance, int pop_sz = 100, int gens = 500,
        int elite_sz = 10, double mut_rate = 0.1)
        : inst(instance), pop_size(pop_sz), generations(gens),
        elite_size(elite_sz), mutation_rate(mut_rate), penalty(10000.0) {
        random_device rd;
        gen = mt19937(rd());
    }

    CVRPSolution solve() {
        cout << "\n--- Iniciando Meta-heuristica AG-VNS ---" << endl;
        auto start_time = chrono::high_resolution_clock::now();

        vector<Solucao> populacao = criarPopulacaoInicial();

        Solucao melhor_solucao_global = *min_element(populacao.begin(), populacao.end(),
            [](const Solucao& a, const Solucao& b) { return a.fitness < b.fitness; });

        for (int gen_atual = 0; gen_atual < generations; gen_atual++) {
            sort(populacao.begin(), populacao.end(),
                [](const Solucao& a, const Solucao& b) { return a.fitness < b.fitness; });

            if (populacao[0].fitness < melhor_solucao_global.fitness) {
                melhor_solucao_global = populacao[0];
            }

            vector<Solucao> nova_populacao;
            for (int i = 0; i < elite_size; i++) {
                nova_populacao.push_back(populacao[i]);
            }

            while (nova_populacao.size() < (size_t)pop_size) {
                Solucao p1 = selecaoTorneio(populacao);
                Solucao p2 = selecaoTorneio(populacao);

                vector<int> filho_cromossomo = crossoverOX(p1.cromossomo, p2.cromossomo);
                filho_cromossomo = mutacaoSwap(filho_cromossomo);

                Solucao filho_sol = avaliarCromossomo(filho_cromossomo);
                Solucao filho_melhorado = vnsLocalSearch(filho_sol);

                nova_populacao.push_back(filho_melhorado);
            }

            populacao = nova_populacao;

            if (gen_atual % 100 == 0) {
                cout << "Geracao " << gen_atual << ": Melhor Fitness = "
                    << fixed << setprecision(2) << melhor_solucao_global.fitness
                    << " (Veiculos: " << melhor_solucao_global.num_veiculos
                    << ", Dist: " << melhor_solucao_global.distancia << ")" << endl;
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;

        cout << "\nSolucao da Meta-heuristica AG-VNS encontrada em "
            << elapsed.count() << "s" << endl;

        CVRPSolution resultado;
        resultado.encontrada = true;
        resultado.distancia_total = melhor_solucao_global.distancia;
        resultado.num_veiculos = melhor_solucao_global.num_veiculos;
        resultado.rotas = melhor_solucao_global.rotas;
        resultado.tempo_execucao = elapsed.count();

        return resultado;
    }
};

// ==================== FUNÇÃO PRINCIPAL ====================

int main() {
    // Parâmetros dos problemas
    vector<int> PROBLEM_SIZES = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    int CAPACIDADE_VEICULO = 50;
    int NUM_VEICULOS_FROTA_MAX = 15;
    double CPLEX_TIME_LIMIT = 180.0;

    // Estrutura para armazenar resultados
    struct Resultado {
        int clientes;
        double dist_cplex;
        int veiculos_cplex;
        double tempo_cplex;
        double dist_agvns;
        int veiculos_agvns;
        double tempo_agvns;
    };

    vector<Resultado> resultados_finais;

    // Arquivo CSV de resultados
    ofstream arquivo_csv("resultados_comparativo.csv");
    arquivo_csv << "Clientes,Dist_CPLEX,Veiculos_CPLEX,Tempo_CPLEX_s,"
        << "Dist_AG_VNS,Veiculos_AG_VNS,Tempo_AG_VNS_s" << endl;

    for (int num_clientes : PROBLEM_SIZES) {
        cout << "\n" << string(70, '=') << endl;
        cout << "EXECUTANDO PARA " << num_clientes << " CLIENTES" << endl;
        cout << string(70, '=') << endl;

        // Gerar instância
        CVRPInstance inst = gerarInstancia(num_clientes, CAPACIDADE_VEICULO, 42);

        // Resolver com CPLEX
        CVRPSolution sol_cplex = resolverCVRP_CPLEX(inst, NUM_VEICULOS_FROTA_MAX, CPLEX_TIME_LIMIT);

        // Resolver com AG-VNS
        AG_VNS_Solver solver_agvns(inst, 100, 500, 10, 0.1);
        CVRPSolution sol_agvns = solver_agvns.solve();

        // Armazenar resultados
        Resultado res;
        res.clientes = num_clientes;
        res.dist_cplex = sol_cplex.encontrada ? sol_cplex.distancia_total : INFINITY;
        res.veiculos_cplex = sol_cplex.encontrada ? sol_cplex.num_veiculos : 0;
        res.tempo_cplex = sol_cplex.tempo_execucao;
        res.dist_agvns = sol_agvns.distancia_total;
        res.veiculos_agvns = sol_agvns.num_veiculos;
        res.tempo_agvns = sol_agvns.tempo_execucao;

        resultados_finais.push_back(res);

        // Escrever no CSV
        arquivo_csv << num_clientes << ","
            << fixed << setprecision(2) << res.dist_cplex << ","
            << res.veiculos_cplex << ","
            << res.tempo_cplex << ","
            << res.dist_agvns << ","
            << res.veiculos_agvns << ","
            << res.tempo_agvns << endl;
    }

    arquivo_csv.close();

    // Tabela de resumo final
    cout << "\n\n" << string(120, '=') << endl;
    cout << "RESUMO COMPARATIVO FINAL" << endl;
    cout << string(120, '=') << endl;
    cout << left << setw(12) << "Clientes"
        << setw(18) << "Dist. CPLEX"
        << setw(18) << "Dist. AG-VNS"
        << setw(20) << "Tempo CPLEX (s)"
        << setw(20) << "Tempo AG-VNS (s)" << endl;
    cout << string(120, '-') << endl;

    for (const auto& res : resultados_finais) {
        cout << left << setw(12) << res.clientes
            << setw(18) << fixed << setprecision(2) << res.dist_cplex
            << setw(18) << res.dist_agvns
            << setw(20) << res.tempo_cplex
            << setw(20) << res.tempo_agvns << endl;
    }
    cout << string(120, '-') << endl;

    cout << "\nResultados salvos em 'resultados_comparativo.csv'" << endl;
    cout << "\nPressione Enter para sair..." << endl;
    cin.get();

    return 0;
}
