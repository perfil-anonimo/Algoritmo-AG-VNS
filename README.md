
## Visão Geral do Projeto

Este repositório contém o código-fonte e o conjunto de dados utilizados na pesquisa que propõe e valida uma solução para o problema de roteamento de veículos focado na coleta fracionada de resíduos eletrônicos. A abordagem desenvolvida é um **Algoritmo Genético Híbrido com Busca em Vizinhança Variável (AG-VNS)**, que demonstrou ser uma ferramenta robusta e escalável, superando um solver exato (IBM CPLEX) em instâncias de maior complexidade.

## Estrutura do Repositório

* **`cvrp_solver.cpp`**: O núcleo da solução. Contém a implementação em C++ do algoritmo AG-VNS e a integração com a biblioteca IBM ILOG CPLEX para fins de comparação.
* **`main.py`**: O script principal em Python, responsável por orquestrar os experimentos, ler os dados da instância, executar o solver C++ e processar os resultados.
* **`instancias_cvrp_organizado.csv`**: O arquivo de dados unificado contendo todas as 10 instâncias de teste utilizadas nos experimentos.
* **`README.md`**: Este arquivo, contendo a documentação do projeto.

## O Conjunto de Dados

O arquivo **`instancias_cvrp_organizado.csv`** centraliza todas as instâncias geradas. Ele foi formatado com ponto e vírgula (`;`) como separador para compatibilidade com softwares como o Microsoft Excel em configurações regionais (Brasil).

#### Estrutura das Colunas:

| Nome da Coluna | Descrição                                                                      |
| :------------- | :----------------------------------------------------------------------------- |
| `instancia`    | O número de clientes na instância (ex: 10, 20, ..., 100).                        |
| `id_local`     | Identificador único do local. `0` é sempre o depósito.                           |
| `tipo`         | "Depósito" ou "Cliente", para facilitar a identificação.                         |
| `demanda`      | A quantidade a ser coletada no local. A demanda do depósito é sempre `0`.      |
| `coord_x`      | A coordenada do local no eixo X.                                               |
| `coord_y`      | A coordenada do local no eixo Y.                                               |

#### Parâmetros de Geração:
* **Espaço de Coordenadas:** Clientes distribuídos em um plano cartesiano de `[0, 100] x [0, 100]`.
* **Localização do Depósito:** Ponto central em `(50, 50)`.
* **Demanda por Cliente:** Um número inteiro sorteado aleatoriamente no intervalo `[5, 20]`.
* **Capacidade do Veículo:** Fixada em `50` unidades.
* **Semente Aleatória (Seed):** `42`, para garantir a reprodutibilidade exata dos dados.

## Como Executar os Experimentos

### Pré-requisitos
1.  **Compilador C++:** Um compilador moderno como g++ ou Clang.
2.  **IBM ILOG CPLEX:** O `cvrp_solver.cpp` depende das bibliotecas do CPLEX. Certifique-se de que o CPLEX esteja instalado e que as variáveis de ambiente estejam configuradas corretamente.
3.  **Python 3:** Com as seguintes bibliotecas instaladas:

    ```bash
    pip install pandas
    ```

### Passos para Execução

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/perfil-anonimo/Algoritmo-AG-VNS.git
    cd cvrp-e-waste-instances
    ```

2.  **Compile o solver C++:**
    Navegue até o diretório do projeto e execute o comando de compilação. Este comando pode variar dependendo do seu sistema e da localização da sua instalação do CPLEX.
    ```bash
    # Exemplo para Linux/macOS. Adapte os caminhos para sua instalação.
    g++ -o cvrp_solver cvrp_solver.cpp -I/path/to/cplex/include -L/path/to/cplex/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lm -lpthread
    ```

3.  **Execute o script principal:**
    O `main.py` provavelmente lê o arquivo CSV e chama o executável `cvrp_solver` para cada instância.
    ```bash
    python main.py
    ```
