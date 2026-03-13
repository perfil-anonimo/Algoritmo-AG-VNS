import os
import pandas as pd

def converter_e_organizar_csv():
    """
    Lê os arquivos .txt, une os dados em um DataFrame,
    e salva em um único arquivo CSV com colunas renomeadas e reordenadas.
    """
    arquivos_txt = [f for f in os.listdir('.') if f.startswith('instancia_') and f.endswith('.txt')]
    
    if not arquivos_txt:
        print("Nenhum arquivo 'instancia_XX.txt' encontrado no diretório.")
        return

    print("Arquivos encontrados. Processando...")
    
    lista_de_dados = []

    for filename in arquivos_txt:
        try:
            num_clientes_instancia = int(filename.split('_')[1].split('.')[0])
            
            with open(filename, 'r') as f:
                lines = f.readlines()
                
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        id_val = int(parts[0])
                        x_val = float(parts[1])
                        y_val = float(parts[2])
                        demanda_val = int(parts[3])
                        tipo = 'Depósito' if id_val == 0 else 'Cliente'
                        
                        lista_de_dados.append([num_clientes_instancia, id_val, tipo, x_val, y_val, demanda_val])
        except Exception as e:
            print(f"Erro ao processar o arquivo {filename}: {e}")

    if not lista_de_dados:
        print("Nenhum dado foi extraído. O arquivo CSV não será gerado.")
        return

    # Cria o DataFrame com nomes de colunas temporários
    df_temp = pd.DataFrame(lista_de_dados, columns=['instancia', 'id_local', 'tipo', 'coord_x', 'coord_y', 'demanda'])
    
    # Define a ordem final e correta das colunas
    ordem_final_colunas = ['instancia', 'id_local', 'tipo', 'demanda', 'coord_x', 'coord_y']
    
    # Reordena o DataFrame
    df_organizado = df_temp[ordem_final_colunas]
    
    # Ordena as linhas por instância e depois por ID
    df_final = df_organizado.sort_values(by=['instancia', 'id_local']).reset_index(drop=True)
    
    # Salva no novo arquivo CSV
    output_filename = 'instancias_cvrp_organizado.csv'
    df_final.to_csv(output_filename, index=False)
    
    print(f"\nArquivo '{output_filename}' gerado com sucesso!")
    print("Abaixo, uma prévia do arquivo com a nova organização:")
    print(df_final.head())

if __name__ == "__main__":
    converter_e_organizar_csv()