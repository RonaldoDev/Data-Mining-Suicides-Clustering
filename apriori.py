from apyori import apriori  
import pandas

def main():
    dataframe = initialize_dataframe()
    dataframe = treat_null_data(dataframe)
    dataframe = remove_columns(dataframe)
    association_results = execute_apriori(dataframe)
    print_result(association_results)

def initialize_dataframe():
    FILENAME = 'suicides.csv'
    return pandas.read_csv('csv/' + FILENAME)


def remove_columns(dataframe):
    return dataframe.drop(columns=['ano', 'municipio', 'raca', 'microregiao', 'motivo', 'local_ocorrencia'])


def treat_null_data(dataframe):
    return dataframe[
        (dataframe['sexo'] != 'NI') & 
        (dataframe['estado_civil'] != 'NI') & 
        (dataframe['escolaridade'] != 'NI') & 
        (dataframe['IDH'] != 'NI') & 
        (dataframe['mesoregiao'] != 'NI') &
        (dataframe['local_cid'] != 'NI') & 
        (dataframe['sexo'] != 'NI') &
        (dataframe['Profissao'] != 'NI') &
        (dataframe['Faixa_etaria'] != 'NI') &
        (dataframe['motivo'] != 'NI') &
        (dataframe['local_ocorrencia'] != 'NI') 
    ]


def execute_apriori(dataframe):
    records = []  
    for i in range(0, 2200):  
        records.append([str(dataframe.values[i,j]) for j in range(0, 8)])

    association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=4)  
    return list(association_rules) 


def print_result(association_results):
    for item in association_results:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0] 
        items = [x for x in pair]
        if (len(items) > 2):
            print("Rule: " + ', '.join([item_rule for item_rule in items[:-1]]) + " -> " + items[-1])

            #second index of the inner list
            print("Support: " + str(item[1]))

            #third index of the list located at 0th
            #of the third index of the inner list

            print("Confidence: " + str(item[2][0][2]))
            print("Lift: " + str(item[2][0][3]))
            print("=====================================")


if __name__ == '__main__':
    main()